# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Helpers of the Flax <-> PyTorch parity tests (need jax + flax next to torch; skipped otherwise)."""
import re

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")
pytest.importorskip("alps_jax")

from alps.convert import flatten  # noqa: E402


def perturb(tree, seed=0, scale=0.02):
    """Move every parameter away from its init (LayerNorm 1 / 0, zero biases) by N(0, scale^2)."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    new = [a + scale * jax.random.normal(k, a.shape, a.dtype) for a, k in zip(leaves, keys)]
    return jax.tree_util.tree_unflatten(treedef, new)


def flat_np(tree):
    return {k: np.asarray(v, np.float32) for k, v in flatten(jax.device_get(tree)).items()}


def rel_err(a, b):
    """max |a - b| / max |b| (0 when both are zero)."""
    a = np.asarray(a.detach().cpu().float() if torch.is_tensor(a) else a, np.float64)
    b = np.asarray(b.detach().cpu().float() if torch.is_tensor(b) else b, np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    den = np.abs(b).max() if b.size else 0.0
    num = np.abs(a - b).max() if b.size else 0.0
    if den == 0.0:
        return float(num)
    return float(num / den)


def assert_close(a, b, tol, what=""):
    e = rel_err(a, b)
    assert e <= tol, f"{what}: relative error {e:.3e} > {tol:.1e}"
    return e


def _exact_zero_grad(key):
    """Gradients that are zero in exact arithmetic: attention key biases (softmax is invariant to q . b_k)."""
    return bool(re.search(r"(key|_k)/bias$", key))


def assert_flat_close(got, ref, tol, what="", floor=1e-3, grads=False):
    """Per-leaf error max|a - b| / max(max|b|, floor * max over all leaves of max|b|) of two flat dicts with the
    same keys (the floor keeps leaves with tiny values from being judged by their rounding noise alone). With
    grads=True, leaves whose exact gradient is zero (attention key biases) must be below 1e-6 of the global scale
    in both. Returns the worst error."""
    assert set(got) == set(ref), f"{what}: keys differ {sorted(set(got) ^ set(ref))[:6]}"
    scale = max(float(np.abs(np.asarray(v, np.float64)).max()) if np.size(v) else 0.0 for v in ref.values())
    worst, wk = 0.0, None
    for k in ref:
        a = np.asarray(got[k].detach().cpu().float() if torch.is_tensor(got[k]) else got[k], np.float64)
        b = np.asarray(ref[k], np.float64)
        assert a.shape == b.shape, (what, k, a.shape, b.shape)
        if not b.size:
            continue
        if grads and _exact_zero_grad(k):
            assert max(np.abs(a).max(), np.abs(b).max()) <= 1e-6 * scale, (what, k, np.abs(a).max(), np.abs(b).max())
            continue
        den = max(float(np.abs(b).max()), floor * scale)
        e = float(np.abs(a - b).max() / den) if den > 0 else float(np.abs(a - b).max())
        if e > worst:
            worst, wk = e, k
    assert worst <= tol, f"{what}: worst leaf {wk} relative error {worst:.3e} > {tol:.1e}"
    return worst


def torch_grads(module, names=None):
    """{name: grad} with zeros for parameters that received no gradient."""
    out = {}
    for n, p in module.named_parameters():
        if names is not None and n not in names:
            continue
        out[n] = p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
    return out


def to_torch(x):
    return torch.from_numpy(np.array(x, copy=True))


def nhwc_to_nchw(x):
    return to_torch(np.asarray(x, np.float32)).permute(0, 3, 1, 2).contiguous()
