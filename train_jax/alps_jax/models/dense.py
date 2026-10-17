# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Dense correspondence objective of Stage 1 (Flax / JAX), as in the original code.

The base view x and a crop + flip g(x) of it are both encoded; a projector maps the tokens to 256 dimensions; for each
patch of g(x) the positive is the bilinear sample of x's features at the same image location, and the negatives are
other locations of the same image (a (2R+1)^2 ring around the positive excluded, optionally subsampled to a fixed
number by Gumbel top-k) in one softmax and 64 locations of other images in a second softmax; L = w_same L_same +
w_cross L_cross with temperature 0.1. Coordinates are normalized (y, x) in [0, 1] with patch centres at (i + 0.5) / g.
"""
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp


class DenseProjector(nn.Module):
    dim: int = 256
    dtype: Any = jnp.bfloat16
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, z):
        h = nn.Dense(self.dim, dtype=self.dtype, param_dtype=self.param_dtype)(z)
        return nn.LayerNorm(use_scale=False, use_bias=False, dtype=jnp.float32)(h)


def patch_center_grid(g):
    c = (jnp.arange(g) + 0.5) / g
    yy, xx = jnp.meshgrid(c, c, indexing="ij")
    return jnp.stack([yy, xx], -1)


def warp_coords(crop_box, flip, g):
    """Centre of every g(x) patch in the base view's normalized frame. crop_box [B, 4] = (y0, x0, h, w)."""
    cy, cx, ch, cw = [crop_box[:, k][:, None, None] for k in range(4)]
    centers = patch_center_grid(g)[None]
    py, px = centers[..., 0], centers[..., 1]
    px = jnp.where(flip[:, None, None], 1.0 - px, px)
    return jnp.stack([cy + py * ch, cx + px * cw], -1)


def bilinear_sample(feat, coords):
    """feat [B, g, g, C], coords [B, g, g, 2] normalized (y, x); align_corners=False, edge clamp."""
    b_, g, _, _ = feat.shape
    fy = coords[..., 0] * g - 0.5
    fx = coords[..., 1] * g - 0.5
    y0 = jnp.floor(fy).astype(jnp.int32); x0 = jnp.floor(fx).astype(jnp.int32)
    y1 = y0 + 1; x1 = x0 + 1
    wy = fy - y0; wx = fx - x0
    clamp = lambda a: jnp.clip(a, 0, g - 1)
    y0c, y1c, x0c, x1c = clamp(y0), clamp(y1), clamp(x0), clamp(x1)
    b = jnp.arange(b_)[:, None, None]
    gather = lambda yy, xx: feat[b, yy, xx]
    top = gather(y0c, x0c) * (1 - wx)[..., None] + gather(y0c, x1c) * wx[..., None]
    bot = gather(y1c, x0c) * (1 - wx)[..., None] + gather(y1c, x1c) * wx[..., None]
    return top * (1 - wy)[..., None] + bot * wy[..., None]


def _l2n(x, eps=1e-6):
    return x / jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def patch_infonce_loss(h_x, h_gx, crop_box, flip, rng, *, temperature=0.1, n_cross=64, exclude_radius=1,
                       n_same=None, split_weights=(0.75, 0.25)):
    """Returns (loss, diagnostics). h_x, h_gx: [B, g, g, D] projected features of x and g(x)."""
    h_x = h_x.astype(jnp.float32); h_gx = h_gx.astype(jnp.float32)
    B, g, _, D = h_x.shape
    N = g * g
    coords = warp_coords(crop_box, flip, g)
    pos = bilinear_sample(h_x, coords)
    q = _l2n(h_gx).reshape(B, N, D)
    p = _l2n(pos).reshape(B, N, D)
    k = _l2n(h_x).reshape(B, N, D)
    HI = jax.lax.Precision.HIGHEST
    pos_logit = (q * p).sum(-1)[..., None]
    same = jnp.einsum("bnd,bmd->bnm", q, k, precision=HI)
    m0 = jnp.round(coords[..., 0] * g - 0.5).reshape(B, N)[:, :, None, None]
    n0 = jnp.round(coords[..., 1] * g - 0.5).reshape(B, N)[:, :, None, None]
    rows = jnp.arange(g)[None, None, :, None]
    cols = jnp.arange(g)[None, None, None, :]
    near = ((jnp.abs(rows - m0) <= exclude_radius) & (jnp.abs(cols - n0) <= exclude_radius)).reshape(B, N, N)
    same_neg = jnp.where(near, -1e9, same)
    if n_same is not None and int(n_same) < N - 1:
        u = jax.random.uniform(jax.random.fold_in(rng, 101), same.shape, minval=1e-9, maxval=1.0)
        score = jnp.where(near, -jnp.inf, -jnp.log(-jnp.log(u)))
        _, sel = jax.lax.top_k(score, int(n_same))
        same_neg = jnp.take_along_axis(same, sel, axis=-1)
    flat = k.reshape(B * N, D)
    idx = jax.random.randint(rng, (n_cross,), 0, B * N)
    cross = jnp.einsum("bnd,kd->bnk", q, flat[idx], precision=HI)
    cross = jnp.where(((idx[None, :] // N) == jnp.arange(B)[:, None])[:, None, :], -1e9, cross)

    def nce(neg):
        lg = jnp.concatenate([pos_logit, neg], -1) / temperature
        return -jax.nn.log_softmax(lg, -1)[..., 0].mean()

    l_same, l_cross = nce(same_neg), nce(cross)
    loss = float(split_weights[0]) * l_same + float(split_weights[1]) * l_cross
    sg = jax.lax.stop_gradient
    return loss, {"pnce_same": sg(l_same), "pnce_cross": sg(l_cross)}
