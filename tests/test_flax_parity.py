# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch modules vs the Flax modules of the training code, on perturbed random weights.

Generate the reference first (JAX environment):
    JAX_PLATFORMS=cpu python tools/parity/flax_reference.py --code train_jax --out /tmp/alps_flax_ref.npz
then:  ALPS_FLAX_REF=/tmp/alps_flax_ref.npz pytest tests/test_flax_parity.py
"""
import os

import numpy as np
import pytest
import torch

from alps.convert import _dense, text_to_torch, vit_to_torch
from alps.text import TextEncoder
from alps.vit import VisionTransformer

REF = os.environ.get("ALPS_FLAX_REF", "")
pytestmark = pytest.mark.skipif(not os.path.exists(REF), reason="set ALPS_FLAX_REF to a reference npz")


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(REF))


def _sub(d, prefix):
    return {k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}


def _close(a, b, tol):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    err = np.abs(a - b).max() / (np.abs(b).max() + 1e-12)
    assert err < tol, f"max relative error {err:.2e} >= {tol:.0e}"


def test_visual_encoder(ref):
    depth = int(ref["depth"])
    enc = VisionTransformer(depth=depth, pos_mode="recompute").eval()
    sd = vit_to_torch(_sub(ref, "vit/"), dst="")
    enc.load_state_dict(sd, strict=True)
    with torch.no_grad():
        o = enc(torch.from_numpy(ref["x224"]).permute(0, 3, 1, 2))
        o_odd = enc(torch.from_numpy(ref["x_odd"]).permute(0, 3, 1, 2))
    _close(o["tokens"], ref["z"], 1e-5)
    _close(o["pooled"], ref["zg"], 1e-5)
    _close(o_odd["tokens"], ref["z_odd"], 1e-5)        # a 10 x 16 grid, sin-cos table recomputed as in JAX
    _close(o_odd["pooled"], ref["zg_odd"], 1e-5)


def test_text_encoder_and_heads(ref):
    depth, vocab = int(ref["depth"]), int(ref["vocab"])
    txt = TextEncoder(vocab_size=vocab, depth=depth).eval()
    txt.load_state_dict(text_to_torch(_sub(ref, "text/"), dst=""), strict=True)
    with torch.no_grad():
        pooled = txt(torch.from_numpy(ref["ids"]).long())
        w_t = _dense(_sub(ref, "txt_proj/"), "proj")
        w_i = _dense(_sub(ref, "img_proj/"), "proj")
        txt_emb = pooled @ w_t.T
        img_emb = torch.from_numpy(ref["z"]).mean(1) @ w_i.T
    _close(pooled, ref["t_pooled"], 1e-5)
    _close(txt_emb, ref["txt_emb"], 1e-5)
    _close(img_emb, ref["img_emb"], 1e-5)
