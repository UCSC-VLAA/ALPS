# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the V-JEPA 2 position handling of the video benchmarks (video_encoders.VJEPA2Video), on a small randomly
initialized VJEPA2Model (transformers >= 4.53; skipped otherwise).

    python eval/video/tests/test_vjepa2_rope.py          (or pytest)

* rope="recompute" at the square grid of the configuration equals the unmodified HF encoder;
* the position ids of a non-square grid: integer (recompute) and scaled to the 16 x 16 pretraining grid (interp);
* the VideoEncoder wrapper (shapes, no input conversion).
"""
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common  # noqa: E402
from eval.video.video_encoders import VideoEncoder, VJEPA2Video  # noqa: E402


def _tiny_model():
    try:
        from transformers import VJEPA2Config, VJEPA2Model
    except ImportError:
        _common.skip("transformers with VJEPA2 (>= 4.53) not installed")
    cfg = VJEPA2Config(patch_size=16, crop_size=64, frames_per_clip=4, tubelet_size=2, hidden_size=96, num_attention_heads=2,
                       num_hidden_layers=2, mlp_ratio=2.0, pred_hidden_size=48, pred_num_attention_heads=2,
                       pred_num_hidden_layers=1, pred_num_mask_tokens=2, attn_implementation="sdpa")
    torch.manual_seed(0)
    return VJEPA2Model(cfg).eval()


def test_recompute_equals_stock_encoder_at_config_grid():
    m = _tiny_model()
    stock = copy.deepcopy(m.encoder)
    v = VJEPA2Video(model=copy.deepcopy(m), rope="recompute").freeze()
    x = torch.randn(2, 4, 3, 64, 64)
    with torch.no_grad():
        a = stock(pixel_values_videos=x).last_hidden_state
        b = v.forward_clip(x)
    assert b.shape == (2, 2, 16, 96) and torch.equal(a, b.flatten(1, 2))


def test_position_ids():
    m = _tiny_model()
    for rope in ("recompute", "interp"):
        v = VJEPA2Video(model=copy.deepcopy(m), rope=rope)
        v.set_grid(3, 5)
        f, h, w = v.attn_modules[0].get_position_ids(torch.zeros(1, 2 * 15, 96))
        ff, hh, ww = torch.arange(30) // 15, (torch.arange(30) % 15) // 5, torch.arange(30) % 5
        assert torch.equal(f, ff)
        if rope == "recompute":
            assert torch.equal(h, hh) and torch.equal(w, ww)
        else:
            assert torch.allclose(h, hh.float() * 16 / 3) and torch.allclose(w, ww.float() * 16 / 5)
            assert float(h.max()) < 16 and float(w.max()) < 16
    x = torch.randn(1, 4, 3, 48, 80)                 # the patched ids are the ones the forward pass uses
    with torch.no_grad():
        y = {r: VJEPA2Video(model=copy.deepcopy(m), rope=r).freeze().forward_clip(x) for r in ("recompute", "interp")}
        stock = copy.deepcopy(m.encoder)(pixel_values_videos=x).last_hidden_state
    assert not torch.allclose(y["recompute"], y["interp"]) and not torch.allclose(y["recompute"].flatten(1, 2), stock)


def test_video_encoder_wrapper():
    m = _tiny_model()
    venc = VideoEncoder(VJEPA2Video(model=copy.deepcopy(m), rope="interp").freeze())
    assert venc.input_mode == "identity" and venc.tubelet == 2 and venc.t_out(16) == 8
    x = torch.randn(1, 4, 3, 48, 80)
    with torch.no_grad():
        tok = venc.forward_clip(x)
        img = venc.forward_images(x[:, 0])
    assert tok.shape == (1, 2, 15, 96) and img.shape == (1, 15, 96)


if __name__ == "__main__":
    sys.exit(_common.run_all(dict(globals())))
