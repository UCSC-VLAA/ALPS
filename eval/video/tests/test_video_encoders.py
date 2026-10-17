# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of video_encoders.VideoEncoder: the conversion of ImageNet-normalized inputs to each encoder's convention
(bit for bit the arithmetic of the original video wrappers), and the clip interface of image encoders.

    pytest eval/video/tests/test_video_encoders.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common  # noqa: E402,F401
from eval import encoders as E  # noqa: E402
from eval.video.video_encoders import IMAGENET, VideoEncoder  # noqa: E402

STATS = {"clip": ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
         "siglip": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), "unit": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), "imagenet": IMAGENET}


class _Enc(E.FrozenEncoder):
    """Returns its input, flattened to tokens of one channel group, so the conversion can be read off the output."""

    def __init__(self, mean, std):
        super().__init__(mean, std)
        self.name, self.patch, self.dim = "x", 1, 3

    def forward(self, x):
        return x.flatten(2).transpose(1, 2)


def _x(seed=0):
    x = torch.rand(4, 3, 8, 12, generator=torch.Generator().manual_seed(seed))
    m0, s0 = torch.tensor(IMAGENET[0]).view(1, 3, 1, 1), torch.tensor(IMAGENET[1]).view(1, 3, 1, 1)
    xi = (x - m0) / s0
    xi[0, :, :2, :2] = torch.randn(3, 2, 2) * 3            # values outside the image range (random erasing in SSv2)
    return xi


def test_conversion_to_each_encoder_convention():
    x = _x()
    for name, (mean, std) in STATS.items():
        venc = VideoEncoder(_Enc(mean, std))
        got = venc.forward_images(x)
        if name == "imagenet":
            assert venc.input_mode == "identity"
            exp = x
        else:                                       # the original timm wrapper: x * ra + rb from the configured statistics
            assert venc.input_mode == "affine"
            m0, s0, m1, s1 = (torch.tensor(v) for v in (IMAGENET[0], IMAGENET[1], mean, std))
            exp = x * (s0 / s1).view(1, 3, 1, 1) + ((m0 - m1) / s1).view(1, 3, 1, 1)
        assert torch.equal(got, exp.flatten(2).transpose(1, 2)), name
    radio = E.RadioEncoder.__new__(E.RadioEncoder)  # C-RADIOv3 without its weights: back to [0, 1] with a clamp
    E.FrozenEncoder.__init__(radio, (0.0,) * 3, (1.0,) * 3)
    radio.name = "cradio_v3_b"
    venc = VideoEncoder(radio)
    m0, s0 = torch.tensor(IMAGENET[0]).view(1, 3, 1, 1), torch.tensor(IMAGENET[1]).view(1, 3, 1, 1)
    assert venc.input_mode == "unit_clamp" and torch.equal(venc.from_imagenet(x), (x * s0 + m0).clamp(0, 1))


def test_clip_interface_of_image_encoders():
    venc = VideoEncoder(_Enc(*STATS["clip"]))
    clip = torch.stack([_x(s) for s in range(3)], 1)       # [4, 3 frames, 3, 8, 12]
    tok = venc.forward_clip(clip)
    assert tok.shape == (4, 3, 96, 3) and venc.tubelet == 1 and venc.t_out(16) == 16
    for t in range(3):
        assert torch.equal(tok[:, t], venc.forward_images(clip[:, t]))


def test_alps_encoder_through_the_wrapper():
    from alps.vit import vit_b16
    enc = E.ALPSEncoder.__new__(E.ALPSEncoder)              # random weights instead of a checkpoint file
    E.FrozenEncoder.__init__(enc)
    torch.manual_seed(0)
    enc.name, enc.weights, enc.model = "alps", "random", vit_b16(pos_mode="interpolate").eval()
    venc = VideoEncoder(enc.freeze())
    assert venc.input_mode == "identity" and venc.patch == 16 and venc.dim == 768
    x = torch.randn(2, 3, 64, 96)
    with torch.no_grad():
        assert torch.equal(venc.forward_images(x), enc.model(x)["tokens"])
    assert venc.meta()["pos_mode"] == "interpolate" and np.isclose(venc.meta()["params_M"], 85.6)


def test_dinov2_row_is_the_timm_model():
    """DINOv2-B/14 of the video tables = eval.encoders "dinov2_b14_timm", created with the arguments of the original video
    registry and read with its token conventions (offline: timm's weights replaced by a random initialization)."""
    try:
        import timm
    except ImportError:
        _common.skip("timm not installed")
    from eval.video import video_encoders as V
    assert V.PAPER_VIDEO_ROWS["DINOv2-B/14"] == "dinov2_b14_timm" and "dinov2_b14" not in V.VIDEO_TIMM
    assert E.TIMM["dinov2_b14_timm"] == "vit_base_patch14_dinov2.lvd142m"
    calls, real = [], timm.create_model

    def create(name, pretrained=False, **kw):       # records the call, builds the same model without downloading weights
        calls.append((name, pretrained, kw))
        torch.manual_seed(0)
        return real(name, pretrained=False, **kw)

    timm.create_model = create
    try:
        venc = V.build_video_encoder("dinov2_b14_timm", device="cpu")
    finally:
        timm.create_model = real
    exp_call = ("vit_base_patch14_dinov2.lvd142m", True, {"num_classes": 0, "img_size": 224, "dynamic_img_size": True})
    assert calls[0] == exp_call, calls
    assert venc.input_mode == "identity" and (venc.patch, venc.dim, venc.enc.n_prefix) == (14, 768, 1)
    x = torch.randn(2, 3, 98, 154)                          # a non-square grid (7 x 11) like the DAVIS frames
    with torch.no_grad():
        tok = venc.forward_images(x)
        assert tok.shape == (2, 77, 768)
        assert torch.equal(tok, venc.enc.model.forward_features(x)[:, 1:])


if __name__ == "__main__":
    sys.exit(_common.run_all(dict(globals())))
