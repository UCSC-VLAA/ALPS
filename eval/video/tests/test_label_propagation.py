# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the DAVIS label propagation (davis.py).

    pytest eval/video/tests/test_label_propagation.py

* propagate() against a direct per-token implementation of DINO's rule;
* frame sizes, clip windows, and the interpolated ALPS position table;
* run_video end to end on a synthetic two-object clip, with an image encoder and a video encoder.
"""
import math
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common  # noqa: E402,F401
from eval.video import davis  # noqa: E402

CPU = torch.device("cpu")


def direct_propagate(feat_tar, feats_src, segs, radius, topk, temp):
    """One target token at a time, in float64: affinities to every context token, kept if within the radius and among
    the topk largest, normalized, then the weighted sum of the context soft masks."""
    n, c, h, w = segs.shape
    ft = F.normalize(feat_tar.double(), dim=1)
    src = feats_src.double()
    out = torch.zeros(c, h * w, dtype=torch.float64)
    for q in range(h * w):
        qi, qj = divmod(q, w)
        cand = []
        for k in range(n):
            for s in range(h * w):
                si, sj = divmod(s, w)
                inside = radius <= 0 or (abs(qi - si) <= radius and abs(qj - sj) <= radius)
                cand.append((math.exp(float(ft[q] @ src[k, :, s]) / temp) if inside else 0.0, k, s))
        thr = sorted((a for a, _, _ in cand), reverse=True)[topk - 1]
        keep = [(a, k, s) for a, k, s in cand if a >= thr]
        z = sum(a for a, _, _ in keep)
        for a, k, s in keep:
            out[:, q] += a / z * segs[k, :, s // w, s % w].double()
    return out.reshape(1, c, h, w)


def _random_case(seed, n=3, c=4, h=5, w=7, d=16):
    g = torch.Generator().manual_seed(seed)
    feat_tar = torch.randn(h * w, d, generator=g)
    src = F.normalize(torch.randn(n, h * w, d, generator=g), dim=2).transpose(1, 2).contiguous()
    segs = torch.rand(n, c, h, w, generator=g)
    return feat_tar, src, segs


def test_propagate_matches_direct_rule():
    for seed, (radius, topk) in enumerate([(2, 5), (1, 3), (12, 5), (0, 4)]):
        feat_tar, src, segs = _random_case(seed)
        h, w = segs.shape[-2:]
        nb = davis.neighborhood_mask(h, w, radius, CPU)
        got = davis.propagate(feat_tar, src, segs, nb, temp=0.1, topk=topk, radius=radius)
        ref = direct_propagate(feat_tar, src, segs, radius, topk, 0.1)
        assert torch.allclose(got.double(), ref, atol=1e-5, rtol=1e-5), (radius, topk, (got.double() - ref).abs().max())


def test_frame_sizes_and_windows():
    assert davis.target_size(480, 854, 16) == (480, 832)           # DAVIS 480p, patch 16: 30 x 52 tokens
    assert davis.target_size(480, 854, 14) == (476, 840)           # patch 14: 34 x 60 tokens
    assert davis.target_size(854, 480, 16) == (832, 480)
    assert davis.clip_window(0, 16) == [0] * 16
    assert davis.clip_window(3, 16) == [0] * 13 + [1, 2, 3]
    assert davis.clip_window(20, 16) == list(range(5, 21))


def _sincos_scaled(h, w, dim, temp=10000.0):
    """The sine-cosine table of the DAVIS / SSv2 runs at grids other than 14 x 14 (indices rescaled into 0..14)."""
    omega = np.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temp ** omega)
    y, x = np.meshgrid(np.arange(h) * 14.0 / h, np.arange(w) * 14.0 / w, indexing="ij")
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = np.concatenate([np.sin(x), np.cos(x), np.sin(y), np.cos(y)], axis=1)
    pe = np.concatenate([np.zeros((1, dim)), pe], axis=0)
    return torch.tensor(pe[None], dtype=torch.float32)


def test_alps_interpolated_positions():
    """The release encoder (pos_mode="interpolate") evaluates the table of the paper's runs at DAVIS grids."""
    from alps.vit import vit_b16
    ref = _sincos_scaled
    enc = vit_b16(pos_mode="interpolate")
    for gh, gw in [(30, 52), (52, 30), (30, 53), (16, 16), (20, 28)]:
        got = enc.position_table(gh, gw)
        exp = ref(gh, gw, 768)
        assert got.shape == exp.shape and torch.allclose(got, exp, atol=1e-6, rtol=0), (gh, gw, (got - exp).abs().max())
    # 14 x 14 (SSv2 at 224 px): the training table, computed in float32 as in the JAX code; the earlier port computed it
    # in float64 (differences below 1e-6)
    y, x = np.meshgrid(np.arange(14.0), np.arange(14.0), indexing="ij")
    omega = 1.0 / (10000.0 ** (np.arange(192) / 191))
    y, x = y.flatten()[:, None] * omega, x.flatten()[:, None] * omega
    exp = torch.tensor(np.concatenate([np.zeros((1, 768)), np.concatenate([np.sin(x), np.cos(x), np.sin(y), np.cos(y)], 1)])[None],
                       dtype=torch.float32)
    assert torch.allclose(enc.position_table(14, 14), exp, atol=1e-6, rtol=0)


class _ImageEnc(torch.nn.Module):
    """A fixed random patch embedding as a stand-in image encoder (ImageNet-normalized input)."""
    tubelet = 1

    def __init__(self, patch=16, dim=32):
        super().__init__()
        self.patch = patch
        self.w = torch.randn(dim, 3, patch, patch, generator=torch.Generator().manual_seed(1)) / patch

    def forward_images(self, x):
        return F.conv2d(x, self.w, stride=self.patch).flatten(2).transpose(1, 2)

    def forward_clip(self, x):
        b, t = x.shape[:2]
        tok = self.forward_images(x.flatten(0, 1))
        return tok.view(b, t, *tok.shape[1:])


class _VideoEnc(_ImageEnc):
    """A stand-in video encoder: tubelets of 2 frames, every token mixed with the clip mean (attention over the clip)."""
    tubelet = 2

    def forward_clip(self, x):
        b, t = x.shape[:2]
        tok = self.forward_images(x.flatten(0, 1)).view(b, t // 2, 2, -1, self.w.shape[0]).mean(2)
        return tok + 0.5 * tok.mean((1, 2), keepdim=True)


def _synthetic_davis(root, name="vid", n_frames=12, h=96, w=170, seed=0):
    import cv2
    from PIL import Image
    rng = np.random.RandomState(seed)
    os.makedirs(os.path.join(root, "JPEGImages", "480p", name))
    os.makedirs(os.path.join(root, "Annotations", "480p", name))
    yy, xx = np.mgrid[0:h, 0:w]
    for t in range(n_frames):                       # two moving blobs on a noisy background
        im = rng.randint(0, 60, (h, w, 3)).astype(np.uint8)
        for c, (cy, cx, r) in enumerate([(30 + t, 40 + 3 * t, 18), (60, 120 - 2 * t, 14)]):
            im[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = [200, 80 + 60 * c, 40]
        cv2.imwrite(os.path.join(root, "JPEGImages", "480p", name, f"{t:05d}.jpg"), im)
    seg = np.zeros((h, w), np.uint8)
    seg[(yy - 30) ** 2 + (xx - 40) ** 2 < 18 * 18] = 1
    seg[(yy - 60) ** 2 + (xx - 120) ** 2 < 14 * 14] = 2
    ann = Image.fromarray(seg, mode="P")
    ann.putpalette([0, 0, 0, 128, 0, 0, 0, 128, 0] + [0] * (256 * 3 - 9))
    ann.save(os.path.join(root, "Annotations", "480p", name, "00000.png"))
    return name


def test_run_video_on_a_synthetic_clip():
    """davis.run_video end to end on a synthetic clip with two moving objects, for an image and a video encoder."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        _common.skip("opencv-python not installed")
    from PIL import Image
    for enc in (_ImageEnc(), _VideoEnc()):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "DAVIS")
            name = _synthetic_davis(root)
            masks = os.path.join(tmp, "masks")
            r = davis.run_video(enc, root, name, masks, CPU, n_last=7, radius=12, topk=5, temp=0.1, short=96, t_hist=4,
                                batch=5, precision="fp32")
            assert r["n_frames"] == 12, r
            frames = sorted(os.listdir(os.path.join(masks, name)))
            assert len(frames) == 12
            palette = Image.open(os.path.join(root, "Annotations", "480p", name, "00000.png")).getpalette()
            first = np.array(Image.open(os.path.join(masks, name, frames[0])))
            assert set(np.unique(first).tolist()) <= {0, 1, 2}
            moved = 0
            for fr in frames:
                m = Image.open(os.path.join(masks, name, fr))
                assert m.getpalette() == palette
                moved += int(not np.array_equal(np.array(m), first))
            assert moved > 0                        # the masks follow the moving objects


if __name__ == "__main__":
    sys.exit(_common.run_all(dict(globals())))
