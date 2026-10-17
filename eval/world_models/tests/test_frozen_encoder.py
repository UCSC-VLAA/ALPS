# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""CPU tests of the DINO-WM frozen-encoder adapter (dino_wm/overlay/models/encoder/frozen_encoder.py)."""
import json
import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from _load import OVERLAY, load, registry

REG = registry()
FE = load(os.path.join(OVERLAY, "models", "encoder", "frozen_encoder.py"), "wm_frozen_encoder")


class Stub(REG.FrozenEncoder):
    """A patch embedding as a stand-in for a ViT; remembers its last input."""

    def __init__(self, patch=16, dim=8, mean=REG.IMAGENET[0], std=REG.IMAGENET[1]):
        super().__init__(mean, std)
        self.name, self.patch, self.dim = "stub", patch, dim
        g = torch.Generator().manual_seed(0)
        self.proj = nn.Conv2d(3, dim, patch, patch)
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(self.proj.weight.shape, generator=g) * 0.1)
            self.proj.bias.zero_()
        self.seen = None

    def forward(self, x):
        self.seen = x.detach().clone()
        return self.proj(x).flatten(2).transpose(1, 2)


def frames(b=2, side=224, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, 3, side, side, generator=g) * 2 - 1          # DINO-WM input range [-1, 1]


def write_stats(path, d, encoder, seed=1):
    g = np.random.default_rng(seed)
    mu, sd = g.normal(size=d).astype(np.float32), (0.5 + g.random(d)).astype(np.float32)
    np.savez(path, mu=mu, sd=sd, meta=json.dumps({"encoder": encoder}))
    return torch.from_numpy(mu), torch.from_numpy(sd)


def test_interface_and_shapes():
    enc = FE.FrozenTokenEncoder("alps", backbone=Stub())
    assert enc.name == "frozen_alps" and enc.latent_ndim == 2 and enc.patch_size == 16
    assert enc.res == 224 and enc.native_grid == 14 and enc.emb_dim == enc.raw_dim == 8
    assert enc(frames()).shape == (2, 196, 8)


def test_imagenet_normalisation_of_the_frame():
    enc = FE.FrozenTokenEncoder("alps", backbone=Stub())
    x = frames()
    enc.forward_raw(x)
    x01 = x * 0.5 + 0.5
    want = (x01 - torch.tensor(REG.IMAGENET[0]).view(1, 3, 1, 1)) / torch.tensor(REG.IMAGENET[1]).view(1, 3, 1, 1)
    assert torch.allclose(enc.backbone.seen, want, atol=1e-6)


def test_dinov2_rules():
    enc = FE.FrozenTokenEncoder("dinov2_b14", backbone=Stub(patch=14, mean=(0.5,) * 3, std=(0.5,) * 3))
    assert "dino" not in enc.name and enc.res == 196 and enc.native_grid == 14     # 196-pixel input = 14 x 14 tokens
    t = enc(frames())
    assert enc.backbone.seen.shape[-1] == 196 and t.shape == (2, 196, 8)
    x = frames(); enc.forward_raw(x)                                                 # mean = std = 0.5 on the resized frame
    import torchvision.transforms.functional as TF
    x01 = TF.resize(x * 0.5 + 0.5, [196, 196], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
    assert torch.allclose(enc.backbone.seen, (x01 - 0.5) / 0.5, atol=1e-6)


def test_cradio_input_is_clamped_to_unit_range():
    enc = FE.FrozenTokenEncoder("cradio_v3_b", backbone=Stub(mean=(0.0,) * 3, std=(1.0,) * 3))
    enc.forward_raw(frames() * 1.01)
    assert enc.backbone.seen.min() >= 0 and enc.backbone.seen.max() <= 1


def test_standardisation_with_stats():
    raw = FE.FrozenTokenEncoder("alps", backbone=Stub())
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "stats.npz"); mu, sd = write_stats(p, 8, "alps")
        enc = FE.FrozenTokenEncoder("alps", stats=p, backbone=Stub())
        x = frames()
        assert torch.allclose(enc(x), (raw(x) - mu) / sd, atol=1e-5)
        assert enc.mu.shape == (1, 1, 8) and enc.sd.shape == (1, 1, 8)


def test_stats_checks():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "stats.npz"); write_stats(p, 8, "dinov3_b16")
        with pytest.raises(ValueError):
            FE.FrozenTokenEncoder("alps", stats=p, backbone=Stub())                    # fitted for another encoder
        q = os.path.join(d, "stats4.npz"); write_stats(q, 4, "alps")
        with pytest.raises(ValueError):
            FE.FrozenTokenEncoder("alps", stats=q, backbone=Stub())                    # wrong number of channels


def test_frozen_and_eval_mode():
    enc = FE.FrozenTokenEncoder("alps", backbone=Stub())
    enc.train()
    assert not enc.backbone.training and all(not p.requires_grad for p in enc.parameters())


def test_grid_resampling_and_tf32_flags():
    t = torch.randn(2, 256, 5)
    assert FE.to_grid14(t, 16).shape == (2, 196, 5) and FE.to_grid14(t[:, :196], 14) is not None
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    with FE.tf32_ctx(True):
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == before


def test_alps_end_to_end_on_random_weights():
    pytest.importorskip("safetensors")
    import alps
    torch.manual_seed(0)
    vit = alps.vit_b16()
    with torch.no_grad():
        for p in vit.parameters():
            p.add_(0.02 * torch.randn_like(p))
    with tempfile.TemporaryDirectory() as d:
        w = os.path.join(d, "alps_random.safetensors")
        alps.save({"visual." + k: v for k, v in vit.state_dict().items()}, w)
        enc = FE.FrozenTokenEncoder("alps", weights=w, tf32=False)
        x = frames(1)
        with torch.no_grad():                    # the same frame arithmetic: [-1, 1] -> [0, 1] -> 0..255 round trip -> ImageNet
            ref = alps.load_encoder(w)(alps.normalize((x * 0.5 + 0.5) * 255.0 / 255.0))["tokens"]
            loose = alps.load_encoder(w)(alps.normalize(x * 0.5 + 0.5))["tokens"]
        got = enc.forward_raw(x)
        assert got.shape == (1, 196, 768) and torch.allclose(got, ref, atol=1e-6, rtol=0)
        assert (got - loose).abs().max() / loose.abs().max() < 1e-4     # the round trip only moves the last float bits
