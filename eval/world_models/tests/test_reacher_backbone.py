# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""CPU tests of the Reacher backbone adapter (reacher/frozen_backbone.py)."""
import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from _load import REACHER, load, registry

REG = registry()
FB = load(os.path.join(REACHER, "frozen_backbone.py"), "wm_frozen_backbone")
IM = (torch.tensor(FB.IMNET_MEAN).view(1, 3, 1, 1), torch.tensor(FB.IMNET_STD).view(1, 3, 1, 1))


class Stub(REG.FrozenEncoder):
    def __init__(self, mean=REG.IMAGENET[0], std=REG.IMAGENET[1]):
        super().__init__(mean, std)
        self.proj = nn.Conv2d(3, 768, 16, 16)
        self.seen = None

    def forward(self, x):
        self.seen = x.detach().clone()
        return self.proj(x).flatten(2).transpose(1, 2)


def pixels(b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(b, 3, 224, 224, generator=g) - IM[0]) / IM[1]            # ImageNet-normalised, as PreJEPA feeds them


def test_alps_reads_the_imagenet_pixels_and_tokens_have_unit_length():
    m = FB.FrozenTokens("alps", backbone=Stub())
    x = pixels()
    out = m(x)
    assert out.last_hidden_state.shape == (2, 197, 768) and torch.equal(out.last_hidden_state[:, 0], torch.zeros(2, 768))
    assert torch.allclose(m.vit.seen, x)
    assert torch.allclose(m.tokens(x).norm(dim=-1), torch.ones(2, 196), atol=1e-5)
    assert m.config.hidden_size == 768


def test_public_encoders_get_their_own_normalisation():
    m = FB.FrozenTokens("siglip2_b16", backbone=Stub(mean=(0.5,) * 3, std=(0.5,) * 3))
    x = pixels(); m(x)
    assert torch.allclose(m.vit.seen, ((x * IM[1] + IM[0]) - 0.5) / 0.5, atol=1e-5)
    r = FB.FrozenTokens("cradio_v3_b", backbone=Stub(mean=(0.0,) * 3, std=(1.0,) * 3))
    r(pixels() * 1.5)
    assert r.vit.seen.min() >= 0 and r.vit.seen.max() <= 1


def test_standardisation_and_state_dict():
    with tempfile.TemporaryDirectory() as d:
        g = np.random.default_rng(0)
        mu, sd = g.normal(size=768).astype(np.float32) * 0.01, (0.5 + g.random(768)).astype(np.float32) * 0.05
        p = os.path.join(d, "reacher_alps.npz"); np.savez(p, mean=mu, std=sd)
        m = FB.FrozenTokens("alps", stats=p, backbone=Stub())
        x = pixels()
        want = (m.tokens(x) - torch.from_numpy(mu)) / torch.from_numpy(sd)
        assert torch.allclose(m(x).last_hidden_state[:, 1:], want, atol=1e-5)
        keys = set(m.state_dict())
        assert {"mu", "sd"} <= keys and "im_mean" not in keys and all(not p_.requires_grad for p_ in m.parameters())
        missing = FB.FrozenTokens("alps", stats=os.path.join(d, "absent.npz"), backbone=Stub())     # buffers then come from the weights file
        assert torch.equal(missing.mu, torch.zeros(768))


def test_unknown_backbone():
    with pytest.raises(ValueError):
        FB.FrozenTokens("dinov2_b14", backbone=Stub())        # Reacher's DINOv2 is dinov2_b14_196 (Hugging Face model, 196 pixels)
