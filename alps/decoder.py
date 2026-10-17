# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Evaluation decoder: ALPS tokens -> SDXL autoencoder latents -> pixels.

The paper's reconstructions (Table 1 l1 / LPIPS, Fig. 4) and the renderings of world-model predictions use a small
convolutional adapter trained on frozen tokens for each encoder: tokens [B, 196, D] -> 14 x 14 grid (bilinear if the
native grid differs) -> 1x1 conv to 512 -> transposed conv to 28 x 28 -> 5 residual blocks (GroupNorm 32, SiLU, 3x3
conv) -> 4 latent channels; the frozen SDXL autoencoder (stabilityai/sdxl-vae, scaling factor 0.13025) decodes the
latent to a 224 x 224 image in [-1, 1]. Train one with eval/reconstruction/train_decoder.py.

    dec = alps.decoder.load_decoder("alps_decoder.pt")
    rgb = alps.decoder.SDXLRenderer(dec)(tokens)            # [B, 3, 224, 224] in [0, 1]

This decoder is separate from the pixel decoder of the Stage-1 reconstruction objective, which is not released.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["EvalDecoder", "SDXLRenderer", "load_decoder", "SDXL_SCALING"]

SDXL_SCALING = 0.13025


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.n1 = nn.GroupNorm(32, ch); self.c1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.n2 = nn.GroupNorm(32, ch); self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        return x + self.c2(F.silu(self.n2(h)))


class EvalDecoder(nn.Module):
    """tokens [B, T, D] (row-major square grid) -> SDXL latent [B, 4, 2 * grid, 2 * grid] (scaled by SDXL_SCALING)."""

    def __init__(self, dim: int = 768, ch: int = 512, out_ch: int = 4, grid: int = 14, nblocks: int = 5):
        super().__init__()
        self.grid = grid
        self.proj = nn.Conv2d(dim, ch, 1)
        self.up = nn.ConvTranspose2d(ch, ch, 4, 2, 1)
        self.blocks = nn.ModuleList([ResBlock(ch) for _ in range(nblocks)])
        self.out = nn.Sequential(nn.GroupNorm(32, ch), nn.SiLU(), nn.Conv2d(ch, out_ch, 3, 1, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, t, d = z.shape
        g = int(round(t ** 0.5))
        x = z.transpose(1, 2).reshape(b, d, g, g)
        if g != self.grid:
            x = F.interpolate(x, size=(self.grid, self.grid), mode="bilinear", align_corners=False)
        x = self.up(self.proj(x))
        for blk in self.blocks:
            x = blk(x)
        return self.out(x)


def load_decoder(path: str, dim: int = 768, ch: int = 512, nblocks: int = 5, device: str = "cpu") -> EvalDecoder:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    dec = EvalDecoder(dim=dim, ch=ch, nblocks=nblocks)
    dec.load_state_dict(sd, strict=True)
    return dec.to(device).eval()


class SDXLRenderer(nn.Module):
    """Decoder + frozen SDXL autoencoder decoder (needs `diffusers`)."""

    def __init__(self, decoder: EvalDecoder, vae: str = "stabilityai/sdxl-vae"):
        super().__init__()
        from diffusers import AutoencoderKL
        self.decoder = decoder.eval()
        self.vae = AutoencoderKL.from_pretrained(vae).eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        img = self.vae.decode(self.decoder(tokens) / SDXL_SCALING).sample.clamp(-1, 1)
        return (img + 1) / 2
