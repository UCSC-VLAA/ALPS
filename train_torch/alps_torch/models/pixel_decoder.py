# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Convolutional pixel decoder of the Stage-1 reconstruction objective.

tokens [B, g*g, D] -> 1x1 conv to 512 -> four up-blocks (3x3 conv to 4c, pixel shuffle x2, GroupNorm(min(32, c)),
SiLU, 3x3 conv to c) for c = 256, 128, 64, 32 -> 3x3 conv to RGB, computed in float32, no output activation.

The pixel shuffle uses the channel order of the JAX code, (r, r, C) channel-last: output pixel (2h + i, 2w + j)
channel k = input channel (2i + j) * C + k. This is NOT the order of torch.nn.PixelShuffle ((C, r, r)), so
weights of the 4c-channel convolutions are only interchangeable with the JAX checkpoints under this order.
Convolutions: Flax default init (lecun normal, zero bias); GroupNorm epsilon 1e-6, scale 1, bias 0.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .. import init as I
from .. import nn as N


def pixel_shuffle_jax(x: torch.Tensor, r: int) -> torch.Tensor:
    """[B, r*r*C, H, W] -> [B, C, r*H, r*W] with the (r, r, C) channel order of the JAX code."""
    b, c4, h, w = x.shape
    c = c4 // (r * r)
    return x.view(b, r, r, c, h, w).permute(0, 3, 4, 1, 5, 2).reshape(b, c, h * r, w * r)


class UpBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, 4 * cout, 3, padding=1)
        self.norm = nn.GroupNorm(min(32, cout), cout, eps=N.LN_EPS)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)

    def forward(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        x = N.conv2d(x, self.conv1.weight, self.conv1.bias, padding=1, dtype=dtype)
        x = pixel_shuffle_jax(x, 2)
        x = N.group_norm(x, self.norm.num_groups, self.norm.weight, self.norm.bias, self.norm.eps, dtype)
        x = torch.nn.functional.silu(x)
        return N.conv2d(x, self.conv2.weight, self.conv2.bias, padding=1, dtype=dtype)


class RGBDecoder(nn.Module):
    """forward(tokens [B, g*g, D], grid (g, g)) -> RGB [B, 3, 16 g, 16 g] (float32)."""

    def __init__(self, in_dim: int = 768, dtype: torch.dtype = torch.float32, widths=(256, 128, 64, 32)):
        super().__init__()
        self.compute_dtype = dtype
        self.stem = nn.Conv2d(in_dim, 512, 1)
        chans = (512,) + tuple(widths)
        self.blocks = nn.ModuleList([UpBlock(chans[i], chans[i + 1]) for i in range(len(widths))])
        self.out = nn.Conv2d(chans[-1], 3, 3, padding=1)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, generator: Optional[torch.Generator] = None) -> None:
        I.flax_default_(self, generator)

    def forward(self, tokens: torch.Tensor, grid) -> torch.Tensor:
        b, n, d = tokens.shape
        gh, gw = grid
        x = tokens.transpose(1, 2).reshape(b, d, gh, gw)
        x = N.conv2d(x, self.stem.weight, self.stem.bias, dtype=self.compute_dtype)
        for blk in self.blocks:
            x = blk(x, self.compute_dtype)
        return N.conv2d(x, self.out.weight, self.out.bias, padding=1, dtype=torch.float32)
