# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Projector and geometry of the Stage-1 dense correspondence objective (the loss is in alps_torch/losses/stage1.py).

Coordinates are normalized (y, x) in [0, 1] of the base view x, patch centres at (i + 0.5) / g. For every patch of
the geometric view g(x) (a crop + optional horizontal flip of x with normalized box (y0, x0, h, w)), the positive is
the bilinear sample of x's projected features at the patch centre mapped into x (align_corners=False, edge clamp).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .. import init as I
from .. import nn as N


class DenseProjector(nn.Module):
    """Linear(D -> 256) (Flax default init) then a LayerNorm without scale and bias, output float32."""

    def __init__(self, in_dim: int = 768, dim: int = 256, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.compute_dtype = dtype
        self.proj = nn.Linear(in_dim, dim)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, generator: Optional[torch.Generator] = None) -> None:
        I.lecun_normal_(self.proj.weight, generator); I.zeros_(self.proj.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = N.dense(z, self.proj.weight, self.proj.bias, self.compute_dtype)
        return N.layer_norm(h, None, None, N.LN_EPS, torch.float32)


def patch_center_grid(g: int, device=None) -> torch.Tensor:
    c = (torch.arange(g, dtype=torch.float32, device=device) + 0.5) / g
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([yy, xx], -1)


def warp_coords(crop_box: torch.Tensor, flip: torch.Tensor, g: int) -> torch.Tensor:
    """Centre of every g(x) patch in the base view's normalized frame: [B, g, g, 2]. crop_box [B, 4] = (y0, x0, h, w)."""
    cy, cx, ch, cw = [crop_box[:, k].float()[:, None, None] for k in range(4)]
    centers = patch_center_grid(g, crop_box.device)[None]
    py, px = centers[..., 0], centers[..., 1]
    px = torch.where(flip.bool()[:, None, None], 1.0 - px, px)
    return torch.stack([cy + py * ch, cx + px * cw], -1)


def bilinear_sample(feat: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """feat [B, g, g, C], coords [B, g, g, 2] normalized (y, x); align_corners=False, edge clamp."""
    b_, g = feat.shape[:2]
    fy = coords[..., 0] * g - 0.5
    fx = coords[..., 1] * g - 0.5
    y0 = torch.floor(fy).long(); x0 = torch.floor(fx).long()
    y1 = y0 + 1; x1 = x0 + 1
    wy = (fy - y0.to(fy.dtype))[..., None]
    wx = (fx - x0.to(fx.dtype))[..., None]
    clamp = lambda a: a.clamp(0, g - 1)
    y0c, y1c, x0c, x1c = clamp(y0), clamp(y1), clamp(x0), clamp(x1)
    bi = torch.arange(b_, device=feat.device)[:, None, None]
    gather = lambda yy, xx: feat[bi, yy, xx]
    top = gather(y0c, x0c) * (1 - wx) + gather(y0c, x1c) * wx
    bot = gather(y1c, x0c) * (1 - wx) + gather(y1c, x1c) * wx
    return top * (1 - wy) + bot * wy
