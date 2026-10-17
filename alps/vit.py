# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""ALPS visual encoder: a plain ViT-B/16 in PyTorch.

This module reproduces the JAX/Flax encoder that ALPS was trained with:

* 16x16 patch embedding (a convolution without bias),
* a class token, followed by a FIXED 2-D sine-cosine position embedding
  (the class-token slot of the table is zero),
* 12 pre-LayerNorm blocks (multi-head attention with biases, MLP with exact GELU),
  LayerNorm epsilon 1e-6 as in Flax,
* the visual state z = the RAW patch tokens after the last block, without a final
  LayerNorm.

`pooled` is the global vector of the Flax model, LayerNorm(mean(z)). The image
embedding of the language interface is computed from mean(z) by the image
projection (see `alps.model.ALPS`), not from `pooled`.

Run the encoder in float32 or bfloat16. Do not run it in float16: the activations
of the deeper blocks overflow the float16 range and produce NaN tokens.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["VisionTransformer", "sincos_2d", "vit_b16"]


def sincos_2d(h: int, w: int, dim: int, temperature: float = 10_000.0, cls_token: bool = True,
              scale_to: int = 0) -> torch.Tensor:
    """Fixed 2-D sine-cosine table, [1, (1 +) h*w, dim], row-major over the grid (row index = y).

    Channels [sin(x), cos(x), sin(y), cos(y)], each dim/4 wide, frequencies 1 / temperature ** (k / (dim/4 - 1)); the
    class-token row is zero. Positions are the integer cell indices (as in training), or with `scale_to = G` the indices
    rescaled into the training grid, i * G / h and j * G / w (the protocol of every ALPS evaluation at other resolutions).
    """
    if dim % 4:
        raise ValueError(f"width {dim} must be a multiple of 4 for the sin-cos table")
    q = dim // 4
    if scale_to:
        omega = 1.0 / (temperature ** (torch.arange(q, dtype=torch.float64) / (q - 1)))
        ys = torch.arange(h, dtype=torch.float64) * scale_to / h
        xs = torch.arange(w, dtype=torch.float64) * scale_to / w
    else:                                        # float32, like the JAX code
        omega = 1.0 / (temperature ** (torch.arange(q, dtype=torch.float32) / (q - 1)))
        ys, xs = torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32)
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=1)
    if cls_token:
        pe = torch.cat([torch.zeros(1, dim, dtype=pe.dtype), pe], dim=0)
    return pe[None].float()


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError(f"width {dim} is not divisible by {heads} heads")
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v)            # softmax(q k^T / sqrt(head_dim)) v
        return self.proj(o.transpose(1, 2).reshape(b, n, d))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int, approximate_gelu: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.approximate = "tanh" if approximate_gelu else "none"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate=self.approximate))


class Block(nn.Module):
    """Pre-LayerNorm transformer block: x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, dim: int, heads: int, mlp_dim: int, eps: float = 1e-6,
                 approximate_gelu: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = Mlp(dim, mlp_dim, approximate_gelu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    """ViT with a class token and a fixed sin-cos position embedding.

    forward(x) takes images that are already normalized (ImageNet mean / std, see
    `alps.transforms`) as [B, 3, H, W] and returns a dict:
        tokens  [B, N, D]  raw patch tokens z (the visual state), row-major over the grid
        cls     [B, D]     output of the class token (not used by any ALPS objective)
        pooled  [B, D]     LayerNorm(mean(tokens)), the "z_global" vector of the Flax model
    The input side must be a multiple of the patch size. For grids other than the training
    grid (14 x 14 at 224 pixels) the sin-cos table is evaluated at cell indices rescaled into the
    training range (`pos_mode="interpolate"`, the protocol of every ALPS evaluation at other
    resolutions, e.g. DAVIS at 480 pixels) or at the integer indices of the new grid
    (`pos_mode="recompute"`, what the JAX model computes).
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16, dim: int = 768,
                 depth: int = 12, heads: int = 12, mlp_dim: int = 3072, eps: float = 1e-6,
                 pos_mode: str = "interpolate"):
        super().__init__()
        self.patch_size, self.dim, self.grid = patch_size, dim, img_size // patch_size
        self.pos_mode = pos_mode
        self.patch_embed = nn.Conv2d(3, dim, patch_size, patch_size, bias=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        # Fixed (not learned): a non-persistent buffer, so state dicts carry no position table.
        self.register_buffer("pos_embed", sincos_2d(self.grid, self.grid, dim), persistent=False)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_dim, eps) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=eps)       # applied to mean(tokens) only

    # -- position table for an arbitrary grid -------------------------------------------------
    def position_table(self, gh: int, gw: int) -> torch.Tensor:
        if (gh, gw) == (self.grid, self.grid):
            return self.pos_embed
        if self.pos_mode == "recompute":         # integer indices beyond the training range (what the JAX model computes)
            return sincos_2d(gh, gw, self.dim).to(self.pos_embed.device)
        if self.pos_mode != "interpolate":
            raise ValueError(f"unknown pos_mode {self.pos_mode!r}")
        # indices rescaled into the 14 x 14 training range: the sin-cos analogue of position-embedding interpolation
        return sincos_2d(gh, gw, self.dim, scale_to=self.grid).to(self.pos_embed.device)

    def forward(self, x: torch.Tensor, return_blocks: Optional[list] = None) -> Dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"expected [B, 3, H, W], got {tuple(x.shape)}")
        if x.shape[-1] % self.patch_size or x.shape[-2] % self.patch_size:
            raise ValueError(f"image side {tuple(x.shape[-2:])} is not a multiple of {self.patch_size}")
        # The patch embedding runs in float32 in the training code; keep it there.
        with torch.autocast(device_type=x.device.type, enabled=False):
            t = self.patch_embed(x.float())
        gh, gw = t.shape[-2:]
        t = t.flatten(2).transpose(1, 2)                                   # [B, N, D]
        t = torch.cat([self.cls_token.expand(t.shape[0], -1, -1).to(t.dtype), t], dim=1)
        t = t + self.position_table(gh, gw).to(t.dtype)
        out: Dict[str, torch.Tensor] = {}
        for i, blk in enumerate(self.blocks):
            t = blk(t)
            if return_blocks is not None and i in return_blocks:
                out[f"block{i}"] = t[:, 1:]
        tokens = t[:, 1:]
        out.update(tokens=tokens, cls=t[:, 0], pooled=self.norm(tokens.mean(dim=1)),
                   grid=torch.tensor([gh, gw]))
        return out


def vit_b16(**kw) -> VisionTransformer:
    """The ALPS architecture: ViT-B/16, width 768, 12 blocks, 12 heads, MLP 3072."""
    return VisionTransformer(patch_size=16, dim=768, depth=12, heads=12, mlp_dim=3072, **kw)
