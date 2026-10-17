# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""The ALPS ViT for training: `alps.vit.VisionTransformer` (same parameters and state-dict keys) with the forward
pass of the Flax training code.

Differences from the inference forward of `alps.vit`:
* the compute dtype is explicit (`dtype`, float32 or bfloat16) and the casts follow the Flax module: patch
  embedding, class token and position table in float32, every block casts its input to the compute dtype;
* `train=True` adds each residual branch in float32 (in the Flax code a rate-0 stochastic-depth layer promoted the
  branch to float32 in training mode); `train=False` adds it in the compute dtype. Stage 1 trains with train=True;
  Stage 2 runs both the student and the teacher with train=False. In float32 both are the same;
* the position table is recomputed for the input grid (integer cell indices), as the JAX model does;
* LayerNorm uses Flax's statistics (alps_torch.nn.layer_norm), attention the explicit Flax order of operations.

`encoder_norm` (here `norm`) only produces the pooled vector, which no training loss reads, so it receives no
gradient in either code base; it is frozen (requires_grad=False), which is exactly equivalent.

Initialization (`reset_parameters`) = the Flax initializers: patch embedding kaiming-uniform, class token
normal(1e-6), q/k/v/out kernels normal(0.02), first MLP layer normal(0.02), second MLP layer
variance_scaling(0.3072, fan_out, truncated normal), biases 0, LayerNorm 1 / 0.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from alps.vit import Block, VisionTransformer, sincos_2d

from .. import init as I
from .. import nn as N


def block_forward(blk: Block, x: torch.Tensor, *, dtype: torch.dtype, train: bool, approximate_gelu: bool,
                  mask: Optional[torch.Tensor] = None, attn_impl: str = "explicit") -> torch.Tensor:
    """One pre-LayerNorm block (alps.vit.Block parameters) with the numerics of the Flax Encoder1DBlock."""
    b, n, d = x.shape
    heads = blk.attn.heads
    x = x.to(dtype)
    y = N.layer_norm(x, blk.norm1.weight, blk.norm1.bias, blk.norm1.eps, dtype)
    qkv = N.dense(y, blk.attn.qkv.weight, blk.attn.qkv.bias, dtype).view(b, n, 3, heads, d // heads)
    o = N.dot_product_attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], mask, dtype, attn_impl)
    y = N.dense(o.reshape(b, n, d), blk.attn.proj.weight, blk.attn.proj.bias, dtype)
    x = x + (y.float() if train else y)
    y = N.layer_norm(x, blk.norm2.weight, blk.norm2.bias, blk.norm2.eps, dtype)
    y = N.dense(y, blk.mlp.fc1.weight, blk.mlp.fc1.bias, dtype)
    y = N.gelu(y, approximate_gelu, dtype)
    y = N.dense(y, blk.mlp.fc2.weight, blk.mlp.fc2.bias, dtype)
    return x + (y.float() if train else y)


class ViT(VisionTransformer):
    """ViT encoder of ALPS for training. forward(x [B, 3, H, W] ImageNet-normalized) -> {"tokens", "pooled"}."""

    def __init__(self, width: int = 768, depth: int = 12, mlp_dim: int = 3072, num_heads: int = 12,
                 patch_size: int = 16, img_size: int = 224, dtype: torch.dtype = torch.float32, remat: bool = False,
                 attn_impl: str = "explicit"):
        super().__init__(img_size=img_size, patch_size=patch_size, dim=width, depth=depth, heads=num_heads,
                         mlp_dim=mlp_dim, eps=N.LN_EPS, pos_mode="recompute")
        self.compute_dtype = dtype
        self.remat = remat
        self.attn_impl = attn_impl
        self.reset_parameters()
        self.norm.requires_grad_(False)          # pooled-vector LayerNorm: zero gradient in training (see above)

    @torch.no_grad()
    def reset_parameters(self, generator: Optional[torch.Generator] = None) -> None:
        I.kaiming_uniform_(self.patch_embed.weight, generator)
        I.normal_(self.cls_token, 1e-6, generator)
        for blk in self.blocks:
            I.normal_(blk.attn.qkv.weight, 0.02, generator)
            I.normal_(blk.attn.proj.weight, 0.02, generator)
            I.normal_(blk.mlp.fc1.weight, 0.02, generator)
            I.variance_scaling_(blk.mlp.fc2.weight, 0.3072, "fan_out", "truncated_normal", generator)
            for lin in (blk.attn.qkv, blk.attn.proj, blk.mlp.fc1, blk.mlp.fc2):
                I.zeros_(lin.bias)
            for ln in (blk.norm1, blk.norm2):
                I.ones_(ln.weight); I.zeros_(ln.bias)
        I.ones_(self.norm.weight); I.zeros_(self.norm.bias)

    def forward(self, x: torch.Tensor, train: bool = False, pooled: bool = False) -> Dict[str, torch.Tensor]:
        p = self.patch_size
        if x.ndim != 4 or x.shape[1] != 3 or x.shape[-1] % p or x.shape[-2] % p:
            raise ValueError(f"expected [B, 3, H, W] with sides multiple of {p}, got {tuple(x.shape)}")
        with torch.autocast(device_type=x.device.type, enabled=False):
            t = N.conv2d(x.float(), self.patch_embed.weight, stride=p, dtype=torch.float32)
        gh, gw = t.shape[-2:]
        t = t.flatten(2).transpose(1, 2)                                            # [B, N, D], row-major grid
        t = torch.cat([self.cls_token.float().expand(t.shape[0], -1, -1), t], dim=1)
        t = t + sincos_2d(gh, gw, self.dim).to(t.device)                            # float32, zero class-token row
        t = t.to(self.compute_dtype)
        for blk in self.blocks:
            if self.remat and torch.is_grad_enabled():
                t = checkpoint(self._block, blk, t, train, use_reentrant=False)
            else:
                t = self._block(blk, t, train)
        out = {"tokens": t[:, 1:], "grid": (gh, gw)}
        if pooled:
            out["pooled"] = N.layer_norm(out["tokens"].mean(dim=1), self.norm.weight, self.norm.bias, self.norm.eps,
                                         self.compute_dtype)
        return out

    def _block(self, blk: Block, t: torch.Tensor, train: bool) -> torch.Tensor:
        return block_forward(blk, t, dtype=self.compute_dtype, train=train, approximate_gelu=False,
                             attn_impl=self.attn_impl)


def vit(variant: str = "B", **kw) -> ViT:
    w, d, m, h = {"S": (384, 12, 1536, 6), "B": (768, 12, 3072, 12), "L": (1024, 24, 4096, 16)}[variant]
    return ViT(width=w, depth=d, mlp_dim=m, num_heads=h, **kw)
