# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Numerical building blocks with the semantics of the Flax layers of the JAX training code.

Every function takes a compute dtype `dtype` (torch.float32 or torch.bfloat16) that plays the role of the `dtype`
attribute of a Flax module: parameters are stored in float32 and cast to `dtype` for the computation, outputs are
in `dtype`. LayerNorm and GroupNorm statistics are always computed in float32 with Flax's fast variance
(E[x^2] - E[x]^2, clipped at 0) and the normalization itself in float32, then cast to the output dtype; epsilon
1e-6 everywhere. With dtype = float32 every cast is a no-op and the functions are plain float32 math, which is the
mode the parity tests check against Flax to float rounding.

bfloat16 mode reproduces the cast points of the Flax code (bf16 inputs and outputs of every Dense / Conv, bf16
attention logits and softmax output in the encoder and text towers, a float32 softmax in the Stage-2 predictor,
float32 islands where the Flax code has them). Inside a single op PyTorch keeps float32 intermediates (e.g. the
softmax is computed in float32 and rounded once), which corresponds to XLA's default `allow_excess_precision`.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

LN_EPS = 1e-6


def dense(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
          dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """flax.linen.Dense(dtype=dtype): inputs, kernel and bias cast to `dtype`. weight is [out, in] (torch layout)."""
    return F.linear(x.to(dtype), weight.to(dtype), None if bias is None else bias.to(dtype))


def conv2d(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, *, stride: int = 1,
           padding: int = 0, groups: int = 1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """flax.linen.Conv(dtype=dtype) on NCHW tensors (cross-correlation; 'SAME' for an odd kernel = padding k // 2)."""
    return F.conv2d(x.to(dtype), weight.to(dtype), None if bias is None else bias.to(dtype), stride=stride,
                    padding=padding, groups=groups)


def layer_norm(x: torch.Tensor, weight: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None,
               eps: float = LN_EPS, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """flax.linen.LayerNorm over the last axis.

    Statistics in float32 with the fast variance max(0, E[x^2] - E[x]^2); y = (x - mean) * (rsqrt(var + eps) * scale)
    + bias in float32; output cast to `dtype` (None: the promoted type of x and the parameters, as in Flax)."""
    x32 = x.float()
    mean = x32.mean(-1, keepdim=True)
    mean2 = (x32 * x32).mean(-1, keepdim=True)
    var = torch.clamp(mean2 - mean * mean, min=0.0)
    y = x32 - mean
    mul = torch.rsqrt(var + eps)
    if weight is not None:
        mul = mul * weight.float()
    y = y * mul
    if bias is not None:
        y = y + bias.float()
    if dtype is None:
        dtype = x.dtype if weight is None and bias is None else torch.promote_types(x.dtype, torch.float32)
    return y.to(dtype)


def group_norm(x: torch.Tensor, num_groups: int, weight: torch.Tensor, bias: torch.Tensor, eps: float = LN_EPS,
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """flax.linen.GroupNorm on an NCHW tensor: groups of contiguous channels, statistics over (H, W, channels of the
    group) in float32 with the fast variance, per-channel scale and bias, output in `dtype`."""
    b, c, h, w = x.shape
    x32 = x.float().reshape(b, num_groups, c // num_groups, h, w)
    mean = x32.mean(dim=(2, 3, 4), keepdim=True)
    mean2 = (x32 * x32).mean(dim=(2, 3, 4), keepdim=True)
    var = torch.clamp(mean2 - mean * mean, min=0.0)
    y = (x32 - mean).reshape(b, c, h, w)
    mul = torch.rsqrt(var + eps).expand(b, num_groups, c // num_groups, 1, 1).reshape(b, c, 1, 1)
    mul = mul * weight.float().view(1, c, 1, 1)
    y = y * mul + bias.float().view(1, c, 1, 1)
    return y.to(dtype)


def gelu(x: torch.Tensor, approximate: bool, dtype: torch.dtype) -> torch.Tensor:
    """nn.gelu(x.astype(dtype), approximate).astype(dtype): exact (erf) GELU in the ViT and the predictor,
    tanh-approximated GELU in the text towers."""
    return F.gelu(x.to(dtype), approximate="tanh" if approximate else "none")


def softmax(x: torch.Tensor) -> torch.Tensor:
    """jax.nn.softmax along the last axis, output in the input dtype."""
    return torch.softmax(x, dim=-1)


def dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor],
                          dtype: torch.dtype, impl: str = "explicit") -> torch.Tensor:
    """flax.linen.dot_product_attention (and the ViT's own copy of it) on [B, L, H, d] tensors.

    q / sqrt(d) in the compute dtype, logits = q k^T in the compute dtype, masked logits set to finfo(dtype).min,
    softmax in the compute dtype, out = attn v. `mask` is a boolean tensor broadcastable to [B, H, Lq, Lk]
    (True = attend). impl="sdpa" uses torch's fused kernel (same math in float32; in bfloat16 its softmax runs in
    float32, a documented deviation that is faster on GPUs)."""
    if impl == "sdpa":
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
        return o.transpose(1, 2)
    if impl != "explicit":
        raise ValueError(f"unknown attention implementation {impl!r}")
    depth = q.shape[-1]
    q = q / torch.tensor(math.sqrt(depth), dtype=torch.float32).to(dtype)
    w = torch.einsum("bqhd,bkhd->bhqk", q, k)
    if mask is not None:
        w = torch.where(mask, w, torch.tensor(torch.finfo(w.dtype).min, dtype=w.dtype, device=w.device))
    w = softmax(w).to(dtype)
    return torch.einsum("bhqk,bkhd->bqhd", w, v).to(dtype)
