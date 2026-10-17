# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Stage-2 cross-view predictor (`CrossViewFusion`) and output head (`FeatHead`).

The student encodes crop A of an image; the frozen Stage-1 teacher encodes a clean crop B. For every patch of B, a
query = learned vector m + fixed sin-cos code of the patch centre in A's frame (in patches of A) + an affine map of
(ln r, ln^2 r), r = side_B / side_A, attends to the student tokens (projected to the predictor width, plus the same
code of A's own grid) and to one constant token (an affine map of `u0`, the frozen flan-T5-base encoding of the empty
instruction, in slot 0 of 32 slots whose other 31 are masked out). Four pre-LayerNorm layers (self-attention over the
queries, cross-attention, MLP 2048), width 512, 8 heads, then LayerNorm + Linear to 768 (`out_proj`) and the head
LayerNorm + Linear (`FeatHead`).

Numerics as in the Flax modules: compute dtype for every Dense and LayerNorm output; attention logits from a
compute-dtype einsum, cast to float32, divided by sqrt(64), masked keys set to finfo(float32).min, softmax in
float32, cast back. Initialization = Flax defaults (lecun normal kernels, zero biases, LayerNorm 1 / 0) and
mask_token normal(0.02).

Flax parameter names (subtree `transition`) map to: scale_proj, src_proj, txt_proj, mask_token, ln_out, out_proj;
ln_sa_{i} -> layers.{i}.ln_sa, sa_{i}_{q,k,v,o} -> layers.{i}.sa.{q,k,v,o}, likewise ca, ln_ca_{i}, ln_mlp_{i},
mlp_{i}/{fc1,fc2} -> layers.{i}.mlp.{fc1,fc2}. Subtree `feat_head`: ln, proj.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .. import init as I
from .. import nn as N


def sincos_2d(pos_yx: torch.Tensor, width: int) -> torch.Tensor:
    """Fixed code of continuous positions [..., 2] = (y, x): [sin y, cos y, sin x, cos x], omega_k = 10000^(-k/d),
    d = width / 4, k = 0 .. d - 1 (float32). Note: differs from the encoder's table (x first, exponent k / (d - 1))."""
    if width % 4:
        raise ValueError(width)
    d = width // 4
    omega = 1.0 / (10000.0 ** (torch.arange(d, dtype=torch.float32, device=pos_yx.device) / d))
    y = pos_yx[..., 0:1].float() * omega
    x = pos_yx[..., 1:2].float() * omega
    return torch.cat([torch.sin(y), torch.cos(y), torch.sin(x), torch.cos(x)], dim=-1)


def crossview_geometry(box_a: torch.Tensor, box_b: torch.Tensor, grid: int) -> Tuple[torch.Tensor, ...]:
    """box_a, box_b [B, 3] = (y0, x0, side) normalized by the side of the image's central square.

    Returns q_pos [B, N, 2] (centres of B's patches in A's frame, in patches of A), valid [B, N] (centre inside A)
    and log_r [B] = ln(side_B / side_A)."""
    g = int(grid)
    c = (torch.arange(g, dtype=torch.float32, device=box_a.device) + 0.5) / g
    box_a = box_a.float(); box_b = box_b.float()
    yc = box_b[:, 0:1] + c[None, :] * box_b[:, 2:3]
    xc = box_b[:, 1:2] + c[None, :] * box_b[:, 2:3]
    v = (yc - box_a[:, 0:1]) / box_a[:, 2:3]
    u = (xc - box_a[:, 1:2]) / box_a[:, 2:3]
    b = box_a.shape[0]
    vv = v[:, :, None].expand(b, g, g)
    uu = u[:, None, :].expand(b, g, g)
    q_pos = torch.stack([vv, uu], dim=-1).reshape(b, g * g, 2) * g
    valid = ((vv >= 0.) & (vv < 1.) & (uu >= 0.) & (uu < 1.)).reshape(b, g * g)
    log_r = torch.log(box_b[:, 2] / box_a[:, 2])
    return q_pos, valid, log_r


class Attention(nn.Module):
    """q / k / v / o Linear layers of one attention of the predictor (Flax `{name}_q` ... `{name}_o`)."""

    def __init__(self, width: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.q, self.k, self.v, self.o = (nn.Linear(width, width) for _ in range(4))

    def forward(self, q: torch.Tensor, kv: torch.Tensor, key_mask: Optional[torch.Tensor], dtype) -> torch.Tensor:
        b, lq, w = q.shape
        lk = kv.shape[1]
        h = self.num_heads
        d = w // h
        qh = N.dense(q, self.q.weight, self.q.bias, dtype).view(b, lq, h, d)
        kh = N.dense(kv, self.k.weight, self.k.bias, dtype).view(b, lk, h, d)
        vh = N.dense(kv, self.v.weight, self.v.bias, dtype).view(b, lk, h, d)
        logits = torch.einsum("bqhd,bkhd->bhqk", qh, kh).float() / torch.tensor(math.sqrt(d), dtype=torch.float32)
        if key_mask is not None:
            logits = torch.where(key_mask[:, None, None, :], logits,
                                 torch.tensor(torch.finfo(torch.float32).min, device=logits.device))
        attn = torch.softmax(logits, dim=-1).to(vh.dtype)
        out = torch.einsum("bhqk,bkhd->bqhd", attn, vh).reshape(b, lq, w)
        return N.dense(out, self.o.weight, self.o.bias, dtype)


class Mlp(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(width, hidden), nn.Linear(hidden, width)

    def forward(self, x: torch.Tensor, dtype) -> torch.Tensor:
        x = N.dense(x, self.fc1.weight, self.fc1.bias, dtype)
        x = N.gelu(x, False, dtype)
        return N.dense(x, self.fc2.weight, self.fc2.bias, dtype)


class FusionLayer(nn.Module):
    def __init__(self, width: int, num_heads: int, hidden: int):
        super().__init__()
        self.ln_sa, self.ln_ca, self.ln_mlp = (nn.LayerNorm(width, eps=N.LN_EPS) for _ in range(3))
        self.sa, self.ca = Attention(width, num_heads), Attention(width, num_heads)
        self.mlp = Mlp(width, hidden)


def _ln(x, ln: nn.LayerNorm, dtype):
    return N.layer_norm(x, ln.weight, ln.bias, ln.eps, dtype)


class CrossViewFusion(nn.Module):
    def __init__(self, width: int = 512, depth: int = 4, num_heads: int = 8, grid: int = 14, token_dim: int = 768,
                 text_dim: int = 768, mlp_ratio: int = 4, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.width, self.grid, self.compute_dtype = width, grid, dtype
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        self.scale_proj = nn.Linear(2, width)
        self.src_proj = nn.Linear(token_dim, width)
        self.txt_proj = nn.Linear(text_dim, width)
        self.layers = nn.ModuleList([FusionLayer(width, num_heads, mlp_ratio * width) for _ in range(depth)])
        self.ln_out = nn.LayerNorm(width, eps=N.LN_EPS)
        self.out_proj = nn.Linear(width, token_dim)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, generator: Optional[torch.Generator] = None) -> None:
        I.flax_default_(self, generator)
        I.normal_(self.mask_token, 0.02, generator)

    def forward(self, q_pos: torch.Tensor, log_r: torch.Tensor, src_tokens: torch.Tensor, txt_tokens: torch.Tensor,
                txt_mask: torch.Tensor) -> torch.Tensor:
        """q_pos [B, Nq, 2]; log_r [B]; src_tokens [B, g*g, token_dim]; txt_tokens [B, L, text_dim]; txt_mask [B, L]."""
        b = q_pos.shape[0]
        g, w, cdt = self.grid, self.width, self.compute_dtype
        if src_tokens.shape[1] != g * g:
            raise ValueError(f"expected {g * g} source tokens, got {src_tokens.shape[1]}")
        log_r = log_r.float()
        scale_in = torch.stack([log_r, torch.square(log_r)], dim=-1).to(cdt)
        scale_emb = N.dense(scale_in, self.scale_proj.weight, self.scale_proj.bias, cdt)
        x = self.mask_token.to(cdt) + sincos_2d(q_pos, w).to(cdt) + scale_emb[:, None, :]
        ar = torch.arange(g, dtype=torch.float32, device=q_pos.device)
        rr, cc = torch.meshgrid(ar, ar, indexing="ij")
        a_pos = torch.stack([rr, cc], dim=-1).reshape(1, g * g, 2) + 0.5
        src_kv = N.dense(src_tokens.to(cdt), self.src_proj.weight, self.src_proj.bias, cdt) + sincos_2d(a_pos, w).to(cdt)
        txt_kv = N.dense(txt_tokens.to(cdt), self.txt_proj.weight, self.txt_proj.bias, cdt)
        kv = torch.cat([src_kv, txt_kv], dim=1)
        key_mask = torch.cat([torch.ones(b, src_kv.shape[1], dtype=torch.bool, device=kv.device),
                              txt_mask.bool()], dim=1)
        for layer in self.layers:
            y = _ln(x, layer.ln_sa, cdt)
            x = x + layer.sa(y, y, None, cdt)
            y = _ln(x, layer.ln_ca, cdt)
            x = x + layer.ca(y, kv, key_mask, cdt)
            y = _ln(x, layer.ln_mlp, cdt)
            x = x + layer.mlp(y, cdt)
        x = _ln(x, self.ln_out, cdt)
        return N.dense(x, self.out_proj.weight, self.out_proj.bias, cdt)


class FeatHead(nn.Module):
    """LayerNorm + Linear(768 -> 768), compute dtype."""

    def __init__(self, in_dim: int = 768, out_dim: int = 768, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.compute_dtype = dtype
        self.ln = nn.LayerNorm(in_dim, eps=N.LN_EPS)
        self.proj = nn.Linear(in_dim, out_dim)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, generator: Optional[torch.Generator] = None) -> None:
        I.flax_default_(self, generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ln(x.to(self.compute_dtype), self.ln, self.compute_dtype)
        return N.dense(x, self.proj.weight, self.proj.bias, self.compute_dtype)
