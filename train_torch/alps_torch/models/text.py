# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Text towers of Stage 1: the contrastive text encoder (`alps.text.TextEncoder` parameters, Flax forward) and the
prefix-LM caption decoder.

Both use pre-LayerNorm blocks with tanh-approximated GELU, no padding mask (padding slots take part in attention),
LayerNorm epsilon 1e-6, and the Flax initializers: token embedding normal(0.02), learned position table
normal(0.01), q/k/v normal(width^-0.5), attention output and second MLP layer normal(width^-0.5 (2 * 12)^-0.5),
first MLP layer normal((2 width)^-0.5), biases 0, LayerNorm 1 / 0. (The `12` is the depth the original code used in
the formula, independent of the actual depth.)

Caption decoder: input = concat(raw image tokens, caption token embeddings + learned positions), no projection and no
extra position code on the image tokens; attention mask image<->image all, text->image all, text->text causal,
image->text none; final LayerNorm and a Linear head (Flax default init: lecun normal, zero bias) over the caption
positions, both in float32.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from alps.text import TextEncoder as _AlpsTextEncoder
from alps.vit import Block

from .. import init as I
from .. import nn as N
from .vit import block_forward

_DEPTH_FOR_INIT = 12


@torch.no_grad()
def init_text_blocks(blocks, width: int, generator: Optional[torch.Generator] = None) -> None:
    std_attn = width ** -0.5
    std_proj = (width ** -0.5) * ((2 * _DEPTH_FOR_INIT) ** -0.5)
    std_fc = (2 * width) ** -0.5
    for blk in blocks:
        I.normal_(blk.attn.qkv.weight, std_attn, generator)
        I.normal_(blk.attn.proj.weight, std_proj, generator)
        I.normal_(blk.mlp.fc1.weight, std_fc, generator)
        I.normal_(blk.mlp.fc2.weight, std_proj, generator)
        for lin in (blk.attn.qkv, blk.attn.proj, blk.mlp.fc1, blk.mlp.fc2):
            I.zeros_(lin.bias)
        for ln in (blk.norm1, blk.norm2):
            I.ones_(ln.weight); I.zeros_(ln.bias)


class TextEncoder(_AlpsTextEncoder):
    """Bidirectional text encoder of the contrastive objective; forward(ids [B, L]) -> the last slot [B, D] (the
    trailing [CLS]) after the final LayerNorm, in float32."""

    def __init__(self, vocab_size: int = 32_000, width: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_dim: int = 3072, max_len: int = 32, dtype: torch.dtype = torch.float32,
                 attn_impl: str = "explicit", remat: bool = False):
        super().__init__(vocab_size=vocab_size, context_length=max_len, dim=width, depth=depth, heads=num_heads,
                         mlp_dim=mlp_dim, eps=N.LN_EPS)
        self.compute_dtype, self.attn_impl, self.remat = dtype, attn_impl, remat
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, generator: Optional[torch.Generator] = None) -> None:
        I.normal_(self.token_embedding.weight, 0.02, generator)
        I.normal_(self.pos_embed, 0.01, generator)
        init_text_blocks(self.blocks, self.pos_embed.shape[-1], generator)
        I.ones_(self.norm.weight); I.zeros_(self.norm.bias)

    def forward(self, ids: torch.Tensor, train: bool = False, return_tokens: bool = False) -> torch.Tensor:
        x = F.embedding(ids.long(), self.token_embedding.weight)                    # float32
        x = (x + self.pos_embed[:, :ids.shape[1]]).to(self.compute_dtype)
        for blk in self.blocks:
            x = _run_block(blk, x, self.compute_dtype, train, None, self.attn_impl, self.remat)
        x = N.layer_norm(x, self.norm.weight, self.norm.bias, self.norm.eps, torch.float32)
        return x if return_tokens else x[:, -1]


def prefix_lm_mask(n_prefix: int, n_text: int, device=None) -> torch.Tensor:
    """[L, L] bool (True = attend): prefix <-> prefix all, text -> prefix all, text -> text causal, prefix -> text
    none."""
    li, lt = n_prefix, n_text
    m = torch.zeros(li + lt, li + lt, dtype=torch.bool, device=device)
    m[:li, :li] = True
    m[li:, :li] = True
    m[li:, li:] = torch.tril(torch.ones(lt, lt, dtype=torch.bool, device=device))
    return m


class CaptionDecoder(nn.Module):
    """Prefix-LM caption decoder. forward(z_img [B, N, D] raw image tokens, text [B, T] ids) -> logits [B, T, V]
    (float32); logits[:, t] predicts text[:, t + 1]."""

    def __init__(self, vocab_size: int = 32_000, width: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_dim: int = 3072, max_len: int = 80, dtype: torch.dtype = torch.float32,
                 attn_impl: str = "explicit", remat: bool = False):
        super().__init__()
        self.width, self.max_len = width, max_len
        self.compute_dtype, self.attn_impl, self.remat = dtype, attn_impl, remat
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, width))
        self.blocks = nn.ModuleList([Block(width, num_heads, mlp_dim, N.LN_EPS, approximate_gelu=True)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(width, eps=N.LN_EPS)
        self.head = nn.Linear(width, vocab_size)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, generator: Optional[torch.Generator] = None) -> None:
        I.normal_(self.token_embedding.weight, 0.02, generator)
        I.normal_(self.pos_embed, 0.01, generator)
        init_text_blocks(self.blocks, self.width, generator)
        I.ones_(self.norm.weight); I.zeros_(self.norm.bias)
        I.lecun_normal_(self.head.weight, generator); I.zeros_(self.head.bias)

    def forward(self, z_img: torch.Tensor, text: torch.Tensor, train: bool = False) -> torch.Tensor:
        if z_img.shape[-1] != self.width:
            raise ValueError(f"image token width {z_img.shape[-1]} != decoder width {self.width} (no projection)")
        n, t = z_img.shape[1], text.shape[1]
        w = F.embedding(text.long(), self.token_embedding.weight) + self.pos_embed[:, :t]
        x = torch.cat([z_img.to(self.compute_dtype), w.to(self.compute_dtype)], dim=1)
        mask = prefix_lm_mask(n, t, x.device)[None, None]
        for blk in self.blocks:
            x = _run_block(blk, x, self.compute_dtype, train, mask, self.attn_impl, self.remat)
        x = N.layer_norm(x[:, n:], self.norm.weight, self.norm.bias, self.norm.eps, torch.float32)
        return N.dense(x, self.head.weight, self.head.bias, torch.float32)


def _block_fn(blk, x, dtype, train, mask, attn_impl):
    return block_forward(blk, x, dtype=dtype, train=train, approximate_gelu=True, mask=mask, attn_impl=attn_impl)


def _run_block(blk, x, dtype, train, mask, attn_impl, remat):
    if remat and torch.is_grad_enabled():
        return checkpoint(_block_fn, blk, x, dtype, train, mask, attn_impl, use_reentrant=False)
    return _block_fn(blk, x, dtype, train, mask, attn_impl)
