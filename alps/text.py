# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""ALPS text encoder (PyTorch), the text side of the contrastive language interface.

Reproduces the Flax text tower trained in Stage 1:

* token embedding (vocabulary 32,000 rows, width 768) plus a LEARNED position embedding
  over 32 slots,
* 12 bidirectional pre-LayerNorm blocks (12 heads, MLP 3072, tanh-approximated GELU as in
  the Flax text MLP), LayerNorm epsilon 1e-6,
* a final LayerNorm, then the LAST slot is the pooled text vector. The tokenizer always
  writes a [CLS] summary token into that slot (see `alps.tokenizer`).

There is NO attention mask: padding slots (id 0) take part in attention exactly as they did
in training, so inputs must use the fixed length and layout produced by
`alps.tokenizer.ALPSTokenizer` (32 ids, [CLS] last).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .vit import Block

__all__ = ["TextEncoder"]


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 32_000, context_length: int = 32, dim: int = 768,
                 depth: int = 12, heads: int = 12, mlp_dim: int = 3072, eps: float = 1e-6):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, context_length, dim))
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_dim, eps, approximate_gelu=True)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, ids: torch.Tensor, return_tokens: bool = False) -> torch.Tensor:
        if ids.ndim != 2 or ids.shape[1] != self.context_length:
            raise ValueError(f"expected token ids [B, {self.context_length}], got {tuple(ids.shape)}; "
                             f"use alps.tokenizer.ALPSTokenizer, which pads to this length")
        x = self.token_embedding(ids.long()) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x if return_tokens else x[:, -1]
