# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Stage-2 trainable model: the student encoder with the cross-view predictor and its head (one module, so that
DistributedDataParallel synchronizes all of them). The frozen teacher is a separate ViT outside the optimizer."""
from __future__ import annotations

import torch
import torch.nn as nn

from .predictor import CrossViewFusion, FeatHead, crossview_geometry
from .vit import ViT


class Stage2Student(nn.Module):
    def __init__(self, encoder: ViT, predictor: CrossViewFusion, head: FeatHead):
        super().__init__()
        self.encoder, self.predictor, self.head = encoder, predictor, head

    def forward(self, x_a: torch.Tensor, box_a: torch.Tensor, box_b: torch.Tensor, u: torch.Tensor,
                u_mask: torch.Tensor):
        """x_a: ImageNet-normalized student view A [B, 3, H, W]. Returns (pred [B, N, D] float32, valid [B, N])."""
        za = self.encoder(x_a, train=False)["tokens"]
        q_pos, valid, log_r = crossview_geometry(box_a, box_b, self.predictor.grid)
        fused = self.predictor(q_pos, log_r, za.float(), u, u_mask).float()
        return self.head(fused).float(), valid, log_r


def constant_token(u0: torch.Tensor, batch: int, slots: int = 32):
    """u0 in slot 0 of `slots` instruction slots whose others are masked out (the empty instruction [EOS, PAD...])."""
    u = torch.zeros(batch, slots, u0.shape[-1], dtype=torch.float32, device=u0.device)
    u[:, 0] = u0.float()
    mask = torch.zeros(batch, slots, dtype=torch.bool, device=u0.device)
    mask[:, 0] = True
    return u, mask
