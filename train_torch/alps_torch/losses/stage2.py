# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Stage-2 loss.

    tgt  = stop_grad((E_0(x_B) - mu) / sigma)        frozen Stage-1 teacher on the clean crop B, fixed statistics
    Za   = E(x_A)                                    student on the augmented crop A
    pred = FeatHead(CrossViewFusion(q_pos, log_r, Za, u))
    L    = mean_images [ sum_{n valid} mean_channels (pred - tgt)^2 / max(#valid, 1) ]

Both encoders run with train=False (the residual stream stays in the compute dtype). The loss is a mean over images
of per-image terms, so data parallelism over equal shards is exact under DDP's gradient averaging.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from alps.transforms import normalize as normalize_image

from ..models.stage2 import Stage2Student, constant_token
from ..models.vit import ViT


@torch.no_grad()
def teacher_targets(teacher: ViT, x_b_rgb: torch.Tensor, zstats: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    zb = teacher(normalize_image(x_b_rgb.float()), train=False)["tokens"]
    mu, sd = zstats
    return ((zb.float() - mu) / sd).detach()


def crossview_loss_from_pred(pred: torch.Tensor, tgt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    err = torch.mean(torch.square(pred.float() - tgt), dim=-1)
    vf = valid.float()
    nv = vf.sum(1)
    return torch.mean((err * vf).sum(1) / torch.clamp(nv, min=1.0))


def crossview_loss(student: torch.nn.Module, teacher: ViT, batch: Dict[str, torch.Tensor], *,
                   zstats: Tuple[torch.Tensor, torch.Tensor], u0: torch.Tensor):
    """student: Stage2Student (or a DDP wrapper of it). batch: x_s_rgb (view A), x_t_rgb (view B) in [0, 1] as
    [B, 3, H, W]; box_a, box_b [B, 3]. Returns (loss, metrics)."""
    tgt = teacher_targets(teacher, batch["x_t_rgb"], zstats)
    u, u_mask = constant_token(u0, tgt.shape[0])
    pred, valid, log_r = student(normalize_image(batch["x_s_rgb"].float()), batch["box_a"], batch["box_b"], u, u_mask)
    loss = crossview_loss_from_pred(pred, tgt, valid)
    nv = valid.float().sum(1)
    return loss, {"loss": loss.detach(), "n_valid": nv.mean().detach(), "log_r": log_r.float().mean().detach()}


__all__ = ["crossview_loss", "crossview_loss_from_pred", "teacher_targets", "normalize_image", "Stage2Student"]
