# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""One training step of each stage (used by the trainers and by the data-parallel tests).

Random draws per step: a per-rank generator for per-image draws (reconstruction noise, Gumbel uniforms of the
same-image negatives) and a generator seeded identically on every rank for the step's global set of cross-image
negative indices. Tests can pass explicit draws instead.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from alps.transforms import normalize

from .distributed import Dist
from .losses.stage1 import cross_indices, stage1_loss
from .losses.stage2 import crossview_loss
from .models.stage1 import project_logit_scale


def step_generator(seed: int, step: int, salt: int, device, rank: Optional[int] = None) -> torch.Generator:
    words = [int(seed), int(step), int(salt)] + ([] if rank is None else [int(rank)])
    s = int(np.random.SeedSequence(words).generate_state(1, np.uint64)[0] >> np.uint64(1))
    g = torch.Generator(device=device)
    g.manual_seed(s)
    return g


def stage1_forward_loss(model, batch: Dict[str, torch.Tensor], cfg, step: int, dist: Dist,
                        draws: Optional[Dict[str, torch.Tensor]] = None):
    """Forward + loss of one Stage-1 step on this rank's shard of the global batch. Returns (loss, metrics)."""
    draws = draws or {}
    dev = batch["x_rgb"].device
    g_local = step_generator(cfg.seed, step, 1, dev, dist.rank)
    g_shared = step_generator(cfg.seed, step, 2, dev)
    out = model(normalize(batch["x_rgb"]), normalize(batch["gx_rgb"]), batch["x_rgb"], batch["crop_box"],
                batch["flip"], batch["itc_labels1"], batch["itc_labels2"], batch["labels"], train=True,
                noise=draws.get("noise"), generator=g_local)
    b, g = out["h_x"].shape[:2]
    cross_idx = draws.get("cross_idx")
    if cross_idx is None:
        cross_idx = cross_indices(b * dist.world * g * g, int(cfg.dense_n_cross), g_shared, dev)
    n_same = int(cfg.dense_n_same) or None
    return stage1_loss(out, cross_idx=cross_idx, cap_w=cfg.cap_w, rec_w=cfg.rec_w, dense_w=cfg.dense_w,
                       dense_radius=int(cfg.dense_radius), dense_n_same=n_same, dense_split=tuple(cfg.dense_split),
                       gumbel_u=draws.get("gumbel_u"), generator=g_local, dist=dist)


def stage1_train_step(model, opt, batch, cfg, step: int, dist: Dist, draws=None) -> Dict[str, float]:
    loss, metrics = stage1_forward_loss(model, batch, cfg, step, dist, draws)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    lr = opt.lr(scale=cfg.peak_lr)
    opt.step()
    project_logit_scale(model)
    out = {k: float(dist.mean_nograd(v)) for k, v in metrics.items()}
    out["lr"] = lr
    return out


def stage2_accumulate(student, teacher, batches: Sequence[Dict[str, torch.Tensor]], *, zstats, u0, dist: Dist):
    """Gradients of the mean loss over `batches` (micro-batches of one update, the JAX accumulation: the sum of the
    micro-batch gradients divided by their number). Under DDP the reduction runs on the last micro-batch only."""
    n = len(batches)
    metrics: List[Dict[str, torch.Tensor]] = []
    for i, mb in enumerate(batches):
        ctx = student.no_sync() if (dist.distributed and i < n - 1 and hasattr(student, "no_sync")) else nullcontext()
        with ctx:
            loss, m = crossview_loss(student, teacher, mb, zstats=zstats, u0=u0)
            loss.backward()
        metrics.append(m)
    params = student.parameters()
    for p in params:
        if p.grad is not None:
            p.grad.div_(n)
    return {k: torch.stack([m[k] for m in metrics]).mean() for k in metrics[0]}


def stage2_train_step(student, teacher, opt, batches, cfg, dist: Dist, *, zstats, u0) -> Dict[str, float]:
    opt.zero_grad(set_to_none=True)
    m = stage2_accumulate(student, teacher, batches, zstats=zstats, u0=u0, dist=dist)
    lr = opt.lr(scale=1.0)
    opt.step()
    out = {k: float(dist.mean_nograd(v)) for k, v in m.items()}
    out.update(lr=lr, grad_norm=float(opt.last_grad_norm) if opt.last_grad_norm is not None else float("nan"))
    return out
