# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Optimizer and learning-rate schedule of both stages, with the semantics of the optax chains of the JAX code.

Per parameter, with c = the number of previous updates (0 at the first update) and t = c + 1:

    g      = gradient (a parameter without gradient counts as g = 0, as in optax)
    [Stage 2: g = g * (max_norm / ||g||_global) if ||g||_global >= max_norm, without epsilon]
    m      = (1 - b1) g + bf16(bf16(b1) * m_bf16)       first moment STORED in bfloat16; the product of the decay with
                                                       the stored moment is a bfloat16 product (optax multiplies the
                                                       bf16 moment by the weakly-typed Python float b1, i.e. by
                                                       bf16(0.9) = 0.8984375, rounding to bf16); the sum is float32
    v      = (1 - b2) g^2 + b2 v                        float32
    u      = (m / (1 - b1^t)) / (sqrt(v / (1 - b2^t)) + eps)        eps 1e-8, bias corrections in float32
    m_bf16 = bf16(m)                                    the update above used the float32 value before rounding
    u      = u + wd * theta                             decoupled weight decay 0.05, Linear / Conv weights only
    theta  = theta - sched(c) * (scale * u)

`scale` is the group's constant factor: the peak learning rate in Stage 1 (schedule base 1), the 0.1 / 1.0 encoder /
predictor multiplier in Stage 2 (schedule base = 1e-4). sched(c) = base * min(1, c / W) * 0.5 (1 + cos(pi
clip((c - W) / (T - W), 0, 1))) in float32, so the first update has learning rate exactly 0 while the moments update.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn

f32 = np.float32


def warmup_cosine(total_steps: int, warmup_steps: int, base: float = 1.0) -> Callable[[int], float]:
    """lr(c) = base * min(1, c / W) * 0.5 * (1 + cos(pi * clip((c - W) / (T - W), 0, 1))), evaluated in float32
    with the order of operations of the JAX schedule."""
    def fn(step: int) -> float:
        step = int(step)
        progress = f32(np.clip(f32(step - warmup_steps) / f32(float(total_steps - warmup_steps)), f32(0), f32(1)))
        lr = f32(base * 0.5) * (f32(1.0) + np.cos(f32(np.pi) * progress, dtype=f32))
        if warmup_steps:
            lr = f32(lr * np.minimum(f32(1.0), f32(step) / f32(warmup_steps)))
        return float(f32(lr))
    return fn


def is_kernel(module: nn.Module, name: str) -> bool:
    """True for the parameters the JAX code decays: Dense / Conv kernels (here Linear / Conv weights)."""
    return name == "weight" and isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d))


def param_groups(model: nn.Module, scale: float = 1.0, weight_decay: float = 0.05, prefix: str = "",
                 only_trainable: bool = True) -> List[dict]:
    """Two groups (decayed kernels, everything else) of `model`'s parameters with the constant factor `scale`."""
    decay, other = [], []
    names_d, names_o = [], []
    seen = set()
    for mname, m in model.named_modules():
        for pname, p in m.named_parameters(recurse=False):
            if id(p) in seen or (only_trainable and not p.requires_grad):
                continue
            seen.add(id(p))
            full = f"{prefix}{mname + '.' if mname else ''}{pname}"
            (decay if is_kernel(m, pname) else other).append(p)
            (names_d if is_kernel(m, pname) else names_o).append(full)
    return [{"params": decay, "names": names_d, "scale": scale, "weight_decay": weight_decay},
            {"params": other, "names": names_o, "scale": scale, "weight_decay": 0.0}]


class AdamWBF16(torch.optim.Optimizer):
    """Adam with a bfloat16 first moment, kernel-only decoupled weight decay and an external schedule (optax chain
    scale_by_adam(mu_dtype=bf16) -> add_decayed_weights(mask) -> scale(scale) -> scale_by_schedule -> scale(-1),
    optionally preceded by clip_by_global_norm). Groups need "scale" and "weight_decay" entries."""

    def __init__(self, groups: Iterable[dict], schedule: Callable[[int], float], b1: float = 0.9, b2: float = 0.95,
                 eps: float = 1e-8, eps_root: float = 0.0, clip_norm: Optional[float] = None):
        defaults = dict(scale=1.0, weight_decay=0.0, names=None)
        super().__init__(list(groups), defaults)
        self.schedule, self.b1, self.b2, self.eps, self.eps_root, self.clip_norm = schedule, b1, b2, eps, eps_root, clip_norm
        self.count = 0
        self.last_grad_norm = None

    # --- checkpointing: the count lives next to the per-parameter state
    def state_dict(self):
        sd = super().state_dict()
        sd["count"] = self.count
        return sd

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        self.count = int(state_dict.pop("count", 0))
        super().load_state_dict(state_dict)
        for st in self.state.values():          # torch casts floating state to the parameter dtype on load
            if "exp_avg" in st:
                st["exp_avg"] = st["exp_avg"].to(torch.bfloat16)

    def lr(self, scale: float = 1.0, count: Optional[int] = None) -> float:
        """The learning rate the next update applies (schedule x group scale)."""
        return self.schedule(self.count if count is None else count) * scale

    @torch.no_grad()
    def global_grad_norm(self) -> torch.Tensor:
        sq = None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    s = torch.sum(p.grad.float() * p.grad.float())
                    sq = s if sq is None else sq + s
        return torch.sqrt(sq) if sq is not None else torch.zeros(())

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("closures are not supported")
        c = self.count
        t = c + 1
        clip = None
        if self.clip_norm is not None:
            gn = self.global_grad_norm()
            self.last_grad_norm = gn
            if float(gn) >= self.clip_norm:          # optax: select(g_norm < max_norm, g, (g / g_norm) * max_norm)
                clip = gn
        b1_bf16 = float(torch.tensor(self.b1, dtype=torch.bfloat16))
        one_m_b1, one_m_b2, b2 = float(f32(1 - self.b1)), float(f32(1 - self.b2)), float(f32(self.b2))
        bc1 = float(f32(1) - f32(self.b1) ** f32(t))
        bc2 = float(f32(1) - f32(self.b2) ** f32(t))
        sched = self.schedule(c)
        for group in self.param_groups:
            scale, wd = float(f32(group["scale"])), float(f32(group["weight_decay"]))
            for p in group["params"]:
                g = p.grad.float() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32)
                if clip is not None:
                    g = (g / clip.to(g.device)) * self.clip_norm
                st = self.state[p]
                if not st:
                    st["exp_avg"] = torch.zeros_like(p, dtype=torch.bfloat16)
                    st["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                m_prev, v = st["exp_avg"], st["exp_avg_sq"]
                m = g * one_m_b1 + (m_prev.float() * b1_bf16).to(torch.bfloat16).float()
                v.copy_((g * g) * one_m_b2 + v * b2)
                u = (m / bc1) / (torch.sqrt(v / bc2 + self.eps_root) + self.eps)
                st["exp_avg"] = m.to(torch.bfloat16)
                if wd:
                    u = u + wd * p.float()
                if scale != 1.0:
                    u = scale * u
                p.sub_((sched * u).to(p.dtype))
        self.count += 1
        return None


def stage1_optimizer(model: nn.Module, peak_lr: float, total_steps: int, warmup_steps: int, weight_decay: float = 0.05,
                     b1: float = 0.9, b2: float = 0.95) -> AdamWBF16:
    """One learning rate for every parameter; no gradient clipping."""
    return AdamWBF16(param_groups(model, peak_lr, weight_decay), warmup_cosine(total_steps, warmup_steps, 1.0), b1, b2)


def stage2_optimizer(student: nn.Module, base_lr: float, total_steps: int, warmup_steps: int,
                     encoder_lr_ratio: float = 0.1, weight_decay: float = 0.05, b1: float = 0.9, b2: float = 0.99,
                     clip_norm: float = 1.0) -> AdamWBF16:
    """student = Stage2Student: encoder at encoder_lr_ratio x, predictor and head at 1 x; global-norm clipping."""
    groups = (param_groups(student.encoder, encoder_lr_ratio, weight_decay, "encoder.")
              + param_groups(student.predictor, 1.0, weight_decay, "predictor.")
              + param_groups(student.head, 1.0, weight_decay, "head."))
    return AdamWBF16(groups, warmup_cosine(total_steps, warmup_steps, base_lr), b1, b2, clip_norm=clip_norm)


def named_state(opt: AdamWBF16) -> Dict[str, Dict[str, torch.Tensor]]:
    """{parameter name: {"exp_avg", "exp_avg_sq"}} for parity checks."""
    out = {}
    for group in opt.param_groups:
        for n, p in zip(group["names"] or [], group["params"]):
            if p in opt.state and opt.state[p]:
                out[n] = opt.state[p]
    return out


__all__ = ["AdamWBF16", "warmup_cosine", "param_groups", "stage1_optimizer", "stage2_optimizer", "is_kernel",
           "named_state"]
