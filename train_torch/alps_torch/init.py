# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Initializers with the distributions of the Flax initializers of the JAX training code.

Fans are those of the Flax kernel ([in, out] for Dense, [kh, kw, in, out] for Conv), which are the same numbers as
PyTorch's fans of the transposed weight ([out, in] / [out, in, kh, kw]):

    normal(s)                         N(0, s^2)
    variance_scaling(scale, mode, d)  variance = scale / fan; d = "truncated_normal": N(0, (sqrt(variance) /
                                      .87962566) ^ 2) truncated to +-2 of that standard deviation (so the samples have
                                      the requested variance); d = "uniform": U(-sqrt(3 variance), sqrt(3 variance))
    lecun_normal                      variance_scaling(1, fan_in, truncated_normal) (Flax Dense / Conv default)
    kaiming_uniform                   variance_scaling(2, fan_in, uniform)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

_TRUNC_STD = 0.87962566103423978      # std of a standard normal truncated to [-2, 2]


def fans(weight: torch.Tensor):
    """(fan_in, fan_out) of a torch Linear [out, in] or Conv [out, in, kh, kw] weight."""
    if weight.ndim < 2:
        raise ValueError("fans need a weight with at least 2 dimensions")
    receptive = math.prod(weight.shape[2:]) if weight.ndim > 2 else 1
    return weight.shape[1] * receptive, weight.shape[0] * receptive


@torch.no_grad()
def normal_(t: torch.Tensor, std: float, generator=None) -> torch.Tensor:
    return t.normal_(0.0, std, generator=generator)


@torch.no_grad()
def variance_scaling_(t: torch.Tensor, scale: float, mode: str, distribution: str, generator=None) -> torch.Tensor:
    fan_in, fan_out = fans(t)
    fan = {"fan_in": fan_in, "fan_out": fan_out, "fan_avg": (fan_in + fan_out) / 2}[mode]
    variance = scale / max(1.0, fan)
    if distribution == "truncated_normal":
        std = math.sqrt(variance) / _TRUNC_STD
        return nn.init.trunc_normal_(t, 0.0, std, -2.0 * std, 2.0 * std, generator=generator)
    if distribution == "normal":
        return t.normal_(0.0, math.sqrt(variance), generator=generator)
    if distribution == "uniform":
        lim = math.sqrt(3.0 * variance)
        return t.uniform_(-lim, lim, generator=generator)
    raise ValueError(distribution)


def lecun_normal_(t: torch.Tensor, generator=None) -> torch.Tensor:
    return variance_scaling_(t, 1.0, "fan_in", "truncated_normal", generator=generator)


def kaiming_uniform_(t: torch.Tensor, generator=None) -> torch.Tensor:
    return variance_scaling_(t, 2.0, "fan_in", "uniform", generator=generator)


@torch.no_grad()
def zeros_(t: torch.Tensor) -> torch.Tensor:
    return t.zero_()


@torch.no_grad()
def ones_(t: torch.Tensor) -> torch.Tensor:
    return t.fill_(1.0)


def flax_default_(module: nn.Module, generator=None) -> None:
    """Flax defaults on every Linear / Conv2d (lecun normal kernel, zero bias) and norm (scale 1, bias 0) inside
    `module`; modules with special initializers are re-initialized by their owners afterwards."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            lecun_normal_(m.weight, generator)
            if m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            if m.weight is not None:
                ones_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)
