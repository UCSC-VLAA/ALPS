# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Data-parallel helpers: process setup for torchrun and autograd-aware collectives for global-batch losses.

The JAX code computes every loss on the GLOBAL batch. Under DistributedDataParallel each rank computes its
"share" of a global loss in a form whose gradient, after DDP's gradient averaging over the W ranks, is exactly the
gradient of the global loss (and whose average over ranks is the global loss value):

* ITC: image and text embeddings are all-gathered WITH gradient (backward = sum of the gradients of all ranks for
  each slot, i.e. a reduce-scatter); each rank takes the cross-entropy of its own rows against the gathered matrix,
  averaged over its rows (OpenCLIP's local_loss + gather_with_grad);
* caption loss: the local sum of the per-token cross-entropies times W, divided by the GLOBAL number of non-pad
  targets (an all-reduce without gradient);
* dense cross-image negatives: one global set of patch indices per step (same random draw on every rank); each
  rank contributes the rows it owns and an all-reduce WITH gradient assembles them (its backward all-reduces the
  gradients, so every owner receives the gradient of every rank's queries);
* all other terms are means over equal per-rank shards, for which DDP's averaging is already exact.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist


class _AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        ctx.rank, ctx.world = dist.get_rank(group), dist.get_world_size(group)
        ctx.n = x.shape[0]
        x = x.contiguous()
        out = [torch.empty_like(x) for _ in range(ctx.world)]
        dist.all_gather(out, x, group=group)
        return torch.cat(out, 0)

    @staticmethod
    def backward(ctx, g):
        g = g.contiguous()
        if dist.get_backend(ctx.group) == "nccl":
            out = torch.empty((ctx.n,) + tuple(g.shape[1:]), dtype=g.dtype, device=g.device)
            dist.reduce_scatter_tensor(out, g, op=dist.ReduceOp.SUM, group=ctx.group)
            return out, None
        g = g.clone()
        dist.all_reduce(g, op=dist.ReduceOp.SUM, group=ctx.group)
        return g[ctx.rank * ctx.n:(ctx.rank + 1) * ctx.n], None


class _AllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        y = x.clone().contiguous()
        dist.all_reduce(y, op=dist.ReduceOp.SUM, group=group)
        return y

    @staticmethod
    def backward(ctx, g):
        g = g.clone().contiguous()
        dist.all_reduce(g, op=dist.ReduceOp.SUM, group=ctx.group)
        return g, None


class Dist:
    """Rank / world size of the default process group (world 1 and identity collectives when not distributed)."""

    def __init__(self, group=None):
        self.enabled = dist.is_available() and dist.is_initialized()
        self.group = group
        self.world = dist.get_world_size(group) if self.enabled else 1
        self.rank = dist.get_rank(group) if self.enabled else 0

    @property
    def distributed(self) -> bool:
        return self.world > 1

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenation of every rank's x along dim 0, differentiable."""
        return _AllGather.apply(x, self.group) if self.distributed else x

    def all_reduce_sum(self, x: torch.Tensor) -> torch.Tensor:
        """Sum over ranks, differentiable."""
        return _AllReduceSum.apply(x, self.group) if self.distributed else x

    def sum_nograd(self, x: torch.Tensor) -> torch.Tensor:
        if not self.distributed:
            return x.detach()
        y = x.detach().clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM, group=self.group)
        return y

    def mean_nograd(self, x: torch.Tensor) -> torch.Tensor:
        return self.sum_nograd(x) / self.world

    def barrier(self):
        if self.distributed:
            dist.barrier(group=self.group)


def init_from_env(backend: Optional[str] = None) -> Dist:
    """Initialize the default process group from the torchrun environment (no-op for a single process)."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
        if backend == "nccl":
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group(backend)
    return Dist()


def device_for_rank() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    return torch.device("cpu")
