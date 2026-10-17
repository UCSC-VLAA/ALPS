# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Small runtime helpers of the trainers: dtype policy, matmul precision, device transfer, metric logging."""
from __future__ import annotations

import json
import os
import time
from typing import Dict

import torch


def compute_dtype(precision: str) -> torch.dtype:
    """'float32' (default; the parity mode) or 'bfloat16' (the compute dtype of the ALPS runs, with the Flax cast
    points: parameters, optimizer state, losses and the float32 islands of the JAX code stay float32)."""
    return {"float32": torch.float32, "fp32": torch.float32, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}[precision]


def set_matmul_precision(allow_tf32: bool = False) -> None:
    """float32 matmuls / convolutions in full precision by default (TF32 off): the JAX code pinned the dense
    similarities to HIGHEST precision and the parity tests assume IEEE float32."""
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def to_device(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


class MetricLogger:
    def __init__(self, workdir: str, enabled: bool = True):
        self.path = os.path.join(workdir, "metrics.jsonl") if enabled else None
        self.t0 = time.time()

    def log(self, step: int, metrics: Dict[str, float]) -> None:
        if self.path is None:
            return
        row = {"step": int(step), "time": round(time.time() - self.t0, 1), **{k: float(v) for k, v in metrics.items()}}
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(" ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
