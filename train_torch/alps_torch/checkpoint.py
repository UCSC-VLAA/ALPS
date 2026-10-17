# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Training checkpoints (torch.save of {step, model / student, optimizer, config}) and loading of Stage-1 weights
from the formats the trainers accept."""
from __future__ import annotations

import glob
import os
import re
from typing import Dict, Optional

import numpy as np
import torch


class CheckpointManager:
    """<directory>/ckpt_<step>.pt, the newest `keep` kept; writes are atomic (temporary file + rename)."""

    def __init__(self, directory: str, keep: int = 3):
        self.dir, self.keep = directory, keep
        os.makedirs(directory, exist_ok=True)

    def steps(self):
        out = []
        for p in glob.glob(os.path.join(self.dir, "ckpt_*.pt")):
            m = re.search(r"ckpt_(\d+)\.pt$", p)
            if m:
                out.append(int(m.group(1)))
        return sorted(out)

    def latest_step(self) -> Optional[int]:
        s = self.steps()
        return s[-1] if s else None

    def path(self, step: int) -> str:
        return os.path.join(self.dir, f"ckpt_{step:08d}.pt")

    def save(self, step: int, state: dict) -> str:
        p = self.path(step)
        tmp = p + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, p)
        for s in self.steps()[:-self.keep] if self.keep else []:
            try:
                os.remove(self.path(s))
            except FileNotFoundError:
                pass
        return p

    def load(self, step: Optional[int] = None, map_location="cpu") -> dict:
        step = self.latest_step() if step is None else step
        if step is None:
            raise FileNotFoundError(f"no checkpoint in {self.dir}")
        return torch.load(self.path(step), map_location=map_location, weights_only=False)


def _latest_file(path: str) -> str:
    if os.path.isdir(path):
        sub = os.path.join(path, "checkpoints") if os.path.isdir(os.path.join(path, "checkpoints")) else path
        m = CheckpointManager.__new__(CheckpointManager)
        m.dir, m.keep = sub, 0
        s = m.latest_step()
        if s is None:
            raise FileNotFoundError(f"no ckpt_*.pt in {sub}")
        return m.path(s)
    return path


def load_stage1_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Stage-1 weights as a Stage1Model state dict (or the subset an `alps` file holds) from: a checkpoint
    directory or file of train_stage1.py (newest step), an `alps` .safetensors / .pt export, or a flat .npz of the
    JAX Stage-1 params (tools/export_flax_params.py)."""
    path = _latest_file(path)
    if path.endswith(".npz"):
        from .convert import stage1_from_flax
        d = np.load(path)
        return stage1_from_flax({k: d[k] for k in d.files})
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def encoder_state_dict(sd: Dict[str, torch.Tensor], prefix: str = "visual.") -> Dict[str, torch.Tensor]:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix) and not k.endswith("pos_embed")}
