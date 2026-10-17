# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""Per-channel token mean / std of a frozen backbone on 4,096 Reacher frames (fixed RNG), for frozen_backbone.FrozenTokens.

  python make_stats.py <encoder> --out <dir> [--h5 $STABLEWM_HOME/reacher.h5] [--n 4096] [--weights <safetensors>]
  -> <dir>/reacher_<encoder>.npz with mean, std [768] (float32), frames, tokens

Frames: numpy RandomState(0) choice of n rows of the pixels of reacher.h5 (224 x 224, uint8), ImageNet-normalised as in
the PreJEPA training; tokens = FrozenTokens.tokens (unit length, float32); statistics in float64 over all 196 tokens of
every frame: mean, std = sqrt(max(E[t^2] - mean^2, 1e-12)).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frozen_backbone import FrozenTokens  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--out", required=True)
    ap.add_argument("--h5", default=os.path.join(os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable-wm")), "reacher.h5"))
    ap.add_argument("--n", type=int, default=4096); ap.add_argument("--weights", default=None)
    a = ap.parse_args()
    import h5py
    import hdf5plugin  # noqa: F401  (Blosc filter of the pixels of reacher.h5)
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"reacher_{a.name}.npz")
    if os.path.exists(out):
        sys.exit(f"exists: {out}")
    f = h5py.File(a.h5, "r"); N = f["pixels"].shape[0]
    idx = np.sort(np.random.RandomState(0).choice(N, a.n, replace=False))
    m = FrozenTokens(a.name, stats=None, weights=a.weights).cuda()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    s1 = torch.zeros(768, dtype=torch.float64, device="cuda"); s2 = torch.zeros_like(s1); cnt = 0; t = time.time()
    for i in range(0, a.n, 128):
        x = torch.from_numpy(f["pixels"][idx[i:i + 128]]).cuda().permute(0, 3, 1, 2).float() / 255.0
        z = m.tokens((x - mean) / std).double().reshape(-1, 768)
        s1 += z.sum(0); s2 += (z * z).sum(0); cnt += z.shape[0]
    mu = (s1 / cnt); sd = (s2 / cnt - mu * mu).clamp_min(1e-12).sqrt()
    np.savez(out, mean=mu.cpu().numpy().astype(np.float32), std=sd.cpu().numpy().astype(np.float32), frames=a.n, tokens=cnt)
    print(f"STATS {a.name}: {a.n} frames, {cnt} tokens, mean|mu| {mu.abs().mean():.4f}, mean sd {sd.mean():.4f}, {time.time() - t:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
