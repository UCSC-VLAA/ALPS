# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Per-channel mean and standard deviation of the frozen Stage-1 tokens (the Stage-2 target standardization).

    python compute_token_stats.py --stage1 runs/stage1b --sources images.jsonl --out token_stats.npz \
        [--images 32768] [--crop alps|teacher] [--precision bfloat16]

As train_jax/compute_token_stats.py: tokens in inference mode (bfloat16 by default, as in the original), accumulated
per channel over all tokens of all images (float32 sums per batch, float64 across batches), std = sqrt(max(E[z^2] -
E[z]^2, 1e-8)). --crop alps (default): the recipe of the released statistics file, one square crop of area U(0.8, 1)
of the central square per image, bilinear without antialias; --crop teacher: the teacher view B of the training
pairs. The statistics of ALPS itself are train_jax/assets/alps_stage1b_token_stats.npz (32,768 images).
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alps_torch  # noqa: E402,F401  (puts the inference package `alps` on the path)
from alps.transforms import normalize  # noqa: E402
from alps_torch.checkpoint import encoder_state_dict, load_stage1_state_dict  # noqa: E402
from alps_torch.data import transforms as X  # noqa: E402
from alps_torch.data.datasets import Stage2Dataset, decode_image, make_loader, make_source  # noqa: E402
from alps_torch.distributed import device_for_rank  # noqa: E402
from alps_torch.models import ViT  # noqa: E402
from alps_torch.runtime import compute_dtype, set_matmul_precision  # noqa: E402


class _Crops(Stage2Dataset):
    def __init__(self, sources, res, crop, seed):
        super().__init__([(s, 1.0) for s in sources], res, seed=seed)
        self.crop = crop

    def process(self, r, rng):
        img = decode_image(self._image(r, rng))
        if self.crop == "alps":
            return {"x_t_rgb": X.token_stats_crop(img, self.res, rng.random(3, dtype=np.float32))}
        return {"x_t_rgb": X.make_crossview_pair(img, self.res, X.crossview_draws(rng), self.cfg)["x_t_rgb"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1", required=True, help="train_stage1.py workdir / checkpoint, alps export or JAX npz")
    ap.add_argument("--sources", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", type=int, default=32_768)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--crop", choices=("alps", "teacher"), default="alps")
    ap.add_argument("--precision", default="bfloat16")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--mlp_dim", type=int, default=3072)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    set_matmul_precision(False)
    device = device_for_rank()
    enc = ViT(a.width, a.depth, a.mlp_dim, a.heads, img_size=a.res, dtype=compute_dtype(a.precision))
    enc.load_state_dict(encoder_state_dict(load_stage1_state_dict(a.stage1)), strict=True)
    enc = enc.to(device).eval()
    for s in a.sources:
        make_source(s)                                     # fail early on a bad source
    loader = make_loader(_Crops(a.sources, a.res, a.crop, a.seed), a.batch, a.workers)
    s1 = s2 = 0.0
    n = 0
    with torch.no_grad():
        for i, b in enumerate(loader):
            if i * a.batch >= a.images:
                break
            z = enc(normalize(b["x_t_rgb"].to(device)), train=False)["tokens"].float().reshape(-1, a.width)
            s1 = s1 + z.sum(0).double().cpu().numpy()
            s2 = s2 + (z * z).sum(0).double().cpu().numpy()
            n += z.shape[0]
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-8))
    np.savez(a.out, mean=mean.astype(np.float32), std=std.astype(np.float32),
             n_images=np.int64(n // ((a.res // 16) ** 2)))
    print(f"wrote {a.out}: {n} tokens; std range {std.min():.3f} .. {std.max():.3f}")


if __name__ == "__main__":
    main()
