# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Paired comparison of two SSv2 probe runs on the same validation videos.

    python eval/video/ssv2_pair.py runs/ssv2/siglip2_b16_stage2 runs/ssv2/siglip2_b16 [--epoch 20]

Reads val_preds_ep<epoch>.npz of both runs, matches the videos by name, and reports top-1 of A and B, the difference
A - B with a bootstrap 95% interval over videos (5,000 resamples, numpy default_rng(--seed)), and the numbers of
videos that only A, only B, or both classify correctly.
"""
import argparse
import json
import os

import numpy as np


def correct(run, epoch):
    p = np.load(os.path.join(run, f"val_preds_ep{epoch}.npz"), allow_pickle=True)
    o = np.argsort(p["video"])
    return p["video"][o], p["label"][o], (p["pred"][o] == p["label"][o]).astype(float)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--epoch", type=int, default=20)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    va, ya, ha = correct(args.a, args.epoch)
    vb, yb, hb = correct(args.b, args.epoch)
    assert (va == vb).all() and (ya == yb).all(), "the two runs were evaluated on different videos or labels"
    d = ha - hb
    rng = np.random.default_rng(args.seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(args.n_boot)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    sig = " *" if lo > 0 or hi < 0 else ""
    out = {"top1_a": float(ha.mean()), "top1_b": float(hb.mean()), "diff": float(d.mean()), "ci95": [float(lo), float(hi)],
           "only_a": int(((ha == 1) & (hb == 0)).sum()), "only_b": int(((ha == 0) & (hb == 1)).sum()),
           "both": int(((ha == 1) & (hb == 1)).sum()), "n_videos": int(len(d))}
    print(f"SSv2 val top-1 at epoch {args.epoch}: A {100 * out['top1_a']:.2f}, B {100 * out['top1_b']:.2f}; A - B "
          f"{100 * out['diff']:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]{sig}; only A {out['only_a']}, only B {out['only_b']}, "
          f"both {out['both']} of {out['n_videos']} videos")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
