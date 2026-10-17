# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""Merges the 5 shards of one Reacher planning run (one encoder, one planner seed) into one JSON line (Tables 3 and S12).

  python merge_reacher.py <$STABLEWM_HOME/<run>_cem<seed>> --encoder <name> --seed <planner seed> [--shards 5] > out.json

Reads shard_<i>_<n>/episode_results.npz (written by eval.py with le_wm.patch): successes in episode order, the dataset rows
of the episodes (the same 50 episodes for every encoder and planner seed: top-level seed 42 of reacher.yaml).
"""
import argparse
import json
import os

import numpy as np


def merge(folder, n_shards=5):
    s, rows, miss = [], [], []
    for i in range(n_shards):
        f = os.path.join(folder, f"shard_{i}_{n_shards}", "episode_results.npz")
        if not os.path.exists(f):
            miss.append(i); continue
        z = np.load(f)
        s.append(z["successes"].astype(bool)); rows.append(np.asarray(z["rows"]))
    x = np.concatenate(s) if s else None
    r = np.concatenate(rows) if rows else None
    return {"missing_shards": miss, "successes": [int(v) for v in x] if x is not None else None,
            "rows": [int(v) for v in r] if r is not None else None, "success_rate": float(x.mean()) if x is not None else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("--encoder", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--shards", type=int, default=5)
    a = ap.parse_args()
    print(json.dumps({"encoder": a.encoder, "planner_seed": a.seed, **merge(a.folder, a.shards)}))


if __name__ == "__main__":
    main()
