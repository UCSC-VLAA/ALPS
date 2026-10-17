# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Table of DAVIS runs (davis.py) and paired comparisons over the videos.

    python eval/video/summarize_davis.py runs/davis/*
    python eval/video/summarize_davis.py runs/davis/* --paired runs/davis/alps runs/davis/alps_stage1

Every run directory holds results.json. A paired comparison A vs B takes the per-video J&F (mean over the objects of a
video of (J + F) / 2) on the videos both runs share and reports the mean difference A - B, its median, the number of
videos on which A is better / worse, and a bootstrap 95% interval of the mean difference over videos (10,000 resamples,
numpy RandomState(--seed), a fresh generator for every pair).
"""
import argparse
import json
import os

import numpy as np


def load(run):
    with open(os.path.join(run, "results.json")) as f:
        return json.load(f)


def paired(a, b, n_boot=10000, seed=0):
    vids = sorted(set(a["per_video"]) & set(b["per_video"]))
    d = np.array([a["per_video"][v]["JF"] - b["per_video"][v]["JF"] for v in vids])
    idx = np.random.RandomState(seed).randint(0, len(d), (n_boot, len(d)))
    m = d[idx].mean(1)
    return {"n_videos": len(d), "mean": float(d.mean()), "median": float(np.median(d)), "better": int((d > 0).sum()),
            "worse": int((d < 0).sum()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run directories (each with results.json)")
    ap.add_argument("--paired", nargs=2, action="append", default=[], metavar=("A", "B"), help="paired comparison A - B")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="", help="also write the numbers to this file")
    a = ap.parse_args()
    R = {r.rstrip("/"): load(r) for r in a.runs if os.path.exists(os.path.join(r, "results.json"))}
    for r in a.runs:
        if r.rstrip("/") not in R:
            print(f"(no results.json in {r})")
    print("| run | J&F | J | F | J recall | J decay | F recall | F decay | objects | videos | precision | time (s) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r, d in sorted(R.items(), key=lambda kv: -kv[1]["summary"]["J&F_mean"]):
        s, m = d["summary"], d["meta"]
        print(f"| {os.path.basename(r)} | {s['J&F_mean']:.3f} | {s['J_mean']:.3f} | {s['F_mean']:.3f} | {s['J_recall']:.3f} | "
              f"{s['J_decay']:.3f} | {s['F_recall']:.3f} | {s['F_decay']:.3f} | {s['n_objects']} | {s['n_videos']} | "
              f"{m.get('precision', '?')} | {m.get('time_total_s', float('nan')):.0f} |")
    out = {"summary": {os.path.basename(r): d["summary"] for r, d in R.items()}, "paired": []}
    if a.paired:
        print(f"\npaired per-video J&F, A - B (bootstrap 95% interval over videos, {a.n_boot:,} resamples, seed {a.seed})")
        for x, y in a.paired:
            p = paired(load(x), load(y), a.n_boot, a.seed)
            sig = " *" if p["ci95"][0] > 0 or p["ci95"][1] < 0 else ""
            print(f"  {os.path.basename(x.rstrip('/'))} - {os.path.basename(y.rstrip('/'))}: {p['mean']:+.3f} "
                  f"[{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]{sig}  median {p['median']:+.3f}  better / worse on "
                  f"{p['better']} / {p['worse']} of {p['n_videos']} videos")
            out["paired"].append({"A": x, "B": y, **p})
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
