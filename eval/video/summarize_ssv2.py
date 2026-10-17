# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Table of SSv2 probe runs (ssv2_probe.py): validation top-1 at epochs 20 and 10, holdout top-1, training accuracy, cost.

    python eval/video/summarize_ssv2.py runs/ssv2/*

Every run directory holds results.json, log.jsonl and meta.json. The reported number of the paper is the validation
top-1 at the last epoch (val_top1_final, epoch 20). Paired comparisons of two runs: ssv2_pair.py.
"""
import argparse
import json
import os


def load(run):
    with open(os.path.join(run, "results.json")) as f:
        res = json.load(f)
    ep = {}
    log = os.path.join(run, "log.jsonl")
    if os.path.exists(log):
        with open(log) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    ep[rec["epoch"]] = rec                   # the last record of an epoch (a resume may log it again)
    world = res.get("world") or res["meta"]["world"]
    gpu_h = sum(r["train_time_s"] + r.get("holdout_time_s", 0) + r.get("val_time_s", 0) for r in ep.values()) * world / 3600
    return res, ep, gpu_h


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run directories (each with results.json)")
    a = ap.parse_args()
    rows = []
    for r in a.runs:
        if not os.path.exists(os.path.join(r, "results.json")):
            print(f"(no results.json in {r})")
            continue
        res, ep, gpu_h = load(r)
        last = max(ep) if ep else None
        hp = max(ep.values(), key=lambda x: x.get("holdout_top1", 0)) if ep else {}
        rows.append((res.get("val_top1_final") or 0.0, os.path.basename(r.rstrip("/")), res, ep, last, hp, gpu_h))

    def pct(x):
        return f"{100 * x:.1f}" if isinstance(x, (int, float)) else "-"

    print("| run | val top-1 (final) | val epoch 10 | holdout (final) | holdout peak (epoch) | train acc (final) | GPUs | GPU-h |")
    print("|---|---|---|---|---|---|---|---|")
    for _, name, res, ep, last, hp, gpu_h in sorted(rows, key=lambda x: -x[0]):
        print(f"| {name} | {pct(res.get('val_top1_final'))} | {pct(res['val_by_epoch'].get('10'))} | "
              f"{pct(ep[last].get('holdout_top1') if last else None)} | {pct(hp.get('holdout_top1'))} ({hp.get('epoch', '-')}) | "
              f"{pct(ep[last].get('train_acc') if last else None)} | {res['meta']['world']} x {res['meta']['per_gpu_batch']} | {gpu_h:.1f} |")


if __name__ == "__main__":
    main()
