# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""The state columns of the paper's Table 1 from the probe results: position error of every cell divided by the error of the
reference encoder (ALPS (Stage 1)) in the same cell, averaged over the two suites ("now" = readout, "1 s", "3 s") and over
all six cells ("mean"). Cells: LIBERO-Object and LIBERO-Spatial x {readout, future 1 s, future 3 s}, budget f100, 16 history
frames, seed 0, dev_test split.
    python table.py <out_dir of train_probe.py> [--ref alps_stage1]
"""
import argparse
import glob
import json
import os

CELLS = [("libero_object", "readout"), ("libero_object", "future_h1"), ("libero_object", "future_h3"),
         ("libero_spatial", "readout"), ("libero_spatial", "future_h1"), ("libero_spatial", "future_h3")]


def load(out_dir):
    R = {}
    for f in glob.glob(os.path.join(out_dir, "results", "*.json")):
        if f.endswith("_persample.json"):
            continue
        r = json.load(open(f))
        if r["budget"] != "f100" or r["n_hist"] != 16 or r["no_action"]:
            continue
        cell = r["mode"] if r["mode"] == "readout" else f"future_h{r['horizon_s']:g}"
        R.setdefault(r["encoder"], {})[(r["suite"], cell)] = r["dev_test"]["all"]["pos_err_cm_mean"]
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--ref", default="alps_stage1")
    a = ap.parse_args()
    R = load(a.out_dir)
    ref = R[a.ref]
    print(f"{'encoder':22s} {'now':>6s} {'1 s':>6s} {'3 s':>6s} {'mean':>6s}   (position error relative to {a.ref})")
    for enc, cells in sorted(R.items()):
        if not all(c in cells for c in CELLS):
            print(f"{enc:22s} incomplete: {len(cells)} of 6 cells")
            continue
        r = [cells[c] / ref[c] for c in CELLS]
        print(f"{enc:22s} {(r[0] + r[3]) / 2:6.3f} {(r[1] + r[4]) / 2:6.3f} {(r[2] + r[5]) / 2:6.3f} {sum(r) / 6:6.3f}")


if __name__ == "__main__":
    main()
