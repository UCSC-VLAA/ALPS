# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Split files of the SSv2 probe from the official Something-Something v2 labels.

    python eval/video/make_ssv2_csv.py --videos <dir with <id>.webm> --labels <dir with labels.json, train.json,
        validation.json> --out <dir>

Writes '<path> <class index>' lists for the vjepa2 VideoDataset:
  train.csv       official train minus the holdout (166,913 videos)
  holdout.csv     2,000 official training videos (np.random.RandomState(0).permutation over the official train order),
                  used for monitoring only, so that the official validation set is only used for the reported number
  val.csv         official validation (24,777 videos)
  train_full.csv  official train
  splits.json     counts and hashes; sha256_16 hashes the file as written (depends on --videos), basename_sha256_16
                  hashes '<id>.webm <class>' lines and does not depend on where the videos are stored
Class indices are those of labels.json (174 classes), looked up by the template of every video with its brackets removed.
"""
import argparse
import hashlib
import json
import os

import numpy as np


def rows(videos, labels, split, classes):
    with open(os.path.join(labels, f"{split}.json")) as f:
        entries = json.load(f)
    out = []
    for e in entries:
        p = os.path.join(videos, f"{e['id']}.webm")
        assert os.path.exists(p), p
        assert " " not in p, f"paths must not contain spaces (space-separated csv): {p}"
        out.append((p, int(classes[e["template"].replace("[", "").replace("]", "")])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--videos", required=True, help="directory of the official <id>.webm files")
    ap.add_argument("--labels", required=True, help="directory with labels.json, train.json, validation.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    with open(os.path.join(a.labels, "labels.json")) as f:
        classes = json.load(f)
    assert len(classes) == 174, len(classes)
    tr, va = rows(a.videos, a.labels, "train", classes), rows(a.videos, a.labels, "validation", classes)
    perm = np.random.RandomState(a.seed).permutation(len(tr))
    ho_idx, tr_idx = sorted(perm[:a.holdout].tolist()), sorted(perm[a.holdout:].tolist())
    os.makedirs(a.out, exist_ok=True)
    splits = {}
    for name, R in [("train", [tr[i] for i in tr_idx]), ("holdout", [tr[i] for i in ho_idx]), ("val", va), ("train_full", tr)]:
        txt = "".join(f"{p} {lab}\n" for p, lab in R)
        with open(os.path.join(a.out, f"{name}.csv"), "w") as f:
            f.write(txt)
        base = "".join(f"{os.path.basename(p)} {lab}\n" for p, lab in R)
        splits[name] = {"n": len(R), "sha256_16": hashlib.sha256(txt.encode()).hexdigest()[:16],
                        "basename_sha256_16": hashlib.sha256(base.encode()).hexdigest()[:16], "n_classes": len({lab for _, lab in R})}
    splits["note"] = (f"train = official train minus holdout ({a.holdout:,} videos, np.random.RandomState({a.seed}).permutation); "
                      "val = official validation (reporting set)")
    with open(os.path.join(a.out, "splits.json"), "w") as f:
        json.dump(splits, f, indent=1)
    print(json.dumps(splits, indent=1))


if __name__ == "__main__":
    main()
