# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""Train-trajectory order files of the world-model experiments: <out>/<task>_100.json for the DINO-WM tasks.

  python make_subsets.py --data $DATASET_DIR --out <subsets dir> [--tasks pusht wall point_maze granular rope]

Each file lists ALL trajectories of the official train split ("indices" = positions in the train split as the DINO-WM
loader returns it), permuted once with numpy RandomState(0). The loaders read it through env.dataset.traj_subset
(datasets/traj_dset.py apply_traj_subset): the train set is the same as the official one, in this order, which fixes the
order of the training slices together with the training seed. The validation split is not touched.
Train split per task: PushT = the train folder of pusht_noise (18,685 trajectories); Wall, PointMaze, Granular, Rope = the
seed-42 random 0.9 split of datasets/traj_dset.py (1,728 / 1,800 / 900 / 900 trajectories); "global_ids" are the
underlying episode ids (information only). The files only depend on the number of trajectories; the md5 sums of the files
used in the paper are listed in EXPECTED_MD5.
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch

SEED = 0
EXPECTED_MD5 = {"pusht_100.json": "f8fc74fee2830bb24f7143a958928211", "wall_100.json": "aa86757c049f697cfa3586c9005af4df",
                "point_maze_100.json": "197b30080117d65b3125d00ff894f5cd", "granular_100.json": "0b03b11c6fd81c8c133afbb7799bcb68",
                "rope_100.json": "dd7ace48e4db1c1b16bf94bec9789b6a"}


def pusht_train_positions(data):
    p = os.path.join(data, "pusht_noise/train/seq_lengths.pkl")
    with open(p, "rb") as f:
        n = len(pickle.load(f))
    return n, None


def split_train_positions(states_path):
    """datasets/traj_dset.py split_traj_datasets(train_fraction=0.9, random_seed=42), without loading images."""
    n_total = torch.load(states_path, weights_only=True).shape[0]
    n_train = int(0.9 * n_total)
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(42)).tolist()
    return n_train, perm[:n_train]           # position i of the train split = episode perm[i]


SOURCES = {
    "pusht": pusht_train_positions,
    "wall": lambda d: split_train_positions(os.path.join(d, "wall_single/states.pth")),
    "point_maze": lambda d: split_train_positions(os.path.join(d, "point_maze/states.pth")),
    "granular": lambda d: split_train_positions(os.path.join(d, "deformable/granular/states.pth")),
    "rope": lambda d: split_train_positions(os.path.join(d, "deformable/rope/states.pth")),
}


def subset_record(task, n_train, global_ids=None):
    perm = np.random.RandomState(SEED).permutation(n_train)
    idx = perm[:n_train].tolist()
    rec = {"task": task, "budget": 100, "seed": SEED, "n_train_total": n_train, "n": n_train, "indices": idx,
           "index_space": "position in the official train split (see make_subsets.py)",
           "nested": "prefix of the 100% list"}
    if global_ids is not None:
        rec["global_ids"] = [global_ids[i] for i in idx]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("DATASET_DIR"), help="DINO-WM data folder (default: env DATASET_DIR)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="+", default=list(SOURCES))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import hashlib
    for task in a.tasks:
        n_train, global_ids = SOURCES[task](a.data)
        p = os.path.join(a.out, f"{task}_100.json")
        with open(p, "w") as f:
            json.dump(subset_record(task, n_train, global_ids), f)
        md5 = hashlib.md5(open(p, "rb").read()).hexdigest()
        ok = "matches the paper's file" if md5 == EXPECTED_MD5.get(os.path.basename(p)) else "DIFFERS from the paper's file"
        print(f"{p}: {n_train} train trajectories, md5 {md5} ({ok})")


if __name__ == "__main__":
    main()
