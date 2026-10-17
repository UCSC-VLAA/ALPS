# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Split files of the SSv2 probe (make_ssv2_csv.py) on a synthetic label set.

    pytest eval/video/tests/test_ssv2_splits.py
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common  # noqa: E402

SCRIPT = os.path.join(_common.REPO, "eval", "video", "make_ssv2_csv.py")


def _synthetic(root, n_train=2600, n_val=300, seed=0):
    rng = np.random.RandomState(seed)
    labels = os.path.join(root, "labels")
    videos = os.path.join(root, "videos")
    os.makedirs(labels)
    os.makedirs(videos)
    classes = {f"Moving something {i} [something]".replace("[", "").replace("]", ""): str(i) for i in range(174)}
    with open(os.path.join(labels, "labels.json"), "w") as f:
        json.dump(classes, f)
    for split, n, off in (("train", n_train, 0), ("validation", n_val, 100000)):
        ents = [{"id": str(off + i), "template": f"Moving something {c} [something]", "placeholders": []}
                for i, c in enumerate(rng.randint(0, 174, n))]
        with open(os.path.join(labels, f"{split}.json"), "w") as f:
            json.dump(ents, f)
        for e in ents:
            open(os.path.join(videos, e["id"] + ".webm"), "w").close()
    return videos, labels


def _read(path):
    with open(path) as f:
        return f.read()


def test_splits():
    with tempfile.TemporaryDirectory() as tmp:
        videos, labels = _synthetic(tmp)
        out = os.path.join(tmp, "csv")
        subprocess.run([sys.executable, SCRIPT, "--videos", videos, "--labels", labels, "--out", out], check=True, capture_output=True)
        rows = {k: [ln.split() for ln in _read(os.path.join(out, f"{k}.csv")).splitlines()] for k in ("train", "holdout", "val", "train_full")}
        assert (len(rows["train"]), len(rows["holdout"]), len(rows["val"]), len(rows["train_full"])) == (600, 2000, 300, 2600)
        full = [r[0] for r in rows["train_full"]]
        perm = np.random.RandomState(0).permutation(len(full))
        assert [r[0] for r in rows["holdout"]] == [full[i] for i in sorted(perm[:2000])]
        assert sorted(r[0] for r in rows["train"] + rows["holdout"]) == sorted(full)
        cls = json.load(open(os.path.join(labels, "labels.json")))
        ents = {e["id"]: e for e in json.load(open(os.path.join(labels, "train.json")))}
        for p, lab in rows["train_full"][:50]:
            assert int(lab) == int(cls[ents[os.path.basename(p)[:-5]]["template"].replace("[", "").replace("]", "")])
        splits = json.load(open(os.path.join(out, "splits.json")))
        assert splits["holdout"]["n"] == 2000 and "basename_sha256_16" in splits["val"]


if __name__ == "__main__":
    sys.exit(_common.run_all(dict(globals())))
