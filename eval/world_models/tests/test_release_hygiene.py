# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""The world-model folder contains no absolute machine paths, no non-English comments and
no recursive removals in its scripts; the patches touch only the expected upstream files."""
import os
import re

from _load import WM

FORBIDDEN = [r"(?<![\w.])/data\d+/", r"(?<![\w.])/data-\d", r"(?<![\w.])/home/\w", r"(?<![\w.])/Users/\w", r"gs://",   # machine paths
             r"rm -r", r"[\u4e00-\u9fff]"]
TEXT = (".py", ".sh", ".yaml", ".md", ".patch", ".example")


def files():
    for root, _, names in os.walk(WM):
        for n in names:
            if n.endswith(TEXT) and n != "test_release_hygiene.py":
                yield os.path.join(root, n)


def test_no_internal_names_or_paths():
    bad = []
    for f in files():
        text = open(f, encoding="utf-8").read()
        for pat in FORBIDDEN:
            if re.search(pat, text):
                bad.append((os.path.relpath(f, WM), pat))
    assert not bad, bad


def test_patches_touch_only_the_expected_files():
    def touched(p):
        return sorted(re.findall(r"^diff --git a/(\S+) b/", open(os.path.join(WM, p)).read(), re.M))
    assert touched("dino_wm/dino_wm.patch") == sorted([
        "datasets/deformable_env_dset.py", "datasets/point_maze_dset.py", "datasets/pusht_dset.py", "datasets/traj_dset.py",
        "datasets/wall_dset.py", "env/__init__.py", "plan.py", "planning/cem.py", "planning/evaluator.py", "train.py"])
    assert touched("reacher/le_wm.patch") == ["eval.py"]


def test_apache_header_on_new_code():
    for f in files():
        if f.endswith((".py", ".sh")) and not f.endswith("env.sh.example"):
            head = open(f).read(400)
            assert "Licensed under the Apache License, Version 2.0" in head, os.path.relpath(f, WM)
