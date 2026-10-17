# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""CPU tests of the training recipe: learning-rate schedule, budgets, data streams (wm_train.py, reacher_wm_train.py)."""
import math
import os

import numpy as np

from _load import OVERLAY, REACHER, load

TRAINER = load(os.path.join(OVERLAY, "wm_train.py"), "wm_train_under_test")
REACHER_TRAINER = load(os.path.join(REACHER, "reacher_wm_train.py"), "reacher_wm_train_under_test")


def test_budgets_of_the_paper():
    assert TRAINER.SCHEDULES["pusht"][:2] == (31000, 29500)
    assert TRAINER.SCHEDULES["wall"][:2] == TRAINER.SCHEDULES["point_maze"][:2] == (15500, 14700)
    assert TRAINER.SCHEDULES["granular"][:2] == TRAINER.SCHEDULES["rope"][:2] == (7750, 7350)
    for task, (total, cos_end, saves) in TRAINER.SCHEDULES.items():
        TRAINER.check_schedule(total, 200, cos_end, 0.01)
        assert saves[-1] == total and saves == sorted(set(saves)) and len([s for s in saves if s >= cos_end]) == 6, task
    assert math.isclose(5e-4 * math.sqrt(256 / TRAINER.REFERENCE_BATCH), 1.41421356e-3, rel_tol=1e-8)


def test_schedule_shape():
    f = lambda u: TRAINER.lr_factor(u, 200, 29500, 0.01)  # noqa: E731
    assert f(0) == 1 / 200 and f(99) == 0.5 and f(199) == 1.0
    vals = [f(u) for u in range(200, 29500)]
    assert all(a >= b for a, b in zip(vals, vals[1:])) and abs(vals[-1] - 0.01) < 1e-12
    assert f(29500) == f(30999) == 0.01
    # update s = 14850 is the midpoint of the cosine: 0.01 + 0.99 / 2
    assert abs(f(14849) - (0.01 + 0.99 * 0.5)) < 1e-12


def test_reacher_schedule_equals_the_dino_wm_one():
    for u in (0, 150, 199, 200, 1000, 24699, 24700, 25999):
        assert REACHER_TRAINER.lr_factor(u, 200, 24700, 0.01) == TRAINER.lr_factor(u, 200, 24700, 0.01)


def test_stream_positions_tile_the_global_batch():
    n, pb, world = 1000, 32, 8
    gb = pb * world
    for u in (0, 1, 3, 7):
        pos = [p for r in range(world) for p in TRAINER.stream_positions(u, r, gb, pb, n)]
        assert len(set(pos)) == gb and pos == [(gb * u + j) % n for j in range(gb)]


def test_reacher_epoch_stream_matches_the_formula():
    seed, n, pb, world = 1, 1000, 32, 8
    gb = pb * world
    for u in (0, 3, 4, 11):
        for r in (0, 5):
            want = []
            for j in range(pb):
                p = gb * u + pb * r + j
                want.append(int(np.random.RandomState(seed * 100003 + p // n).permutation(n)[p % n]))
            assert REACHER_TRAINER.epoch_stream(seed, n, u, r, gb, pb) == want
