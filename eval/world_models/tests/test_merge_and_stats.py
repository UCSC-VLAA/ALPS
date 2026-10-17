# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""CPU tests of the result tools: merging of planning shards (tools/merge_plan_shards.py, tools/merge_reacher.py) and
the paired statistics of tools/summarize.py."""
import json
import os
import tempfile

import numpy as np

from _load import TOOLS, load

MP = load(os.path.join(TOOLS, "merge_plan_shards.py"), "wm_merge_plan_shards")
MR = load(os.path.join(TOOLS, "merge_reacher.py"), "wm_merge_reacher")
SU = load(os.path.join(TOOLS, "summarize.py"), "wm_summarize")


def pusht_attempt(succ, dist, rc=0):
    s = ", ".join("True" if x else "False" for x in succ); d = ", ".join("%.3f" % x for x in dist)
    return (f"START 2026-01-01T00:00:00Z cfg=plan_pusht\nSuccess rate:  0.1\n{{'success': array([False]), 'state_dist': array([9.0])}}\n"
            f"Success rate:  {np.mean(succ)}\n{{'success': array([{s}]), 'state_dist': array([{d}])}}\nEND 2026-01-01T01:00:00Z rc={rc} wallclock_s=1\n")


def test_merge_success_tasks_uses_the_last_attempt_and_final_evaluation():
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "pusht_alps_0_2.log"); b = os.path.join(d, "pusht_alps_2_4.log")
        open(a, "w").write(pusht_attempt([0, 0], [5, 5], rc=1) + pusht_attempt([1, 0], [1.0, 30.0]))   # failed first attempt, then rc=0
        open(b, "w").write(pusht_attempt([1, 1], [2.0, 3.0]))
        out = MP.merge([a, b])
        assert out["n"] == 4 and out["missing"] == [] and out["success"] == [1, 0, 1, 1]
        assert abs(out["success_rate"] - 0.75) < 1e-12 and abs(out["mean_state_dist"] - 9.0) < 1e-9


def test_merge_chamfer_tasks_and_missing_shards():
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "rope_alps_0_2.log"); b = os.path.join(d, "rope_alps_2_4.log")
        open(a, "w").write("START x\nCD:  0.9\nCD:  0.8\nCD:  0.5\nCD:  0.7\nEND x rc=0 wallclock_s=1\n")
        open(b, "w").write("START x\nCD:  0.4\nEND x rc=139 wallclock_s=1\n")                     # crashed without a final evaluation
        out = MP.merge([a, b])
        assert out["cd"] == [0.5, 0.7] and out["missing"] == ["rope_alps_2_4.log"] and abs(out["median_cd"] - 0.6) < 1e-12


def test_merge_reacher():
    with tempfile.TemporaryDirectory() as d:
        for i, (s, r) in enumerate((([1, 0], [10, 11]), ([0, 1], [12, 13]))):
            os.makedirs(os.path.join(d, f"shard_{i}_2"))
            np.savez(os.path.join(d, f"shard_{i}_2", "episode_results.npz"), successes=np.array(s, bool), rows=np.array(r))
        out = MR.merge(d, 2)
        assert out["successes"] == [1, 0, 0, 1] and out["rows"] == [10, 11, 12, 13] and out["success_rate"] == 0.5


def test_paired_statistics():
    n10, n01, p = SU.mcnemar([1, 1, 1, 0, 0], [0, 0, 1, 1, 0])
    assert (n10, n01) == (2, 1) and p == 1.0
    assert SU.mcnemar([1, 1], [1, 1])[2] == 1.0
    assert abs(SU.mcnemar([1] * 6, [0] * 6)[2] - 2 / 64) < 1e-12
    assert SU.signflip_p([0, 0, 0]) == 1.0 and abs(SU.signflip_p([1, 1, 1, 1]) - 2 / 16) < 1e-12
    x = np.arange(20, dtype=float); traj = np.repeat(np.arange(5), 4)
    m, lo, hi = SU.boot(x, traj, 1)
    assert m == x.mean() and lo < m < hi
    assert SU.boot(x, traj, 1) == SU.boot(x, traj, 1)                                # seeded
    m2, lo2, hi2 = SU.boot_pair(x, x - 1.0)
    assert m2 == lo2 == hi2 == 1.0
