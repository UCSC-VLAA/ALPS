# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""The 100 held-out Reacher episodes of the prediction evaluation.

  python reacher_collect_heldout.py <out.h5>

The PreJEPA world models are trained on a random split of the CLIPS of reacher.h5, so no episode of that file is held out.
This collects 100 new episodes with the procedure of the dataset (stable-worldmodel scripts/data/collect_reacher.py:
swm/ReacherDMControl-v0, RandomPolicy, 200 steps, 224 x 224 pixels, 10 parallel environments); only the seed differs
(numpy default_rng(20261001) draws the policy seed and the collection seed). Rendering: MuJoCo EGL.
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import hdf5plugin  # noqa: E402,F401  (Blosc filter of the Reacher HDF5 files)
import numpy as np  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.policy import RandomPolicy  # noqa: E402


def main():
    out = sys.argv[1]
    assert not os.path.exists(out), out
    world = swm.World("swm/ReacherDMControl-v0", num_envs=10, max_episode_steps=200, image_shape=[224, 224])
    rng = np.random.default_rng(20261001)
    world.set_policy(RandomPolicy(seed=rng.integers(0, 1_000_000).item()))
    world.collect(out, episodes=100, seed=rng.integers(0, 1_000_000).item(), options=None, format="hdf5")
    print("[collect] done", out, flush=True)


if __name__ == "__main__":
    main()
