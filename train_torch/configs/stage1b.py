# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Stage 1b of ALPS: 224 pixels (1.0 billion samples), from all Stage-1a parameters with a fresh optimizer."""
from stage1a import get_config as _stage1a


def get_config():
    c = _stage1a()
    c.res = 224
    c.global_batch = 16_384
    c.total_steps = 62_556
    c.warmup_steps = 3_127
    c.peak_lr = 2.56e-5
    c.rec_w = 1.0
    c.dense_radius = 1              # 14 x 14 grid: the 3 x 3 neighbourhood of the positive is excluded
    c.dense_n_same = 64             # 64 same-image negatives, drawn uniformly by Gumbel top-k
    c.init_from = "REQUIRED: <stage1a workdir>"
    return c
