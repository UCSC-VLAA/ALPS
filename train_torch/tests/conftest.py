# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_TORCH = os.path.dirname(HERE)
REPO = os.path.dirname(TRAIN_TORCH)
TRAIN_JAX = os.path.join(REPO, "train_jax")

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
for p in (TRAIN_TORCH, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
if os.path.isdir(TRAIN_JAX) and TRAIN_JAX not in sys.path:
    sys.path.append(TRAIN_JAX)          # alps_jax, for the parity tests
