# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""ALPS training code (PyTorch): Stage 1 (image-text pretraining) and Stage 2 (cross-view refinement).

A re-implementation of the JAX/Flax training code in `train_jax/`, with the same parameter layout, initializers,
numerics, losses, optimizer and data transforms. The model classes build on the inference package `alps`
(`alps.vit.VisionTransformer`, `alps.text.TextEncoder`), so trained weights export directly to the `alps` format.
"""
import os
import sys

try:                                    # the inference package lives at the repository root
    import alps  # noqa: F401
except ImportError:                     # pragma: no cover - running from a source checkout without `pip install -e .`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import alps  # noqa: F401

__version__ = "0.1.0"
