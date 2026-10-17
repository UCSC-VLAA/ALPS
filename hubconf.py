# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""torch.hub entry points:

    model = torch.hub.load("UCSC-VLAA/ALPS", "alps_vitb16")                                    # weights from Hugging Face
    model = torch.hub.load("UCSC-VLAA/ALPS", "alps_vitb16", weights="alps_vitb16.safetensors")  # a local file
"""
dependencies = ["torch", "numpy", "safetensors"]


def alps_vitb16(weights: str = "alps-vitb16", device: str = "cpu", with_text: bool = True):
    """ALPS (Stage 1 + Stage 2): encoder and language interface."""
    from alps import load
    return load(weights, device=device, with_text=with_text)


def alps_stage1_vitb16(weights: str = "alps-stage1-vitb16", device: str = "cpu", with_text: bool = True):
    """ALPS (Stage 1), before cross-view refinement."""
    from alps import load
    return load(weights, device=device, with_text=with_text)
