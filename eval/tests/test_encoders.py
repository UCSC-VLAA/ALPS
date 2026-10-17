# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the unified encoder interface (random weights; no downloads)."""
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import alps  # noqa: E402
from eval.encoders import PAPER_ROWS, build  # noqa: E402


@pytest.fixture(scope="module")
def alps_file():
    torch.manual_seed(0)
    m = alps.ALPS()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "alps.safetensors")
        alps.save(m.state_dict(), path)
        yield path, m.eval()


def test_alps_entry_matches_the_package(alps_file):
    path, m = alps_file
    enc = build("alps", device="cpu", weights=path)
    fr = np.random.default_rng(0).integers(0, 256, (2, 224, 224, 3), dtype=np.uint8)
    with torch.no_grad():
        ref = m.encode_image(alps.normalize(torch.from_numpy(fr).permute(0, 3, 1, 2).float() / 255.0))["tokens"]
    assert torch.allclose(enc.encode_frames(fr), ref, atol=1e-5)
    assert enc.encode_clip(np.stack([fr, fr], 1)).shape == (2, 196, 768)
    assert enc.encode_frames(fr[:, :160, :208]).shape == (2, 10 * 13, 768)       # other grid: rescaled sin-cos table


def test_timm_entry_shapes():
    pytest.importorskip("timm")
    enc = build("dinov3_b16", device="cpu", pretrained=False)
    tok = enc.encode_frames(np.zeros((1, 224, 224, 3), np.uint8))
    assert tok.shape == (1, 196, 768) and enc.meta()["output"].startswith("forward_features")


def test_paper_rows_cover_table1():
    assert len(PAPER_ROWS) == 17 and PAPER_ROWS["ALPS"] == "alps"
