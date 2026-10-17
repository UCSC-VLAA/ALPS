# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Export trained weights to the `alps` inference format (safetensors), readable by alps.load / alps.load_encoder.

    python export.py --stage1 runs/stage1b --out alps_stage1_vitb16.safetensors
    python export.py --stage1 runs/stage1b --stage2 runs/stage2 --out alps_vitb16.safetensors
    python export.py --stage2 runs/stage2 --encoder-only --out alps_encoder.safetensors

--stage1: a train_stage1.py workdir or checkpoint file, an `alps` export, or a flat .npz of JAX Stage-1 params.
--stage2: a train_stage2.py workdir or checkpoint file; its encoder replaces the Stage-1 encoder (the released ALPS
model = the Stage-2 encoder with the unchanged Stage-1 language interface).
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alps_torch import convert  # noqa: E402
from alps_torch.checkpoint import _latest_file, encoder_state_dict, load_stage1_state_dict  # noqa: E402


def stage2_encoder(path: str):
    st = torch.load(_latest_file(path), map_location="cpu", weights_only=False)
    sd = st["student"] if "student" in st else st
    return encoder_state_dict(sd, "encoder.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1")
    ap.add_argument("--stage2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder-only", action="store_true")
    ap.add_argument("--name", default="")
    a = ap.parse_args(argv)
    if not (a.stage1 or a.stage2):
        raise SystemExit("pass --stage1 and/or --stage2")
    s1 = load_stage1_state_dict(a.stage1) if a.stage1 else None
    enc = stage2_encoder(a.stage2) if a.stage2 else None
    meta = {"name": a.name or os.path.basename(a.out),
            "sources": " ".join(os.path.basename(os.path.normpath(p)) for p in (a.stage1, a.stage2) if p)}
    sd = convert.export_alps(a.out, s1, encoder_sd=enc, metadata=meta, encoder_only=a.encoder_only or s1 is None)
    n = sum(v.numel() for v in sd.values())
    print(f"[ok] wrote {a.out}: {len(sd)} tensors, {n / 1e6:.1f}M parameters")


if __name__ == "__main__":
    main()
