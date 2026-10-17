# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Convert checkpoints of the JAX training code into `alps` safetensors files.

Sources
  --stage1 NPZ          Stage-1 pretrainer parameters as a flat npz ("a/b/c" keys), written from the orbax
                        train state by tools/export_flax_params.py (JAX environment).
  --encoder NPZ         An encoder dump (keys "param/...", plus ref_input / ref_z / ref_zglobal), as written
                        by the Stage-1 and Stage-2 encoder dumpers. With --stage1 it replaces the Stage-1
                        encoder: the final ALPS model is the Stage-2 encoder with the Stage-1 language
                        interface.
  --legacy-port PT      A state dict of the earlier PyTorch port of the encoder.

Examples
  python tools/export_flax_params.py --ckpt runs/stage1b/checkpoints --out stage1b_params.npz        # JAX env
  python train_jax/export_encoder.py --ckpt runs/stage2/checkpoints --subtree encoder --out stage2_encoder.npz  # JAX env
  python tools/convert_checkpoint.py --stage1 stage1b_params.npz --encoder stage2_encoder.npz --out alps_vitb16.safetensors
  python tools/convert_checkpoint.py --stage1 stage1b_params.npz --out alps_stage1_vitb16.safetensors
  python tools/convert_checkpoint.py --encoder stage2_encoder.npz --out alps_encoder.safetensors

When the encoder dump carries a reference forward, the converted encoder is run on it and the agreement is
printed (a float32 reference must agree to float rounding; a bfloat16 reference to a cosine above 0.99 per token).
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from alps.convert import flatten, legacy_port_to_torch, pretrainer_to_torch, vit_to_torch  # noqa: E402
from alps.hub import save  # noqa: E402
from alps.vit import VisionTransformer, vit_b16  # noqa: E402

def read_stage1(path):
    """Flat npz of the Stage-1 pretrainer parameters, as written by tools/export_flax_params.py."""
    d = np.load(path)
    return {k[len("params/"):] if k.startswith("params/") else k: d[k] for k in d.files if k != "step"}


def read_encoder_dump(path):
    d = np.load(path)
    P = {k[len("param/"):]: d[k] for k in d.files if k.startswith("param/")}
    ref = {k: d[k] for k in ("ref_input", "ref_z", "ref_zglobal") if k in d.files}
    return P, ref


def check_against_reference(sd, ref):
    depth = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("visual.blocks."))
    enc = vit_b16(pos_mode="recompute") if depth == 12 else VisionTransformer(depth=depth, pos_mode="recompute")
    enc = enc.eval()
    enc.load_state_dict({k[len("visual."):]: v for k, v in sd.items() if k.startswith("visual.")}, strict=True)
    from alps.transforms import normalize
    x = normalize(torch.from_numpy(ref["ref_input"].astype(np.float32) / 255.0).permute(0, 3, 1, 2))
    with torch.no_grad():
        o = enc(x)
    z, rz = o["tokens"].numpy(), ref["ref_z"].astype(np.float32)
    cos_tok = (z * rz).sum(-1) / (np.linalg.norm(z, axis=-1) * np.linalg.norm(rz, axis=-1) + 1e-12)
    zg, rzg = o["pooled"].numpy(), ref["ref_zglobal"].astype(np.float32)
    cos_g = (zg * rzg).sum(-1) / (np.linalg.norm(zg, axis=-1) * np.linalg.norm(rzg, axis=-1) + 1e-12)
    print(f"[check] per-token cosine: min {cos_tok.min():.5f} mean {cos_tok.mean():.5f}; pooled cosine min {cos_g.min():.6f}")
    if cos_tok.min() < 0.99 or cos_g.min() < 0.999:
        raise SystemExit("[check] FAILED: the converted encoder does not reproduce the reference forward")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1")
    ap.add_argument("--encoder")
    ap.add_argument("--legacy-port")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    ref = {}
    enc_tree = None
    if a.encoder:
        enc_tree, ref = read_encoder_dump(a.encoder)
    if a.stage1:
        sd = pretrainer_to_torch(read_stage1(a.stage1), visual=enc_tree)
    elif enc_tree is not None:
        sd = vit_to_torch(enc_tree)
    elif a.legacy_port:
        old = torch.load(a.legacy_port, map_location="cpu", weights_only=True)
        sd = legacy_port_to_torch(old)
    else:
        raise SystemExit("pass --stage1, --encoder or --legacy-port")
    if ref:
        check_against_reference(sd, ref)
    meta = {"format": "alps-torch-v1", "name": a.name or os.path.basename(a.out),
            "sources": " ".join(os.path.basename(p) for p in (a.stage1, a.encoder, a.legacy_port) if p)}
    save(sd, a.out, meta)
    n = sum(v.numel() for v in sd.values())
    print(f"[ok] wrote {a.out}: {len(sd)} tensors, {n / 1e6:.1f}M parameters")


if __name__ == "__main__":
    main()
