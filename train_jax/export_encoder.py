# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Write an encoder of a train-state checkpoint to an npz for tools/convert_checkpoint.py.

    python export_encoder.py --ckpt runs/stage2/checkpoints --subtree encoder --out stage2_encoder.npz
    python export_encoder.py --ckpt runs/stage1b/checkpoints --subtree visual_encoder --out stage1_encoder.npz

Layout: param/<flax path> (float32) for every encoder leaf, plus a float32 reference forward on 4 fixed random inputs
(ref_input in [0, 255], ref_z = patch tokens, ref_zglobal = LayerNorm of the mean token), computed on the CPU with
float32 activations so that the PyTorch conversion can be checked to float rounding.
For the language interface, run tools/export_flax_params.py on the Stage-1 checkpoint.
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np

from alps_jax import runtime
from alps_jax.losses.stage2 import normalize_image
from alps_jax.models.vit import ViT


def flat(tree, prefix=""):
    out = {}
    for k, v in tree.items():
        key = f"{prefix}/{k}" if prefix else k
        out.update(flat(v, key)) if hasattr(v, "items") else out.__setitem__(key, np.asarray(v, np.float32))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subtree", default="encoder", help="encoder (Stage 2) or visual_encoder (Stage 1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--heads", type=int, default=0, help="attention heads (default: width / 64)")
    a = ap.parse_args()
    params, step = runtime.restore_params(a.ckpt, subtrees=(a.subtree,))
    p = params[a.subtree]
    ref = np.asarray(jax.random.uniform(jax.random.PRNGKey(12345), (4, a.res, a.res, 3)) * 255.0, np.float32)
    width = p["embedding"]["kernel"].shape[-1]
    enc = ViT(width=width, depth=len(p["Transformer"]), mlp_dim=p["Transformer"]["encoderblock_0"]["MlpBlock_0"]["Dense_0"]["kernel"].shape[1],
              num_heads=a.heads or width // 64, patch_size=p["embedding"]["kernel"].shape[0], dtype=jnp.float32,
              param_dtype=jnp.float32)
    with jax.default_matmul_precision("highest"):
        zg, z = enc.apply({"params": p}, normalize_image(ref / 255.0), train=False)
    out = {f"param/{k}": v for k, v in flat(p).items()}
    out.update(ref_input=ref, ref_z=np.asarray(z, np.float32), ref_zglobal=np.asarray(zg, np.float32),
               step=np.int64(step))
    np.savez(a.out, **out)
    print(f"wrote {a.out}: {len(out) - 4} parameter arrays from step {step}")


if __name__ == "__main__":
    main()
