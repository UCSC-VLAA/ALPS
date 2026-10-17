# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Step 1 of the weight conversion (JAX environment, no torch needed).

Reads an orbax train-state directory of the JAX training code and writes the parameters needed for inference to a
flat npz ("a/b/c" keys, float32):

    python tools/export_flax_params.py --ckpt runs/stage1b/checkpoints --out stage1b_params.npz

--ckpt is the CheckpointManager directory of train_jax (<workdir>/checkpoints; the original ALPS runs named this
orbax directory checkpoint.npz).

Only params/{visual_encoder, text_encoder, image_projection_head, text_projection_head, logit_scale} are read; the
optimizer state and the training-only heads are not read (partial restore), so the restore reads a fraction of the checkpoint.
Step 2: tools/convert_checkpoint.py --stage1 stage1b_params.npz [--encoder stage2_encoder.npz] --out ....safetensors
"""
import argparse
import os

import numpy as np

KEEP = ("visual_encoder", "text_encoder", "image_projection_head", "text_projection_head", "logit_scale")


def flatten(tree, prefix=""):
    out = {}
    for k, v in tree.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if hasattr(v, "items"):
            out.update(flatten(v, key))
        else:
            out[key] = np.asarray(v, dtype=np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="orbax CheckpointManager directory (contains <step>/default)")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", default=",".join(KEEP))
    a = ap.parse_args()
    import jax
    import orbax.checkpoint as ocp
    keep = tuple(a.keep.split(","))
    path = a.ckpt if "://" in a.ckpt else os.path.abspath(a.ckpt)
    mngr = ocp.CheckpointManager(path, options=ocp.CheckpointManagerOptions(read_only=True))
    step = a.step if a.step is not None else mngr.latest_step()
    meta = mngr.item_metadata(step)
    if meta is None:                         # orbax >= 0.12: managers without registered items return None
        from etils import epath
        base = epath.Path(path) / str(step) / "default"
        meta = ocp.StandardCheckpointHandler().metadata(base)
    elif "default" in meta:
        meta = meta["default"]
    meta = getattr(meta, "tree", meta)
    cpu = jax.sharding.SingleDeviceSharding(jax.local_devices(backend="cpu")[0])

    def abstract(m):
        return jax.ShapeDtypeStruct(tuple(m.shape), m.dtype, sharding=cpu)

    found = [k for k in keep if k in meta["params"]]
    target = {"params": {k: jax.tree.map(abstract, meta["params"][k], is_leaf=lambda x: hasattr(x, "shape"))
                         for k in found}}
    print(f"[orbax] step {step}: restoring params/{{{','.join(found)}}}")
    # Partial restore straight from the item directory: the optimizer state and the other heads are not read.
    from etils import epath
    item_dir = epath.Path(path) / str(step) / "default"
    restore_args = ocp.checkpoint_utils.construct_restore_args(target)
    restored = ocp.PyTreeCheckpointer().restore(
        item_dir, args=ocp.args.PyTreeRestore(item=target, restore_args=restore_args, partial_restore=True))
    flat = flatten({k: v for k, v in jax.device_get(restored["params"]).items() if k in keep})
    np.savez(a.out, step=np.int64(step), **flat)
    n = sum(v.size for v in flat.values())
    print(f"[ok] wrote {a.out}: {len(flat)} arrays, {n / 1e6:.1f}M parameters")


if __name__ == "__main__":
    main()
