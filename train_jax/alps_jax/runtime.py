# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Device mesh, batch sharding, checkpoints and logging for the training scripts (pure data parallelism).

Both stages ran data-parallel only: parameters and optimizer state replicated, the batch split over every device.
On a multi-host TPU slice, call `init_distributed()` first; each host feeds its local share of the global batch.
"""
import json
import os
import time
from typing import Any, Dict, Optional

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def init_distributed():
    if os.environ.get("ALPS_SINGLE_HOST") == "1":
        return
    try:
        jax.distributed.initialize()
    except Exception as e:                       # single process (CPU, one GPU host) is fine
        print(f"[runtime] jax.distributed.initialize skipped: {e}")


def make_mesh():
    mesh = Mesh(np.asarray(jax.devices()), ("data",))
    return mesh, NamedSharding(mesh, P("data")), NamedSharding(mesh, P())


def global_batch(local: Dict[str, np.ndarray], data_sharding) -> Dict[str, jax.Array]:
    """Host-local numpy batch -> global jax.Arrays sharded over the data axis."""
    return {k: jax.make_array_from_process_local_data(data_sharding, v) for k, v in local.items()}


def checkpoint_manager(directory: str, max_to_keep: int = 3, save_interval_steps: int = 1000):
    import orbax.checkpoint as ocp
    path = directory if "://" in directory else os.path.abspath(directory)
    return ocp.CheckpointManager(path, options=ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep, save_interval_steps=save_interval_steps, create=True))


def save(mngr, step: int, state: Any, force: bool = False):
    import orbax.checkpoint as ocp
    mngr.save(step, args=ocp.args.StandardSave(state), force=force)


def restore(mngr, target: Any, step: Optional[int] = None):
    """Restore the latest (or a given) step into the structure and shardings of `target`."""
    import orbax.checkpoint as ocp
    step = mngr.latest_step() if step is None else step
    abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), target)
    return mngr.restore(step, args=ocp.args.StandardRestore(abstract)), step


def restore_params(directory: str, subtrees=None, step: Optional[int] = None, sharding=None):
    """Read params (or a subset of top-level params subtrees) from a train-state checkpoint of these scripts or of
    the original training code (orbax, item "default")."""
    import orbax.checkpoint as ocp
    from etils import epath
    path = directory if "://" in directory else os.path.abspath(directory)
    mngr = ocp.CheckpointManager(path, options=ocp.CheckpointManagerOptions(read_only=True))
    step = mngr.latest_step() if step is None else step
    meta = mngr.item_metadata(step)
    if meta is None:
        meta = ocp.StandardCheckpointHandler().metadata(epath.Path(path) / str(step) / "default")
    elif "default" in meta:
        meta = meta["default"]
    meta = getattr(meta, "tree", meta)
    place = sharding or jax.sharding.SingleDeviceSharding(jax.local_devices(backend="cpu")[0])
    keys = list(meta["params"].keys()) if subtrees is None else list(subtrees)
    target = {"params": {k: jax.tree.map(lambda m: jax.ShapeDtypeStruct(tuple(m.shape), m.dtype, sharding=place),
                                         meta["params"][k], is_leaf=lambda x: hasattr(x, "shape")) for k in keys}}
    restored = ocp.PyTreeCheckpointer().restore(
        epath.Path(path) / str(step) / "default",
        args=ocp.args.PyTreeRestore(item=target, restore_args=ocp.checkpoint_utils.construct_restore_args(target),
                                    partial_restore=True))
    return restored["params"], step


class MetricLogger:
    def __init__(self, workdir: str):
        self.path = os.path.join(workdir, "metrics.jsonl") if jax.process_index() == 0 else None
        self.t0 = time.time()

    def log(self, step: int, metrics: Dict[str, Any]):
        if self.path is None:
            return
        row = {"step": int(step), "time": round(time.time() - self.t0, 1),
               **{k: float(np.asarray(v)) for k, v in metrics.items()}}
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(" ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
