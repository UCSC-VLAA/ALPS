# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Stage 1 of ALPS: image-text pretraining with contrastive, caption, reconstruction and dense objectives.

    python train_stage1.py --config=configs/stage1a.py --workdir=runs/stage1a --config.data.files='<data>/*.tfrecord'
    python train_stage1.py --config=configs/stage1b.py --workdir=runs/stage1b --config.init_from=runs/stage1a/checkpoints

The global batch must be divisible by the number of devices. A smaller batch needs a proportionally smaller
learning rate (the original configs scale it linearly with the batch).
"""
import functools
import os

import jax
import jax.numpy as jnp
import optax
from absl import app, flags
from ml_collections import config_flags

from alps_jax import runtime
from alps_jax.data.pipelines import stage1_dataset
from alps_jax.losses.stage1 import stage1_loss
from alps_jax.losses.stage2 import normalize_image
from alps_jax.models.stage1 import Stage1Model
from alps_jax.optim import project_logit_scale, stage1_optimizer

config_flags.DEFINE_config_file("config", None, "training config")
flags.DEFINE_string("workdir", None, "output directory")
FLAGS = flags.FLAGS


def build_model(cfg):
    return Stage1Model(**cfg.model.to_dict(), dtype=jnp.bfloat16, param_dtype=jnp.float32)


def init_params(model, cfg, rng_init):
    b, r = 2, cfg.res
    x = jnp.zeros((b, r, r, 3), jnp.float32)
    ids32, ids80 = jnp.zeros((b, cfg.model.itc_len), jnp.int32), jnp.zeros((b, cfg.model.caption_len), jnp.int32)
    rngs = {k: rng_init for k in ("params", "dropout", "drop_path", "random_mask")}
    return model.init(rngs, x_enc=x, gx_enc=x, x_rgb=x, crop_box=jnp.zeros((b, 4)), flip=jnp.zeros((b,), bool),
                      text=ids32, itc_text2=ids32, cap_text=ids80, train=True)["params"]


def make_train_step(model, tx, cfg, rng_loop):
    loss_kw = dict(cap_w=cfg.cap_w, rec_w=cfg.rec_w, dense_w=cfg.dense_w, dense_n_cross=cfg.dense_n_cross,
                   dense_radius=cfg.dense_radius, dense_n_same=(cfg.dense_n_same or None),
                   dense_split=tuple(cfg.dense_split))

    def step(state, batch):
        r = jax.random.fold_in(rng_loop, state["step"])          # per-step key, as in the original code

        def loss_fn(params):
            out = model.apply({"params": params}, x_enc=normalize_image(batch["x_rgb"]),
                              gx_enc=normalize_image(batch["gx_rgb"]), x_rgb=batch["x_rgb"],
                              crop_box=batch["crop_box"], flip=batch["flip"], text=batch["itc_labels1"],
                              itc_text2=batch["itc_labels2"], cap_text=batch["labels"], train=True,
                              rngs={"dropout": r, "drop_path": r, "random_mask": r})
            return stage1_loss(out, r, **loss_kw)

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state["params"])
        updates, opt = tx.update(grads, state["opt"], state["params"])
        params = project_logit_scale(optax.apply_updates(state["params"], updates))
        return {"params": params, "opt": opt, "step": state["step"] + 1}, metrics
    return step


def main(_):
    cfg, workdir = FLAGS.config, FLAGS.workdir
    runtime.init_distributed()
    os.makedirs(workdir, exist_ok=True)
    mesh, data_sh, repl = runtime.make_mesh()
    if cfg.global_batch % jax.device_count():
        raise ValueError(f"global batch {cfg.global_batch} is not divisible by {jax.device_count()} devices")
    model = build_model(cfg)
    rng1, rng_init = jax.random.split(jax.random.PRNGKey(cfg.seed))
    _, rng_loop = jax.random.split(rng1, 2)
    params = jax.jit(functools.partial(init_params, model, cfg), out_shardings=repl)(rng_init)
    if cfg.init_from:                                             # Stage 1b: every subtree from Stage 1a
        loaded, s = runtime.restore_params(cfg.init_from, sharding=repl)
        if set(loaded) != set(params):
            raise ValueError(f"checkpoint subtrees {sorted(loaded)} != model {sorted(params)}")
        params = jax.tree.map(lambda a, b: jnp.asarray(b, a.dtype), params, loaded)
        print(f"[stage1] initialized from {cfg.init_from} step {s}; optimizer and schedule start fresh")
    params = project_logit_scale(params)
    tx, lr_fn = stage1_optimizer(cfg.peak_lr, cfg.total_steps, cfg.warmup_steps, cfg.weight_decay, cfg.b1, cfg.b2)
    state = {"params": params, "opt": jax.jit(tx.init, out_shardings=repl)(params), "step": jnp.int32(0)}
    mngr = runtime.checkpoint_manager(os.path.join(workdir, "checkpoints"), cfg.ckpt_keep, cfg.ckpt_steps)
    if mngr.latest_step() is not None:
        state, s = runtime.restore(mngr, state)
        print(f"[stage1] resumed from step {s}")
    step_fn = jax.jit(make_train_step(model, tx, cfg, rng_loop), donate_argnums=(0,),
                      in_shardings=(repl, data_sh), out_shardings=(repl, repl))
    ds = stage1_dataset(res=cfg.res, per_host_batch=cfg.global_batch // jax.process_count(),
                        files=cfg.data.files or None, tfds_dir=cfg.data.tfds_dir or None, tfds_split=cfg.data.tfds_split,
                        image_key=cfg.data.image_key, web_key=cfg.data.web_key, synthetic_key=cfg.data.synthetic_key,
                        itc_len=cfg.model.itc_len, caption_len=cfg.model.caption_len,
                        shuffle_buffer=cfg.data.shuffle_buffer, process_index=jax.process_index(),
                        process_count=jax.process_count())
    it = ds.as_numpy_iterator()
    log = runtime.MetricLogger(workdir)
    start = int(state["step"])
    for step in range(start, cfg.total_steps):
        state, metrics = step_fn(state, runtime.global_batch(next(it), data_sh))
        if (step + 1) % cfg.log_steps == 0 or step == start:
            log.log(step + 1, {**metrics, "lr": lr_fn(step)})
        runtime.save(mngr, step + 1, state, force=(step + 1 == cfg.total_steps))
    mngr.wait_until_finished()


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
