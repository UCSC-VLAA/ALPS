# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Stage 2 of ALPS: cross-view self-distillation on single images.

    python train_stage2.py --config=configs/stage2.py --workdir=runs/stage2 \
        --config.stage1_ckpt=runs/stage1b/checkpoints --config.data.sources="(('<data>/*.tfrecord', 1.0),)"

The student encoder starts from the Stage-1 encoder; a frozen copy of it is the teacher. The language interface of
Stage 1 is not touched: the released model combines the Stage-2 encoder with the Stage-1 text encoder and projections.
"""
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags
from ml_collections import config_flags

from alps_jax import runtime
from alps_jax.data.pipelines import stage2_dataset
from alps_jax.losses.stage2 import crossview_loss
from alps_jax.models.predictor import CrossViewFusion, FeatHead
from alps_jax.models.vit import ViT
from alps_jax.optim import stage2_optimizer

config_flags.DEFINE_config_file("config", None, "training config")
flags.DEFINE_string("workdir", None, "output directory")
FLAGS = flags.FLAGS


def build_models(cfg):
    ec = cfg.encoder
    enc = ViT(width=ec.width, depth=ec.depth, mlp_dim=ec.mlp_dim, num_heads=ec.num_heads, patch_size=ec.patch_size,
              dtype=jnp.bfloat16, param_dtype=jnp.float32)
    pc = cfg.predictor
    pred = CrossViewFusion(width=pc.width, depth=pc.depth, num_heads=pc.num_heads, grid=cfg.res // ec.patch_size,
                           token_dim=ec.width, text_dim=pc.text_dim, mlp_ratio=pc.mlp_ratio, dtype=jnp.bfloat16,
                           param_dtype=jnp.float32)
    head = FeatHead(ec.width, dtype=jnp.bfloat16, param_dtype=jnp.float32)
    return enc, pred, head


def init_heads(pred, head, rng_init, n_tokens, width, text_dim, b=2, text_len=32):
    """Fresh predictor and head, with the key derivation of the original code (predictor: split(rng, 4)[2],
    head: fold_in(rng, 19))."""
    _, _, r_pred, _ = jax.random.split(rng_init, 4)
    z = jnp.zeros((b, n_tokens, width), jnp.float32)
    p_tr = pred.init({"params": r_pred}, jnp.zeros((b, n_tokens, 2)), jnp.zeros((b,)), z,
                     jnp.zeros((b, text_len, text_dim)), jnp.ones((b, text_len), bool))["params"]
    p_head = head.init({"params": jax.random.fold_in(rng_init, 19)}, z)["params"]
    return p_tr, p_head


def main(_):
    cfg, workdir = FLAGS.config, FLAGS.workdir
    runtime.init_distributed()
    os.makedirs(workdir, exist_ok=True)
    mesh, data_sh, repl = runtime.make_mesh()
    micro = cfg.global_batch // cfg.accum_steps
    if cfg.global_batch % cfg.accum_steps or micro % jax.device_count():
        raise ValueError("global_batch / accum_steps must be a multiple of the device count")
    enc, pred, head = build_models(cfg)
    loaded, s = runtime.restore_params(cfg.stage1_ckpt, subtrees=("visual_encoder",), sharding=repl)
    teacher = jax.tree.map(lambda x: jnp.asarray(x, jnp.float32), loaded["visual_encoder"])
    print(f"[stage2] student and teacher from {cfg.stage1_ckpt} step {s}")
    _, rng_init = jax.random.split(jax.random.PRNGKey(cfg.seed))
    n_tok = (cfg.res // cfg.encoder.patch_size) ** 2
    p_tr, p_head = jax.jit(functools.partial(init_heads, pred, head, n_tokens=n_tok, width=cfg.encoder.width,
                                             text_dim=cfg.predictor.text_dim), out_shardings=repl)(rng_init)
    params = {"encoder": jax.tree.map(jnp.copy, teacher), "transition": p_tr, "feat_head": p_head}
    zs = np.load(cfg.token_stats)
    zstats = (jnp.asarray(zs["mean"], jnp.float32)[None, None], jnp.asarray(zs["std"], jnp.float32)[None, None])
    u0 = jnp.asarray(np.load(cfg.empty_token)[cfg.empty_token_key], jnp.float32)
    if zstats[0].shape[-1] != cfg.encoder.width or u0.shape[-1] != cfg.predictor.text_dim:
        raise ValueError("token statistics / constant token do not match the encoder / predictor widths")
    tx, lr_fn = stage2_optimizer(cfg.base_lr, cfg.total_steps, cfg.warmup_steps, cfg.encoder_lr_ratio,
                                 cfg.weight_decay, cfg.b1, cfg.b2, cfg.clip_norm)
    state = {"params": params, "opt": jax.jit(tx.init, out_shardings=repl)(params), "step": jnp.int32(0)}
    mngr = runtime.checkpoint_manager(os.path.join(workdir, "checkpoints"), cfg.ckpt_keep, cfg.ckpt_steps)
    if mngr.latest_step() is not None:
        state, s = runtime.restore(mngr, state)
        print(f"[stage2] resumed from step {s}")

    loss_fn = functools.partial(crossview_loss, encoder=enc, predictor=pred, head=head, zstats=zstats, u0=u0)

    @functools.partial(jax.jit, in_shardings=(repl, repl, data_sh), out_shardings=(repl, repl))
    def grad_fn(params, teacher, batch):
        (_, m), g = jax.value_and_grad(loss_fn, has_aux=True)(params, teacher, batch)
        return g, m

    @functools.partial(jax.jit, donate_argnums=(0,), in_shardings=(repl, repl), out_shardings=repl)
    def apply_fn(state, grads):
        updates, opt = tx.update(grads, state["opt"], state["params"])
        return {"params": optax.apply_updates(state["params"], updates), "opt": opt, "step": state["step"] + 1}

    ds = stage2_dataset(sources=[tuple(x) for x in cfg.data.sources], res=cfg.res,
                        per_host_batch=micro // jax.process_count(), crossview_cfg=cfg.crossview.to_dict(),
                        shuffle_buffer=cfg.data.shuffle_buffer, process_index=jax.process_index(),
                        process_count=jax.process_count())
    it = ds.as_numpy_iterator()
    log = runtime.MetricLogger(workdir)
    start = int(state["step"])
    for step in range(start, cfg.total_steps):
        grads, ms = None, []
        for _ in range(cfg.accum_steps):                  # gradient of the mean over the global batch
            g, m = grad_fn(state["params"], teacher, runtime.global_batch(next(it), data_sh))
            grads = g if grads is None else jax.tree.map(jnp.add, grads, g)
            ms.append(m)
        grads = jax.tree.map(lambda x: x / cfg.accum_steps, grads)
        state = apply_fn(state, grads)
        if (step + 1) % cfg.log_steps == 0 or step == start:
            log.log(step + 1, {**{k: np.mean([float(m[k]) for m in ms]) for k in ms[0]}, "lr": lr_fn(step)})
        runtime.save(mngr, step + 1, state, force=(step + 1 == cfg.total_steps))
    mngr.wait_until_finished()


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
