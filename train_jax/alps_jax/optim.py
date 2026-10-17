# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Optimizers and learning-rate schedules of Stage 1 and Stage 2 (optax), as in the original training code.

Both stages use Adam with a bfloat16 first moment, eps 1e-8 (optax scale_by_adam with mu_dtype=bfloat16: the stored
moment is decayed in bfloat16, b1 * mu with b1 rounded to bfloat16, then added to the float32 (1 - b1) g; the update
uses that float32 value, which is rounded to bfloat16 for storage), decoupled weight decay 0.05 on leaves named `kernel` only (not biases, LayerNorm parameters,
embeddings, the class token, position tables or the logit scale), and a linear warm-up from 0 followed by a cosine to
0. The first update therefore has learning rate exactly 0.

Stage 1: b2 = 0.95, no gradient clipping, one learning rate for all parameters.
Stage 2: b2 = 0.99, global-norm clipping at 1.0, learning rate 1e-4 for the predictor and head and 0.1x for the
encoder; the frozen teacher is not part of the optimized tree.
"""
import re

import jax
import jax.numpy as jnp
import optax


def warmup_cosine(total_steps, warmup_steps, base=1.0):
    """lr(c) = base * min(1, c / W) * 0.5 * (1 + cos(pi * clip((c - W) / (T - W), 0, 1)))."""
    def fn(step):
        progress = jnp.clip((step - warmup_steps) / float(total_steps - warmup_steps), 0.0, 1.0)
        lr = base * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        if warmup_steps:
            lr = lr * jnp.minimum(1.0, step / warmup_steps)
        return jnp.asarray(lr, dtype=jnp.float32)
    return fn


def kernel_mask(tree):
    """True on leaves whose name is `kernel` (the original code matched '.*/kernel$')."""
    return jax.tree_util.tree_map_with_path(lambda path, _: getattr(path[-1], "key", None) == "kernel", tree)


def stage1_optimizer(peak_lr, total_steps, warmup_steps, weight_decay=0.05, b1=0.9, b2=0.95):
    sched = warmup_cosine(total_steps, warmup_steps)
    tx = optax.chain(
        optax.scale_by_adam(b1=b1, b2=b2, eps=1e-8, eps_root=0.0, mu_dtype=jnp.bfloat16),
        optax.add_decayed_weights(weight_decay, mask=kernel_mask),
        optax.scale(peak_lr),
        optax.scale_by_schedule(sched),
        optax.scale(-1.0))
    return tx, (lambda step: peak_lr * sched(step))


def stage2_optimizer(base_lr, total_steps, warmup_steps, encoder_lr_ratio=0.1, weight_decay=0.05, b1=0.9, b2=0.99,
                     clip_norm=1.0, depth=12):
    """Parameter tree {"encoder", "transition", "feat_head"}; encoder blocks share the encoder ratio (no layer decay)."""
    sched = warmup_cosine(total_steps, warmup_steps, base=base_lr)

    def adamw(mult):
        return optax.chain(
            optax.scale_by_adam(b1=b1, b2=b2, eps=1e-8, eps_root=0.0, mu_dtype=jnp.bfloat16),
            optax.add_decayed_weights(weight_decay, mask=kernel_mask),
            optax.scale(mult),
            optax.scale_by_schedule(sched),
            optax.scale(-1.0))

    txs = {"transition": adamw(1.0), "feat_head": adamw(1.0), "enc_rest": adamw(encoder_lr_ratio)}
    for i in range(depth):
        txs[f"enc_{i}"] = adamw(encoder_lr_ratio)
    block_re = re.compile(r"encoderblock_(\d+)")

    def label(path, _):
        keys = [str(getattr(p, "key", getattr(p, "idx", ""))) for p in path]
        if keys[0] == "encoder":
            m = block_re.search("/".join(keys))
            return f"enc_{int(m.group(1))}" if m else "enc_rest"
        return keys[0]

    tx = optax.chain(optax.clip_by_global_norm(clip_norm),
                     optax.multi_transform(txs, lambda params: jax.tree_util.tree_map_with_path(label, params)))
    return tx, sched


def project_logit_scale(params, max_scale=100.0):
    """Keep exp(logit_scale) in [1, max_scale] by projecting the parameter after every update (Stage 1)."""
    if "logit_scale" not in params:
        return params
    p = dict(params)
    p["logit_scale"] = jnp.clip(params["logit_scale"], 0.0, jnp.log(max_scale))
    return p
