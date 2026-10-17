# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Stage-2 loss (JAX), as in the original training code.

    tgt  = stop_grad((E_0(x_B) - mu) / sigma)        frozen Stage-1 teacher on the clean crop B, fixed statistics
    Za   = E(x_A)                                    student on the augmented crop A
    pred = FeatHead(CrossViewFusion(q_pos, log_r, Za, u))
    L    = mean_images [ sum_{n valid} mean_channels (pred - tgt)^2 / max(#valid, 1) ]
Both encoders run with train=False (no dropout; the residual stream stays in the compute dtype).
"""
import jax
import jax.numpy as jnp

from ..models.predictor import crossview_geometry

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_image(x):
    x = jnp.asarray(x, dtype=jnp.float32)
    return (x - jnp.asarray(IMAGENET_MEAN)[None, None, None, :]) / jnp.asarray(IMAGENET_STD)[None, None, None, :]


def crossview_loss(params, teacher, batch, *, encoder, predictor, head, zstats, u0):
    """params = {"encoder", "transition", "feat_head"}; teacher = frozen encoder params; u0 [768] constant token.

    batch: x_s_rgb (student view A) and x_t_rgb (teacher view B) in [0, 1], box_a / box_b [B, 3]."""
    f32 = jnp.float32
    sg = jax.lax.stop_gradient
    x_a = normalize_image(batch["x_s_rgb"])
    x_b = normalize_image(batch["x_t_rgb"])
    _, zb = encoder.apply({"params": teacher}, x_b, train=False)
    mu, sd = zstats
    tgt = sg((zb.astype(f32) - mu) / sd)
    b_, n_, d_ = tgt.shape
    g_ = int(round(n_ ** 0.5))
    _, za = encoder.apply({"params": params["encoder"]}, x_a, train=False)
    q_pos, valid, log_r = crossview_geometry(batch["box_a"], batch["box_b"], g_)
    # the constant token: u0 in slot 0 of a 32-slot instruction whose other slots are masked out
    u = jnp.zeros((b_, 32, u0.shape[-1]), f32).at[:, 0].set(u0.astype(f32))
    u_mask = jnp.zeros((b_, 32), bool).at[:, 0].set(True)
    fused = predictor.apply({"params": params["transition"]}, q_pos, log_r, za.astype(f32), u, u_mask).astype(f32)
    pred = head.apply({"params": params["feat_head"]}, fused).astype(f32)
    err = jnp.mean(jnp.square(pred - tgt), axis=-1)
    vf = valid.astype(f32)
    nv = jnp.sum(vf, axis=1)
    loss = jnp.mean(jnp.sum(err * vf, axis=1) / jnp.maximum(nv, 1.0))
    return loss, {"loss": loss, "n_valid": sg(jnp.mean(nv)), "log_r": sg(jnp.mean(log_r))}
