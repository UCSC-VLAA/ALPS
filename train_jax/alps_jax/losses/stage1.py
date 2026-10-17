# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Stage-1 losses (JAX), as in the original training code.

    L = L_itc + w_cap L_cap + w_rec L_rec + w_dense L_dense
    L_itc   = 1/2 [ITC(v, t_web) + ITC(v, t_sentence)]            multi-positive, one [B, B] softmax per caption
    L_cap   = next-token cross-entropy of the caption decoder over the non-pad targets
    L_rec   = rho(D(z), x) + 0.25 rho(D(z + noise), x),  rho = l1 + 0.2 (1 - SSIM_7x7) + 0.1 Sobel-gradient l1
    L_dense = 0.75 L_same + 0.25 L_cross                           (alps_jax/models/dense.py)
All means are over the global batch.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..models.dense import patch_infonce_loss


def _l2_normalize(x, axis=-1, eps=1e-6):
    return x / (jnp.sqrt(jnp.sum(x * x, axis=axis, keepdims=True)) + eps)


def _xent(logits, labels):
    return -jnp.sum(jax.nn.one_hot(labels, logits.shape[-1]) * jax.nn.log_softmax(logits, axis=-1), axis=-1).mean()


def itc_loss(image_emb, text_emb, logit_scale):
    v = _l2_normalize(image_emb.astype(jnp.float32))
    t = _l2_normalize(text_emb.astype(jnp.float32))
    logits = jnp.dot(v, t.T) * logit_scale.astype(jnp.float32)
    labels = jnp.arange(v.shape[0])
    return 0.5 * (_xent(logits, labels) + _xent(logits.T, labels))


def itc_loss_multipos(image_emb, text_emb1, text_emb2, logit_scale):
    return 0.5 * (itc_loss(image_emb, text_emb1, logit_scale) + itc_loss(image_emb, text_emb2, logit_scale))


def caption_ntp_loss(logits, text, pad_id=0):
    """logits[:, t] predicts text[:, t + 1]; mean over the non-pad targets (the trailing [CLS] target counts)."""
    text = text.astype("int32")
    tgt = text[:, 1:]
    lg = logits[:, :-1].astype(jnp.float32)
    valid = (tgt != pad_id).astype(jnp.float32)
    ce = optax.softmax_cross_entropy_with_integer_labels(lg, tgt)
    return (ce * valid).sum() / jnp.maximum(valid.sum(), 1.0)


def l1_loss(p, t):
    return jnp.abs(p.astype(jnp.float32) - t.astype(jnp.float32)).mean()


def _avgpool(x, k=7):
    x = jax.lax.reduce_window(x, 0.0, jax.lax.add, (1, k, 1, 1), (1, 1, 1, 1), "VALID")
    x = jax.lax.reduce_window(x, 0.0, jax.lax.add, (1, 1, k, 1), (1, 1, 1, 1), "VALID")
    return x / (k * k)


def ssim_loss(p, t, k=7, c1=0.01 ** 2, c2=0.03 ** 2):
    p = p.astype(jnp.float32); t = t.astype(jnp.float32)
    mp, mt = _avgpool(p, k), _avgpool(t, k)
    sp = _avgpool(p * p, k) - mp * mp
    st = _avgpool(t * t, k) - mt * mt
    spt = _avgpool(p * t, k) - mp * mt
    ssim = ((2 * mp * mt + c1) * (2 * spt + c2)) / ((mp * mp + mt * mt + c1) * (sp + st + c2))
    return 1.0 - ssim.mean()


_SX = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]], np.float32)
_SY = _SX.T


def _grad(img):
    c = img.shape[-1]

    def dw(kern):
        w = jnp.broadcast_to(kern[:, :, None, None], (3, 3, 1, c))
        return jax.lax.conv_general_dilated(img, w, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"),
                                            feature_group_count=c)
    return dw(_SX), dw(_SY)


def sobel_gradient_loss(p, t):
    p = p.astype(jnp.float32); t = t.astype(jnp.float32)
    px, py = _grad(p); tx, ty = _grad(t)
    return (jnp.abs(px - tx) + jnp.abs(py - ty)).mean()


def recon_loss(p, t):
    return l1_loss(p, t) + 0.2 * ssim_loss(p, t) + 0.1 * sobel_gradient_loss(p, t)


def stage1_loss(out, rng, *, cap_w=1.0, rec_w=1.0, dense_w=0.25, dense_n_cross=64, dense_radius=1,
                dense_n_same=64, dense_split=(0.75, 0.25), pad_id=0):
    """Total Stage-1 loss from the outputs of Stage1Model (train=True). Returns (loss, metrics)."""
    if "txt_emb2" in out:
        l_itc = itc_loss_multipos(out["img_emb"], out["txt_emb"], out["txt_emb2"], out["logit_scale"])
    else:
        l_itc = itc_loss(out["img_emb"], out["txt_emb"], out["logit_scale"])
    l_rec = recon_loss(out["rec_pred"], out["x_rgb"]) + 0.25 * recon_loss(out["rec_pred_noisy"], out["x_rgb"])
    l_dense, diag = patch_infonce_loss(out["h_x"], out["h_gx"], out["crop_box"], out["flip"],
                                       jax.random.fold_in(rng, 1), n_cross=dense_n_cross,
                                       exclude_radius=dense_radius, n_same=dense_n_same, split_weights=dense_split)
    l_cap = jnp.zeros((), jnp.float32)
    if "cap_logits" in out:
        l_cap = caption_ntp_loss(out["cap_logits"], out["cap_text"], pad_id=pad_id)
    loss = l_itc + cap_w * l_cap + rec_w * l_rec + dense_w * l_dense
    return loss, {"loss": loss, "L_itc": l_itc, "L_cap": l_cap, "L_rec": l_rec, "L_dense": l_dense,
                  "logit_scale": out["logit_scale"], "latent_rms": out["latent_rms"], **diag}
