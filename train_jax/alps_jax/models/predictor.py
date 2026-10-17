# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Stage-2 cross-view predictor and output head (Flax), as in the original training code.

The student encodes crop A of an image; the frozen Stage-1 teacher encodes a clean crop B. For every patch of B, a
query = learned vector m + fixed sin-cos code of the patch centre in A's frame (in patches of A) + an affine map of
(ln r, ln^2 r), r = side_B / side_A, attends to the student tokens (projected to the predictor width, plus the same code
of A's own grid) and to one constant token (an affine map of the frozen flan-T5 encoding of the empty instruction,
`u0`). Four pre-LayerNorm layers (self-attention over the queries, cross-attention, MLP 2048), width 512, 8 heads, then
LayerNorm + Linear to 768 (`transition/out_proj`) and the head LayerNorm + Linear (`feat_head`).

Parameter names are those of the original code (subtrees "transition" and "feat_head" of the Stage-2 train state).
"""
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

Dtype = Any


def sincos_2d(pos_yx, width):
    """Fixed code of continuous positions pos_yx [..., 2] = (y, x): [sin y, cos y, sin x, cos x], omega_k = 10000^(-k/d)."""
    assert width % 4 == 0, width
    d = width // 4
    omega = 1.0 / (10000.0 ** (jnp.arange(d, dtype=jnp.float32) / d))
    y = pos_yx[..., 0:1].astype(jnp.float32) * omega
    x = pos_yx[..., 1:2].astype(jnp.float32) * omega
    return jnp.concatenate([jnp.sin(y), jnp.cos(y), jnp.sin(x), jnp.cos(x)], axis=-1)


def crossview_geometry(box_a, box_b, grid):
    """box_a, box_b [B, 3] = (y0, x0, side) normalized by the side of the image's central square.

    Returns q_pos [B, N, 2] (centres of B's patches in A's frame, in patches of A), valid [B, N] (centre inside A) and
    log_r [B] = ln(side_B / side_A)."""
    g = int(grid)
    c = (jnp.arange(g, dtype=jnp.float32) + 0.5) / g
    box_a = box_a.astype(jnp.float32); box_b = box_b.astype(jnp.float32)
    yc = box_b[:, 0:1] + c[None, :] * box_b[:, 2:3]
    xc = box_b[:, 1:2] + c[None, :] * box_b[:, 2:3]
    v = (yc - box_a[:, 0:1]) / box_a[:, 2:3]
    u = (xc - box_a[:, 1:2]) / box_a[:, 2:3]
    b = box_a.shape[0]
    vv = jnp.broadcast_to(v[:, :, None], (b, g, g))
    uu = jnp.broadcast_to(u[:, None, :], (b, g, g))
    q_pos = jnp.stack([vv, uu], axis=-1).reshape(b, g * g, 2) * g
    valid = ((vv >= 0.) & (vv < 1.) & (uu >= 0.) & (uu < 1.)).reshape(b, g * g)
    log_r = jnp.log(box_b[:, 2] / box_a[:, 2])
    return q_pos, valid, log_r


def _mha(q, kv, key_mask, num_heads, name, *, dtype, param_dtype):
    """Multi-head attention with q/k/v/o Dense layers named {name}_q ... {name}_o; softmax in float32."""
    B, Lq, W = q.shape
    Lk = kv.shape[1]
    H = num_heads
    d = W // H
    qh = nn.Dense(W, use_bias=True, dtype=dtype, param_dtype=param_dtype, name=f"{name}_q")(q).reshape(B, Lq, H, d)
    kh = nn.Dense(W, use_bias=True, dtype=dtype, param_dtype=param_dtype, name=f"{name}_k")(kv).reshape(B, Lk, H, d)
    vh = nn.Dense(W, use_bias=True, dtype=dtype, param_dtype=param_dtype, name=f"{name}_v")(kv).reshape(B, Lk, H, d)
    logits = jnp.einsum("bqhd,bkhd->bhqk", qh, kh).astype(jnp.float32) / jnp.sqrt(d).astype(jnp.float32)
    if key_mask is not None:
        logits = jnp.where(key_mask[:, None, None, :], logits, jnp.finfo(jnp.float32).min)
    attn = jax.nn.softmax(logits, axis=-1).astype(vh.dtype)
    out = jnp.einsum("bhqk,bkhd->bqhd", attn, vh).reshape(B, Lq, W)
    return nn.Dense(W, use_bias=True, dtype=dtype, param_dtype=param_dtype, name=f"{name}_o")(out)


class _MlpBlock(nn.Module):
    hidden: int
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        d = x.shape[-1]
        x = nn.Dense(self.hidden, dtype=self.dtype, param_dtype=self.param_dtype, name="fc1")(x)
        x = nn.gelu(x, approximate=False)
        return nn.Dense(d, dtype=self.dtype, param_dtype=self.param_dtype, name="fc2")(x)


class CrossViewFusion(nn.Module):
    width: int = 512
    depth: int = 4
    num_heads: int = 8
    grid: int = 14
    token_dim: int = 768
    text_dim: int = 768
    mlp_ratio: int = 4
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, q_pos, log_r, src_tokens, txt_tokens, txt_mask):
        """q_pos [B, Nq, 2]; log_r [B]; src_tokens [B, g*g, token_dim]; txt_tokens [B, L, text_dim]; txt_mask [B, L]."""
        B = q_pos.shape[0]
        g, W, cdt = self.grid, self.width, self.dtype
        assert src_tokens.shape[1] == g * g, (src_tokens.shape, g)
        mask_token = self.param("mask_token", nn.initializers.normal(0.02), (1, 1, W), self.param_dtype)
        scale_in = jnp.stack([log_r, jnp.square(log_r)], axis=-1).astype(cdt)
        scale_emb = nn.Dense(W, use_bias=True, dtype=cdt, param_dtype=self.param_dtype, name="scale_proj")(scale_in)
        x = mask_token.astype(cdt) + sincos_2d(q_pos, W).astype(cdt) + scale_emb[:, None, :]
        rr, cc = jnp.meshgrid(jnp.arange(g, dtype=jnp.float32), jnp.arange(g, dtype=jnp.float32), indexing="ij")
        a_pos = jnp.stack([rr, cc], axis=-1).reshape(1, g * g, 2) + 0.5
        src_kv = nn.Dense(W, use_bias=True, dtype=cdt, param_dtype=self.param_dtype, name="src_proj")(
            src_tokens.astype(cdt)) + sincos_2d(a_pos, W).astype(cdt)
        txt_kv = nn.Dense(W, use_bias=True, dtype=cdt, param_dtype=self.param_dtype, name="txt_proj")(
            txt_tokens.astype(cdt))
        kv = jnp.concatenate([src_kv, txt_kv], axis=1)
        key_mask = jnp.concatenate([jnp.ones((B, src_kv.shape[1]), bool), txt_mask.astype(bool)], axis=1)
        for i in range(self.depth):
            y = nn.LayerNorm(dtype=cdt, param_dtype=self.param_dtype, name=f"ln_sa_{i}")(x)
            x = x + _mha(y, y, None, self.num_heads, f"sa_{i}", dtype=cdt, param_dtype=self.param_dtype)
            y = nn.LayerNorm(dtype=cdt, param_dtype=self.param_dtype, name=f"ln_ca_{i}")(x)
            x = x + _mha(y, kv, key_mask, self.num_heads, f"ca_{i}", dtype=cdt, param_dtype=self.param_dtype)
            y = nn.LayerNorm(dtype=cdt, param_dtype=self.param_dtype, name=f"ln_mlp_{i}")(x)
            x = x + _MlpBlock(self.mlp_ratio * W, dtype=cdt, param_dtype=self.param_dtype, name=f"mlp_{i}")(y)
        x = nn.LayerNorm(dtype=cdt, param_dtype=self.param_dtype, name="ln_out")(x)
        return nn.Dense(self.token_dim, use_bias=True, dtype=cdt, param_dtype=self.param_dtype, name="out_proj")(x)


class FeatHead(nn.Module):
    out_dim: int = 768
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype, name="ln")(x.astype(self.dtype))
        return nn.Dense(self.out_dim, use_bias=True, dtype=self.dtype, param_dtype=self.param_dtype, name="proj")(x)
