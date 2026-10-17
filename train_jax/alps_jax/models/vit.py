# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""ViT encoder of ALPS (Flax).

A plain ViT with a class token and a fixed 2-D sine-cosine position embedding. Parameter names and shapes are those
of the original training code, so checkpoints load into either code base:

    embedding/kernel (16, 16, 3, D)     patch embedding, a convolution without bias, computed in float32
    cls (1, 1, D)
    Transformer/encoderblock_{i}/{LayerNorm_0, MultiHeadDotProductAttention_0/{query,key,value,out}, LayerNorm_1,
                                  MlpBlock_0/{Dense_0, Dense_1}}
    encoder_norm/{scale,bias}           LayerNorm of the mean token (the pooled vector only)

`__call__` returns (pooled, tokens): tokens are the RAW patch tokens after the last block (no final LayerNorm), the
visual state of ALPS; pooled = encoder_norm(mean(tokens)).

Numerics follow the original code exactly: every block casts its input to the compute dtype; with `train=True` the
residual branches are added in float32 (the original code's stochastic-depth layer, at rate 0, promoted them), with
`train=False` in the compute dtype. Stage 1 trains with train=True; Stage 2 runs the student and the teacher with
train=False.
"""
from typing import Any, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

Dtype = Any

VARIANTS = {   # width, depth, MLP width, heads
    "S": (384, 12, 1536, 6), "B": (768, 12, 3072, 12), "L": (1024, 24, 4096, 16),
}


def posemb_sincos_2d(h, w, width, temperature=10_000., dtype=jnp.float32, cls_token=False):
    """[1, (1 +) h*w, width]: [sin x, cos x, sin y, cos y] with omega_k = 1 / temperature^(k / (width/4 - 1))."""
    y, x = jnp.mgrid[:h, :w]
    assert width % 4 == 0, "width must be a multiple of 4 for the sin-cos table"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1. / (temperature ** omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    if cls_token:
        pe = jnp.concatenate([jnp.zeros([1, width]), pe], axis=0)
    return jnp.asarray(pe, dtype)[None, :, :]


class MultiHeadDotProductAttention(nn.Module):
    """Self-attention with fused 2-D q/k/v kernels [D, D] (named query/key/value/out)."""
    num_heads: int
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        b, n, d = x.shape
        hd = d // self.num_heads
        dense = lambda name, init: nn.Dense(d, name=name, dtype=self.dtype, param_dtype=self.param_dtype,
                                            kernel_init=init, bias_init=nn.initializers.zeros)
        init = nn.initializers.normal(stddev=0.02)
        q = dense("query", init)(x).reshape(b, n, self.num_heads, hd).astype(self.dtype)
        k = dense("key", init)(x).reshape(b, n, self.num_heads, hd).astype(self.dtype)
        v = dense("value", init)(x).reshape(b, n, self.num_heads, hd).astype(self.dtype)
        q = q / jnp.sqrt(hd).astype(self.dtype)                     # scale the query in the compute dtype
        w = jnp.einsum("...qhd,...khd->...hqk", q, k)
        w = jax.nn.softmax(w).astype(self.dtype)                    # softmax in the compute dtype, as in training
        o = jnp.einsum("...hqk,...khd->...qhd", w, v).astype(self.dtype).reshape(b, n, d)
        return dense("out", init)(o)


class MlpBlock(nn.Module):
    mlp_dim: int
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        d = x.shape[-1]
        x = nn.Dense(self.mlp_dim, kernel_init=nn.initializers.normal(stddev=0.02), bias_init=nn.initializers.zeros,
                     dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = nn.gelu(x.astype(self.dtype), approximate=False).astype(self.dtype)
        return nn.Dense(d, kernel_init=nn.initializers.variance_scaling(0.3072, "fan_out", "truncated_normal"),
                        bias_init=nn.initializers.zeros, dtype=self.dtype, param_dtype=self.param_dtype)(x)


class Encoder1DBlock(nn.Module):
    mlp_dim: int
    num_heads: int
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x, train=False):
        x = x.astype(self.dtype)
        ln = lambda: nn.LayerNorm(epsilon=1e-6, dtype=self.dtype, param_dtype=self.param_dtype)
        y = MultiHeadDotProductAttention(self.num_heads, dtype=self.dtype, param_dtype=self.param_dtype)(ln()(x))
        x = x + (y.astype(jnp.float32) if train else y)
        y = MlpBlock(self.mlp_dim, dtype=self.dtype, param_dtype=self.param_dtype)(ln()(x))
        return x + (y.astype(jnp.float32) if train else y)


class Encoder(nn.Module):
    depth: int
    mlp_dim: int
    num_heads: int
    remat: bool = False
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x, train=False):
        Block = nn.remat(Encoder1DBlock, prevent_cse=True, static_argnums=(2,)) if self.remat else Encoder1DBlock
        for i in range(self.depth):
            x = Block(self.mlp_dim, self.num_heads, dtype=self.dtype, param_dtype=self.param_dtype,
                      name=f"encoderblock_{i}")(x, train)
        return x


class ViT(nn.Module):
    width: int = 768
    depth: int = 12
    mlp_dim: int = 3072
    num_heads: int = 12
    patch_size: int = 16
    remat: bool = False
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, image, *, train=False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """image: [B, H, W, 3], ImageNet-normalized. Returns (pooled [B, D], tokens [B, N, D])."""
        p = self.patch_size
        x = nn.Conv(self.width, (p, p), strides=(p, p), padding="VALID", use_bias=False, dtype=jnp.float32,
                    param_dtype=self.param_dtype, kernel_init=nn.initializers.kaiming_uniform(),
                    name="embedding")(image)
        n, h, w, c = x.shape
        x = x.reshape(n, h * w, c)
        cls = self.param("cls", nn.initializers.normal(1e-6), (1, 1, c), x.dtype)
        x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)
        x = x + posemb_sincos_2d(h, w, c, dtype=x.dtype, cls_token=True)
        x = x.astype(self.dtype)
        x = Encoder(self.depth, self.mlp_dim, self.num_heads, remat=self.remat, dtype=self.dtype,
                    param_dtype=self.param_dtype, name="Transformer")(x, train)
        tokens = x[:, 1:]
        pooled = nn.LayerNorm(name="encoder_norm", dtype=self.dtype, param_dtype=self.param_dtype)(
            jnp.mean(tokens, axis=1))
        return pooled, tokens


def vit(variant: str = "B", **kw) -> ViT:
    w, d, m, h = VARIANTS[variant]
    return ViT(width=w, depth=d, mlp_dim=m, num_heads=h, **kw)
