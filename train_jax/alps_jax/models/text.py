# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Text transformer blocks, the contrastive text encoder and the caption decoder of Stage 1 (Flax).

Same parameter names, initializers and numerics as the original training code:
* pre-LayerNorm blocks (epsilon 1e-6), attention via `flax.linen.dot_product_attention` (query scaled by 1/sqrt(64),
  softmax in the compute dtype), MLP with tanh-approximated GELU;
* init: q/k/v normal(width^-0.5); attention output and second MLP layer normal(width^-0.5 * (2 * 12)^-0.5);
  first MLP layer normal((2 * width)^-0.5); biases 0;
* `train=True` adds the residual branches in float32 (see models/vit.py);
* no padding mask: padding slots take part in attention.
"""
from typing import Any, Optional

import flax.linen as nn
import jax.numpy as jnp

Dtype = Any


class MultiHeadDotProductAttention(nn.Module):
    num_heads: int
    attn_kernel_init: Any = nn.initializers.normal(stddev=0.01)
    proj_kernel_init: Any = nn.initializers.normal(stddev=0.01)
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x, mask=None):
        b, n, d = x.shape
        hd = d // self.num_heads
        dense = lambda name, init: nn.Dense(d, name=name, dtype=self.dtype, param_dtype=self.param_dtype,
                                            kernel_init=init, bias_init=nn.initializers.zeros)
        q = dense("query", self.attn_kernel_init)(x).reshape(b, n, self.num_heads, hd).astype(self.dtype)
        k = dense("key", self.attn_kernel_init)(x).reshape(b, n, self.num_heads, hd).astype(self.dtype)
        v = dense("value", self.attn_kernel_init)(x).reshape(b, n, self.num_heads, hd).astype(self.dtype)
        o = nn.dot_product_attention(q, k, v, mask=mask, deterministic=True, dtype=self.dtype)
        return dense("out", self.proj_kernel_init)(o.astype(self.dtype).reshape(b, n, d))


class MlpBlock(nn.Module):
    mlp_dim: int
    fc_init: Any = nn.initializers.xavier_uniform()
    proj_init: Any = nn.initializers.xavier_uniform()
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        d = x.shape[-1]
        x = nn.Dense(self.mlp_dim, kernel_init=self.fc_init, bias_init=nn.initializers.zeros, dtype=self.dtype,
                     param_dtype=self.param_dtype)(x)
        x = nn.gelu(x.astype(self.dtype), approximate=True).astype(self.dtype)
        return nn.Dense(d, kernel_init=self.proj_init, bias_init=nn.initializers.zeros, dtype=self.dtype,
                        param_dtype=self.param_dtype)(x)


class Encoder1DBlock(nn.Module):
    mlp_dim: int
    num_heads: int
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x, mask=None, train=False):
        width = x.shape[-1]
        depth_for_init = 12          # the original code used the block's default depth (12) in the init formula
        std_attn = width ** -0.5
        std_proj = (width ** -0.5) * ((2 * depth_for_init) ** -0.5)
        std_fc = (2 * width) ** -0.5
        x = x.astype(self.dtype)
        ln = lambda: nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype)
        y = MultiHeadDotProductAttention(self.num_heads, nn.initializers.normal(std_attn),
                                         nn.initializers.normal(std_proj), dtype=self.dtype,
                                         param_dtype=self.param_dtype)(ln()(x), mask)
        x = x + (y.astype(jnp.float32) if train else y)
        y = MlpBlock(self.mlp_dim, nn.initializers.normal(std_fc), nn.initializers.normal(std_proj),
                     dtype=self.dtype, param_dtype=self.param_dtype)(ln()(x))
        return x + (y.astype(jnp.float32) if train else y)


class Transformer(nn.Module):
    depth: int
    mlp_dim: int
    num_heads: int
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x, mask=None, train=False):
        for i in range(self.depth):
            x = Encoder1DBlock(self.mlp_dim, self.num_heads, dtype=self.dtype, param_dtype=self.param_dtype,
                               name=f"encoderblock_{i}")(x, mask, train)
        return x


class TextEncoder(nn.Module):
    """Bidirectional text encoder of the contrastive objective; pooled = last slot (the trailing [CLS])."""
    vocab_size: int = 32_000
    width: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_dim: Optional[int] = None
    max_len: int = 32
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, text, *, train=False, return_tokens=False):
        text = text.astype("int32")
        x = nn.Embed(self.vocab_size, self.width, dtype=jnp.float32, param_dtype=self.param_dtype,
                     embedding_init=nn.initializers.normal(stddev=0.02), name="token_embedding")(text)
        pos = self.param("pos_embedding", nn.initializers.normal(stddev=0.01), (1, self.max_len, self.width),
                         self.param_dtype)
        x = (x + pos[:, :x.shape[1]]).astype(self.dtype)
        x = Transformer(self.depth, self.mlp_dim or 4 * self.width, self.num_heads, dtype=self.dtype,
                        param_dtype=self.param_dtype, name="Transformer")(x, None, train)
        x = nn.LayerNorm(name="encoder_norm")(x)
        return x if return_tokens else x[:, -1]


def prefix_lm_mask(batch, num_heads, n_prefix, n_text):
    """[B, H, L, L] bool: prefix <-> prefix all, text -> prefix all, text -> text causal, prefix -> text none."""
    li, lt = n_prefix, n_text
    causal = jnp.tril(jnp.ones((lt, lt), bool))
    m = jnp.zeros((li + lt, li + lt), bool)
    m = m.at[:li, :li].set(True).at[li:, li:].set(causal).at[li:, :li].set(True)
    return jnp.broadcast_to(m[None, None], (batch, num_heads, li + lt, li + lt))


class CaptionDecoder(nn.Module):
    """Prefix-LM caption decoder on concat(raw image tokens, caption tokens); logits for the caption positions."""
    vocab_size: int = 32_000
    width: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_dim: Optional[int] = None
    max_len: int = 80
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, z_img, text, *, train=False):
        if z_img.shape[-1] != self.width:
            raise ValueError(f"image token width {z_img.shape[-1]} != decoder width {self.width} (no projection)")
        text = text.astype("int32")
        b, n, _ = z_img.shape
        t = text.shape[1]
        w = nn.Embed(self.vocab_size, self.width, dtype=jnp.float32, param_dtype=self.param_dtype,
                     embedding_init=nn.initializers.normal(stddev=0.02), name="token_embedding")(text)
        pos = self.param("pos_embedding", nn.initializers.normal(stddev=0.01), (1, self.max_len, self.width),
                         self.param_dtype)
        w = w + pos[:, :t]
        x = jnp.concatenate([z_img.astype(self.dtype), w.astype(self.dtype)], axis=1)
        x = Transformer(self.depth, self.mlp_dim or 4 * self.width, self.num_heads, dtype=self.dtype,
                        param_dtype=self.param_dtype, name="decoder")(
            x, prefix_lm_mask(b, self.num_heads, n, t), train)
        x = nn.LayerNorm(dtype=jnp.float32, param_dtype=self.param_dtype, name="decoder_norm")(x[:, n:])
        return nn.Dense(self.vocab_size, dtype=jnp.float32, param_dtype=self.param_dtype, name="head")(x)
