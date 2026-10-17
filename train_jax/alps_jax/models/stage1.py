# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Stage-1 model of ALPS (Flax): the encoder with its four training objectives' heads.

Parameter tree (identical to the original training code, so checkpoints interchange):
    visual_encoder, image_projection_head/proj, logit_scale, text_encoder, text_projection_head/proj,
    caption_decoder, RGBDecoder_0, DenseProjector_0
Only visual_encoder, text_encoder, the two projection heads and logit_scale are kept after training.

The module returns the tensors the losses need (alps_jax/losses/stage1.py). Inputs: x_enc / gx_enc = ImageNet-
normalized base view and geometric view, x_rgb = the raw [0, 1] base view (reconstruction target), crop_box / flip =
the geometry of g(x), text / itc_text2 = the two contrastive captions (32 ids), cap_text = the caption-decoder
caption (80 ids).
"""
import math
from typing import Any, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from .dense import DenseProjector
from .pixel_decoder import RGBDecoder
from .text import CaptionDecoder, TextEncoder
from .vit import ViT


class ProjectionHead(nn.Module):
    proj_dim: int
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.proj_dim, use_bias=False, dtype=self.dtype, param_dtype=self.param_dtype, name="proj")(x)


class Stage1Model(nn.Module):
    # visual encoder
    width: int = 768
    depth: int = 12
    mlp_dim: int = 3072
    num_heads: int = 12
    patch_size: int = 16
    remat: bool = True
    # language
    vocab_size: int = 32_000
    text_width: int = 768
    text_depth: int = 12
    text_heads: int = 12
    text_mlp_dim: int = 3072
    itc_len: int = 32
    caption_len: int = 80
    caption: bool = True                 # build the caption decoder (False: "no captioning" ablation)
    cap_depth: int = 12                  # caption decoder width = encoder width (tokens enter without projection)
    cap_heads: int = 12
    cap_mlp_dim: int = 3072
    itc_proj_dim: int = 768
    itc_temperature: float = 0.07        # init of the learned logit scale exp(t) = 1 / 0.07
    dense_proj_dim: int = 256
    noise_scale: float = 0.05            # z_noisy = z + noise_scale * rms(z) * eps
    dtype: Any = jnp.bfloat16
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x_enc, gx_enc=None, x_rgb=None, crop_box=None, flip=None, text=None, itc_text2=None,
                 cap_text=None, *, train=False, return_embeds_only=False):
        out = {}
        E = ViT(self.width, self.depth, self.mlp_dim, self.num_heads, self.patch_size, remat=self.remat,
                dtype=self.dtype, param_dtype=self.param_dtype, name="visual_encoder")
        z_global_x, z_x = E(x_enc, train=train)
        out["img_emb"] = ProjectionHead(self.itc_proj_dim, dtype=self.dtype, param_dtype=self.param_dtype,
                                        name="image_projection_head")(jnp.mean(z_x, axis=1))
        t = self.param("logit_scale", lambda k, s, d: jnp.log(1.0 / self.itc_temperature) * jnp.ones(s, d),
                       (1,), jnp.float32)
        out["logit_scale"] = jnp.exp(t)[0]     # bounded to [1, 100] by a projection after every update
        if text is not None or itc_text2 is not None:
            text_enc = TextEncoder(self.vocab_size, self.text_width, self.text_depth, self.text_heads,
                                   self.text_mlp_dim, self.itc_len, dtype=self.dtype, param_dtype=self.param_dtype,
                                   name="text_encoder")
            head = ProjectionHead(self.itc_proj_dim, dtype=self.dtype, param_dtype=self.param_dtype,
                                  name="text_projection_head")
            if text is not None:
                out["txt_emb"] = head(text_enc(text, train=train))
            if itc_text2 is not None:
                out["txt_emb2"] = head(text_enc(itc_text2, train=train))
        if cap_text is not None and self.caption:
            out["cap_logits"] = CaptionDecoder(self.vocab_size, self.width, self.cap_depth, self.cap_heads,
                                               self.cap_mlp_dim, self.caption_len, dtype=self.dtype,
                                               param_dtype=self.param_dtype, name="caption_decoder")(
                z_x, cap_text, train=train)
            out["cap_text"] = cap_text
        if return_embeds_only:
            out["z"], out["z_global"] = z_x, z_global_x
            return out

        _, z_gx = E(gx_enc, train=train)
        b, n, d = z_x.shape
        g = int(round(math.sqrt(n)))
        z_x_sp, z_gx_sp = z_x.reshape(b, g, g, d), z_gx.reshape(b, g, g, d)

        decoder = RGBDecoder(dtype=self.dtype, param_dtype=self.param_dtype)
        rng_e = self.make_rng("random_mask")
        z32 = z_x_sp.astype(jnp.float32)
        rms = jax.lax.stop_gradient(jnp.sqrt(jnp.mean(z32 ** 2, axis=-1, keepdims=True) + 1e-12))   # per token
        eps = (self.noise_scale * rms * jax.random.normal(rng_e, z_x_sp.shape, jnp.float32)).astype(z_x_sp.dtype)
        out["latent_rms"] = jax.lax.stop_gradient(rms.mean())
        out["rec_pred"] = decoder(z_x_sp)
        out["rec_pred_noisy"] = decoder(z_x_sp + eps)
        out["x_rgb"] = x_rgb

        dense_proj = DenseProjector(dim=self.dense_proj_dim, dtype=self.dtype, param_dtype=self.param_dtype)
        out["h_x"], out["h_gx"] = dense_proj(z_x_sp), dense_proj(z_gx_sp)
        out["crop_box"], out["flip"] = crop_box, flip
        return out


KEEP_AFTER_STAGE1 = ("visual_encoder", "text_encoder", "image_projection_head", "text_projection_head", "logit_scale")
