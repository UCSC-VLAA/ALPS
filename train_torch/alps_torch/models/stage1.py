# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Stage-1 model of ALPS: the encoder with the heads of its four training objectives.

Submodules (state-dict prefixes) and their Flax counterparts:

    visual            visual_encoder          ViT-B/16 (alps.vit.VisionTransformer parameters)
    image_projection  image_projection_head   Linear(D -> 768, no bias) of the MEAN raw patch token
    logit_scale       logit_scale             scalar, init ln(1 / 0.07); exp() is the contrastive scale, kept in
                                              [1, 100] by a projection of the parameter after every update
    text              text_encoder            contrastive text encoder (alps.text.TextEncoder parameters)
    text_projection   text_projection_head    Linear(768 -> 768, no bias) of the last text slot
    caption_decoder   caption_decoder         prefix-LM caption decoder
    pixel_decoder     RGBDecoder_0            pixel decoder of the reconstruction objective
    dense_projector   DenseProjector_0        projector of the dense correspondence objective

The first five use the key layout of `alps.ALPS`, so `alps_torch.convert.export_alps` writes them as an `alps`
checkpoint unchanged. `forward` returns the tensors the Stage-1 loss needs (alps_torch/losses/stage1.py).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .. import init as I
from .. import nn as N
from .dense import DenseProjector
from .pixel_decoder import RGBDecoder
from .text import CaptionDecoder, TextEncoder
from .vit import ViT

KEEP_AFTER_STAGE1 = ("visual", "text", "image_projection", "text_projection", "logit_scale")


class Stage1Model(nn.Module):
    def __init__(self, width: int = 768, depth: int = 12, mlp_dim: int = 3072, num_heads: int = 12,
                 patch_size: int = 16, remat: bool = True, vocab_size: int = 32_000, text_width: int = 768,
                 text_depth: int = 12, text_heads: int = 12, text_mlp_dim: int = 3072, itc_len: int = 32,
                 caption_len: int = 80, caption: bool = True, cap_depth: int = 12, cap_heads: int = 12,
                 cap_mlp_dim: int = 3072, itc_proj_dim: int = 768, itc_temperature: float = 0.07,
                 dense_proj_dim: int = 256, noise_scale: float = 0.05, dtype: torch.dtype = torch.float32,
                 attn_impl: str = "explicit", img_size: int = 224):
        super().__init__()
        self.compute_dtype = dtype
        self.noise_scale = noise_scale
        self.itc_temperature = itc_temperature
        self.visual = ViT(width, depth, mlp_dim, num_heads, patch_size, img_size=img_size, dtype=dtype, remat=remat,
                          attn_impl=attn_impl)
        self.image_projection = nn.Linear(width, itc_proj_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.zeros(()))
        self.text = TextEncoder(vocab_size, text_width, text_depth, text_heads, text_mlp_dim, itc_len, dtype=dtype,
                                attn_impl=attn_impl)
        self.text_projection = nn.Linear(text_width, itc_proj_dim, bias=False)
        self.caption_decoder = (CaptionDecoder(vocab_size, width, cap_depth, cap_heads, cap_mlp_dim, caption_len,
                                               dtype=dtype, attn_impl=attn_impl) if caption else None)
        self.pixel_decoder = RGBDecoder(width, dtype)
        self.dense_projector = DenseProjector(width, dense_proj_dim, dtype)
        self._reset_own_parameters()

    @torch.no_grad()
    def _reset_own_parameters(self, generator: Optional[torch.Generator] = None) -> None:
        I.lecun_normal_(self.image_projection.weight, generator)
        I.lecun_normal_(self.text_projection.weight, generator)
        self.logit_scale.fill_(float(torch.tensor(1.0 / self.itc_temperature, dtype=torch.float32).log()))

    @torch.no_grad()
    def reset_parameters(self, seed: int = 0) -> None:
        """Re-draw every parameter with the Flax initializers from one torch generator."""
        g = torch.Generator().manual_seed(seed)
        dev = self.logit_scale.device
        if dev.type != "cpu":
            g = torch.Generator(dev).manual_seed(seed)
        for m in (self.visual, self.text, self.caption_decoder, self.pixel_decoder, self.dense_projector):
            if m is not None:
                m.reset_parameters(g)
        self._reset_own_parameters(g)

    def forward(self, x_enc: torch.Tensor, gx_enc: Optional[torch.Tensor] = None, x_rgb: Optional[torch.Tensor] = None,
                crop_box: Optional[torch.Tensor] = None, flip: Optional[torch.Tensor] = None,
                text: Optional[torch.Tensor] = None, itc_text2: Optional[torch.Tensor] = None,
                cap_text: Optional[torch.Tensor] = None, *, train: bool = True, noise: Optional[torch.Tensor] = None,
                generator: Optional[torch.Generator] = None,
                return_embeds_only: bool = False) -> Dict[str, torch.Tensor]:
        """Images are [B, 3, H, W]: x_enc / gx_enc ImageNet-normalized, x_rgb in [0, 1] (the reconstruction target).
        `noise` [B, N, D] = the standard-normal draws of the reconstruction noise (drawn from `generator` if None)."""
        out: Dict[str, torch.Tensor] = {}
        enc = self.visual(x_enc, train=train, pooled=return_embeds_only)
        z_x, grid = enc["tokens"], enc["grid"]
        out["img_emb"] = self._project(self.image_projection, z_x.mean(dim=1))
        out["logit_scale"] = torch.exp(self.logit_scale)
        if text is not None:
            out["txt_emb"] = self._project(self.text_projection, self.text(text, train=train))
        if itc_text2 is not None:
            out["txt_emb2"] = self._project(self.text_projection, self.text(itc_text2, train=train))
        if cap_text is not None and self.caption_decoder is not None:
            out["cap_logits"] = self.caption_decoder(z_x, cap_text, train=train)
            out["cap_text"] = cap_text
        if return_embeds_only:
            out["z"], out["z_global"] = z_x, enc["pooled"]
            return out

        z_gx = self.visual(gx_enc, train=train)["tokens"]
        z32 = z_x.float()
        rms = torch.sqrt(torch.mean(z32 ** 2, dim=-1, keepdim=True) + 1e-12).detach()       # per token
        if noise is None:
            noise = torch.randn(z_x.shape, generator=generator, device=z_x.device, dtype=torch.float32)
        eps = (self.noise_scale * rms * noise.float()).to(z_x.dtype)
        out["latent_rms"] = rms.mean().detach()
        out["rec_pred"] = self.pixel_decoder(z_x, grid)
        out["rec_pred_noisy"] = self.pixel_decoder(z_x + eps, grid)
        out["x_rgb"] = x_rgb
        b, n, d = z_x.shape
        out["h_x"] = self.dense_projector(z_x).reshape(b, grid[0], grid[1], -1)
        out["h_gx"] = self.dense_projector(z_gx).reshape(b, grid[0], grid[1], -1)
        out["crop_box"], out["flip"] = crop_box, flip
        return out

    def _project(self, lin: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        return N.dense(x, lin.weight, None, self.compute_dtype)


@torch.no_grad()
def project_logit_scale(model: nn.Module, max_scale: float = 100.0) -> None:
    """Keep exp(logit_scale) in [1, max_scale]: clip the parameter to [0, ln(max_scale)] (float32)."""
    m = getattr(model, "module", model)
    if getattr(m, "logit_scale", None) is not None:
        m.logit_scale.clamp_(0.0, float(torch.tensor(max_scale, dtype=torch.float32).log()))


def stage1_model_from_config(mc, dtype=torch.float32, attn_impl="explicit", img_size=224) -> Stage1Model:
    keys = ("width", "depth", "mlp_dim", "num_heads", "patch_size", "remat", "vocab_size", "text_width", "text_depth",
            "text_heads", "text_mlp_dim", "itc_len", "caption_len", "caption", "cap_depth", "cap_heads", "cap_mlp_dim",
            "itc_proj_dim", "itc_temperature", "dense_proj_dim", "noise_scale")
    kw = {k: mc[k] for k in keys if k in mc}
    return Stage1Model(**kw, dtype=dtype, attn_impl=attn_impl, img_size=img_size)

