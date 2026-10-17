# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Use the ALPS encoder as a standard timm VisionTransformer (timm >= 1.0).

    vit = alps.timm_compat.create_timm_model("alps_vitb16.safetensors")
    feats = vit.forward_features(x)        # [B, 1 + 196, 768]; feats[:, 1:] are the raw ALPS tokens
    pooled = vit.forward_head(feats, pre_logits=True)    # LayerNorm(mean of patch tokens)

The ALPS encoder is a timm ViT-B/16 with a zero patch-embedding bias, the fixed sin-cos table
stored in `pos_embed`, average pooling with the final LayerNorm applied after pooling (fc_norm),
and no LayerNorm on the tokens themselves.
"""
from __future__ import annotations

from typing import Dict

import torch

__all__ = ["to_timm_state_dict", "create_timm_model"]


def to_timm_state_dict(sd: Dict[str, torch.Tensor], prefix: str = "visual.") -> Dict[str, torch.Tensor]:
    from .vit import sincos_2d
    v = {k[len(prefix):]: t for k, t in sd.items() if k.startswith(prefix)}
    dim = v["cls_token"].shape[-1]
    grid = 14
    out = {"cls_token": v["cls_token"], "pos_embed": sincos_2d(grid, grid, dim),
           "patch_embed.proj.weight": v["patch_embed.weight"],
           "patch_embed.proj.bias": torch.zeros(dim),
           "fc_norm.weight": v["norm.weight"], "fc_norm.bias": v["norm.bias"]}
    for k, t in v.items():
        if k.startswith("blocks."):
            out[k] = t
    return out


def create_timm_model(path: str, **kw):
    import timm
    from .hub import read_state_dict
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0, global_pool="avg",
                              fc_norm=True, **kw)
    missing, unexpected = model.load_state_dict(to_timm_state_dict(read_state_dict(path)), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"timm state dict mismatch: missing={missing} unexpected={unexpected}")
    return model.eval()
