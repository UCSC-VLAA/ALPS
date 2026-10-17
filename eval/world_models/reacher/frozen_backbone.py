# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""Frozen backbones for the PreJEPA world model of stable-worldmodel / LeWorldModel on Reacher (Table 3, Tables S12, S16).

Interface expected by stable_worldmodel.wm.PreJEPA and scripts/train/prejepa.py:
  backbone(pixels, **kw) -> object with .last_hidden_state [B, 1 + P, D]  (PreJEPA drops index 0 as the class token)
  backbone.config.hidden_size
The pixels arrive ImageNet-normalised at 224 pixels (prejepa.py get_img_preprocessor / eval.py img_transform).

Tokens of every encoder: the native last-layer patch tokens of eval/encoders.py (env ALPS_ROOT), 196 tokens of 768 channels:
  alps, alps_stage1   raw tokens of the last block, ImageNet input
  dinov3_b16,         timm forward_features with the model's own mean / std (the pixels are mapped back to [0, 1] first)
  siglip2_b16
  cradio_v3_b         backbone spatial features, RGB in [0, 1] (clamped)
  dinov2_b14_196      DINOv2-B/14 of Hugging Face transformers (facebook/dinov2-base, last_hidden_state = after the final
                      LayerNorm, class token dropped) on the ImageNet-normalised pixels resized to 196 (bilinear, antialiased),
                      a 14 x 14 grid; the model of the paper's Reacher runs, not the torch-hub DINOv2 of eval/encoders.py
then each token is normalised to unit length and standardised per channel, (t - mu) / sd, with fixed statistics of 4,096
training frames (make_stats.py; mu / sd are buffers, so they are also saved in and restored from the world-model weights).
The encoder runs in float32 inside the predictor's fp16 autocast.
"""
import importlib.util
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
ENCODERS = ("alps", "alps_stage1", "dinov2_b14_196", "dinov3_b16", "siglip2_b16", "cradio_v3_b")
HF_DINOV2 = "facebook/dinov2-base"


def release_encoders():
    """eval/encoders.py of the ALPS repository (env ALPS_ROOT), loaded by path: LeWorldModel has its own eval.py."""
    name = "alps_eval_encoders"
    if name not in sys.modules:
        root = os.environ.get("ALPS_ROOT")
        if not root:
            raise RuntimeError("set ALPS_ROOT to the root of the ALPS repository (the folder that contains eval/encoders.py)")
        spec = importlib.util.spec_from_file_location(name, os.path.join(root, "eval", "encoders.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


class _Out:
    def __init__(self, h):
        self.last_hidden_state = h


class FrozenTokens(nn.Module):
    def __init__(self, name, stats=None, weights=None, backbone=None):
        super().__init__()
        if name not in ENCODERS:
            raise ValueError(f"unknown backbone {name}; known: {ENCODERS}")
        self.name = name
        self.register_buffer("im_mean", torch.tensor(IMNET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("im_std", torch.tensor(IMNET_STD).view(1, 3, 1, 1), persistent=False)
        if backbone is not None:
            self.vit, self.kind = backbone, "registry"
        elif name == "dinov2_b14_196":
            from transformers import AutoModel
            self.vit, self.kind = AutoModel.from_pretrained(HF_DINOV2), "hf196"
        else:
            self.vit, self.kind = release_encoders().build(name, device="cpu", weights=weights, res=224), "registry"
        self.clamp01 = name.startswith("cradio")
        self.config = types.SimpleNamespace(hidden_size=768)
        self.register_buffer("mu", torch.zeros(768))
        self.register_buffer("sd", torch.ones(768))
        self.stats = stats
        if stats is not None:
            if os.path.exists(stats):
                z = np.load(stats)
                self.mu.copy_(torch.from_numpy(z["mean"]).float()); self.sd.copy_(torch.from_numpy(z["std"]).float())
            else:   # e.g. world-model weights loaded on another machine: the buffers come from the weights file
                print(f"[FrozenTokens] stats {stats} not found: mu / sd from the loaded state dict", flush=True)
        self.eval(); self.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.vit.eval()                                  # frozen: never in training mode
        return self

    def _encode(self, x):
        if self.kind == "hf196":
            x196 = F.interpolate(x, size=(196, 196), mode="bilinear", align_corners=False, antialias=True)
            return self.vit(pixel_values=x196).last_hidden_state[:, 1:]
        if self.name in ("alps", "alps_stage1"):
            return self.vit(x)                               # ImageNet-normalised input, as the pixels arrive
        x01 = x * self.im_std + self.im_mean                 # back to RGB in [0, 1], then the encoder's own constants
        xn = self.vit.normalize(x01)
        return self.vit(xn.clamp(0, 1) if self.clamp01 else xn)

    @torch.no_grad()
    def tokens(self, x):
        """ImageNet-normalised pixels [B, 3, 224, 224] -> unit-length tokens [B, 196, 768], float32 (also inside autocast)."""
        with torch.autocast(device_type="cuda", enabled=False):
            z = self._encode(x.float())
        z = z.float()
        assert z.shape[-1] == 768 and z.shape[1] == 196 and torch.isfinite(z).all(), (self.name, tuple(z.shape))
        return F.normalize(z, dim=-1)

    @torch.no_grad()
    def forward(self, pixels, **kw):
        z = self.tokens(pixels)
        z = (z - self.mu) / self.sd
        return _Out(torch.cat([torch.zeros_like(z[:, :1]), z], dim=1))
