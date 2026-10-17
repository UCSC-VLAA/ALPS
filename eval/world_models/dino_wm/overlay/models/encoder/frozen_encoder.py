# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""Frozen visual encoder of the ALPS evaluation behind the DINO-WM encoder interface.

The backbone comes from the unified encoder registry of the ALPS repository (`eval/encoders.py`, found through the
environment variable ALPS_ROOT = the repository root). This module adds the conventions of the world-model experiments:

  x [B, 3, 224, 224] in [-1, 1]           (DINO-WM image transform: Normalize(mean=0.5, std=0.5))
    -> x * 0.5 + 0.5, resized to the encoder's input size `res` (torchvision bilinear, antialiased) when it differs:
       224 for every patch-16 encoder, 196 for DINOv2-B/14 so that it sees a 14 x 14 grid, the rule of DINO-WM
    -> the encoder's own input normalisation (ALPS: ImageNet; timm models: their pretrained mean / std;
       DINOv2-B/14: mean = std = 0.5, the DINO-WM transform; C-RADIOv3-B: RGB in [0, 1], clamped)
    -> frozen encoder, float32 with TF32 allowed inside the encoder forward only (`tf32`)
    -> native last-layer patch tokens [B, 196, D], prefix tokens dropped (ALPS: raw tokens of the last block, no final
       LayerNorm; timm: forward_features; DINOv2: x_norm_patchtokens; C-RADIOv3: backbone spatial features)
    -> bilinear resampling of the token grid to 14 x 14 if it differs (none of the six encoders of the paper needs it)
    -> per-channel standardisation (t - mu) / sd with statistics of training frames (fit_stats.py), unless stats is None
       (raw tokens, used by fit_stats.py itself)

The world model works on these D = 768 standardised channels; the learned projections D -> 384 -> D are part of the
predictor (models/vit_proj.py). The encoder is never trained and stays in eval mode.
"""
import contextlib
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# encoders of the paper's world-model tables (names of eval/encoders.py)
PAPER_ENCODERS = ("alps", "alps_stage1", "dinov2_b14", "dinov3_b16", "siglip2_b16", "cradio_v3_b")
# input size of the encoder; every other encoder: 224
NATIVE_RES = {"dinov2_b14": 196}
GRID = 14

_REGISTRY = None


def release_encoders():
    """eval/encoders.py of the ALPS repository, loaded by path under a private module name (a plain `import eval`
    could resolve to another module of that name, e.g. eval.py of LeWorldModel)."""
    global _REGISTRY
    if _REGISTRY is None:
        root = os.environ.get("ALPS_ROOT")
        if not root:
            raise RuntimeError("set ALPS_ROOT to the root of the ALPS repository (the folder that contains eval/encoders.py)")
        path = os.path.join(root, "eval", "encoders.py")
        spec = importlib.util.spec_from_file_location("alps_eval_encoders", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _REGISTRY = mod
    return _REGISTRY


def load_stats(path):
    """stats npz of fit_stats.py -> (mu [D], sd [D], meta dict)."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return torch.from_numpy(np.asarray(z["mu"])).float(), torch.from_numpy(np.asarray(z["sd"])).float(), meta


@contextlib.contextmanager
def tf32_ctx(enabled):
    """Allow TF32 matmuls / convolutions for the frozen encoder forward only; the previous flags are restored afterwards,
    so the predictor keeps the precision of the caller."""
    if not enabled:
        yield
        return
    m, c = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = m
        torch.backends.cudnn.allow_tf32 = c


def to_grid14(tokens, g):
    """[B, g * g, D] -> [B, 196, D] by bilinear resampling of the token grid (identity for g = 14)."""
    if g == GRID:
        return tokens
    b, n, d = tokens.shape
    t = tokens.transpose(1, 2).reshape(b, d, g, g)
    t = F.interpolate(t, size=(GRID, GRID), mode="bilinear", align_corners=False)
    return t.reshape(b, d, GRID * GRID).transpose(1, 2)


class FrozenTokenEncoder(nn.Module):
    """DINO-WM encoder interface (emb_dim, latent_ndim, name, patch_size, forward) around a frozen encoder of
    eval/encoders.py. `stats`: npz of fit_stats.py (None = raw tokens); `res`: input size (default NATIVE_RES or 224);
    `weights`: safetensors file of alps / alps_stage1 (default: env ALPS_WEIGHTS / ALPS_STAGE1_WEIGHTS); `backbone`: an
    already built FrozenEncoder (tests)."""

    def __init__(self, encoder, stats=None, res=None, tf32=True, weights=None, backbone=None):
        super().__init__()
        self.encoder_name = encoder
        # VWorldModel resizes the input itself whenever "dino" is part of encoder.name; the resize is done here instead,
        # so the name must not contain "dino" (dinov2 -> dnv2). patch_size = 16 would make that resize a no-op anyway.
        self.name = "frozen_" + encoder.replace("dino", "dn")
        self.patch_size = 16
        self.latent_ndim = 2
        self.res = int(res) if res is not None else NATIVE_RES.get(encoder, 224)
        self.tf32 = bool(tf32)
        if backbone is None:
            kw = {"dinov2_norm": "half"} if encoder == "dinov2_b14" else {}
            backbone = release_encoders().build(encoder, device="cpu", weights=weights, res=224, **kw)
        self.backbone = backbone
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if self.res % int(self.backbone.patch):
            raise ValueError(f"{encoder}: input size {self.res} is not a multiple of the patch size {self.backbone.patch}")
        self.native_grid = self.res // int(self.backbone.patch)
        self.raw_dim = int(self.backbone.dim)
        self.clamp01 = encoder.startswith("cradio")           # C-RADIOv3 reads RGB in [0, 1]
        self.stats_path = stats
        self.stats_meta = None
        if stats is not None:
            mu, sd, self.stats_meta = load_stats(stats)
            if mu.numel() != self.raw_dim:
                raise ValueError(f"stats {stats}: {mu.numel()} channels, encoder {encoder} has {self.raw_dim}")
            if self.stats_meta.get("encoder") not in (None, encoder):
                raise ValueError(f"stats {stats} were fitted for {self.stats_meta['encoder']}, not {encoder}")
            self.register_buffer("mu", mu.view(1, 1, -1))
            self.register_buffer("sd", sd.view(1, 1, -1))
        self.emb_dim = self.raw_dim
        print(f"[FrozenTokenEncoder] {encoder}: res {self.res}, grid {self.native_grid}, D {self.raw_dim}, stats {stats}, "
              f"tf32 {self.tf32}, params {sum(p.numel() for p in self.backbone.parameters()) / 1e6:.1f}M", flush=True)

    def train(self, mode=True):
        """VWorldModel.train() recurses into its children: keep the frozen encoder in eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def prep(self, x):
        """x [B, 3, H, W] in [-1, 1] -> normalised encoder input at self.res."""
        x01 = x.float() * 0.5 + 0.5                      # inverse of Normalize(0.5, 0.5)
        if x01.shape[-2] != self.res or x01.shape[-1] != self.res:
            x01 = TF.resize(x01, [self.res, self.res], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        x01 = (x01 * 255.0) / 255.0                      # the 0..255 round trip of the frame path of the original runs
        xn = self.backbone.normalize(x01)
        return xn.clamp(0, 1) if self.clamp01 else xn

    def forward_raw(self, x):
        """Native tokens before standardisation: [B, 196, D] float32."""
        with torch.no_grad(), tf32_ctx(self.tf32):
            f = self.backbone(self.prep(x)).float()
            return to_grid14(f, self.native_grid)

    def standardize(self, f):
        return (f - self.mu) / self.sd

    def forward(self, x):
        with torch.no_grad():
            f = self.forward_raw(x)
            return f if self.stats_path is None else self.standardize(f)

    def meta(self):
        return {"encoder": self.encoder_name, "res": self.res, "native_grid": self.native_grid, "raw_dim": self.raw_dim,
                "stats": self.stats_path, "tf32": self.tf32, "backbone": self.backbone.meta()}
