# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Frozen encoders of the ALPS evaluation behind one interface.

    enc = build("alps", weights="alps_vitb16.safetensors")       # or one of PUBLIC below
    tok = enc.encode_frames(frames)     # uint8 [B, H, W, 3] -> native last-layer patch tokens [B, N, D] (float32)
    tok = enc.encode_clip(clips)        # uint8 [B, T, H, W, 3] -> tokens of the current (last) frame; video encoders
                                        # read the whole window, image encoders the last frame only
    x = enc.normalize(x01); tok = enc(x)  # the same in two steps, for pipelines that resize in torch
    enc.meta()

Frames are passed at the resolution the benchmark chose (the encoder does not resize). Tokens are each model's native
last-layer patch tokens with prefix tokens dropped, as in every table of the paper:

  alps, alps_stage1   raw tokens of the last block (no final LayerNorm), ImageNet mean / std; for grids other than 14 x 14
                      the sine-cosine table is evaluated at indices rescaled into the 14 x 14 training range
  timm models         forward_features (after the final norm), the model's pretrained mean / std
  dinov2_b14          torch-hub DINOv2-B/14 `x_norm_patchtokens` (the DINO-WM code path), ImageNet mean / std
                      (norm="half": mean = std = 0.5, the DINO-WM transform, used in the world-model experiments);
                      the LIBERO state probe and reconstruction use this entry. The LIBERO policies and the video
                      benchmarks used timm's DINOv2 instead (`dinov2_b14_timm`, a timm model like the rows below)
  cradio_v3_b         C-RADIOv3-B backbone spatial features (no adaptor), input in [0, 1]
  ov1_b16, ov2_l14,   OpenVision `output_tokens` (before the final norm); OpenVision 3 encodes FLUX.1 VAE latents
  ov3_b2
  vjepa2_vitl         V-JEPA 2 ViT-L/16 tokens of the last 2-frame tubelet of a 16-frame window, 256 px
  dinotxt_l14         dino.txt patch tokens after its vision head

Encoders that are not on timm need their official code: C-RADIOv3 (github.com/NVlabs/RADIO, env RADIO_REPO and
RADIO_CKPT), OpenVision (github.com/UCSC-VLAA/OpenVision, env OV_ROOT), VC-1 (env VC1_CKPT), dino.txt and DINOv2 (torch
hub). Run every encoder in float32 unless a benchmark says otherwise.
"""
import os
import sys
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# paper row -> registry name (Table 1 order)
PAPER_ROWS = {
    "TIPSv2-B/14": "tipsv2_b14", "DINOv2-B/14": "dinov2_b14", "PE-Spatial-B/16": "pe_spatial_b16",
    "DINOv3-B/16": "dinov3_b16", "C-RADIOv3-B": "cradio_v3_b", "CLIP-B/16": "clip_b16_qg", "SigLIP 2-B/16": "siglip2_b16",
    "PE-Core-B/16": "pe_core_b16", "OpenVision-B/16": "ov1_b16", "MAE-B/16": "mae_b16", "VC-1-B": "vc1_b16",
    "OpenVision 3-B": "ov3_b2", "OpenVision 2-L/14": "ov2_l14", "V-JEPA 2-L/16": "vjepa2_vitl", "dino.txt-L/14": "dinotxt_l14",
    "ALPS (Stage 1)": "alps_stage1", "ALPS": "alps",
}

TIMM = {
    "tipsv2_b14": "vit_base_patch14_reg1_tipsv2.webli",       # native 448; at 224: 16 x 16 tokens
    "dinov3_b16": "vit_base_patch16_dinov3.lvd1689m",
    "clip_b16_qg": "vit_base_patch16_clip_quickgelu_224.openai",
    "siglip2_b16": "vit_base_patch16_siglip_224.v2_webli",
    "pe_core_b16": "vit_pe_core_base_patch16_224.fb",
    "pe_spatial_b16": "vit_pe_spatial_base_patch16_512.fb",   # native 512; evaluated at 224 like every row
    "mae_b16": "vit_base_patch16_224.mae",
    "vc1_b16": "vit_base_patch16_224.mae",                    # architecture; weights = the VC-1 checkpoint
    "dinov2_b14_timm": "vit_base_patch14_dinov2.lvd142m",     # timm DINOv2-B/14 (LIBERO policies, DAVIS / SSv2)
}


class FrozenEncoder(nn.Module):
    """Base class: subclasses set mean / std (or override normalize), patch, dim, and implement forward(x)."""
    name = ""
    t_hist = 1
    patch = 16
    dim = 768

    def __init__(self, mean=IMAGENET[0], std=IMAGENET[1]):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        return self.eval()

    def train(self, mode=True):                  # frozen: never leaves eval mode
        return super().train(False)

    @property
    def device(self):
        return self.mean.device

    def normalize(self, x01: torch.Tensor) -> torch.Tensor:
        return (x01 - self.mean) / self.std

    @torch.no_grad()
    def encode_frames(self, frames_u8) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(frames_u8)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        return self(self.normalize(x)).float()

    @torch.no_grad()
    def encode_clip(self, clip_u8) -> torch.Tensor:
        return self.encode_frames(np.asarray(clip_u8)[:, -1])

    def meta(self) -> Dict:
        return {"name": self.name, "dim": self.dim, "patch": self.patch, "t_hist": self.t_hist,
                "mean": self.mean.flatten().tolist(), "std": self.std.flatten().tolist(),
                "params_M": round(sum(p.numel() for p in self.parameters()) / 1e6, 1)}


class ALPSEncoder(FrozenEncoder):
    def __init__(self, name, weights, pos_mode="interpolate"):
        super().__init__()
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        import alps
        self.name, self.weights = name, weights
        self.model = alps.load_encoder(weights, pos_mode=pos_mode)

    def forward(self, x):
        return self.model(x)["tokens"]

    def meta(self):
        return {**super().meta(), "source": os.path.basename(self.weights), "output": "raw last-block patch tokens (no final LayerNorm)",
                "pos_mode": self.model.pos_mode}


class TimmEncoder(FrozenEncoder):
    def __init__(self, name, res=224, dynamic_img_size=False, pretrained=True):
        import timm
        self.name = name
        model = timm.create_model(TIMM[name], pretrained=pretrained, num_classes=0, img_size=res,
                                  **({"dynamic_img_size": True} if dynamic_img_size else {}))
        cfg = timm.data.resolve_model_data_config(model)
        super().__init__(cfg["mean"], cfg["std"])
        self.model = model
        if name == "vc1_b16":
            self._load_vc1()
        self.n_prefix = getattr(model, "num_prefix_tokens", 1)
        ps = model.patch_embed.patch_size
        self.patch = ps[0] if isinstance(ps, (tuple, list)) else ps
        self.dim = model.num_features

    def _load_vc1(self):
        path = os.environ.get("VC1_CKPT", "vc1_vitb.pth")
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("model", sd.get("state_dict", sd)) if isinstance(sd, dict) else sd
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if not k.startswith("decoder") and k != "mask_token"}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if [m for m in missing if not m.startswith("head")] or unexpected:
            raise RuntimeError(f"VC-1 checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")

    def forward(self, x):
        return self.model.forward_features(x)[:, self.n_prefix:]

    def meta(self):
        return {**super().meta(), "source": f"timm {TIMM[self.name]}", "output": "forward_features (after the final norm), prefix dropped"}


class DinoV2Hub(FrozenEncoder):
    patch = 14

    def __init__(self, name="dinov2_b14", norm="imagenet"):
        super().__init__(*(IMAGENET if norm == "imagenet" else ((0.5,) * 3, (0.5,) * 3)))
        self.name, self.norm = name, norm
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

    def forward(self, x):
        return self.model.forward_features(x)["x_norm_patchtokens"]

    def meta(self):
        return {**super().meta(), "source": "torch hub facebookresearch/dinov2 dinov2_vitb14", "output": "x_norm_patchtokens"}


class RadioEncoder(FrozenEncoder):
    """C-RADIOv3-B via the official hubconf; the model normalizes internally, so normalize() only maps to [0, 1]."""

    def __init__(self, name="cradio_v3_b"):
        super().__init__((0.0,) * 3, (1.0,) * 3)
        self.name = name
        repo, ckpt = os.environ.get("RADIO_REPO", "NVlabs_RADIO"), os.environ.get("RADIO_CKPT", "c-radio_v3-b_half.pth.tar")
        self.model = torch.hub.load(repo, "radio_model", source="local", version=ckpt, progress=False, skip_validation=True)

    def forward(self, x):
        out = self.model(x)
        return out[1] if isinstance(out, tuple) else out.features

    def meta(self):
        return {**super().meta(), "source": "NVlabs/RADIO hubconf + nvidia/C-RADIOv3-B", "output": "backbone spatial features (no adaptor)"}


class OpenVisionEncoder(FrozenEncoder):
    REPO = {"ov1_b16": "UCSC-VLAA/openvision-vit-base-patch16-224",
            "ov2_l14": "UCSC-VLAA/openvision2-vit-large-patch14-224-vision-only",
            "ov3_b2": "UCSC-VLAA/openvision3-vit-base-patch2-28"}

    def __init__(self, name):
        super().__init__()
        root = os.environ.get("OV_ROOT", "OpenVision")
        for p in (os.path.join(root, "code", "OpenVision-main"), os.path.join(root, "code")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from src.convert_upload.open_clip.factory import create_vision_encoder_and_transforms
        self.name = name
        self.vit = create_vision_encoder_and_transforms(model_name="hf-hub:" + self.REPO[name])
        self.vit.output_tokens = True
        self.vae = None
        if name == "ov3_b2":
            from flux_vae_enc import FluxVAEEncoder
            self.vae = FluxVAEEncoder.from_hf(os.path.join(root, "hf"))
        self.dim = int(self.vit.transformer.width)
        self.patch = 14 if name == "ov2_l14" else 16

    def normalize(self, x01):
        return x01                                 # OpenVision normalizes inside forward (ImageNet, or [-1, 1] for the VAE)

    def forward(self, x01):
        z = self.vae(x01 * 2.0 - 1.0) if self.vae is not None else (x01 - self.mean) / self.std
        out = self.vit(z)
        return out[1] if isinstance(out, (tuple, list)) else out


class VJEPA2Encoder(FrozenEncoder):
    """V-JEPA 2 ViT-L/16 (facebook/vjepa2-vitl-fpc64-256), a video encoder; tokens of the last tubelet (the current time).
    LIBERO probe: the last 16 frames resized to 256 px (bicubic, antialiased). Reconstruction: one image at 224 px repeated
    into one 2-frame tubelet (res=224; the model is built with crop_size=res so that its RoPE grid matches)."""
    t_hist = 16

    def __init__(self, name="vjepa2_vitl", hf_id="facebook/vjepa2-vitl-fpc64-256", res=256):
        super().__init__()
        from transformers import VJEPA2Model
        self.name, self.hf_id, self.res = name, hf_id, res
        self.model = VJEPA2Model.from_pretrained(hf_id, crop_size=res, torch_dtype=torch.float32)
        self.tubelet = int(getattr(self.model.config, "tubelet_size", 2))
        self.dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode_clip(self, clip_u8):
        x = torch.from_numpy(np.ascontiguousarray(clip_u8)).to(self.device).permute(0, 1, 4, 2, 3).float() / 255.0
        b, t = x.shape[:2]
        if x.shape[-1] != self.res:
            x = F.interpolate(x.flatten(0, 1), size=(self.res, self.res), mode="bicubic", align_corners=False,
                              antialias=True).view(b, t, 3, self.res, self.res)
        x = (x - self.mean[:, None]) / self.std[:, None]
        h = self.model(pixel_values_videos=x).last_hidden_state
        g = self.res // 16
        return h.view(b, t // self.tubelet, g * g, self.dim)[:, -1].float()

    @torch.no_grad()
    def encode_frames(self, frames_u8):           # a single image as one tubelet (diagnostic only)
        return self.encode_clip(np.repeat(np.asarray(frames_u8)[:, None], self.tubelet, 1))


class DinoTxtEncoder(FrozenEncoder):
    patch = 14
    dim = 1024

    def __init__(self, name="dinotxt_l14"):
        super().__init__()
        self.name = name
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l")

    def forward(self, x):
        return self.model.get_visual_class_and_patch_tokens(x)[1]


PUBLIC = list(TIMM) + ["dinov2_b14", "cradio_v3_b", "ov1_b16", "ov2_l14", "ov3_b2", "vjepa2_vitl", "dinotxt_l14"]


def build(name: str, device: str = "cuda", weights: Optional[str] = None, res: int = 224,
          dynamic_img_size: bool = False, pos_mode: str = "interpolate", dinov2_norm: str = "imagenet",
          vjepa2_probe: bool = False, pretrained: bool = True) -> FrozenEncoder:
    """Frozen encoder on `device`. `weights`: the safetensors file for alps / alps_stage1 (default: env ALPS_WEIGHTS /
    ALPS_STAGE1_WEIGHTS). `res` / `dynamic_img_size`: timm creation (position embeddings resampled to `res` at load time;
    dynamic = interpolated at every forward, for non-square or variable inputs such as DAVIS frames). `vjepa2_probe`: V-JEPA 2
    at 256 px over 16-frame windows (LIBERO probe) instead of `res`."""
    if name in ("alps", "alps_stage1"):
        w = weights or os.environ.get("ALPS_WEIGHTS" if name == "alps" else "ALPS_STAGE1_WEIGHTS")
        if not w:
            raise ValueError(f"{name}: pass weights= or set ALPS_WEIGHTS / ALPS_STAGE1_WEIGHTS")
        enc = ALPSEncoder(name, w, pos_mode)
    elif name in TIMM:
        enc = TimmEncoder(name, res, dynamic_img_size, pretrained=pretrained)
    elif name == "dinov2_b14":
        enc = DinoV2Hub(name, dinov2_norm)
    elif name == "cradio_v3_b":
        enc = RadioEncoder(name)
    elif name in OpenVisionEncoder.REPO:
        enc = OpenVisionEncoder(name)
    elif name == "vjepa2_vitl":
        enc = VJEPA2Encoder(name, res=res if res != 224 or not vjepa2_probe else 256)
    elif name == "dinotxt_l14":
        enc = DinoTxtEncoder(name)
    else:
        raise KeyError(f"unknown encoder {name!r}; known: alps, alps_stage1, {', '.join(PUBLIC)}")
    return enc.freeze().to(device)
