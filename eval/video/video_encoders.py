# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Frozen encoders of the video benchmarks (DAVIS label propagation, SSv2 attentive probe), built on eval/encoders.py.

    venc = build_video_encoder("alps", weights="alps_vitb16.safetensors")
    tok = venc.forward_images(x)   # x [N, 3, H, W], ImageNet-normalized -> [N, (H/p) * (W/p), D]
    tok = venc.forward_clip(x)     # x [B, T, 3, H, W] -> [B, T', (H/p) * (W/p), D]: T' = T for image encoders (every
                                   # frame encoded alone), T / 2 for V-JEPA 2 (tubelets of 2 frames, attention over the clip)

Both benchmarks normalize their frames with the ImageNet mean and standard deviation for every encoder (the vjepa2 data
pipeline of the SSv2 probe does so, and DAVIS uses the same convention). `VideoEncoder.from_imagenet` maps these inputs
to the convention of each encoder exactly as the runs of the paper did:
  ImageNet statistics (ALPS, DINOv2, DINOv3, V-JEPA 2)   x unchanged
  other statistics m, s (TIPSv2, CLIP, SigLIP 2, PE-Core)  x * (s0 / s) + (m0 - m) / s, with ImageNet m0, s0
  C-RADIOv3 (normalizes internally)                       clamp(x * s0 + m0, 0, 1)
The clamp matters only where SSv2's random erasing writes values outside the image range.

Encoders and their position handling at frame grids other than the pretraining grid:
  alps, alps_stage1   eval.encoders, pos_mode="interpolate": the sine-cosine table at cell indices rescaled into the
                      14 x 14 training range
  timm models         eval.encoders with res=224 and dynamic_img_size=True: position embeddings resampled to 224 px
                      when the model is created and interpolated to every frame grid at run time. DINOv2-B/14 is
                      dinov2_b14_timm (timm vit_base_patch14_dinov2.lvd142m); dinov2_b14 of eval.encoders is the
                      torch-hub model of the world-model experiments, whose position interpolation differs, and is not
                      used for the video tables
  dino_b16            timm vit_base_patch16_224.dino, as above: the anchor of the DAVIS pipeline (DINO paper: J&F .623)
  cradio_v3_b         eval.encoders (C-RADIOv3 handles any resolution itself)
  vjepa2_vitl         HF facebook/vjepa2-vitl-fpc64-256, encoder only, created with crop_size 224 (VJEPA2Video below).
                      Its 3-D RoPE takes the token grid from crop_size and assumes a square grid; here the position ids
                      are recomputed from the actual grid (gh, gw) of every input, and with rope="interp" (the protocol
                      of the paper) the height and width positions are scaled to the 16 x 16 pretraining grid,
                      (i * 16 / gh, j * 16 / gw); rope="recompute" keeps the integer positions of the grid.
"""
import os
import sys
import types
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from eval import encoders as E  # noqa: E402

IMAGENET = E.IMAGENET

# Rows of the video table of the paper (Tab. S19) -> names accepted by build_video_encoder.
PAPER_VIDEO_ROWS = {
    "TIPSv2-B/14": "tipsv2_b14", "DINOv2-B/14": "dinov2_b14_timm", "DINOv3-B/16": "dinov3_b16", "C-RADIOv3-B": "cradio_v3_b",
    "CLIP-B/16": "clip_b16_qg", "SigLIP 2-B/16": "siglip2_b16", "PE-Core-B/16": "pe_core_b16",
    "V-JEPA 2-L/16": "vjepa2_vitl", "ALPS (Stage 1)": "alps_stage1", "ALPS": "alps",
}

# timm models of the video runs that eval.encoders does not register: dino_b16 = DINO ViT-B/16, the anchor of the DAVIS
# pipeline (not in the paper's tables): J&F .622 in our runs, .623 in the DINO paper
VIDEO_TIMM = {"dino_b16": "vit_base_patch16_224.dino"}

VJEPA2_HF = "facebook/vjepa2-vitl-fpc64-256"
VJEPA2_PRETRAIN_GRID = 16          # 256 px / patch 16 at pretraining


class TimmIdEncoder(E.FrozenEncoder):
    """A timm model given by its id, with the conventions of eval.encoders.TimmEncoder (forward_features after the
    final norm, prefix tokens dropped, the model's pretrained mean / std)."""

    def __init__(self, name: str, timm_id: str, res: int = 224, dynamic_img_size: bool = True):
        import timm
        model = timm.create_model(timm_id, pretrained=True, num_classes=0, img_size=res,
                                  **({"dynamic_img_size": True} if dynamic_img_size else {}))
        cfg = timm.data.resolve_model_data_config(model)
        super().__init__(cfg["mean"], cfg["std"])
        self.name, self.timm_id, self.model = name, timm_id, model
        self.n_prefix = getattr(model, "num_prefix_tokens", 1)
        ps = model.patch_embed.patch_size
        self.patch = ps[0] if isinstance(ps, (tuple, list)) else ps
        self.dim = model.num_features

    def forward(self, x):
        return self.model.forward_features(x)[:, self.n_prefix:]

    def meta(self):
        return {**super().meta(), "source": f"timm {self.timm_id}", "output": "forward_features (after the final norm), prefix dropped"}


def _rope_position_ids(mode: str):
    """get_position_ids of HF VJEPA2RopeAttention, for the token grid (gh, gw) stored on the module."""

    def get_position_ids(self, x, masks=None):
        ids = torch.arange(x.size(1), device=x.device) if masks is None else masks.unsqueeze(1).repeat(1, self.num_attention_heads, 1)
        tpf = self.gh * self.gw
        f = ids // tpf
        r = ids - tpf * f
        h = r // self.gw
        w = r - self.gw * h
        if mode == "interp":
            return f, h.float() * (VJEPA2_PRETRAIN_GRID / self.gh), w.float() * (VJEPA2_PRETRAIN_GRID / self.gw)
        return f, h, w

    return get_position_ids


class VJEPA2Video(E.FrozenEncoder):
    """V-JEPA 2 ViT-L/16 as a video encoder over clips [B, T, 3, H, W] (ImageNet-normalized, T even, H and W multiples
    of 16) -> tokens of every tubelet [B, T / 2, (H/16) * (W/16), 1024] (encoder output after its final LayerNorm)."""
    t_hist = 16
    patch = 16
    dim = 1024

    def __init__(self, name: str = "vjepa2_vitl", hf_id: str = VJEPA2_HF, img_size: int = 224, rope: str = "interp",
                 model: Optional[nn.Module] = None):
        super().__init__()                         # ImageNet mean / std
        if rope not in ("interp", "recompute"):
            raise ValueError(f"rope must be 'interp' or 'recompute', got {rope!r}")
        if model is None:
            from transformers import VJEPA2Model
            model = VJEPA2Model.from_pretrained(hf_id, crop_size=img_size, attn_implementation="sdpa", torch_dtype=torch.float32)
        self.name, self.hf_id, self.rope = name, hf_id, rope
        self.enc, self.cfg = model.encoder, model.config          # the predictor is not used
        self.dim, self.tubelet, self.patch = int(self.cfg.hidden_size), int(self.cfg.tubelet_size), int(self.cfg.patch_size)
        self.attn_modules = [m for m in self.enc.modules() if type(m).__name__ == "VJEPA2RopeAttention"]
        if len(self.attn_modules) != self.cfg.num_hidden_layers:
            raise RuntimeError(f"found {len(self.attn_modules)} RoPE attention modules, expected {self.cfg.num_hidden_layers}")
        fn = _rope_position_ids(rope)
        for m in self.attn_modules:
            m.gh = m.gw = m.grid_size
            m.get_position_ids = types.MethodType(fn, m)
        self._grid = None

    def set_grid(self, gh: int, gw: int):
        for m in self.attn_modules:
            m.gh, m.gw = gh, gw
        self._grid = (gh, gw)

    def forward_clip(self, x):
        gh, gw = x.shape[-2] // self.patch, x.shape[-1] // self.patch
        if (gh, gw) != self._grid:
            self.set_grid(gh, gw)
        h = self.enc(pixel_values_videos=x).last_hidden_state
        return h.view(x.shape[0], x.shape[1] // self.tubelet, gh * gw, self.dim)

    def forward(self, x):                          # one image as a static clip of one tubelet (diagnostic only)
        return self.forward_clip(x[:, None].repeat(1, self.tubelet, 1, 1, 1))[:, 0]

    def meta(self):
        return {**super().meta(), "source": f"HF {self.hf_id} (encoder only)", "tubelet": self.tubelet,
                "rope": self.rope, "output": "encoder last_hidden_state of every tubelet",
                "positions": "3-D RoPE ids recomputed for the token grid" + (", height / width scaled to the 16 x 16 pretraining grid"
                                                                             if self.rope == "interp" else "")}


class VideoEncoder(nn.Module):
    """A frozen encoder of eval/encoders.py (or VJEPA2Video) behind the interface of the video benchmarks, with
    ImageNet-normalized inputs (see the module docstring)."""

    def __init__(self, enc: E.FrozenEncoder):
        super().__init__()
        self.enc = enc
        self.name = enc.name
        self.patch, self.dim = int(enc.patch), int(enc.dim)
        self.tubelet = int(enc.tubelet) if isinstance(enc, VJEPA2Video) else 1
        m0 = torch.tensor(IMAGENET[0]).view(1, 3, 1, 1)
        s0 = torch.tensor(IMAGENET[1]).view(1, 3, 1, 1)
        m1 = enc.mean.detach().float().cpu().view(1, 3, 1, 1)
        s1 = enc.std.detach().float().cpu().view(1, 3, 1, 1)
        if isinstance(enc, E.RadioEncoder):
            self.input_mode = "unit_clamp"
        elif type(enc).normalize is not E.FrozenEncoder.normalize:
            self.input_mode = "unit"               # encoders that normalize [0, 1] inputs themselves (not in the video tables)
        elif np.allclose(m1.flatten().numpy(), IMAGENET[0]) and np.allclose(s1.flatten().numpy(), IMAGENET[1]):
            self.input_mode = "identity"
        else:
            self.input_mode = "affine"
        self.register_buffer("m0", m0, persistent=False)
        self.register_buffer("s0", s0, persistent=False)
        self.register_buffer("ra", s0 / s1, persistent=False)
        self.register_buffer("rb", (m0 - m1) / s1, persistent=False)

    def train(self, mode=True):                    # frozen: never leaves eval mode
        return super().train(False)

    def from_imagenet(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "identity":
            return x
        if self.input_mode == "affine":
            return x * self.ra + self.rb
        x01 = x * self.s0 + self.m0
        if self.input_mode == "unit_clamp":
            x01 = x01.clamp(0, 1)
        return self.enc.normalize(x01)

    def forward_images(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(self.from_imagenet(x))

    def forward_clip(self, x: torch.Tensor) -> torch.Tensor:
        if self.tubelet > 1:                       # video encoder: attention over the clip
            return self.enc.forward_clip(self.from_imagenet(x))
        b, t = x.shape[:2]
        tok = self.forward_images(x.flatten(0, 1))
        return tok.view(b, t, *tok.shape[1:])

    def t_out(self, t: int) -> int:
        return t // self.tubelet

    def grid(self, h: int, w: int):
        return h // self.patch, w // self.patch

    def n_params(self) -> int:
        return sum(p.numel() for p in self.enc.parameters())

    def meta(self) -> Dict:
        m = self.enc.meta()
        m.update(tubelet=self.tubelet, input_from_imagenet=self.input_mode, params_M=round(self.n_params() / 1e6, 1))
        if getattr(self.enc, "state_dict_path", None):
            m["state_dict"] = os.path.basename(self.enc.state_dict_path)
        return m


def _read_state_dict(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


def build_video_encoder(name: str, device: str = "cuda", weights: Optional[str] = None, state_dict: Optional[str] = None,
                        res: int = 224, vjepa2_rope: str = "interp") -> VideoEncoder:
    """Frozen encoder for DAVIS / SSv2 on `device`.
    weights:     ALPS safetensors for alps / alps_stage1 (default env ALPS_WEIGHTS / ALPS_STAGE1_WEIGHTS); any checkpoint
                 of the ALPS architecture (e.g. the Stage-1 ablation variants) runs as name="alps" with its own file.
    state_dict:  weights loaded (strict) into a timm encoder after it is created, e.g. SigLIP 2 / CLIP refined with
                 Stage 2 (Sec. 5.2).
    vjepa2_rope: "interp" (the paper's protocol) or "recompute" (integer RoPE positions of the frame grid)."""
    if name in VIDEO_TIMM:
        enc = TimmIdEncoder(name, VIDEO_TIMM[name], res=res).freeze()
    elif name == "vjepa2_vitl":
        enc = VJEPA2Video(name, img_size=res, rope=vjepa2_rope).freeze()
    else:
        enc = E.build(name, device="cpu", weights=weights, res=res, dynamic_img_size=True, pos_mode="interpolate")
    if state_dict:
        if not isinstance(enc, (E.TimmEncoder, TimmIdEncoder)):
            raise ValueError(f"--state_dict applies to timm encoders, not to {name}")
        enc.model.load_state_dict(_read_state_dict(state_dict), strict=True)
        enc.state_dict_path = state_dict
    return VideoEncoder(enc).to(device)
