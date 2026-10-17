# Copyright 2026 The ALPS Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Frozen-encoder image module of the LIBERO policies of the ALPS paper (Tab. 2, S8-S10).

`FrozenEncoderPool` replaces LIBERO's ResnetEncoder in `BCTransformerPolicy`: same constructor surface, same call
`encoder(x[B*T, 3, 128, 128], langs=task_emb[B*T, L]) -> [B*T, output_size]`. LIBERO instantiates it by name from the
image-encoder config (libero/configs/policy/image_encoder/frozen_encoder_pool.yaml, added by patches/libero_8f1084e3.patch,
whose libero/lifelong/models/modules/frozen_encoder_pool.py imports this module). The frozen network comes from eval/encoders.py;
everything around it is the module of the paper's runs, step by step:

  x                     float images in [0, 1] as LIBERO feeds them (checked on the first batch)
  flip_ud               torch.flip(x, dims=[-2]): LIBERO stores and renders frames upside down; the encoder sees them upright
  resize                F.interpolate(x, (224, 224), mode="bilinear", align_corners=False), no antialiasing
  normalization         the encoder's own (FrozenEncoder.normalize: ImageNet mean / std for ALPS, DINOv2, DINOv3; the timm
                        mean / std of CLIP and SigLIP 2; none for C-RADIOv3, whose input conditioner normalizes)
  frozen forward        torch.no_grad() and torch.autocast(bfloat16) on CUDA (no autocast on the CPU), fp32 weights;
                        C-RADIOv3 input clamped to [0, 1]; output = the native last-layer patch tokens of eval/encoders.py
                        (196 tokens at 224 px, DINOv2-B/14: 256), cast to float32
  attention pool        trainable, per camera: query = Linear(L -> D)(task embedding), nn.MultiheadAttention(D, 4 heads,
                        batch_first) with keys = values = the tokens (no position encoding added), then Linear(D -> 64)

Defaults and the settings of the paper's runs. The defaults are a truly frozen encoder and the released model's arithmetic;
scripts/run_paper_policies.sh passes the settings of the runs explicitly:
  * `encoder_train_mode` ("eval" by default; "follow" in the runs, except those of C-RADIOv3-B). In the runs the frozen
    network was a registered submodule of the policy, so LIBERO's `policy.train()` put it in training mode for every training
    epoch and `policy.eval()` back for the rollouts. Of the paper's encoders only C-RADIOv3-B computes differently in training
    mode: RADIO's cropped position embedding (CPE) samples a random crop of its position-embedding grid for every image and
    drops the position embedding of 10% of the images (NVlabs/RADIO `ViTPatchGenerator`). ALPS and the timm encoders (CLIP,
    SigLIP 2, DINOv2, DINOv3) have no training-mode randomness at their default settings. "follow" reproduces the runs; "eval"
    keeps the frozen network in eval mode throughout, as in the paper's C-RADIOv3-B policies (trained with this module).
  * `alps_port_numerics` (False by default; True in the runs of ALPS and ALPS (Stage 1)). Those runs used an earlier PyTorch
    port of the encoder: the released weights, but LayerNorm epsilon 1e-5 (training value 1e-6), the patch embedding inside
    the bfloat16 autocast (the release ViT keeps it in float32), the sine-cosine table computed in float64 and normalization
    constants computed as float32(255 c) / 255. True applies that arithmetic to the release encoder; the tests reproduce the
    port's tokens bit for bit on the CPU, in float32 and under bfloat16 autocast. With False the release encoder runs as in
    every other benchmark of the release: in float32 its tokens stay within a cosine of 0.99999 of the port's, under the
    policy's bfloat16 autocast they differ by about as much as bfloat16 rounding itself. No effect on the other encoders.

Behaviour of the runs that the module always keeps:
  * TF32. Importing the module enables TF32 for float32 matrix products and cuDNN convolutions in the whole process
    (`torch.backends.cuda.matmul.allow_tf32 = True`); every training and evaluation of the paper ran with it, so the float32
    policy (attention pool, transformer, heads) used TF32 matrix products on the Ampere GPUs of the runs.
  * Construction order. The frozen network is built first (its random initialization, overwritten by the loaded weights,
    consumes the generator exactly as in the runs; for ALPS the release ViT draws as many numbers as the earlier port, see
    tests/test_bridge.py), then query_proj, pool and out_proj, so that their initial weights are the same function of the
    training seed as in the runs. One frozen instance serves both cameras.

Difference to the module of the runs (it does not change the trained policy): the frozen network is held outside the module
tree. Policy checkpoints hold the 8.8M trainable parameters only (about 35 MB instead of 380 MB); `policy.to(device)` does not
move the frozen network, the first forward does; LIBERO's printed parameter and FLOP counts cover the trainable part only.
Checkpoints of the research code load as well: their frozen-network keys (`*.vit.*`) and normalization buffers (`*.im_mean`,
`*.im_std`) are dropped on loading, the buffers after a check against the encoder's normalization.
"""
import hashlib
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# Kept from the module of the runs: imported by LIBERO's bc_transformer_policy.py in every training and evaluation process.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

__all__ = ["FrozenEncoderPool", "PAPER_ENCODERS", "build_frozen_encoder", "apply_port_numerics", "clear_cache"]

# The encoders of the paper's policies: bridge name -> (paper row, what the runs called it in network_kwargs.encoder_ckpt)
PAPER_ENCODERS: Dict[str, Tuple[str, str]] = {
    "alps": ("ALPS", "torch port of the ALPS encoder (.pt)"),
    "alps_stage1": ("ALPS (Stage 1)", "torch port of the Stage-1 encoder (.pt)"),
    "clip_b16_qg": ("CLIP-B/16", "timm:vit_base_patch16_clip_quickgelu_224.openai"),
    "siglip2_b16": ("SigLIP 2-B/16 (with weights: SigLIP 2-B/16 + Stage 2)", "timm:vit_base_patch16_siglip_224.v2_webli"),
    "dinov2_b14_timm": ("DINOv2-B/14", "timm:vit_base_patch14_dinov2.lvd142m"),
    "dinov3_b16": ("DINOv3-B/16", "timm:vit_base_patch16_dinov3.lvd1689m"),
    "cradio_v3_b": ("C-RADIOv3-B", "radio:<c-radio_v3-b_half.pth.tar>"),
}
# The policies used timm's DINOv2 (eval/encoders.py "dinov2_b14_timm": vit_base_patch14_dinov2.lvd142m, position embedding
# resampled to 16 x 16 when the model is created), not the torch-hub model of "dinov2_b14".
_ALPS = ("alps", "alps_stage1")
_FROZEN_CACHE: Dict[tuple, nn.Module] = {}


def clear_cache():
    """Forget the shared frozen networks (tests)."""
    _FROZEN_CACHE.clear()


def _encoders():
    from .. import encoders                                   # eval/encoders.py of this repository
    return encoders


class _PortPatchEmbed(nn.Conv2d):
    """Patch embedding of the earlier ALPS port: under the bridge's autocast it ran in the autocast dtype (output upcast to
    float32 by the following concatenation with the float32 class token). The release ViT keeps it in float32."""
    amp_dtype: Optional[torch.dtype] = None

    def forward(self, x):
        if self.amp_dtype is None:
            return super().forward(x)
        w = self.weight.to(self.amp_dtype)
        return F.conv2d(x.to(self.amp_dtype), w, None, self.stride, self.padding, self.dilation, self.groups).float()


def _port_sincos(h: int, w: int, dim: int, temperature: float = 10000.0) -> torch.Tensor:
    """The sine-cosine table of the earlier port: numpy float64, cast to float32 (class-token row zero)."""
    omega = np.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = np.concatenate([np.sin(x), np.cos(x), np.sin(y), np.cos(y)], axis=1)
    pe = np.concatenate([np.zeros((1, dim)), pe], axis=0)
    return torch.tensor(pe[None], dtype=torch.float32)


def apply_port_numerics(enc: nn.Module) -> nn.Module:
    """Make a release ALPS encoder (eval/encoders.py ALPSEncoder) compute like the earlier port used by the paper's runs."""
    vit = enc.model
    for m in vit.modules():
        if isinstance(m, nn.LayerNorm):
            m.eps = 1e-5                                         # torch's default epsilon, which the port used
    with torch.no_grad():
        vit.pos_embed.copy_(_port_sincos(vit.grid, vit.grid, vit.dim).to(vit.pos_embed.device))
        mean = torch.tensor(np.array([0.485, 0.456, 0.406]) * 255.0, dtype=torch.float32).view(1, 3, 1, 1) / 255.0
        std = torch.tensor(np.array([0.229, 0.224, 0.225]) * 255.0, dtype=torch.float32).view(1, 3, 1, 1) / 255.0
        enc.mean.copy_(mean.to(enc.mean.device))
        enc.std.copy_(std.to(enc.std.device))                  # differs from float32(0.224) by one ulp in the green channel
    vit.patch_embed.__class__ = _PortPatchEmbed
    return enc


def _port_patch_embed(enc) -> Optional[_PortPatchEmbed]:
    pe = getattr(getattr(enc, "model", None), "patch_embed", None)
    return pe if isinstance(pe, _PortPatchEmbed) else None


def build_frozen_encoder(encoder: str, weights: str = "", alps_port_numerics: bool = False) -> nn.Module:
    """The frozen network through eval/encoders.py, on the CPU, frozen and in eval mode.

    weights: alps / alps_stage1: the safetensors file (default: env ALPS_WEIGHTS / ALPS_STAGE1_WEIGHTS); cradio_v3_b: the
    checkpoint c-radio_v3-b_half.pth.tar (default: env RADIO_CKPT; the RADIO code: env RADIO_REPO); a timm encoder: an
    optional full state dict of the timm model, loaded strictly (SigLIP 2-B/16 + Stage 2 of Sec. S3).
    alps_port_numerics: alps / alps_stage1 only (see the module docstring); ignored for the other encoders."""
    enc_mod = _encoders()
    if encoder in _ALPS:
        enc = enc_mod.build(encoder, device="cpu", weights=weights or None)
        if alps_port_numerics:
            apply_port_numerics(enc)
        return enc
    if encoder == "cradio_v3_b":
        prev = os.environ.get("RADIO_CKPT")
        if weights:
            os.environ["RADIO_CKPT"] = weights
        try:
            return enc_mod.build(encoder, device="cpu")
        finally:
            if weights:
                if prev is None:
                    os.environ.pop("RADIO_CKPT", None)
                else:
                    os.environ["RADIO_CKPT"] = prev
    enc = enc_mod.build(encoder, device="cpu")
    if weights:
        if encoder not in enc_mod.TIMM:
            raise ValueError(f"weights= for {encoder!r}: a state dict can be loaded into timm encoders only")
        enc.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return enc


def _resolved_weights(encoder: str, weights: str) -> str:
    if weights:
        return os.path.abspath(os.path.expanduser(weights))
    env = {"alps": "ALPS_WEIGHTS", "alps_stage1": "ALPS_STAGE1_WEIGHTS", "cradio_v3_b": "RADIO_CKPT"}.get(encoder)
    return os.path.abspath(os.environ[env]) if env and os.environ.get(env) else ""


_AMP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "none": None, "float32": None, "": None}
_LEGACY_KWARGS = ("encoder_ckpt", "port_encoder_path")


class FrozenEncoderPool(nn.Module):
    """Frozen encoder + language-conditioned attention pool, one instance per camera (see the module docstring).

    encoder: a name of eval/encoders.py (the paper's: PAPER_ENCODERS); weights: see build_frozen_encoder. input_shape,
    output_size, language_dim: filled in by BCTransformerPolicy. language_fusion: accepted for the config surface of LIBERO's
    encoders, unused (the pool is always language-conditioned). num_queries > 1 concatenates K pooled tokens before out_proj
    (the paper uses 1). encoder_train_mode, alps_port_numerics: see the module docstring (the paper's runs: "follow", "eval"
    for C-RADIOv3-B; True)."""

    def __init__(self, input_shape, output_size, language_dim=768, language_fusion="film", encoder="", weights="",
                 num_queries=1, pool_heads=4, flip_ud=False, resize=224, autocast_dtype="bfloat16",
                 encoder_train_mode="eval", alps_port_numerics=False, **unknown):
        super().__init__()
        bad = sorted(unknown)
        if bad:
            hint = (" (encoder_ckpt of the paper's run configs is replaced by encoder= and weights=; scripts/evaluate_policy.py converts them)"
                    if any(k in _LEGACY_KWARGS for k in bad) else "")
            raise TypeError(f"FrozenEncoderPool: unknown network_kwargs {bad}{hint}")
        if not encoder:
            raise ValueError("FrozenEncoderPool: set policy.image_encoder.network_kwargs.encoder (e.g. alps, dinov3_b16)")
        if len(input_shape) != 3 or int(input_shape[0]) != 3:
            raise ValueError(f"FrozenEncoderPool expects 3-channel per-camera input, got {list(input_shape)}")
        if encoder_train_mode not in ("follow", "eval"):
            raise ValueError(f"encoder_train_mode must be 'follow' or 'eval', got {encoder_train_mode!r}")
        if str(autocast_dtype) not in _AMP:
            raise ValueError(f"autocast_dtype must be one of {sorted(k for k in _AMP if k)}, got {autocast_dtype!r}")

        # 1. the frozen network (built once per process and encoder, shared by the cameras)
        self.encoder_name = str(encoder)
        self.weights = _resolved_weights(self.encoder_name, str(weights or ""))
        self.alps_port_numerics = bool(alps_port_numerics) and self.encoder_name in _ALPS
        key = (self.encoder_name, self.weights, self.alps_port_numerics)
        if key not in _FROZEN_CACHE:
            _FROZEN_CACHE[key] = build_frozen_encoder(self.encoder_name, self.weights, self.alps_port_numerics)
        object.__setattr__(self, "_frozen", _FROZEN_CACHE[key])      # not a submodule: outside state_dict / parameters
        dim = int(self._frozen.dim)

        self.flip_ud = bool(flip_ud)
        self.resize = int(resize)
        self.encoder_train_mode = str(encoder_train_mode)
        self._amp_dtype = _AMP[str(autocast_dtype)]
        self._autocast_on_cpu = False                               # tests only: the runs used autocast on CUDA only
        self._clamp01 = self.encoder_name == "cradio_v3_b"
        self._range_checked = False

        # 2. the trainable language-conditioned attention pool (created in the order of the runs)
        self.num_queries = int(num_queries)
        self.query_proj = nn.Linear(int(language_dim), dim * self.num_queries)
        self.pool = nn.MultiheadAttention(dim, int(pool_heads), batch_first=True)
        self.out_proj = nn.Linear(dim * self.num_queries, int(output_size))
        self.out_dim = int(output_size)
        self._register_load_state_dict_pre_hook(self._drop_research_keys)

    # -- frozen network -------------------------------------------------------------------------------------------
    @property
    def frozen(self) -> nn.Module:
        return self._frozen

    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder_train_mode == "follow":
            nn.Module.train(self._frozen, mode)                    # bypasses FrozenEncoder.train, which pins eval mode
        return self

    def state_hash(self) -> str:
        """sha256 over the frozen weights (16 hex digits), to check that they did not change during training."""
        h = hashlib.sha256()
        for k, v in sorted(self._frozen.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return h.hexdigest()[:16]

    def _drop_research_keys(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for k in [k for k in state_dict if k.startswith(prefix + "vit.")]:
            del state_dict[k]
        for name, ref in (("im_mean", self._frozen.mean), ("im_std", self._frozen.std)):
            v = state_dict.pop(prefix + name, None)
            if v is not None and not torch.allclose(v.detach().float().cpu().reshape(-1), ref.float().cpu().reshape(-1),
                                                    atol=1e-6):
                error_msgs.append(f"{prefix}{name}: checkpoint {v.reshape(-1).tolist()} != encoder "
                                  f"{ref.reshape(-1).tolist()} ({self.encoder_name}); trained with another encoder?")

    def tokens(self, x01: torch.Tensor) -> torch.Tensor:
        """Frozen tokens [B, N, D] (float32) of float images in [0, 1] as LIBERO feeds them (flip, resize, normalize, forward)."""
        if not self._range_checked:
            mx = float(x01.max())
            assert mx <= 1.001, f"[FrozenEncoderPool] expected float images in [0, 1], saw max={mx:.3f}"
            self._range_checked = True
        x = x01
        if self.flip_ud:
            x = torch.flip(x, dims=[-2])
        x = F.interpolate(x, size=(self.resize, self.resize), mode="bilinear", align_corners=False)
        enc = self._frozen
        if enc.device != x.device:
            enc.to(x.device)
        x = enc.normalize(x)
        amp = self._amp_dtype is not None and (x.is_cuda or self._autocast_on_cpu)
        pe = _port_patch_embed(enc)
        if pe is not None:
            pe.amp_dtype = self._amp_dtype if amp else None
        with torch.no_grad(), torch.autocast(device_type="cuda" if x.is_cuda else "cpu",
                                             dtype=self._amp_dtype or torch.bfloat16, enabled=amp):
            if self._clamp01:
                x = x.clamp(0, 1)
            z = enc(x)
        return z.detach().float()

    # -- policy interface -----------------------------------------------------------------------------------------
    def forward(self, x, langs=None):
        z = self.tokens(x)
        B = z.shape[0]
        if langs is None:                                        # never the case in BCTransformerPolicy
            langs = torch.zeros(B, self.query_proj.in_features, device=z.device)
        q = self.query_proj(langs).view(B, self.num_queries, -1)
        pooled, _ = self.pool(q, z, z)
        return self.out_proj(pooled.reshape(B, -1))

    def output_shape(self, input_shape, shape_meta=None):
        return (self.out_dim,)
