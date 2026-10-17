# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Loading and saving ALPS weights.

    model = alps.load("alps_vitb16.safetensors")              # encoder + language interface
    enc = alps.load_encoder("alps_vitb16.safetensors")        # visual encoder only
    model = alps.load("hf-hub:<org>/<repo>/<file>")          # a file of a Hugging Face model repository
    model = alps.load("alps-vitb16")                          # a released model by name (see PRETRAINED)

Files are safetensors (preferred) or torch state dicts with the key layout of `alps.ALPS`
(visual.*, image_projection.weight, logit_scale, text.*, text_projection.weight). A file may hold
only the visual.* keys; it then loads with `with_text=False`.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import torch

from .model import ALPS
from .vit import VisionTransformer, vit_b16

__all__ = ["load", "load_encoder", "save", "read_state_dict", "PRETRAINED"]

# The released weights, in the Hugging Face repository UCSC-VLAA/ALPS (published with the paper).
PRETRAINED: Dict[str, Optional[str]] = {
    "alps-vitb16": "hf-hub:UCSC-VLAA/ALPS/alps_vitb16.safetensors",                 # ALPS (Stage 1 + Stage 2), the model of the paper
    "alps-stage1-vitb16": "hf-hub:UCSC-VLAA/ALPS/alps_stage1_vitb16.safetensors",   # ALPS (Stage 1), before cross-view refinement
}
_MISSING = ("RepositoryNotFoundError", "EntryNotFoundError", "RevisionNotFoundError", "GatedRepoError")


def _cache_dir() -> str:
    return os.environ.get("ALPS_HOME", os.path.expanduser("~/.cache/alps"))


def _from_hub(spec: str) -> str:
    """"<org>/<repo>/<file>" (default file: model.safetensors) -> local path, through huggingface_hub when installed
    (its cache and authentication), else a plain download into ALPS_HOME."""
    parts = spec.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"expected hf-hub:<org>/<repo>[/<file>], got hf-hub:{spec}")
    repo, filename = "/".join(parts[:2]), "/".join(parts[2:]) or "model.safetensors"
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        cache = os.path.join(_cache_dir(), repo.replace("/", "--"), filename)
        if not os.path.exists(cache):
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            try:
                torch.hub.download_url_to_file(f"https://huggingface.co/{repo}/resolve/main/{filename}", cache)
            except Exception as e:                              # HTTP 401 / 404: no such (public) file
                raise FileNotFoundError(f"{repo}/{filename} is not available from Hugging Face ({e}); pass a local file") from e
        return cache
    try:
        return hf_hub_download(repo, filename)
    except Exception as e:
        if type(e).__name__ in _MISSING:
            raise FileNotFoundError(f"{repo}/{filename} is not available on Hugging Face ({type(e).__name__}); "
                                    "pass a local file") from e
        raise


def _resolve(name_or_path: str) -> str:
    if os.path.exists(name_or_path):
        return name_or_path
    if name_or_path.startswith("hf-hub:"):
        return _from_hub(name_or_path[len("hf-hub:"):])
    if name_or_path in PRETRAINED:
        loc = PRETRAINED[name_or_path]
        if loc is None:
            raise FileNotFoundError(f"weights for {name_or_path!r} are not published yet; pass a local file")
        if loc.startswith("hf-hub:"):
            return _from_hub(loc[len("hf-hub:"):])
        cache = os.path.join(_cache_dir(), os.path.basename(loc))
        if not os.path.exists(cache):
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            torch.hub.download_url_to_file(loc, cache)
        return cache
    raise FileNotFoundError(name_or_path)


def read_state_dict(path: str) -> Dict[str, torch.Tensor]:
    path = _resolve(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    return sd.get("state_dict", sd)


def _arch(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, int]:
    """Width / depth / MLP width / heads (head width 64) of a ViT or text tower, read from the tensor shapes."""
    blocks = {int(k[len(prefix):].split(".")[1]) for k in sd if k.startswith(prefix + "blocks.")}
    width = sd[prefix + "blocks.0.norm1.weight"].shape[0]
    return {"dim": width, "depth": 1 + max(blocks), "mlp_dim": sd[prefix + "blocks.0.mlp.fc1.weight"].shape[0],
            "heads": width // 64}


def _vit_from(sd: Dict[str, torch.Tensor], prefix: str, pos_mode: str) -> VisionTransformer:
    a = _arch(sd, prefix)
    patch = sd[prefix + "patch_embed.weight"].shape[-1]
    return VisionTransformer(patch_size=patch, pos_mode=pos_mode, **a)


def load(name_or_path: str, device: str = "cpu", with_text: Optional[bool] = None,
         pos_mode: str = "interpolate") -> ALPS:
    """The model of a weights file; the architecture is read from the tensor shapes (ViT-B/16 for the released files)."""
    sd = read_state_dict(name_or_path)
    has_text = any(k.startswith("text.") for k in sd)
    if with_text is None:
        with_text = has_text
    if with_text and not has_text:
        raise ValueError(f"{name_or_path} holds no text encoder; load it with with_text=False")
    model = ALPS(with_text=with_text, pos_mode=pos_mode)
    if _arch(sd, "visual.") != _arch(model.state_dict(), "visual.") or "visual.patch_embed.weight" not in sd or \
            sd["visual.patch_embed.weight"].shape != model.visual.patch_embed.weight.shape:
        model.visual = _vit_from(sd, "visual.", pos_mode)
        model.image_projection = torch.nn.Linear(model.visual.dim, sd["image_projection.weight"].shape[0], bias=False) \
            if "image_projection.weight" in sd else model.image_projection
    if with_text and _arch(sd, "text.") != _arch(model.state_dict(), "text."):
        from .text import TextEncoder
        a = _arch(sd, "text.")
        model.text = TextEncoder(vocab_size=sd["text.token_embedding.weight"].shape[0],
                                 context_length=sd["text.pos_embed"].shape[1], **a)
        model.text_projection = torch.nn.Linear(a["dim"], sd["text_projection.weight"].shape[0], bias=False)
    if not with_text:
        sd = {k: v for k, v in sd.items() if not k.startswith(("text.", "text_projection."))}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    allowed_missing = set() if any(k.startswith("image_projection.") for k in sd) else {"image_projection.weight", "logit_scale"}
    bad = [k for k in missing if k not in allowed_missing]
    if bad or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={bad[:8]} unexpected={unexpected[:8]}")
    return model.to(device).eval()


def load_encoder(name_or_path: str, device: str = "cpu", pos_mode: str = "interpolate") -> VisionTransformer:
    sd = read_state_dict(name_or_path)
    vis = {k[len("visual."):]: v for k, v in sd.items() if k.startswith("visual.")}
    enc = _vit_from(sd, "visual.", pos_mode)
    enc.load_state_dict(vis, strict=True)
    return enc.to(device).eval()


def save(state_dict: Dict[str, torch.Tensor], path: str, metadata: Optional[Dict[str, str]] = None) -> None:
    sd = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    if path.endswith(".safetensors"):
        from safetensors.torch import save_file
        save_file(sd, path, metadata={k: str(v) for k, v in (metadata or {}).items()})
    else:
        torch.save({"state_dict": sd, "metadata": json.dumps(metadata or {})}, path)
