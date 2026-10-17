# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Convert JAX/Flax parameters of the training code into the state dict of `alps.ALPS`.

Inputs are FLAT dicts {"a/b/c": array} of the Flax parameter tree (a nested dict can be flattened
with `flatten`). Three sources are supported:

* the Stage-1 pretrainer tree (keys under visual_encoder/, text_encoder/, image_projection_head/,
  text_projection_head/, logit_scale), from which `pretrainer_to_torch` keeps the encoder and the
  language interface and drops the training-only heads (caption decoder, pixel decoder, dense
  projector);
* a visual-encoder tree alone (keys embedding/kernel, cls, Transformer/encoderblock_*/...,
  encoder_norm/...), e.g. the encoder dumps of Stage 1 and Stage 2 (`param/` prefix stripped);
* the state dict of the earlier PyTorch port of the encoder (keys conv.weight, cls,
  blocks.{i}.{ln1,q,k,v,proj,ln2,fc1,fc2}.*, enc_norm.*), `legacy_port_to_torch`.
"""
from __future__ import annotations

import re
from typing import Dict, Mapping

import numpy as np
import torch

__all__ = ["flatten", "vit_to_torch", "text_to_torch", "pretrainer_to_torch", "legacy_port_to_torch"]


def flatten(tree: Mapping, prefix: str = "") -> Dict[str, np.ndarray]:
    out = {}
    for k, v in tree.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(flatten(v, key))
        else:
            out[key] = np.asarray(v)
    return out


def _t(a) -> torch.Tensor:
    return torch.from_numpy(np.array(a, dtype=np.float32, copy=True))


def _dense(P, name):                    # Flax Dense kernel [in, out] -> torch Linear weight [out, in]
    return _t(P[f"{name}/kernel"]).t().contiguous()


def _blocks(P: Mapping, prefix: str, dst: str, sd: Dict[str, torch.Tensor]) -> int:
    idx = sorted({int(m.group(1)) for k in P for m in [re.match(rf"{re.escape(prefix)}encoderblock_(\d+)/", k)] if m})
    if not idx or idx != list(range(len(idx))):
        raise KeyError(f"no contiguous encoderblock_* under {prefix!r}")
    for i in idx:
        b = f"{prefix}encoderblock_{i}"
        a = f"{b}/MultiHeadDotProductAttention_0"
        o = f"{dst}blocks.{i}"
        if P[f"{a}/query/kernel"].ndim != 2:
            raise ValueError("expected fused 2-D q/k/v kernels (use_dense_general=False)")
        sd[f"{o}.norm1.weight"], sd[f"{o}.norm1.bias"] = _t(P[f"{b}/LayerNorm_0/scale"]), _t(P[f"{b}/LayerNorm_0/bias"])
        sd[f"{o}.norm2.weight"], sd[f"{o}.norm2.bias"] = _t(P[f"{b}/LayerNorm_1/scale"]), _t(P[f"{b}/LayerNorm_1/bias"])
        sd[f"{o}.attn.qkv.weight"] = torch.cat([_dense(P, f"{a}/{n}") for n in ("query", "key", "value")], 0)
        sd[f"{o}.attn.qkv.bias"] = torch.cat([_t(P[f"{a}/{n}/bias"]).reshape(-1) for n in ("query", "key", "value")])
        sd[f"{o}.attn.proj.weight"], sd[f"{o}.attn.proj.bias"] = _dense(P, f"{a}/out"), _t(P[f"{a}/out/bias"])
        sd[f"{o}.mlp.fc1.weight"], sd[f"{o}.mlp.fc1.bias"] = _dense(P, f"{b}/MlpBlock_0/Dense_0"), _t(P[f"{b}/MlpBlock_0/Dense_0/bias"])
        sd[f"{o}.mlp.fc2.weight"], sd[f"{o}.mlp.fc2.bias"] = _dense(P, f"{b}/MlpBlock_0/Dense_1"), _t(P[f"{b}/MlpBlock_0/Dense_1/bias"])
    return len(idx)


def vit_to_torch(P: Mapping, dst: str = "visual.") -> Dict[str, torch.Tensor]:
    """Flax `models/vit.py` encoder (flat keys relative to the encoder) -> `alps.vit.VisionTransformer`."""
    sd = {f"{dst}patch_embed.weight": _t(np.transpose(P["embedding/kernel"], (3, 2, 0, 1))).contiguous(),
          f"{dst}cls_token": _t(P["cls"]).reshape(1, 1, -1),
          f"{dst}norm.weight": _t(P["encoder_norm/scale"]), f"{dst}norm.bias": _t(P["encoder_norm/bias"])}
    if "embedding/bias" in P:
        raise ValueError("this encoder has a patch-embedding bias; ALPS has none")
    _blocks(P, "Transformer/", dst, sd)
    return sd


def text_to_torch(P: Mapping, dst: str = "text.") -> Dict[str, torch.Tensor]:
    """Flax Stage-1 `TextEncoder` (flat keys relative to the text encoder) -> `alps.text.TextEncoder`."""
    sd = {f"{dst}token_embedding.weight": _t(P["token_embedding/embedding"]),
          f"{dst}pos_embed": _t(P["pos_embedding"]),
          f"{dst}norm.weight": _t(P["encoder_norm/scale"]), f"{dst}norm.bias": _t(P["encoder_norm/bias"])}
    _blocks(P, "Transformer/", dst, sd)
    return sd


def _sub(P: Mapping, prefix: str) -> Dict[str, np.ndarray]:
    n = len(prefix)
    return {k[n:]: v for k, v in P.items() if k.startswith(prefix)}


def pretrainer_to_torch(P: Mapping, visual: Mapping = None) -> Dict[str, torch.Tensor]:
    """Stage-1 pretrainer tree -> full `alps.ALPS` state dict.

    `visual` optionally replaces the visual encoder (e.g. the Stage-2 encoder): the final ALPS model
    is the Stage-2 encoder with the unchanged Stage-1 language interface.
    """
    enc = visual if visual is not None else _sub(P, "visual_encoder/")
    sd = vit_to_torch(enc)
    sd.update(text_to_torch(_sub(P, "text_encoder/")))
    sd["image_projection.weight"] = _dense(P, "image_projection_head/proj")
    sd["text_projection.weight"] = _dense(P, "text_projection_head/proj")
    sd["logit_scale"] = _t(P["logit_scale"]).reshape(())
    return sd


def legacy_port_to_torch(old: Mapping[str, torch.Tensor], dst: str = "visual.") -> Dict[str, torch.Tensor]:
    """State dict of the earlier torch port (separate q/k/v Linear layers) -> `alps.vit.VisionTransformer`."""
    sd = {f"{dst}patch_embed.weight": old["conv.weight"], f"{dst}cls_token": old["cls"].reshape(1, 1, -1),
          f"{dst}norm.weight": old["enc_norm.weight"], f"{dst}norm.bias": old["enc_norm.bias"]}
    depth = 1 + max(int(k.split(".")[1]) for k in old if k.startswith("blocks."))
    for i in range(depth):
        b, o = f"blocks.{i}", f"{dst}blocks.{i}"
        for src, tgt in (("ln1", "norm1"), ("ln2", "norm2")):
            sd[f"{o}.{tgt}.weight"], sd[f"{o}.{tgt}.bias"] = old[f"{b}.{src}.weight"], old[f"{b}.{src}.bias"]
        sd[f"{o}.attn.qkv.weight"] = torch.cat([old[f"{b}.{n}.weight"] for n in "qkv"], 0)
        sd[f"{o}.attn.qkv.bias"] = torch.cat([old[f"{b}.{n}.bias"] for n in "qkv"], 0)
        sd[f"{o}.attn.proj.weight"], sd[f"{o}.attn.proj.bias"] = old[f"{b}.proj.weight"], old[f"{b}.proj.bias"]
        for n in ("fc1", "fc2"):
            sd[f"{o}.mlp.{n}.weight"], sd[f"{o}.mlp.{n}.bias"] = old[f"{b}.{n}.weight"], old[f"{b}.{n}.bias"]
    return sd
