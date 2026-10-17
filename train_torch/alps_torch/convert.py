# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Parameter mapping between the Flax trees of the JAX training code and the PyTorch modules, and export to `alps`.

The mapping is a table of rules (Flax key(s) -> torch key, kind); it is linear and invertible, so the same table
converts parameters, gradients and optimizer moments in both directions:

    dense   Flax Dense kernel [in, out]          -> torch Linear weight [out, in] (transpose)
    conv    Flax Conv kernel [kh, kw, in, out]   -> torch Conv2d weight [out, in, kh, kw]
    qkv_w   Flax query/key/value kernels         -> fused qkv weight [3 D, D] (rows: q, k, v; alps.vit layout)
    qkv_b   Flax query/key/value biases          -> fused qkv bias [3 D]
    scalar  Flax [1]                             -> torch scalar ()
    copy    same shape

Flax keys are '/'-joined paths of the parameter tree (see `flatten`). Covered trees: the Stage-1 pretrainer
(`stage1_rules`), the ViT alone (`vit_rules`), the Stage-2 train state {encoder, transition, feat_head}
(`stage2_rules`).
"""
from __future__ import annotations

import re
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from alps.convert import flatten  # noqa: F401  (re-exported)

Rule = Tuple[Union[str, Tuple[str, ...]], str, str]


# --------------------------------------------------------------------------------------------------- rule tables
def _block_rules(fp: str, tp: str, depth: int) -> List[Rule]:
    rules: List[Rule] = []
    for i in range(depth):
        b, o = f"{fp}encoderblock_{i}/", f"{tp}blocks.{i}."
        a = b + "MultiHeadDotProductAttention_0/"
        rules += [
            (b + "LayerNorm_0/scale", o + "norm1.weight", "copy"), (b + "LayerNorm_0/bias", o + "norm1.bias", "copy"),
            ((a + "query/kernel", a + "key/kernel", a + "value/kernel"), o + "attn.qkv.weight", "qkv_w"),
            ((a + "query/bias", a + "key/bias", a + "value/bias"), o + "attn.qkv.bias", "qkv_b"),
            (a + "out/kernel", o + "attn.proj.weight", "dense"), (a + "out/bias", o + "attn.proj.bias", "copy"),
            (b + "LayerNorm_1/scale", o + "norm2.weight", "copy"), (b + "LayerNorm_1/bias", o + "norm2.bias", "copy"),
            (b + "MlpBlock_0/Dense_0/kernel", o + "mlp.fc1.weight", "dense"),
            (b + "MlpBlock_0/Dense_0/bias", o + "mlp.fc1.bias", "copy"),
            (b + "MlpBlock_0/Dense_1/kernel", o + "mlp.fc2.weight", "dense"),
            (b + "MlpBlock_0/Dense_1/bias", o + "mlp.fc2.bias", "copy"),
        ]
    return rules


def _ln(fp: str, tp: str) -> List[Rule]:
    return [(fp + "/scale", tp + ".weight", "copy"), (fp + "/bias", tp + ".bias", "copy")]


def _lin(fp: str, tp: str, bias: bool = True, kind: str = "dense") -> List[Rule]:
    r: List[Rule] = [(fp + "/kernel", tp + ".weight", kind)]
    return r + ([(fp + "/bias", tp + ".bias", "copy")] if bias else [])


def vit_rules(fp: str = "", tp: str = "", depth: int = 12) -> List[Rule]:
    """Flax ViT (alps_jax/models/vit.py) -> alps_torch.models.ViT / alps.vit.VisionTransformer."""
    return ([(fp + "embedding/kernel", tp + "patch_embed.weight", "conv"), (fp + "cls", tp + "cls_token", "copy")]
            + _ln(fp + "encoder_norm", tp + "norm") + _block_rules(fp + "Transformer/", tp, depth))


def text_rules(fp: str = "", tp: str = "", depth: int = 12) -> List[Rule]:
    return ([(fp + "token_embedding/embedding", tp + "token_embedding.weight", "copy"),
             (fp + "pos_embedding", tp + "pos_embed", "copy")]
            + _ln(fp + "encoder_norm", tp + "norm") + _block_rules(fp + "Transformer/", tp, depth))


def caption_decoder_rules(fp: str = "", tp: str = "", depth: int = 12) -> List[Rule]:
    return ([(fp + "token_embedding/embedding", tp + "token_embedding.weight", "copy"),
             (fp + "pos_embedding", tp + "pos_embed", "copy")]
            + _ln(fp + "decoder_norm", tp + "norm") + _lin(fp + "head", tp + "head")
            + _block_rules(fp + "decoder/", tp, depth))


def rgb_decoder_rules(fp: str = "", tp: str = "", n_blocks: int = 4) -> List[Rule]:
    r = _lin(fp + "Conv_0", tp + "stem", kind="conv") + _lin(fp + "Conv_1", tp + "out", kind="conv")
    for i in range(n_blocks):
        u, o = f"{fp}_UpBlock_{i}/", f"{tp}blocks.{i}."
        r += (_lin(u + "Conv_0", o + "conv1", kind="conv") + _ln(u + "GroupNorm_0", o + "norm")
              + _lin(u + "Conv_1", o + "conv2", kind="conv"))
    return r


def stage1_rules(depth: int = 12, text_depth: int = 12, cap_depth: int = 12, caption: bool = True) -> List[Rule]:
    """Stage-1 pretrainer tree (alps_jax/models/stage1.py) -> alps_torch.models.Stage1Model."""
    r = (vit_rules("visual_encoder/", "visual.", depth)
         + _lin("image_projection_head/proj", "image_projection", bias=False)
         + [("logit_scale", "logit_scale", "scalar")]
         + text_rules("text_encoder/", "text.", text_depth)
         + _lin("text_projection_head/proj", "text_projection", bias=False)
         + rgb_decoder_rules("RGBDecoder_0/", "pixel_decoder.")
         + _lin("DenseProjector_0/Dense_0", "dense_projector.proj"))
    if caption:
        r += caption_decoder_rules("caption_decoder/", "caption_decoder.", cap_depth)
    return r


def predictor_rules(fp: str = "", tp: str = "", depth: int = 4) -> List[Rule]:
    """CrossViewFusion (Flax subtree `transition`) -> alps_torch.models.CrossViewFusion."""
    r: List[Rule] = [(fp + "mask_token", tp + "mask_token", "copy")]
    r += _lin(fp + "scale_proj", tp + "scale_proj") + _lin(fp + "src_proj", tp + "src_proj")
    r += _lin(fp + "txt_proj", tp + "txt_proj") + _ln(fp + "ln_out", tp + "ln_out") + _lin(fp + "out_proj", tp + "out_proj")
    for i in range(depth):
        o = f"{tp}layers.{i}."
        for n in ("sa", "ca", "mlp"):
            r += _ln(f"{fp}ln_{n}_{i}", o + f"ln_{n}")
        for n in ("sa", "ca"):
            for p in "qkvo":
                r += _lin(f"{fp}{n}_{i}_{p}", o + f"{n}.{p}")
        r += _lin(f"{fp}mlp_{i}/fc1", o + "mlp.fc1") + _lin(f"{fp}mlp_{i}/fc2", o + "mlp.fc2")
    return r


def feat_head_rules(fp: str = "", tp: str = "") -> List[Rule]:
    return _ln(fp + "ln", tp + "ln") + _lin(fp + "proj", tp + "proj")


def stage2_rules(depth: int = 12, pred_depth: int = 4) -> List[Rule]:
    """Stage-2 train state {encoder, transition, feat_head} -> alps_torch Stage2Student {encoder, predictor, head}."""
    return (vit_rules("encoder/", "encoder.", depth) + predictor_rules("transition/", "predictor.", pred_depth)
            + feat_head_rules("feat_head/", "head."))


def count_blocks(keys, prefix: str, pattern: str = r"encoderblock_(\d+)") -> int:
    idx = {int(m.group(1)) for k in keys if k.startswith(prefix) for m in [re.search(pattern, k[len(prefix):])] if m}
    return 1 + max(idx) if idx else 0


# ----------------------------------------------------------------------------------------------------- conversion
def _np(a) -> np.ndarray:
    return np.asarray(a.detach().cpu().float().numpy() if torch.is_tensor(a) else a, dtype=np.float32)


def flax_to_torch(flat: Mapping[str, np.ndarray], rules: Sequence[Rule], strict: bool = True,
                  dtype=torch.float32) -> Dict[str, torch.Tensor]:
    """Flat Flax dict {"a/b/c": array} -> torch state dict, by `rules`."""
    sd: Dict[str, torch.Tensor] = {}
    used = set()
    for src, dst, kind in rules:
        srcs = (src,) if isinstance(src, str) else tuple(src)
        if any(s not in flat for s in srcs):
            if strict:
                raise KeyError(f"missing Flax keys for {dst}: {[s for s in srcs if s not in flat]}")
            continue
        a = [_np(flat[s]) for s in srcs]
        used.update(srcs)
        if kind == "copy":
            t = a[0]
        elif kind == "scalar":
            t = a[0].reshape(())
        elif kind == "dense":
            t = a[0].T
        elif kind == "conv":
            t = np.transpose(a[0], (3, 2, 0, 1))
        elif kind == "qkv_w":
            t = np.concatenate([x.T for x in a], 0)
        elif kind == "qkv_b":
            t = np.concatenate([x.reshape(-1) for x in a], 0)
        else:
            raise ValueError(kind)
        sd[dst] = torch.from_numpy(np.array(t, dtype=np.float32, copy=True)).to(dtype)
    if strict:
        extra = sorted(set(flat) - used)
        if extra:
            raise KeyError(f"Flax keys without a rule: {extra[:8]}")
    return sd


def torch_to_flax(sd: Mapping[str, torch.Tensor], rules: Sequence[Rule], strict: bool = True,
                  flax_shapes: Optional[Mapping[str, tuple]] = None) -> Dict[str, np.ndarray]:
    """torch state dict (or any dict of tensors with the same keys: gradients, moments) -> flat Flax dict."""
    out: Dict[str, np.ndarray] = {}
    for src, dst, kind in rules:
        if dst not in sd:
            if strict:
                raise KeyError(f"missing torch key {dst}")
            continue
        t = _np(sd[dst])
        if kind == "copy":
            vals = [t]
        elif kind == "scalar":
            vals = [t.reshape(1)]
        elif kind == "dense":
            vals = [t.T]
        elif kind == "conv":
            vals = [np.transpose(t, (2, 3, 1, 0))]
        elif kind == "qkv_w":
            vals = [x.T for x in np.split(t, 3, 0)]
        elif kind == "qkv_b":
            vals = list(np.split(t, 3, 0))
        else:
            raise ValueError(kind)
        srcs = (src,) if isinstance(src, str) else tuple(src)
        for s, v in zip(srcs, vals):
            if flax_shapes is not None and s in flax_shapes:
                v = v.reshape(flax_shapes[s])
            out[s] = np.ascontiguousarray(v)
    return out


def unflatten(flat: Mapping[str, np.ndarray]) -> dict:
    tree: dict = {}
    for k, v in flat.items():
        node = tree
        parts = k.split("/")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    return tree


def stage1_from_flax(flat: Mapping[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """Stage-1 pretrainer params (flat Flax keys, e.g. from tools/export_flax_params.py) -> Stage1Model state dict."""
    flat = {k[len("params/"):] if k.startswith("params/") else k: v for k, v in flat.items() if k != "step"}
    keys = list(flat)
    rules = stage1_rules(count_blocks(keys, "visual_encoder/"), count_blocks(keys, "text_encoder/"),
                         count_blocks(keys, "caption_decoder/"), caption=any(k.startswith("caption_decoder/") for k in keys))
    return flax_to_torch(flat, rules)


# ---------------------------------------------------------------------------------------------------------- export
ALPS_PREFIXES = ("visual.", "text.", "image_projection.", "text_projection.", "logit_scale")


def alps_state_dict(stage1_sd: Mapping[str, torch.Tensor],
                    encoder_sd: Optional[Mapping[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
    """The `alps.ALPS` state dict of a Stage-1 model, optionally with its encoder replaced by `encoder_sd` (the keys
    of a ViT / alps.vit.VisionTransformer, without prefix): the released ALPS model is the Stage-2 encoder with the
    unchanged Stage-1 language interface (text encoder, both projections, logit scale)."""
    sd = {k: v.detach().float().cpu().clone() for k, v in stage1_sd.items()
          if k.startswith(ALPS_PREFIXES) and not k.startswith("visual.pos_embed")}
    if "logit_scale" in sd:
        sd["logit_scale"] = sd["logit_scale"].reshape(())
    if encoder_sd is not None:
        sd = {k: v for k, v in sd.items() if not k.startswith("visual.")}
        sd.update({"visual." + k: v.detach().float().cpu().clone() for k, v in encoder_sd.items()
                   if not k.startswith("pos_embed")})
    return sd


def export_alps(path: str, stage1_sd: Optional[Mapping[str, torch.Tensor]] = None,
                encoder_sd: Optional[Mapping[str, torch.Tensor]] = None, metadata: Optional[dict] = None,
                encoder_only: bool = False) -> Dict[str, torch.Tensor]:
    """Write an `alps` checkpoint (safetensors or .pt) that `alps.load` / `alps.load_encoder` read."""
    from alps.hub import save
    if encoder_only or stage1_sd is None:
        if encoder_sd is None:
            encoder_sd = {k[len("visual."):]: v for k, v in stage1_sd.items() if k.startswith("visual.")}
        sd = {"visual." + k: v.detach().float().cpu().clone() for k, v in encoder_sd.items()
              if not k.startswith("pos_embed")}
    else:
        sd = alps_state_dict(stage1_sd, encoder_sd)
    meta = {"format": "alps-torch-v1", **(metadata or {})}
    save(sd, path, meta)
    return sd
