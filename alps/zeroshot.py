# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Zero-shot ImageNet classification with the ALPS language interface.

protocol="paper" (default; Table 1): the class names and the 80 prompt templates of OpenAI CLIP as distributed with
CLIP_benchmark; each class embedding is the mean of the L2-normalized embeddings of its 80 prompts, normalized again.
protocol="big_vision": the evaluator of the training code (81 templates, canonicalized lower-case text without
punctuation), which the training logs report.
"""
from __future__ import annotations

import json
import os
import re
import string
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

__all__ = ["canonicalize", "imagenet_prompts", "build_classifier", "imagenet_classifier"]

_ASSET = os.path.join(os.path.dirname(__file__), "assets", "imagenet_zeroshot.json")


def canonicalize(text: str, keep: Optional[str] = None) -> str:
    text = text.replace("_", " ")
    strip = str.maketrans("", "", string.punctuation)
    text = keep.join(p.translate(strip) for p in text.split(keep)) if keep else text.translate(strip)
    return re.sub(r"\s+", " ", text.lower()).strip()


def imagenet_prompts(protocol: str = "paper"):
    d = json.load(open(_ASSET))
    if protocol == "paper":
        return d["clip_benchmark"]["class_names"], [t.replace("{c}", "{}") for t in d["clip_benchmark"]["templates"]]
    if protocol == "big_vision":
        names = [canonicalize(n, keep=",") for n in d["big_vision"]["class_names"]]
        return names, [canonicalize(t, keep="{}") for t in d["big_vision"]["templates"]]
    raise ValueError(protocol)


@torch.no_grad()
def build_classifier(model, class_names: Sequence[str], templates: Sequence[str], batch_size: int = 1000,
                     split_aliases: bool = False) -> torch.Tensor:
    """[num_classes, D]: per class, the normalized mean of the normalized prompt embeddings. `split_aliases`: a class
    name with commas is several aliases (big_vision protocol)."""
    out: List[torch.Tensor] = []
    for name in class_names:
        aliases = name.split(",") if split_aliases else [name]
        prompts = [t.replace("{}", alias) for alias in aliases for t in templates]
        embs = torch.cat([model.text_embedding(prompts[i:i + batch_size]) for i in range(0, len(prompts), batch_size)])
        out.append(F.normalize(embs.mean(dim=0), dim=-1))
    return torch.stack(out)


def imagenet_classifier(model, protocol: str = "paper", batch_size: int = 1000) -> torch.Tensor:
    names, templates = imagenet_prompts(protocol)
    return build_classifier(model, names, templates, batch_size, split_aliases=(protocol == "big_vision"))
