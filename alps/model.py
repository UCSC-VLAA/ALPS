# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""ALPS: the visual state encoder together with its native language interface.

    model = alps.load("alps_vitb16.safetensors")
    out = model.encode_image(images)           # images: [B, 3, 224, 224], normalized
    z = out["tokens"]                          # [B, 196, 768] raw patch tokens (the state)
    img = model.image_embedding(tokens=z)      # [B, 768], L2-normalized
    txt = model.text_embedding(["a robot arm opens the drawer"])
    logits = model.logit_scale.exp().clamp(1, 100) * img @ txt.T

Language interface (Stage 1, unchanged by Stage 2): the image embedding is a linear projection
(no bias) of the MEAN of the raw patch tokens, the text embedding a linear projection (no bias)
of the last slot of the text encoder; both are L2-normalized and compared by a dot product scaled
by exp(logit_scale), which the training code bounded to [1, 100].
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .text import TextEncoder
from .tokenizer import ALPSTokenizer
from .vit import VisionTransformer, vit_b16

__all__ = ["ALPS"]


class ALPS(nn.Module):
    def __init__(self, with_text: bool = True, embed_dim: int = 768, pos_mode: str = "interpolate"):
        super().__init__()
        self.visual: VisionTransformer = vit_b16(pos_mode=pos_mode)
        self.image_projection = nn.Linear(self.visual.dim, embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.zeros(()))      # log of the contrastive scale
        self.text: Optional[TextEncoder] = TextEncoder() if with_text else None
        self.text_projection = nn.Linear(768, embed_dim, bias=False) if with_text else None
        self._tokenizer: Optional[ALPSTokenizer] = None

    @property
    def tokenizer(self) -> ALPSTokenizer:
        if self._tokenizer is None:
            self._tokenizer = ALPSTokenizer(context_length=self.text.context_length if self.text else 32)
        return self._tokenizer

    # -- vision ---------------------------------------------------------------------------------
    def encode_image(self, images: torch.Tensor, **kw) -> Dict[str, torch.Tensor]:
        """Raw patch tokens ("tokens"), class-token output ("cls") and LayerNorm(mean(tokens)) ("pooled")."""
        return self.visual(images, **kw)

    def image_embedding(self, images: Optional[torch.Tensor] = None, tokens: Optional[torch.Tensor] = None,
                        normalize: bool = True) -> torch.Tensor:
        if tokens is None:
            if images is None:
                raise ValueError("pass images or tokens")
            tokens = self.visual(images)["tokens"]
        e = self.image_projection(tokens.mean(dim=1))
        return F.normalize(e, dim=-1) if normalize else e

    # -- language -------------------------------------------------------------------------------
    def tokenize(self, texts: Union[str, Sequence[str]]) -> torch.Tensor:
        return self.tokenizer(texts)

    def text_embedding(self, texts: Union[str, Sequence[str], torch.Tensor], normalize: bool = True) -> torch.Tensor:
        if self.text is None:
            raise RuntimeError("this model was loaded without the text encoder")
        ids = texts if torch.is_tensor(texts) else self.tokenize(texts)
        ids = ids.to(self.text_projection.weight.device)
        e = self.text_projection(self.text(ids))
        return F.normalize(e, dim=-1) if normalize else e

    def contrastive_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(1.0, 100.0)

    def forward(self, images: torch.Tensor, texts) -> torch.Tensor:
        """Image-to-text logits [B_img, B_txt]."""
        return self.contrastive_scale() * self.image_embedding(images) @ self.text_embedding(texts).T
