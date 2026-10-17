# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Image preprocessing for ALPS.

The encoder expects RGB images scaled to [0, 1] and normalized with the ImageNet mean and standard
deviation, as [B, 3, H, W] with H and W multiples of 16 (224 x 224 during training).

Two resizing conventions are used in the evaluations:
* natural images (retrieval, zero-shot classification, reconstruction): resize the shorter side to
  256 with area averaging, then take the central 224 x 224 crop (`center_crop_transform`);
* frames of simulated environments (LIBERO, DINO-WM tasks): resize the whole frame to 224 x 224
  (`resize_transform`).
"""
from __future__ import annotations

from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "normalize", "center_crop_transform", "resize_transform",
           "to_float_tensor"]


def to_float_tensor(img) -> torch.Tensor:
    """PIL image, uint8 array [H, W, 3] / [B, H, W, 3], or tensor -> float tensor [B, 3, H, W] in [0, 1]."""
    if not torch.is_tensor(img):
        arr = np.asarray(img)
        if arr.ndim == 3:
            arr = arr[None]
        if arr.shape[-1] != 3:
            raise ValueError(f"expected channels-last RGB, got shape {arr.shape}")
        t = torch.from_numpy(np.array(arr, order="C", copy=True)).permute(0, 3, 1, 2)   # PIL arrays are read-only
    else:
        t = img if img.ndim == 4 else img[None]
    if t.dtype == torch.uint8:
        return t.float() / 255.0
    return t.float()


def normalize(x: torch.Tensor) -> torch.Tensor:
    """[B, 3, H, W] in [0, 1] -> ImageNet-normalized."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def center_crop_transform(img, resize: int = 256, crop: int = 224) -> torch.Tensor:
    """Shorter side to `resize` (area averaging for downsampling, bilinear otherwise), central crop."""
    if not torch.is_tensor(img) and hasattr(img, "resize"):        # PIL: BOX = area averaging
        from PIL import Image
        w, h = img.size
        r = resize / min(w, h)
        nw, nh = int(round(w * r)), int(round(h * r))
        img = img.convert("RGB").resize((nw, nh), Image.BOX if r < 1 else Image.BILINEAR)
        x = to_float_tensor(img)
    else:
        x = to_float_tensor(img)
        h, w = x.shape[-2:]
        r = resize / min(h, w)
        nh, nw = int(round(h * r)), int(round(w * r))
        x = F.interpolate(x, size=(nh, nw), mode="area" if r < 1 else "bilinear",
                          **({} if r < 1 else {"align_corners": False}))
    h, w = x.shape[-2:]
    top, left = (h - crop) // 2, (w - crop) // 2
    return normalize(x[..., top:top + crop, left:left + crop])


def resize_transform(img, size: Union[int, tuple] = 224) -> torch.Tensor:
    """Resize the whole frame to `size` (bilinear, antialiased) and normalize."""
    x = to_float_tensor(img)
    size = (size, size) if isinstance(size, int) else tuple(size)
    if tuple(x.shape[-2:]) != size:
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)
    return normalize(x)
