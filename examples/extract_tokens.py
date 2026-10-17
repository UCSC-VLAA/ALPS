# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Encode images with ALPS and save the patch tokens.

    python examples/extract_tokens.py --weights alps_vitb16.safetensors --out tokens.npy img1.jpg img2.png ...

Saves float32 [N, 196, 768] raw patch tokens (the visual state), and `--pooled` additionally saves
LayerNorm(mean(tokens)) as <out>_pooled.npy. Use `--frames` for simulator frames (whole frame resized to 224)
instead of the natural-image protocol (shorter side 256, central 224 crop).
"""
import argparse

import numpy as np
import torch
from PIL import Image

import alps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", action="store_true")
    ap.add_argument("--pooled", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    enc = alps.load_encoder(a.weights, device=a.device)
    tf = alps.resize_transform if a.frames else alps.center_crop_transform
    toks, pooled = [], []
    for i in range(0, len(a.images), a.batch):
        x = torch.cat([tf(Image.open(p).convert("RGB")) for p in a.images[i:i + a.batch]]).to(a.device)
        with torch.no_grad():
            o = enc(x)
        toks.append(o["tokens"].float().cpu().numpy())
        pooled.append(o["pooled"].float().cpu().numpy())
    np.save(a.out, np.concatenate(toks))
    if a.pooled:
        np.save(a.out.replace(".npy", "") + "_pooled.npy", np.concatenate(pooled))
    print(f"saved {sum(len(t) for t in toks)} x {toks[0].shape[1:]} tokens to {a.out}")


if __name__ == "__main__":
    main()
