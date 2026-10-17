# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Score a trained evaluation decoder on the held-out images of a reconstruction cache (the last --n_val entries).

    python score_decoder.py --enc alps --weights alps_vitb16.safetensors --decoder runs/recon/alps_decoder.pt \
        --cache imagenet224_20k.npy [--device cuda]

Prints the mean l1 (on [-1, 1]) and LPIPS (VGG) of the decoded images, as train_decoder.py reports at the end.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from alps.decoder import SDXL_SCALING, load_decoder  # noqa: E402
from eval.encoders import build  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--vae", default="stabilityai/sdxl-vae")
    ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--ch", type=int, default=512)
    ap.add_argument("--nblocks", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N held-out images")
    a = ap.parse_args()
    import lpips as lpips_lib
    from diffusers import AutoencoderKL
    dev = a.device
    vae = AutoencoderKL.from_pretrained(a.vae).to(dev).eval()
    lp = lpips_lib.LPIPS(net="vgg", verbose=False).to(dev).eval()
    E = build(a.enc, device=dev, weights=a.weights, res=224, dynamic_img_size=True)
    dec = load_decoder(a.decoder, dim=E.dim, ch=a.ch, nblocks=a.nblocks, device=dev)
    cache = np.load(a.cache, mmap_mode="r")
    N = len(cache)
    idx = np.arange(N - a.n_val, N)[: a.limit or None]
    l1s, lps = [], []
    with torch.no_grad():
        for i in range(0, len(idx), a.bs):
            u = torch.from_numpy(np.ascontiguousarray(cache[idx[i:i + a.bs]])).to(dev).permute(0, 3, 1, 2).float()
            img11 = u / 127.5 - 1.0
            z = E(E.normalize(u / 255.0)).float()
            img = vae.decode(dec(z) / SDXL_SCALING).sample.clamp(-1, 1)
            l1s.append((img - img11).abs().mean(dim=(1, 2, 3)).cpu())
            lps.append(lp(img, img11).flatten().cpu())
    l1s, lps = torch.cat(l1s).numpy(), torch.cat(lps).numpy()
    print(f"[{a.enc}] {len(l1s)} held-out images: l1 {l1s.mean():.4f}  LPIPS {lps.mean():.4f}")


if __name__ == "__main__":
    main()
