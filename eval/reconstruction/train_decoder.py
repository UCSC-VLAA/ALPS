# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Pixel reconstruction from frozen tokens: train the evaluation decoder of one encoder and score the held-out images.

    python train_decoder.py --enc alps --weights alps_vitb16.safetensors --cache imagenet224_20k.npy --out runs/recon
    python train_decoder.py --enc dinov3_b16 --cache imagenet224_20k.npy --out runs/recon
    python train_decoder.py --enc alps --weights ... --cache imagenet224_50k.npy --ch 1024 --nblocks 10 --steps 60000 --tag _large

The protocol of the paper (identical for every encoder): the decoder of alps/decoder.py (tokens -> SDXL latents), loss =
MSE on the scaled SDXL latent + LPIPS(VGG) of the decoded image, token-noise augmentation z + U(0, 1) std(z) eps per
image, AdamW lr 1e-4, batch 8, 20,000 steps, float32; the last 1,000 images of the cache are held out and scored with
l1 on [-1, 1] and LPIPS(VGG). Encoders receive the 224 px images with their own normalization (eval/encoders.py).
Outputs: <tag>.json (l1, lpips, history), <tag>_per_image.npz, <tag>_decoder.pt.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from alps.decoder import SDXL_SCALING, EvalDecoder  # noqa: E402
from eval.encoders import build  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", required=True)
    ap.add_argument("--weights", default=None, help="safetensors file for alps / alps_stage1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--vae", default="stabilityai/sdxl-vae")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam_lpips", type=float, default=1.0)
    ap.add_argument("--z_noise", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--ch", type=int, default=512)
    ap.add_argument("--nblocks", type=int, default=5)
    ap.add_argument("--tag", default="", help="suffix of the output names")
    a = ap.parse_args()
    import lpips as lpips_lib
    from diffusers import AutoencoderKL
    dev = "cuda"
    tag = a.enc + a.tag
    os.makedirs(a.out, exist_ok=True)
    res_path = os.path.join(a.out, f"{tag}.json")
    if os.path.exists(res_path):
        print(f"[{tag}] already done: {res_path}")
        return
    vae = AutoencoderKL.from_pretrained(a.vae).to(dev).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    lp = lpips_lib.LPIPS(net="vgg", verbose=False).to(dev).eval()
    E = build(a.enc, device=dev, weights=a.weights, res=224, dynamic_img_size=True)
    meta = E.meta()
    cache = np.load(a.cache, mmap_mode="r")
    N = len(cache)
    ntr = N - a.n_val
    print(f"[{tag}] cache N={N} train={ntr} held out={a.n_val} | encoder dim {E.dim}", flush=True)
    dec = EvalDecoder(dim=E.dim, ch=a.ch, nblocks=a.nblocks).to(dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=a.lr)
    g = torch.Generator().manual_seed(0)
    torch.manual_seed(0)

    def batch(idx):
        u = torch.from_numpy(np.ascontiguousarray(cache[np.sort(idx)])).to(dev).permute(0, 3, 1, 2).float()
        return u, u / 127.5 - 1.0

    @torch.no_grad()
    def enc(u255):
        return E(E.normalize(u255 / 255.0)).float()

    @torch.no_grad()
    def evaluate(per_image=False):
        dec.eval()
        l1s, lps = [], []
        for i in range(ntr, N, a.bs):
            u, img11 = batch(np.arange(i, min(i + a.bs, N)))
            img = vae.decode(dec(enc(u)) / SDXL_SCALING).sample.clamp(-1, 1)
            l1s.append((img - img11).abs().mean(dim=(1, 2, 3)).cpu())
            lps.append(lp(img, img11).flatten().cpu())
        dec.train()
        l1s, lps = torch.cat(l1s).numpy(), torch.cat(lps).numpy()
        return (l1s, lps) if per_image else (float(l1s.mean()), float(lps.mean()))

    t0, hist = time.time(), []
    for step in range(a.steps + 1):
        idx = torch.randint(ntr, (a.bs,), generator=g).numpy()
        u, img11 = batch(idx)
        z = enc(u)
        with torch.no_grad():
            tgt = vae.encode(img11).latent_dist.sample() * SDXL_SCALING
        if a.z_noise > 0:
            z = z + (torch.rand(z.shape[0], 1, 1, device=dev) * a.z_noise) * z.std() * torch.randn_like(z)
        pred = dec(z)
        img = vae.decode(pred / SDXL_SCALING).sample.clamp(-1, 1)
        loss = F.mse_loss(pred, tgt) + a.lam_lpips * lp(img, img11).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print(f"[{tag}] step {step}/{a.steps} loss {loss.item():.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if step % a.eval_every == 0 and step > 0:
            vl1, vlp = evaluate()
            hist.append({"step": step, "l1": vl1, "lpips": vlp})
            print(f"[{tag}] held-out @ {step}: l1 {vl1:.4f}  LPIPS {vlp:.4f}", flush=True)
            torch.save(dec.state_dict(), os.path.join(a.out, f"{tag}_decoder.pt"))
    l1s, lps = evaluate(per_image=True)
    np.savez(os.path.join(a.out, f"{tag}_per_image.npz"), l1=l1s, lpips=lps, index=np.arange(ntr, N))
    json.dump({"enc": tag, "l1": float(l1s.mean()), "lpips": float(lps.mean()), "history": hist, "steps": a.steps,
               "bs": a.bs, "lr": a.lr, "z_noise": a.z_noise, "n_val": a.n_val, "ch": a.ch, "nblocks": a.nblocks,
               "minutes": (time.time() - t0) / 60, "meta": meta}, open(res_path, "w"), indent=1, default=str)
    print(f"[{tag}] DONE l1 {l1s.mean():.4f} LPIPS {lps.mean():.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
