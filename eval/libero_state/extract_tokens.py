# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Cache the frozen patch tokens of every frame of every LIBERO probe episode: <cache>/<encoder>/epNNNNN.npz
{tokens [T, N, D] float16, ts, frame_idx, episode, camera} + encoder_meta.json.

    python extract_tokens.py --encoder alps --weights alps_vitb16.safetensors --episodes_dir <episodes> --cache_dir <cache>

Frames (128 x 128 in the episode files) are resized to 224 with PIL bicubic for every encoder. Image encoders encode frame
by frame; the video encoder (V-JEPA 2) encodes the window of the last 16 frames ending at t (front-padded with frame 0),
never a whole trajectory. Float32 with TF32 off (--tf32 off, the default). The token caches of the paper's first probe
round (ALPS (Stage 1), DINOv3, SigLIP 2, CLIP) were computed with PyTorch's default of TF32 cuDNN convolutions
(--tf32 conv) and V-JEPA 2 with TF32 convolutions and matrix products (--tf32 all); the differences in the tokens are
small (mean cosine 0.99999). Encoders and their token conventions: eval/encoders.py.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from eval.encoders import build  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--weights", default=None, help="safetensors file for alps / alps_stage1")
    ap.add_argument("--episodes_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--max_eps", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tf32", choices=("off", "conv", "all"), default="off")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = a.tf32 == "all"
    torch.backends.cudnn.allow_tf32 = a.tf32 in ("conv", "all")
    enc = build(a.encoder, device=a.device, weights=a.weights, vjepa2_probe=True)
    out = os.path.join(a.cache_dir, a.encoder)
    os.makedirs(out, exist_ok=True)
    idx = json.load(open(os.path.join(a.episodes_dir, "episodes.json")))
    eps = idx["episodes"][:a.max_eps] if a.max_eps else idx["episodes"]
    k, n = map(int, a.shard.split("/"))
    eps = eps[k::n]
    json.dump({**enc.meta(), "tf32": a.tf32}, open(os.path.join(out, "encoder_meta.json"), "w"), indent=1, default=str)
    todo = [e for e in eps if not os.path.exists(os.path.join(out, e["episode"] + ".npz"))]
    print(f"[{a.encoder}] {len(todo)}/{len(eps)} episodes; t_hist {enc.t_hist}", flush=True)
    t0, nf = time.time(), 0
    for i, e in enumerate(todo):
        ep = np.load(os.path.join(a.episodes_dir, e["episode"] + ".npz"))
        fr, ts = ep["frames"], ep["ts"]
        if fr.shape[1] != a.res:
            fr = np.stack([np.asarray(Image.fromarray(f).resize((a.res, a.res), Image.BICUBIC), np.uint8) for f in fr])
        toks = []
        if enc.t_hist == 1:
            for j in range(0, len(fr), a.batch):
                toks.append(enc.encode_frames(fr[j:j + a.batch]).to(torch.float16).cpu().numpy())
        else:
            T = enc.t_hist
            pad = np.concatenate([np.repeat(fr[:1], T - 1, 0), fr])
            clips = np.stack([pad[t:t + T] for t in range(len(fr))])          # clip t = frames t-T+1 .. t
            for j in range(0, len(fr), a.batch):
                toks.append(enc.encode_clip(clips[j:j + a.batch]).to(torch.float16).cpu().numpy())
        toks = np.concatenate(toks)
        np.savez_compressed(os.path.join(out, e["episode"] + ".npz"), tokens=toks, ts=ts, frame_idx=np.arange(len(fr)),
                            episode=e["episode"], camera=idx["camera"])
        nf += len(fr)
        if i % 50 == 0:
            print(f"[{a.encoder}] {i}/{len(todo)} {toks.shape} {nf / (time.time() - t0):.1f} frames/s", flush=True)
    print(f"[{a.encoder}] done: {nf} frames, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
