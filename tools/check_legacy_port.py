# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Compare an earlier PyTorch port of the ALPS encoder with the release module, on real weights.

    python tools/check_legacy_port.py --port <port state dict .pt> [--images DIR] [--n 32] [--out enc.safetensors]

The earlier port (the PyTorch module that ran the paper's evaluations) used torch's default LayerNorm epsilon
1e-5, while the Flax model was trained with 1e-6. The script reports
  (1) release module with epsilon 1e-5 vs the old module: must be identical up to float rounding (mapping check);
  (2) release module with epsilon 1e-6 (the training value) vs the old module: the size of the epsilon effect.
"""
import argparse
import glob
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from alps.convert import legacy_port_to_torch  # noqa: E402
from alps.transforms import center_crop_transform  # noqa: E402
from alps.vit import vit_b16  # noqa: E402


def legacy_sincos2d(h, w, dim, temp=10000.0):
    omega = np.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temp ** omega)
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = np.concatenate([np.sin(x), np.cos(x), np.sin(y), np.cos(y)], axis=1)
    pe = np.concatenate([np.zeros((1, dim)), pe], axis=0)
    return torch.tensor(pe[None], dtype=torch.float32)


class LegacyBlock(nn.Module):          # verbatim structure of the earlier port
    def __init__(self, dim, heads, mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim); self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, mlp); self.fc2 = nn.Linear(mlp, dim)
        self.heads, self.hd = heads, dim // heads

    def forward(self, x):
        h = self.ln1(x)
        B, L, D = h.shape
        q = self.q(h).view(B, L, self.heads, self.hd).transpose(1, 2)
        k = self.k(h).view(B, L, self.heads, self.hd).transpose(1, 2)
        v = self.v(h).view(B, L, self.heads, self.hd).transpose(1, 2)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.hd), dim=-1)
        o = (a @ v).transpose(1, 2).reshape(B, L, D)
        x = x + self.proj(o)
        h = self.ln2(x)
        return x + self.fc2(F.gelu(self.fc1(h)))


class LegacyViT(nn.Module):
    def __init__(self, dim=768, depth=12, heads=12, mlp=3072, patch=16, res=224):
        super().__init__()
        self.conv = nn.Conv2d(3, dim, patch, patch, bias=False)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_buffer("pos", legacy_sincos2d(res // patch, res // patch, dim))
        self.blocks = nn.ModuleList([LegacyBlock(dim, heads, mlp) for _ in range(depth)])
        self.enc_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.conv(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1) + self.pos
        for b in self.blocks:
            x = b(x)
        tok = x[:, 1:]
        return tok, self.enc_norm(tok.mean(1))


def stats(a, b):
    a, b = a.double(), b.double()
    cos = F.cosine_similarity(a.flatten(0, -2), b.flatten(0, -2), dim=-1)
    rel = (a - b).abs().max() / b.abs().max()
    return f"per-token cosine min {cos.min():.7f} mean {cos.mean():.7f}; max rel err {rel:.2e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--images", default="")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--out", default="")
    ap.add_argument("--release", default="", help="a release safetensors file: compare its encoder weights with the port's")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    old_sd = torch.load(a.port, map_location="cpu", weights_only=True)
    old = LegacyViT().eval()
    old.load_state_dict(old_sd, strict=True)
    new_sd = legacy_port_to_torch(old_sd, dst="")
    if a.images:
        from PIL import Image
        paths = sorted(p for p in glob.glob(os.path.join(a.images, "**", "*"), recursive=True)
                       if p.lower().endswith((".jpg", ".jpeg", ".png")))[: a.n]
        x = torch.cat([center_crop_transform(Image.open(p).convert("RGB")) for p in paths])
    else:
        g = torch.Generator().manual_seed(0)
        x = (torch.rand(a.n, 3, 224, 224, generator=g) - torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)) / \
            torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    dev = a.device
    old = old.to(dev)
    with torch.no_grad():
        z_old, g_old = old(x.to(dev))
        for eps in (1e-5, 1e-6):
            new = vit_b16(eps=eps).eval()
            new.load_state_dict(new_sd, strict=True)
            o = new.to(dev)(x.to(dev))
            print(f"[eps {eps:.0e}] tokens: {stats(o['tokens'], z_old)}")
            print(f"[eps {eps:.0e}] pooled: {stats(o['pooled'], g_old)}")
    if a.release:
        from alps.hub import read_state_dict
        rel = {k[len("visual."):]: v for k, v in read_state_dict(a.release).items() if k.startswith("visual.")}
        assert set(rel) == set(new_sd), (sorted(set(rel) ^ set(new_sd))[:5])
        diff = max(float((rel[k].float() - new_sd[k].float()).abs().max()) for k in rel)
        same = all(torch.equal(rel[k].float(), new_sd[k].float()) for k in rel)
        print(f"[weights] release {os.path.basename(a.release)} vs port: {len(rel)} tensors, identical={same}, max|diff|={diff:.3e}")
    if a.out:
        from alps.hub import save
        save({f"visual.{k}": v for k, v in new_sd.items()}, a.out, {"format": "alps-torch-v1",
                                                                  "source": os.path.basename(a.port)})
        print("wrote", a.out)


if __name__ == "__main__":
    main()
