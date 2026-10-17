# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""State probe of the Reacher prediction evaluation (Table 4, Table S16) and the helpers that reacher_rollout_eval.py shares.

  python reacher_predeval.py frames --out <dir> [--h5 $STABLEWM_HOME/reacher.h5]
  python reacher_predeval.py probe --name <encoder> --policy <run>/weights_s26000.pt --frames <dir> --out <probe.pt>

frames: the probe frames, cached once: 1,100 random chunks (HDF5 chunks of 100 rows) of reacher.h5 (numpy default_rng(0)),
        every 5th row -> 22,000 frames; the frames of 100 of these chunks are the model-selection set (2,000 frames).
probe : attentive probe (2 cross-attention blocks with 8 heads, 8 learned queries, width 256; the probe of the DINO-WM tasks)
        on the tokens the predictor sees (FrozenTokens: unit length + per-channel standardisation, float32), read through
        the world model given by --policy (a save_pretrained file under $STABLEWM_HOME/checkpoints; only its frozen encoder
        and statistics matter, so one probe serves every world model of the encoder); target sin / cos of the two joint
        angles and the fingertip (x, y); MSE on targets standardised with training statistics; AdamW 3e-4, weight decay
        0.05, cosine to zero over 6,000 updates, batch 128, seed 0; the checkpoint with the lowest validation loss (every
        500 updates) is kept.
Run with this folder and the stable-worldmodel clone importable (scripts/reacher_env.sh.example).
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
K, HIST, FS, PDIM = 5, 3, 5, 768
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def default_h5():
    return os.path.join(os.environ.get("STABLEWM_HOME", os.path.expanduser("~/.stable-wm")), "reacher.h5")


class Block(nn.Module):
    def __init__(self, d, heads=8):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.nk = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, q, kv):
        q = q + self.att(self.n1(q), self.nk(kv), self.nk(kv), need_weights=False)[0]
        return q + self.mlp(self.n2(q))


class AttnProbe(nn.Module):
    """2 cross-attention layers, 8 learned queries, width 256 (the probe of the DINO-WM tasks), on (B, P, D) tokens."""
    def __init__(self, d_in, n_tok, n_out, d=256, n_q=8, layers=2):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.zeros(1, n_tok, d))
        self.q = nn.Parameter(torch.randn(1, n_q, d) * .02)
        self.blocks = nn.ModuleList([Block(d) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(n_q * d), nn.Linear(n_q * d, n_out))

    def forward(self, z):
        kv = self.inp(z) + self.pos
        q = self.q.expand(z.shape[0], -1, -1)
        for b in self.blocks:
            q = b(q, kv)
        return self.out(q.flatten(1))


def target(qpos, finger):
    """(..., 2) joint angles, (..., 2) fingertip -> (..., 6)"""
    return torch.cat([torch.sin(qpos[..., :1]), torch.cos(qpos[..., :1]), torch.sin(qpos[..., 1:2]), torch.cos(qpos[..., 1:2]), finger], -1)


def wrap(d):
    return torch.remainder(d + math.pi, 2 * math.pi) - math.pi


def errors(pred, qpos, finger):
    """pred (..., 6) in raw units; true joint angles and fingertip -> dict of per-item errors"""
    e1 = wrap(torch.atan2(pred[..., 0], pred[..., 1]) - qpos[..., 0]).abs() * 180 / math.pi
    e2 = wrap(torch.atan2(pred[..., 2], pred[..., 3]) - qpos[..., 1]).abs() * 180 / math.pi
    return {"joints_deg": (e1 + e2) / 2, "joint1_deg": e1, "joint2_deg": e2, "finger": torch.linalg.norm(pred[..., 4:6] - finger, dim=-1)}


def load_model(policy, device):
    import hdf5plugin  # noqa: F401  (Blosc filter of the pixels of reacher.h5)
    import stable_worldmodel as swm
    return swm.wm.utils.load_pretrained(policy).to(device).eval()


def preprocess(pix):
    """uint8 (N, H, W, 3) -> float (N, 3, H, W), ImageNet-normalised (the preprocessing of the PreJEPA training; 224 px in)"""
    x = torch.as_tensor(pix).permute(0, 3, 1, 2).float() / 255.0
    assert x.shape[-1] == 224 and x.shape[-2] == 224, x.shape
    return (x - MEAN) / STD


@torch.no_grad()
def tokens(model, x, device):
    """x (B, T, 3, H, W) normalised -> pixel tokens (B, T, 196, 768) as the predictor sees them"""
    info = model.encode({"pixels": x.to(device)}, emb_keys=[], target="emb")
    return info["pixels_emb"].float()


def heldout_dataset():
    """The action normaliser of the PreJEPA training (z-score of the action column of reacher.h5)."""
    import stable_pretraining  # noqa: F401  (as in the training script)
    import stable_worldmodel as swm
    from stable_worldmodel.data import column_normalizer
    train = swm.data.load_dataset("reacher", num_steps=HIST + 1, frameskip=FS, transform=None, keys_to_load=["pixels", "action"],
                                  keys_to_cache=["action"])
    return column_normalizer(train, "action", "action")


def do_frames(a, device):
    import h5py
    import hdf5plugin  # noqa: F401
    os.makedirs(a.out, exist_ok=True)
    f = h5py.File(a.h5, "r")
    n_rows = f["pixels"].shape[0]; ch = f["pixels"].chunks[0]
    rng = np.random.default_rng(0)
    chunks = np.sort(rng.choice(n_rows // ch, 1100, replace=False))
    val_chunks = set(rng.choice(chunks, 100, replace=False).tolist())
    pix = np.lib.format.open_memmap(os.path.join(a.out, "pixels.npy"), mode="w+", dtype=np.uint8, shape=(1100 * (ch // 5), 224, 224, 3))
    rows, isval = [], []
    t0 = time.time()
    for i, c in enumerate(chunks):
        r = np.arange(c * ch, (c + 1) * ch, 5)
        pix[i * len(r):(i + 1) * len(r)] = f["pixels"][c * ch:(c + 1) * ch][::5]
        rows.append(r); isval.append(np.full(len(r), c in val_chunks))
        if i % 100 == 0:
            print(f"[frames] {i} chunks in {time.time() - t0:.0f} s", flush=True)
    pix.flush()
    rows = np.concatenate(rows); isval = np.concatenate(isval)
    np.savez(os.path.join(a.out, "meta.npz"), rows=rows, isval=isval, qpos=f["qpos"][:][rows], finger=f["finger_pos"][:][rows], chunks=chunks)
    print(f"[frames] {len(rows)} frames ({isval.sum()} for model selection) in {time.time() - t0:.0f} s", flush=True)


def do_probe(a, device):
    model = load_model(a.policy, device)
    meta = np.load(os.path.join(a.frames, "meta.npz")); pix = np.load(os.path.join(a.frames, "pixels.npy"), mmap_mode="r")
    Y = target(torch.as_tensor(meta["qpos"]).float(), torch.as_tensor(meta["finger"]).float()); isval = torch.as_tensor(meta["isval"])
    Z = []; t0 = time.time()
    for s in range(0, len(pix), 64):
        Z.append(tokens(model, preprocess(np.asarray(pix[s:s + 64]))[:, None], device)[:, 0].half().cpu())
    Z = torch.cat(Z)
    print(f"[probe] {a.name}: encoded {tuple(Z.shape)} in {time.time() - t0:.0f} s", flush=True)
    tr = torch.nonzero(~isval)[:, 0]; va = torch.nonzero(isval)[:, 0]
    mu, sd = Y[tr].mean(0), Y[tr].std(0) + 1e-6
    Yn = (Y - mu) / sd
    torch.manual_seed(a.seed)
    probe = AttnProbe(Z.shape[-1], Z.shape[1], Y.shape[-1]).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    best, best_state = 1e9, None
    g = torch.Generator().manual_seed(a.seed)
    for step in range(1, a.steps + 1):
        b = tr[torch.randint(0, len(tr), (128,), generator=g)]
        loss = F.mse_loss(probe(Z[b].to(device).float()), Yn[b].to(device))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0 or step == a.steps:
            probe.eval()
            with torch.no_grad():
                v = torch.cat([probe(Z[va[i:i + 256]].to(device).float()).cpu() for i in range(0, len(va), 256)])
            vl = F.mse_loss(v, Yn[va]).item(); probe.train()
            if vl < best:
                best, best_state = vl, {k: x.detach().cpu().clone() for k, x in probe.state_dict().items()}
            print(f"[probe] step {step} train {loss.item():.4f} val {vl:.4f}", flush=True)
    probe.load_state_dict(best_state); probe.eval()
    with torch.no_grad():
        v = torch.cat([probe(Z[va[i:i + 256]].to(device).float()).cpu() for i in range(0, len(va), 256)]) * sd + mu
    e = errors(v, torch.as_tensor(meta["qpos"][va.numpy()]).float(), torch.as_tensor(meta["finger"][va.numpy()]).float())
    summ = {k: float(x.mean()) for k, x in e.items()}
    print(f"[probe] best val {best:.4f}; held-out frame errors {summ}", flush=True)
    torch.save({"state_dict": best_state, "mu": mu, "sd": sd, "d_in": Z.shape[-1], "n_tok": Z.shape[1], "n_out": Y.shape[-1],
                "name": a.name, "policy": a.policy, "val_errors": summ, "seed": a.seed}, a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["frames", "probe"])
    ap.add_argument("--out", required=True); ap.add_argument("--name"); ap.add_argument("--frames"); ap.add_argument("--policy")
    ap.add_argument("--h5", default=default_h5()); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=6000)
    a = ap.parse_args()
    if a.mode == "probe" and not (a.name and a.frames and a.policy):
        ap.error("probe needs --name, --frames and --policy")
    {"frames": do_frames, "probe": do_probe}[a.mode](a, torch.device("cuda:0"))


if __name__ == "__main__":
    main()
