# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""State probes of the world-model prediction evaluation (PushT, Wall) and the helpers that the evaluation shares.

  python wm_predeval.py probe --run <run dir> --out <probe.pt> [--stats <npz>] [--data <task data folder>] [--epoch s<T>]

The probe is trained on REAL tokens: the frozen encoder and the token statistics of the given world-model run, i.e. the
standardised space in which its predictor works, on single frames of the training trajectories (20,000 frames for
training, 2,000 other training frames for model selection; frames come from 8-frame windows with the run's frame skip).
It depends only on the encoder and its statistics, not on the trained predictor, so one probe per encoder and task serves
every world model of that encoder. Probe: 2 cross-attention blocks (8 heads), 8 learned queries, width 256, a LayerNorm and
a linear layer on the concatenated queries; AdamW 3e-4, weight decay 0.05, cosine to zero over 6,000 updates, batch 128,
seed 0; the checkpoint with the lowest validation loss (every 500 updates) is kept.
Targets: PushT agent and block position and the block angle as (sin, cos), Wall the agent position; MSE on targets
standardised with training statistics. The open-loop evaluation is wm_rollout_eval.py. --stats / --data override the
statistics file and the data folder stored in the run's hydra.yaml (runs moved to another machine).
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import load_model  # noqa: E402

K = 5   # world-model steps of the open-loop evaluation


def load_cfg(run, stats=None, data=None):
    cfg = OmegaConf.load(os.path.join(run, "hydra.yaml"))
    with open_dict(cfg):
        if stats:
            cfg.encoder.stats = stats
        if data:
            cfg.env.dataset.data_path = data
    return cfg


def make_model(run, cfg, device, epoch="final"):
    ckpt = Path(run) / "checkpoints" / f"model_{epoch}.pth"
    assert ckpt.exists(), ckpt
    m = load_model(ckpt, cfg, cfg.num_action_repeat, device=device)
    m.eval()
    return m


def datasets(cfg, num_hist, num_pred, seed):
    np.random.seed(seed)  # TrajSlicerDataset permutes its slices with the global numpy RNG
    d, _ = hydra.utils.call(cfg.env.dataset, num_hist=num_hist, num_pred=num_pred, frameskip=cfg.frameskip)
    return d


def target(env, state):
    """state (..., S) -> probe target (..., T)"""
    if env == "pusht":  # agent x, y, block x, y, block angle (rad)
        a = state[..., 4]
        return torch.stack([state[..., 0], state[..., 1], state[..., 2], state[..., 3], torch.sin(a), torch.cos(a)], -1)
    if env == "wall":   # agent x, y
        return state[..., :2]
    raise ValueError(env)


def errors(env, pred, true):
    """pred / true: (..., T) in raw units -> dict of per-item errors"""
    if env == "pusht":
        out = {"agent": torch.linalg.norm(pred[..., 0:2] - true[..., 0:2], dim=-1),
               "block": torch.linalg.norm(pred[..., 2:4] - true[..., 2:4], dim=-1)}
        ap = torch.atan2(pred[..., 4], pred[..., 5]); at = torch.atan2(true[..., 4], true[..., 5])
        d = torch.remainder(ap - at + math.pi, 2 * math.pi) - math.pi
        out["angle_deg"] = d.abs() * 180 / math.pi
        return out
    return {"agent": torch.linalg.norm(pred - true, dim=-1)}


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
    """2 cross-attention layers, 8 learned queries, width 256, on (B, P, D) tokens."""
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


@torch.no_grad()
def encode_frames(model, obs, device):
    """obs dict (B, T, ...) -> standardised visual tokens (B, T, P, D) as the predictor sees them"""
    o = {k: v.to(device) for k, v in obs.items()}
    return model.encode_obs(o)["visual"].float()


def collate(items):
    obs = {k: torch.stack([it[0][k] for it in items]) for k in items[0][0]}
    return obs, torch.stack([it[1] for it in items]), torch.stack([it[2] for it in items])


def do_probe(a, device):
    cfg = load_cfg(a.run, a.stats, a.data); env = cfg.env.name
    model = make_model(a.run, cfg, device, a.epoch)
    NF = 8  # frames per sampled training window (frame skip of the run)
    d = datasets(cfg, NF, 0, a.seed)
    tr = d["train"]; rng = np.random.default_rng(a.seed)
    nw_tr, nw_va = a.n_train // NF, a.n_val // NF
    idx = rng.choice(len(tr), nw_tr + nw_va, replace=False)   # the first nw_tr windows train the probe, the rest = validation
    Z, Y = [], []
    t0 = time.time()
    for s in range(0, len(idx), a.bs):
        obs, act, st = collate([tr[int(i)] for i in idx[s:s + a.bs]])
        Z.append(encode_frames(model, obs, device).flatten(0, 1).half().cpu()); Y.append(target(env, st.float()).flatten(0, 1))
    Z = torch.cat(Z); Y = torch.cat(Y)
    a.n_train = nw_tr * NF
    print(f"[probe] {env}: encoded {len(Z)} frames {tuple(Z.shape)} in {time.time() - t0:.0f} s", flush=True)
    mu, sd = Y[:a.n_train].mean(0), Y[:a.n_train].std(0) + 1e-6
    Yn = (Y - mu) / sd
    torch.manual_seed(a.seed)
    probe = AttnProbe(Z.shape[-1], Z.shape[1], Y.shape[-1]).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    best, best_state = 1e9, None
    g = torch.Generator().manual_seed(a.seed)
    for step in range(1, a.steps + 1):
        b = torch.randint(0, a.n_train, (128,), generator=g)
        loss = F.mse_loss(probe(Z[b].to(device).float()), Yn[b].to(device))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0 or step == a.steps:
            probe.eval()
            with torch.no_grad():
                v = torch.cat([probe(Z[i:i + 256].to(device).float()).cpu() for i in range(a.n_train, len(Z), 256)])
            vl = F.mse_loss(v, Yn[a.n_train:]).item(); probe.train()
            if vl < best:
                best, best_state = vl, {k: x.detach().cpu().clone() for k, x in probe.state_dict().items()}
            print(f"[probe] step {step} train {loss.item():.4f} val {vl:.4f}", flush=True)
    probe.load_state_dict(best_state); probe.eval()
    with torch.no_grad():
        v = torch.cat([probe(Z[i:i + 256].to(device).float()).cpu() for i in range(a.n_train, len(Z), 256)]) * sd + mu
    e = errors(env, v, Y[a.n_train:])
    summ = {k: float(x.mean()) for k, x in e.items()}
    print(f"[probe] best val {best:.4f}; held-out frame errors {summ}", flush=True)
    torch.save({"state_dict": best_state, "mu": mu, "sd": sd, "env": env, "d_in": Z.shape[-1], "n_tok": Z.shape[1],
                "n_out": Y.shape[-1], "run": a.run, "stats": str(cfg.encoder.stats), "val_errors": summ, "seed": a.seed}, a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe"])
    ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--stats"); ap.add_argument("--data")
    ap.add_argument("--epoch", default="final", help="checkpoint model_<epoch>.pth of the run (any; the probe reads only the encoder)")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--bs", type=int, default=25)
    ap.add_argument("--n_train", type=int, default=20000); ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=6000)
    a = ap.parse_args()
    do_probe(a, torch.device("cuda:0"))


if __name__ == "__main__":
    main()
