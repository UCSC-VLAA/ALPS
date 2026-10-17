# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""State probes of the world-model prediction evaluation on the deformable tasks (Granular, Rope).

  python wm_predeval_deform.py probe --run <run dir> --out <probe.pt> [--stats <npz>] [--data <deformable folder>] [--epoch s<T>]

Same protocol as wm_predeval.py (PushT, Wall), but the state is a particle set: the probe outputs 256 points (x, y, z) and
is trained and scored with the symmetric Chamfer distance of DINO-WM's scorer (env/deformable_env/FlexEnvWrapper:
mean nearest-neighbour Euclidean distance on xyz in both directions, summed) against a fixed particle subset (all particles
if there are at most 2,048, Rope: 1,965; otherwise 2,048 drawn once with seed 20261001, Granular). Episodes have 20 frames
and no frame skip; the probe frames come from 8-frame windows.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wm_predeval import AttnProbe, collate, datasets, encode_frames, load_cfg, make_model  # noqa: E402

N_PTS = 256        # points predicted by the probe
N_SUB = 2048       # particles of the reference set
SUB_SEED = 20261001


def subset(n_particles):
    if n_particles <= N_SUB:
        return np.arange(n_particles)
    return np.sort(np.random.default_rng(SUB_SEED).choice(n_particles, N_SUB, replace=False))


def target(state, sub):
    """state (..., P * 4), flattened as in DeformDataset -> (..., S, 3): xyz of the reference particles"""
    s = state.reshape(*state.shape[:-1], -1, 4)[..., :3]
    return s[..., torch.as_tensor(sub, device=state.device), :]


def chamfer(x, y):
    """x (B, N, 3), y (B, M, 3) -> (B,) symmetric Chamfer distance as in DINO-WM's scorer"""
    d = torch.cdist(x, y)
    return d.min(2)[0].mean(1) + d.min(1)[0].mean(1)


def task_of(cfg):
    t = cfg.env.dataset.object_name
    assert cfg.env.name == "deformable_env" and t in ("granular", "rope"), (cfg.env.name, t)
    return t


def do_probe(a, device):
    cfg = load_cfg(a.run, a.stats, a.data); task = task_of(cfg)
    model = make_model(a.run, cfg, device, a.epoch)
    NF = 8  # frames per sampled training window (episodes have 20 frames, frame skip 1)
    d = datasets(cfg, NF, 0, a.seed)
    tr = d["train"]; rng = np.random.default_rng(a.seed)
    nw_tr, nw_va = a.n_train // NF, a.n_val // NF
    idx = rng.choice(len(tr), nw_tr + nw_va, replace=False)   # the first nw_tr windows train the probe, the rest = validation
    sub = None; Z, Y = [], []
    t0 = time.time()
    for s in range(0, len(idx), a.bs):
        obs, act, st = collate([tr[int(i)] for i in idx[s:s + a.bs]])
        if sub is None:
            sub = subset(st.shape[-1] // 4)
        Z.append(encode_frames(model, obs, device).flatten(0, 1).half().cpu()); Y.append(target(st.float(), sub).flatten(0, 1))
    Z = torch.cat(Z); Y = torch.cat(Y)                                                   # (N, P, D), (N, S, 3)
    a.n_train = nw_tr * NF
    print(f"[probe] {task}: encoded {len(Z)} frames {tuple(Z.shape)}, reference set {tuple(Y.shape[1:])}, in {time.time() - t0:.0f} s", flush=True)
    mu = Y[:a.n_train].reshape(-1, 3).mean(0); sd = Y[:a.n_train].reshape(-1, 3).std(0) + 1e-6
    torch.manual_seed(a.seed)
    probe = AttnProbe(Z.shape[-1], Z.shape[1], N_PTS * 3).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    mu_d, sd_d = mu.to(device), sd.to(device)

    def points(z):
        return probe(z).view(len(z), N_PTS, 3) * sd_d + mu_d

    def val():
        probe.eval()
        with torch.no_grad():
            v = torch.cat([chamfer(points(Z[i:i + 128].to(device).float()), Y[i:i + 128].to(device)).cpu() for i in range(a.n_train, len(Z), 128)])
        probe.train()
        return v
    best, best_state = 1e9, None
    g = torch.Generator().manual_seed(a.seed)
    for step in range(1, a.steps + 1):
        b = torch.randint(0, a.n_train, (128,), generator=g)
        loss = chamfer(points(Z[b].to(device).float()), Y[b].to(device)).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0 or step == a.steps:
            vl = val().mean().item()
            if vl < best:
                best, best_state = vl, {k: x.detach().cpu().clone() for k, x in probe.state_dict().items()}
            print(f"[probe] step {step} train {loss.item():.4f} val chamfer {vl:.4f}", flush=True)
    probe.load_state_dict(best_state)
    v = val()
    print(f"[probe] best val chamfer {best:.4f}; held-out frame chamfer {v.mean().item():.4f}", flush=True)
    torch.save({"state_dict": best_state, "mu": mu, "sd": sd, "env": task, "d_in": Z.shape[-1], "n_tok": Z.shape[1],
                "n_out": N_PTS * 3, "sub": sub, "run": a.run, "stats": str(cfg.encoder.stats), "val_errors": {"chamfer": float(v.mean())},
                "seed": a.seed}, a.out)


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
