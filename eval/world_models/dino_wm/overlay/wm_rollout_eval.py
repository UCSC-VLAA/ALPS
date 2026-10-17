# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""Prediction accuracy of a DINO-WM world model (Table 4 and the PushT, Wall, Granular and Rope blocks of Table S16).

  python wm_rollout_eval.py --run <run dir> --probe <probe.pt> --epoch s<T> --out <npz> [--stats <npz>] [--data <task data folder>]

Windows: numpy default_rng(seed 0) draws min(2000, all) of the validation slices with the run's history n and K = 5
world-model steps, sorted (PushT 1,695 windows = all, Wall 2,000 of 7,872, Granular and Rope 1,500 = all). Per window the
world model rolls out open loop from the real history with the recorded actions (VWorldModel.rollout: every prediction,
with its predicted proprioceptive channels, is fed back; the action channels are refilled with the recorded actions);
the frozen probe of the encoder (wm_predeval.py / wm_predeval_deform.py) reads the state from
  real_<m>         the real tokens of the frames k = 0..5 (readout reference; k = 0 = the last observed frame),
  pred_<m>         the predicted tokens of k = 1..5,
  static_tok_<m>   the real tokens of k = 0, scored against the true state at k = 1..5 ("last observation repeated"),
  static_state_<m> the true state at k = 0 scored against the true state at k (no probe),
and token_mse is the MSE between predicted and real standardised tokens (k = 1..5).
m = block / agent / angle_deg (PushT; Table 4 = block at k = 5), agent (Wall), chamfer (Granular, Rope); errors in the units
of the environment. Also written: traj (validation trajectory of every window, the bootstrap cluster) and window.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wm_predeval as W  # noqa: E402
import wm_predeval_deform as WD  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--probe", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--stats", default=None); ap.add_argument("--data", default=None); ap.add_argument("--epoch", default="final")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--bs", type=int, default=25); ap.add_argument("--n_windows", type=int, default=2000)
    a = ap.parse_args()
    dev = torch.device("cuda:0"); K = W.K

    cfg = W.load_cfg(a.run, a.stats, a.data); n = int(cfg.num_hist)
    deform = cfg.env.name == "deformable_env"
    env = WD.task_of(cfg) if deform else cfg.env.name
    model = W.make_model(a.run, cfg, dev, a.epoch)
    P = torch.load(a.probe, map_location="cpu")
    assert P["env"] == env, (P["env"], env)
    probe = W.AttnProbe(P["d_in"], P["n_tok"], P["n_out"]).to(dev); probe.load_state_dict(P["state_dict"]); probe.eval()
    mu, sd = P["mu"].to(dev), P["sd"].to(dev); sub = P.get("sub")
    d = W.datasets(cfg, n, K, a.seed)["valid"]
    rng = np.random.default_rng(a.seed)
    idx = np.sort(rng.choice(len(d), min(a.n_windows, len(d)), replace=False))
    traj = np.array([int(d.slices[i][0]) for i in idx])
    rec = {}

    def add(key, x):
        rec.setdefault(key, []).append(x.cpu())
    t0 = time.time()
    for s in range(0, len(idx), a.bs):
        obs, act, st = W.collate([d[int(i)] for i in idx[s:s + a.bs]])
        obs = {k: v.to(dev) for k, v in obs.items()}; act = act.to(dev); st = st.to(dev).float()
        with torch.no_grad():
            z_real = W.encode_frames(model, obs, dev)                                       # (B, n+K, P, D)
            z_obs, _ = model.rollout({k: v[:, :n] for k, v in obs.items()}, act[:, :n + K - 1])
            z_pred = z_obs["visual"][:, n:n + K].float()                                    # predictions of frames n .. n+K-1
            assert z_pred.shape[1] == K, z_pred.shape
            B = z_real.shape[0]
            if deform:
                y_true = WD.target(st[:, n - 1:n + K], sub)                                 # (B, K+1, S, 3)

                def read(z):
                    return probe(z.flatten(0, 1)).view(-1, WD.N_PTS, 3) * sd + mu
                r_real = read(z_real[:, n - 1:n + K]).view(B, K + 1, WD.N_PTS, 3)
                add("real_chamfer", WD.chamfer(r_real.flatten(0, 1), y_true.flatten(0, 1)).view(B, K + 1))
                add("pred_chamfer", WD.chamfer(read(z_pred), y_true[:, 1:].flatten(0, 1)).view(B, K))
                add("static_tok_chamfer", torch.stack([WD.chamfer(r_real[:, 0], y_true[:, k]) for k in range(1, K + 1)], 1))
                add("static_state_chamfer", torch.stack([WD.chamfer(y_true[:, 0], y_true[:, k]) for k in range(1, K + 1)], 1))
            else:
                y_true = W.target(env, st[:, n - 1:n + K])                                  # (B, K+1, T)

                def read(z):
                    Bz, T = z.shape[:2]
                    return (probe(z.flatten(0, 1)) * sd + mu).view(Bz, T, -1)
                y_real = read(z_real[:, n - 1:n + K]); y_pred = read(z_pred)
                er = W.errors(env, y_real, y_true); ep = W.errors(env, y_pred, y_true[:, 1:])
                es = W.errors(env, y_real[:, :1].expand(-1, K, -1), y_true[:, 1:])
                et = W.errors(env, y_true[:, :1].expand(-1, K, -1), y_true[:, 1:])
                for m in er:
                    add("real_" + m, er[m]); add("pred_" + m, ep[m]); add("static_tok_" + m, es[m]); add("static_state_" + m, et[m])
            add("token_mse", ((z_pred - z_real[:, n:n + K]) ** 2).mean(dim=(2, 3)))
        if s == 0:
            print(f"[rollout_eval] first batch in {time.time() - t0:.0f} s", flush=True)
    out = {k: torch.cat(v).numpy() for k, v in rec.items()}
    out["traj"] = traj; out["window"] = idx
    np.savez(a.out, **out, meta=json.dumps({"run": a.run, "probe": a.probe, "env": env, "num_hist": n, "K": K, "frameskip": int(cfg.frameskip),
                                             "seed": a.seed, "stats": str(cfg.encoder.stats), "epoch": a.epoch, "n_windows": int(len(idx))}))
    msg = {k: [round(float(x), 4) for x in v.mean(0)] for k, v in out.items() if k not in ("traj", "window")}
    print(f"[rollout_eval] {os.path.basename(a.run.rstrip('/'))} epoch {a.epoch}: {len(idx)} windows from {len(set(traj))} trajectories in {time.time() - t0:.0f} s")
    print("[rollout_eval] means per k: " + json.dumps(msg), flush=True)
    print("WM_ROLLOUT_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
