# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""Prediction accuracy of a Reacher world model (Table 4 Reacher column, the Reacher block of Table S16).

  python reacher_rollout_eval.py --name <encoder> --policy <run>/weights_s26000.pt --probe <probe.pt> --heldout <h5> --out <npz>

Windows: the 100 fresh episodes of reacher_collect_heldout.py (the world models were trained on a random split of the clips
of reacher.h5, so no episode of that file is held out), 20 windows per episode (numpy default_rng(0)) of 8 frames 5 steps
apart; three real history frames, the recorded action blocks normalised with the training normaliser, open loop k = 1..5
with the autoregressive scheme of PreJEPA.rollout (predicted tokens fed back, action channels replaced by the recorded
actions). The frozen probe of the encoder (reacher_predeval.py) reads
  real_<m>        the real tokens of the frames k = 0..5,
  pred_<m>        the predicted tokens k = 1..5,
  static_tok_<m>  the real tokens of k = 0, scored against the true state at k = 1..5 ("last observation"),
  nochange_<m>    the true state at k = 0 scored against the true state at k (no probe),
m = joints_deg (Table 4: mean of the two absolute wrapped joint-angle errors in degrees, k = 5), joint1_deg, joint2_deg,
finger; plus token_mse (k = 1..5), tf_loss (the one-step teacher-forced loss, a check against training), traj (episode),
window, start.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reacher_predeval as RP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--policy", required=True); ap.add_argument("--probe", required=True)
    ap.add_argument("--heldout", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--per_ep", type=int, default=20)
    a = ap.parse_args()
    import h5py
    import hdf5plugin  # noqa: F401
    device = torch.device("cuda:0"); K, HIST, FS, PDIM = RP.K, RP.HIST, RP.FS, RP.PDIM
    model = RP.load_model(a.policy, device)
    P = torch.load(a.probe, map_location="cpu"); assert P["name"] == a.name
    probe = RP.AttnProbe(P["d_in"], P["n_tok"], P["n_out"]).to(device); probe.load_state_dict(P["state_dict"]); probe.eval()
    mu, sd = P["mu"].to(device), P["sd"].to(device)
    norm = RP.heldout_dataset()
    f = h5py.File(a.heldout, "r")
    lens, offs = f["ep_len"][:], f["ep_offset"][:]
    span = (HIST + K) * FS                                     # 8 frames, 5 steps apart = 40 rows (the 8th action block is unused)
    rng = np.random.default_rng(a.seed)
    wins = [(e, int(s)) for e in range(len(lens)) for s in np.sort(rng.choice(lens[e] - span + 1, a.per_ep, replace=False))]
    traj = np.array([w[0] for w in wins])
    rec = {}; tok, tf_loss = [], []
    t0 = time.time()
    for s0 in range(0, len(wins), a.bs):
        W = wins[s0:s0 + a.bs]
        rows = [np.arange(offs[e] + s, offs[e] + s + span) for e, s in W]
        pix = np.stack([f["pixels"][r[0]:r[-1] + 1][::FS] for r in rows])                       # (B, 8, H, W, 3)
        act = np.stack([f["action"][r[0]:r[-1] + 1] for r in rows]).astype(np.float64)          # (B, 40, 2)
        act = torch.stack([norm({"action": torch.as_tensor(x)})["action"] for x in act]).float()
        act = torch.nan_to_num(act, 0.0).reshape(len(W), HIST + K, -1).to(device)             # (B, 8, 10): block j = steps between frames j, j+1
        qpos = torch.as_tensor(np.stack([f["qpos"][r[0]:r[-1] + 1][::FS] for r in rows])).float().to(device)
        fing = torch.as_tensor(np.stack([f["finger_pos"][r[0]:r[-1] + 1][::FS] for r in rows])).float().to(device)
        x = torch.stack([RP.preprocess(p) for p in pix])                                         # (B, 8, 3, H, W)
        with torch.no_grad():
            z_real = RP.tokens(model, x, device)                                                # (B, 8, 196, 768)
            info = model.encode({"pixels": x[:, :HIST].to(device), "action": act[:, :HIST]}, target="emb")
            z = info["emb"]                                                                      # (B, 3, 196, 778); context blocks 0, 1, 2
            tf = model.predict(z)                                                                # teacher-forced, as in training
            tf_loss.append(((tf[:, :, :, :PDIM] - z_real[:, 1:HIST + 1]) ** 2).mean(dim=(1, 2, 3)).cpu())
            for t in range(K - 1):                                                               # PreJEPA.rollout
                pred = model.predict(z[:, -HIST:])[:, -1:]
                pred = model.replace_action_in_embedding(pred.unsqueeze(0), act[None, :, HIST + t:HIST + t + 1])[0]
                z = torch.cat([z, pred], 1)
            z = torch.cat([z, model.predict(z[:, -HIST:])[:, -1:]], 1)
            z_pred = z[:, HIST:HIST + K, :, :PDIM].float()                                       # frames 3 .. 7 = k 1 .. 5

            def read(zz):
                B, T = zz.shape[:2]
                return (probe(zz.flatten(0, 1)) * sd + mu).view(B, T, -1)
            y_real = read(z_real[:, HIST - 1:])
            er = RP.errors(y_real, qpos[:, HIST - 1:], fing[:, HIST - 1:])                      # k = 0 .. 5
            ep = RP.errors(read(z_pred), qpos[:, HIST:], fing[:, HIST:])                        # k = 1 .. 5
            q0, f0 = qpos[:, HIST - 1:HIST].expand(-1, K, -1), fing[:, HIST - 1:HIST].expand(-1, K, -1)
            en = RP.errors(torch.cat([torch.sin(q0[..., :1]), torch.cos(q0[..., :1]), torch.sin(q0[..., 1:]), torch.cos(q0[..., 1:]), f0], -1),
                           qpos[:, HIST:], fing[:, HIST:])                                      # nothing moves
            es = RP.errors(y_real[:, :1].expand(-1, K, -1), qpos[:, HIST:], fing[:, HIST:])      # last observation, through the probe
            for m in er:
                for w, d in (("real", er), ("pred", ep), ("nochange", en), ("static_tok", es)):
                    rec.setdefault(f"{w}_{m}", []).append(d[m].cpu())
            tok.append(((z_pred - z_real[:, HIST:]) ** 2).mean(dim=(2, 3)).cpu())
        if s0 == 0:
            print(f"[rollout_eval] first batch in {time.time() - t0:.0f} s", flush=True)
    out = {k: torch.cat(v).numpy() for k, v in rec.items()}
    out["token_mse"] = torch.cat(tok).numpy(); out["tf_loss"] = torch.cat(tf_loss).numpy()
    out["traj"] = traj; out["window"] = np.arange(len(wins)); out["start"] = np.array([w[1] for w in wins])
    np.savez(a.out, **out, meta=json.dumps({"name": a.name, "policy": a.policy, "probe": a.probe, "heldout": a.heldout, "K": K, "hist": HIST,
                                             "frameskip": FS, "seed": a.seed, "per_ep": a.per_ep}))
    msg = {k: [round(float(v_), 3) for v_ in v.mean(0)] for k, v in out.items() if k not in ("traj", "window", "start", "tf_loss")}
    print(f"[rollout_eval] {a.name} {a.policy}: {len(wins)} windows from {len(set(traj))} fresh episodes in {time.time() - t0:.0f} s; "
          f"one-step teacher-forced loss {out['tf_loss'].mean():.4f}")
    print("[rollout_eval] means per k: " + json.dumps(msg), flush=True)
    print("WM_ROLLOUT_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
