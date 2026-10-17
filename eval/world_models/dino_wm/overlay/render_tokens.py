# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""PushT rendering evaluation and rollout diagnosis, token side (Tables S15 and S17, Fig. 6). No training.

  python render_tokens.py --run <PushT run dir> --epoch s31000 --probe <probe_pusht_<enc>.pt> --out <enc>_diag.npz
                          --tok_out <enc>_tokens.npz [--n_windows 200] [--device cpu]

Windows: the validation slices of DINO-WM's PushT data (history 3, frame skip 5, K = 5), numpy default_rng(0) choice of
200, sorted. Per window the 8 frames are j = 0..7 = simulator steps -10, -5, 0, +5, ..., +25; k = j - 2 (k = 0 = the last
observation). Conditions, all in the standardised token space of the run:
  real   : tokens of the true frames k = 0..5 (reference level for the probe and the decoder)
  m1     : the open-loop rollout of Table 4, model.rollout(history, recorded actions)
  m1c    : the same rollout re-implemented step by step from the jointly encoded 8 frames (numerical check of m2 / m3)
  m2     : recursive rollout in which the proprioceptive channels of every fed-back prediction are replaced by the
           embedding of the RECORDED proprioception of that frame (agent position + velocity)
  m3     : one-step predictions with the REAL history (teacher forcing): frame k from the real frames k-3..k-1
  static : the last observed tokens (real, k = 0) repeated for k = 1..5
--out: probe readouts (y_<condition>, probe outputs in raw units), true states, token statistics, the agent position read
back from the proprioceptive channels, checks. --tok_out: the true frames k = 0..5 (uint8), the real tokens k = 0..5 and the
m1 / m2 / m3 tokens k = 1..5 converted back to the encoder's raw token space (z * sd + mu, float16), the input of
tools/render_decode.py. The paper ran this on the CPU (float32 throughout).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wm_predeval as W  # noqa: E402

CONDS = ("m1", "m1c", "m2", "m3", "static")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--tok_out", required=True)
    ap.add_argument("--stats", default=None); ap.add_argument("--data", default=None)
    ap.add_argument("--epoch", default="final"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n_windows", type=int, default=200)
    ap.add_argument("--bs", type=int, default=10); ap.add_argument("--device", default="cpu"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); dev = torch.device(a.device); K = W.K
    cfg = W.load_cfg(a.run, a.stats, a.data); n = int(cfg.num_hist); assert cfg.env.name == "pusht" and n == 3 and K == 5
    model = W.make_model(a.run, cfg, dev, a.epoch)
    enc = model.encoder; mu, sd = enc.mu.float().to(dev), enc.sd.float().to(dev)                     # (1, 1, D)
    D = int(enc.emb_dim); PD = int(model.proprio_dim); AD = int(model.action_dim); assert model.concat_dim == 1
    P = torch.load(a.probe, map_location="cpu"); assert P["env"] == "pusht"
    probe = W.AttnProbe(P["d_in"], P["n_tok"], P["n_out"]).to(dev); probe.load_state_dict(P["state_dict"]); probe.eval()
    pmu, psd = P["mu"].to(dev), P["sd"].to(dev)
    d = W.datasets(cfg, n, K, a.seed)["valid"]
    rng = np.random.default_rng(a.seed); idx = np.sort(rng.choice(len(d), min(a.n_windows, len(d)), replace=False))
    traj = np.array([int(d.slices[i][0]) for i in idx])
    if a.limit:
        idx, traj = idx[:a.limit], traj[:a.limit]                                                   # smoke test on the first windows only
    pm_mean, pm_std = d.dataset.proprio_mean.float().to(dev), d.dataset.proprio_std.float().to(dev)   # proprio normalisation of the dataset
    Wp = model.proprio_encoder.patch_embed.weight[:, :, 0].float(); bp = model.proprio_encoder.patch_embed.bias.float()   # emb = Wp p + bp
    Wp_pinv = torch.linalg.pinv(Wp)                                                                    # least-squares inverse

    @torch.no_grad()
    def readout(z):                                     # (B, T, P, D) standardised visual tokens -> probe outputs (B, T, 6) in raw units
        B, T = z.shape[:2]
        return (probe(z.flatten(0, 1)) * psd + pmu).view(B, T, -1)

    def proprio_agent(pe):                              # proprio embedding (B, T, PD) -> agent (x, y) in simulator units, by least squares
        p = (pe - bp) @ Wp_pinv.T
        return (p * pm_std + pm_mean)[..., :2]

    @torch.no_grad()
    def rollout_from(z_hist, act_next, prop_emb_next=None):
        """z_hist (B, n, P, D + PD + AD): embeddings of the n history frames; act_next (B, K - 1, A): recorded actions of the
        predicted frames k = 1..K-1; prop_emb_next (B, K - 1, PD) or None: if given, the proprio channels of each fed-back
        prediction are overwritten with the embedding of the recorded proprio (m2). Returns the K predictions (B, K, P, C)."""
        z = z_hist; out = []
        for t in range(K):
            zp = model.predict(z[:, -model.num_hist:])[:, -1:]
            out.append(zp.clone())
            if t < K - 1:
                zp = zp.clone()
                if prop_emb_next is not None:
                    zp[..., D:D + PD] = prop_emb_next[:, t:t + 1, None, :].expand(-1, -1, zp.shape[2], -1)
                zp = model.replace_actions_from_z(zp, act_next[:, t:t + 1])
                z = torch.cat([z, zp], dim=1)
        return torch.cat(out, dim=1)

    def tokstats(zc, zr):                                # (B, K, P, D) condition vs real (same k): mse, cosine, norm (means over patches)
        return (((zc - zr) ** 2).mean(dim=(2, 3)), F.cosine_similarity(zc, zr, dim=-1).mean(2), zc.norm(dim=-1).mean(2))

    R = {k: [] for k in ["state", "y_real", "tok_norm_real", "pagent_real"] + ["y_" + c for c in CONDS] + ["tok_mse_" + c for c in CONDS] +
         ["tok_cos_" + c for c in CONDS] + ["tok_norm_" + c for c in CONDS] + ["pagent_" + c for c in ("m1", "m1c", "m2", "m3")] +
         ["tok_step_real", "tok_spread_real"]}
    TOK = {"frames": [], "real": [], "m1": [], "m2": [], "m3": []}
    chk = {"m1_vs_m1c_maxabs_std": 0.0, "k1_m1c_vs_m2_maxabs": 0.0, "k1_m1c_vs_m3_maxabs": 0.0,
           "proprio_inverse_real_max_err_units": 0.0, "m1_proprio_patch_spread_mean": 0.0}
    t0 = time.time()
    for s in range(0, len(idx), a.bs):
        obs, act, st = W.collate([d[int(i)] for i in idx[s:s + a.bs]])
        obs = {k: v.to(dev) for k, v in obs.items()}; act = act.to(dev); st = st.to(dev).float(); B = act.shape[0]
        with torch.no_grad():
            z_real_all = W.encode_frames(model, obs, dev)                                         # (B, 8, P, D) standardised
            z_obs, _ = model.rollout({k: v[:, :n] for k, v in obs.items()}, act[:, :n + K - 1])
            m1 = z_obs["visual"][:, n:n + K].float(); m1_prop = z_obs["proprio"][:, n:n + K].float()
            z_all = model.encode(obs, act).float()                                                # (B, 8, P, D + PD + AD)
            prop_emb = model.encode_proprio(obs["proprio"]).float()                               # (B, 8, PD) embedding of the recorded proprio
            m1c_full = rollout_from(z_all[:, :n], act[:, n:n + K - 1])
            m2_full = rollout_from(z_all[:, :n], act[:, n:n + K - 1], prop_emb[:, n:n + K - 1])
            m3_full = torch.cat([model.predict(z_all[:, t:t + n])[:, -1:] for t in range(K)], dim=1)   # frame n + t from real frames t..t+n-1
            zr = z_real_all[:, n - 1:n + K]                                                       # real k = 0..5
            conds = {"m1": m1, "m1c": m1c_full[..., :D], "m2": m2_full[..., :D], "m3": m3_full[..., :D],
                     "static": zr[:, :1].expand(-1, K, -1, -1)}
            props = {"m1": m1_prop, "m1c": m1c_full[:, :, 0, D:D + PD], "m2": m2_full[:, :, 0, D:D + PD], "m3": m3_full[:, :, 0, D:D + PD]}
            chk["m1_vs_m1c_maxabs_std"] = max(chk["m1_vs_m1c_maxabs_std"], float((m1 - conds["m1c"]).abs().max()))
            chk["k1_m1c_vs_m2_maxabs"] = max(chk["k1_m1c_vs_m2_maxabs"], float((m1c_full[:, 0] - m2_full[:, 0]).abs().max()))
            chk["k1_m1c_vs_m3_maxabs"] = max(chk["k1_m1c_vs_m3_maxabs"], float((m1c_full[:, 0] - m3_full[:, 0]).abs().max()))
            pa_real = proprio_agent(prop_emb[:, n - 1:n + K])                                     # (B, 6, 2): must equal the true agent position
            chk["proprio_inverse_real_max_err_units"] = max(chk["proprio_inverse_real_max_err_units"], float((pa_real - st[:, n - 1:n + K, 0:2]).norm(dim=-1).max()))
            chk["m1_proprio_patch_spread_mean"] += float(m1c_full[..., D:D + PD].std(dim=2).mean()) * B
            R["state"].append(st.cpu()); R["y_real"].append(readout(zr).cpu())
            R["tok_norm_real"].append(zr.norm(dim=-1).mean(2).cpu())
            R["pagent_real"].append((pa_real - st[:, n - 1:n + K, 0:2]).norm(dim=-1).cpu())
            dz = z_real_all[:, 1:] - z_real_all[:, :-1]                                            # real change per world-model step, all 7 steps
            R["tok_step_real"].append((dz ** 2).mean(dim=(2, 3)).cpu())
            R["tok_spread_real"].append(((zr - zr.mean(dim=1, keepdim=True)) ** 2).mean(dim=(1, 2, 3)).cpu())   # within-window spread k = 0..5
            for c, zc in conds.items():
                R["y_" + c].append(readout(zc).cpu())
                m, co, no = tokstats(zc, zr[:, 1:]); R["tok_mse_" + c].append(m.cpu()); R["tok_cos_" + c].append(co.cpu()); R["tok_norm_" + c].append(no.cpu())
            for c, pe in props.items():
                R["pagent_" + c].append((proprio_agent(pe) - st[:, n:n + K, 0:2]).norm(dim=-1).cpu())
            fr = obs["visual"][:, n - 1:n + K].float()                                            # (B, K+1, 3, H, W) in [-1, 1]
            TOK["frames"].append(((fr * .5 + .5).clamp(0, 1) * 255).round().byte().permute(0, 1, 3, 4, 2).cpu().numpy())
            TOK["real"].append((zr * sd + mu).half().cpu().numpy())
            for c in ("m1", "m2", "m3"):
                TOK[c].append((conds[c] * sd + mu).half().cpu().numpy())
        if s == 0:
            print("first batch %.0f s" % (time.time() - t0), flush=True)
    chk["m1_proprio_patch_spread_mean"] /= len(idx)
    out = {k: torch.cat(v).numpy() for k, v in R.items()}
    meta = {"run": a.run, "epoch": a.epoch, "stats": str(cfg.encoder.stats), "encoder": str(cfg.encoder.encoder), "probe": a.probe, "seed": a.seed,
            "num_hist": n, "K": K, "frameskip": int(cfg.frameskip), "device": a.device, "threads": torch.get_num_threads(), "checks": chk,
            "state_layout": "agent x, agent y, block x, block y, block angle (rad), agent vx, agent vy; frames j = 0..7 = k + 2",
            "probe_output": "agent x, agent y, block x, block y, sin(angle), cos(angle)"}
    np.savez(a.out, **out, window=idx, traj=traj, meta=json.dumps(meta))
    np.savez(a.tok_out, **{c: np.concatenate(v) for c, v in TOK.items()}, window=idx, traj=traj,
             meta=json.dumps({"encoder": str(cfg.encoder.encoder), "run": a.run, "epoch": a.epoch, "token_space": "raw (z * sd + mu), float16"}))
    print("CHECKS " + json.dumps(chk), flush=True)
    print("RENDER_TOKENS_DONE %s windows %d %.0f s" % (os.path.basename(a.run.rstrip("/")), len(idx), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
