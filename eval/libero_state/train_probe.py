# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""LIBERO state-probe trainer (frozen encoder; only the attentive probe trains). One invocation = (encoder, suite, mode, horizon, seed).
mode=readout : tokens(t-n_hist+1..t)                          -> object state at t   (positions, orientations, predicates)
mode=future  : tokens(t-n_hist+1..t) + raw 20 Hz actions(2t+1..2j) -> object state at frame j = t + H (H = 1 s -> 10 frames / 3 s -> 30)
Object slots = union of object names in the suite (fixed order); presence mask per episode; losses/metrics only on present objects with
label_exact. Per-channel token standardisation from the TRAIN split of this encoder (cached). References written by the same run:
train-mean (positions/orientations) + majority class (predicates), hold-current (future mode), no-action (future mode, --no_action).
Outputs: heads/<tag>.pt, results/<tag>.json (per field, per task, changed-subset), results/<tag>_perep.json (per-episode metrics for pairing)."""
import argparse, json, os, sys, time, random, hashlib
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe import AttentiveProbe, orientation_error_deg, orientation_cos, rot6d_to_matrix, quat_to_matrix
HERE = os.path.dirname(os.path.abspath(__file__))


def load_manifest(a):
    man = json.load(open(a.manifest)); idx = json.load(open(os.path.join(a.episodes_dir, "episodes.json")))
    meta = {e["episode"]: e["meta"] for e in idx["episodes"]}; eps = {s: [e for e in man["episodes"][s] if meta[e]["suite"] == a.suite] for s in ("train", "train_f010", "val", "dev_test")}
    return man, meta, eps


def build_slots(lab_manifest, suite, lab_dir, eps_by_split):
    """Object slots = union of MOVABLE object names over the suite (fixed sorted order); symmetry per slot from the label manifest's
    object_symmetry (by obj_category); predicate set = union of pred_names over the suite. Scanned from the label files themselves."""
    names, cats, preds = set(), {}, set()
    for split, eps in eps_by_split.items():
        for e in eps:
            L = np.load(os.path.join(lab_dir, e + ".npz"), allow_pickle=True)
            for n, c, mv in zip(L["obj_names"], L["obj_category"], L["obj_movable"]):
                if bool(mv): names.add(str(n)); cats[str(n)] = str(c)
            preds.update(str(p) for p in L["pred_names"])
    names = sorted(names); sym_map = lab_manifest.get("object_symmetry", {})
    def sym_of(n):
        c = cats[n]; s = sym_map.get(c, sym_map.get(n, "none")); return s if isinstance(s, str) else s.get("symmetry_class", "none")
    return names, {n: sym_of(n) for n in names}, sorted(preds)


class EpisodeStore:
    """Standardised tokens (fp16) of a list of episodes, concatenated into ONE GPU tensor (fast gathers); labels stay on CPU."""
    def __init__(self, eps, enc_dir, lab_dir, ep_dir, slots, preds, mu, sd, dev="cuda"):
        self.eps, self.slots, self.preds, self.dev = eps, slots, preds, dev; self.lab, self.act, self.off, self.len = {}, {}, {}, {}; chunks = []; off = 0
        for e in eps:
            z = np.load(os.path.join(enc_dir, e + ".npz")); t = z["tokens"].astype(np.float32); t = np.clip((t - mu) / sd, -20, 20).astype(np.float16)
            chunks.append(torch.from_numpy(t)); self.off[e] = off; self.len[e] = len(t); off += len(t)
            L = np.load(os.path.join(lab_dir, e + ".npz"), allow_pickle=True); names = [str(n) for n in L["obj_names"]]; K = len(slots)
            pos = np.full((len(t), K, 3), np.nan, np.float32); quat = np.full((len(t), K, 4), np.nan, np.float32); present = np.zeros(K, bool)
            for j, n in enumerate(names):
                if n in slots and bool(L["obj_movable"][j]): k = slots.index(n); pos[:, k] = L["obj_pos"][:, j]; quat[:, k] = L["obj_quat_wxyz"][:, j]; present[k] = True
            pv = np.zeros((len(t), len(preds)), np.float32); pm = np.zeros(len(preds), bool)
            for j, p in enumerate([str(x) for x in L["pred_names"]]):
                if p in preds: pv[:, preds.index(p)] = L["pred_vals"][:, j]; pm[preds.index(p)] = True
            self.lab[e] = {"pos": pos, "quat": quat, "present": present, "pred": pv, "pred_mask": pm, "exact": L["label_exact"].astype(bool), "eef": L["eef_pos"].astype(np.float32)}
            self.act[e] = L["raw_actions"].astype(np.float32)
        big = torch.cat(chunks)                          # [sum T, N, D] fp16
        self.gpu_store = (big.numel() * 2 < 11e9) and os.environ.get("LIBERO_CPU_STORE", "0") != "1"
        self.big = big.to(dev) if self.gpu_store else big.pin_memory()   # large-token encoders (vjepa2) stay on CPU
    def n_frames(self): return int(self.big.shape[0])
    def tok(self, e): return self.big[self.off[e]: self.off[e] + self.len[e]]


def make_samples(store, mode, H, n_hist):
    S = []
    for e in store.eps:
        T = store.len[e]
        for i in range(T):
            j = i + H if mode == "future" else i
            if j >= T or not store.lab[e]["exact"][j]: continue
            S.append((e, i, j))
    return S


def batch_from(store, samples, mode, H, n_hist, act_len, dev):
    idx, acts, ys = [], [], {"pos": [], "quat": [], "pred": [], "present": [], "pred_mask": [], "cur_pos": [], "cur_quat": []}
    for e, i, j in samples:
        o = store.off[e]; idx.append([o + max(0, i - n_hist + 1 + k) for k in range(n_hist)]); L = store.lab[e]
        ys["pos"].append(L["pos"][j]); ys["quat"].append(L["quat"][j]); ys["pred"].append(L["pred"][j]); ys["present"].append(L["present"]); ys["pred_mask"].append(L["pred_mask"]); ys["cur_pos"].append(L["pos"][i]); ys["cur_quat"].append(L["quat"][i])
        if mode == "future":
            ra = store.act[e]; w = ra[2 * i + 1: 2 * j + 1]           # obs[2i] -> obs[2j] driven by actions[2i+1 .. 2j]
            if len(w) < act_len: w = np.concatenate([w, np.repeat(w[-1:], act_len - len(w), 0)]) if len(w) else np.zeros((act_len, ra.shape[1]), np.float32)
            acts.append(w[:act_len])
    ix = torch.tensor(idx, device=dev if store.gpu_store else 'cpu')
    X = (store.big[ix].to(dev) if not store.gpu_store else store.big[ix]).float()   # [B, n_hist, N, D]
    A = torch.from_numpy(np.stack(acts)).to(dev) if acts else None
    Y = {k: torch.from_numpy(np.stack(v)).to(dev) for k, v in ys.items()}
    return X, A, Y


def losses(out, Y, K, sym_list, w_rot=0.1, w_pred=1.0):
    pos = out["pos"].view(-1, K, 3); rot = out["rot"].view(-1, K, 6); m = Y["present"] & ~torch.isnan(Y["pos"][..., 0])
    l_pos = (F.l1_loss(pos, torch.nan_to_num(Y["pos"]), reduction="none").sum(-1) * m).sum() / m.sum().clamp(min=1)
    l_rot = ((1.0 - orientation_cos_batch(rot, Y["quat"], sym_list)) * m).sum() / m.sum().clamp(min=1)          # smooth 1 - cos (no acos in the graph)
    pm = Y["pred_mask"]; l_pred = (F.binary_cross_entropy_with_logits(out["pred"], Y["pred"], reduction="none") * pm).sum() / pm.sum().clamp(min=1)
    return l_pos + w_rot * l_rot + w_pred * l_pred, {"pos": float(l_pos), "rot": float(l_rot), "pred": float(l_pred)}


def orientation_cos_batch(rot6d, quat, sym_list):
    out = []
    for k, s in enumerate(sym_list):
        q = torch.nan_to_num(quat[:, k]); q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6); out.append(orientation_cos(rot6d[:, k], q, s))
    return torch.stack(out, 1)


def orientation_error_batch(rot6d, quat, sym_list):
    errs = []
    for k, s in enumerate(sym_list):
        q = torch.nan_to_num(quat[:, k]); q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6); errs.append(orientation_error_deg(rot6d[:, k], q, s))
    return torch.stack(errs, 1)


@torch.no_grad()
def evaluate(model, store, samples, mode, H, n_hist, act_len, K, sym_list, preds, meta, dev, bs=128, no_action=False):
    model.eval(); rec = []
    for b in range(0, len(samples), bs):
        sm = samples[b:b + bs]; X, A, Y = batch_from(store, sm, mode, H, n_hist, act_len, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16): out = model(X, None if no_action else A)
        out = {k: v.float() for k, v in out.items()}; pos = out["pos"].view(-1, K, 3); rot = out["rot"].view(-1, K, 6)
        m = (Y["present"] & ~torch.isnan(Y["pos"][..., 0])).cpu().numpy(); perr = (pos - torch.nan_to_num(Y["pos"])).norm(dim=-1).cpu().numpy() * 100.0
        rerr = orientation_error_batch(rot, Y["quat"], sym_list).cpu().numpy(); pp = (out["pred"] > 0).cpu().numpy(); pt = (Y["pred"] > 0.5).cpu().numpy(); pmask = Y["pred_mask"].cpu().numpy()
        moved = ((Y["pos"] - Y["cur_pos"]).norm(dim=-1) > 0.01).cpu().numpy() if mode == "future" else None
        hold_err = (Y["cur_pos"] - Y["pos"]).norm(dim=-1).cpu().numpy() * 100.0 if mode == "future" else None
        for n, (e, i, j) in enumerate(sm):
            rec.append({"ep": e, "task": meta[e]["task"], "i": i, "j": j, "pos_err": perr[n].tolist(), "rot_err": rerr[n].tolist(), "mask": m[n].tolist(), "moved": (moved[n].tolist() if moved is not None else None), "hold_err": (hold_err[n].tolist() if hold_err is not None else None), "pred_hit": (pp[n] == pt[n]).tolist(), "pred_true": pt[n].tolist(), "pred_pos": pp[n].tolist(), "pred_mask": pmask[n].tolist()})
    model.train(); return rec


def summarise(rec, slots, preds):
    def agg(sel):
        pe = [x["pos_err"][k] for x in sel for k in range(len(slots)) if x["mask"][k]]; re_ = [x["rot_err"][k] for x in sel for k in range(len(slots)) if x["mask"][k]]
        out = {"n_frames": len(sel), "pos_err_cm_mean": float(np.mean(pe)) if pe else None, "pos_err_cm_median": float(np.median(pe)) if pe else None, "rot_err_deg_mean": float(np.mean(re_)) if re_ else None}
        if sel and sel[0]["moved"] is not None:
            pm = [x["pos_err"][k] for x in sel for k in range(len(slots)) if x["mask"][k] and x["moved"][k]]; hm = [x["hold_err"][k] for x in sel for k in range(len(slots)) if x["mask"][k] and x["moved"][k]]; hall = [x["hold_err"][k] for x in sel for k in range(len(slots)) if x["mask"][k]]
            out.update({"changed_n": len(pm), "changed_pos_err_cm": float(np.mean(pm)) if pm else None, "changed_hold_current_err_cm": float(np.mean(hm)) if hm else None, "hold_current_err_cm_all": float(np.mean(hall)) if hall else None})
        f1s = []
        for p in range(len(preds)):
            tp = sum(1 for x in sel if x["pred_mask"][p] and x["pred_true"][p] and x["pred_pos"][p]); fp = sum(1 for x in sel if x["pred_mask"][p] and not x["pred_true"][p] and x["pred_pos"][p]); fn = sum(1 for x in sel if x["pred_mask"][p] and x["pred_true"][p] and not x["pred_pos"][p]); n = sum(1 for x in sel if x["pred_mask"][p])
            if n: f1s.append(2 * tp / max(1, 2 * tp + fp + fn) if (tp + fp + fn) else 1.0)
        out["pred_macro_f1"] = float(np.mean(f1s)) if f1s else None; out["per_object_pos_err_cm"] = {s: float(np.mean([x["pos_err"][k] for x in sel if x["mask"][k]])) for k, s in enumerate(slots) if any(x["mask"][k] for x in sel)}
        return out
    tasks = sorted({x["task"] for x in rec}); eps = sorted({x["ep"] for x in rec})
    return {"all": agg(rec), "per_task": {t: agg([x for x in rec if x["task"] == t]) for t in tasks}, "per_episode": {e: agg([x for x in rec if x["ep"] == e]) for e in eps}}


def main(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); dev = "cuda"
    man, meta, eps = load_manifest(a); labm = json.load(open(a.labels_manifest)); slots, sym, preds = build_slots(labm, a.suite, a.labels_dir, eps); K = len(slots); sym_list = [sym[s] for s in slots]
    enc_dir = os.path.join(a.cache_dir, a.encoder); tr_eps = eps["train_f010"] if a.budget == "f010" else eps["train"]
    stats_p = os.path.join(enc_dir, f"train_stats_{a.suite}_{a.budget}_{man['sha256_16']}.npz")
    if not os.path.exists(stats_p):
        xs = np.concatenate([np.load(os.path.join(enc_dir, e + ".npz"))["tokens"].astype(np.float32).reshape(-1, np.load(os.path.join(enc_dir, e + ".npz"))["tokens"].shape[-1]) for e in tr_eps[::max(1, len(tr_eps) // 60)]])
        np.savez(stats_p, mu=xs.mean(0), sd=np.maximum(xs.std(0), 1e-3), n=len(xs)); print(f"[stats] {stats_p} from {len(xs)} tokens", flush=True)
    st = np.load(stats_p); mu, sd = st["mu"][None, None], st["sd"][None, None]
    H = 0 if a.mode == "readout" else int(round(a.horizon_s * 10)); act_len = 0 if (a.mode == "readout" or a.no_action) else 2 * H
    stores = {s: EpisodeStore(eps[s] if s != "train" else tr_eps, enc_dir, a.labels_dir, a.episodes_dir, slots, preds, mu, sd) for s in ("train", "val", "dev_test")}
    samples = {s: make_samples(stores[s], a.mode, H, a.n_hist) for s in stores}
    d_in = int(stores["train"].big.shape[-1]); n_tok = int(stores["train"].big.shape[1])
    fields = {"pos": 3 * K, "rot": 6 * K, "pred": len(preds)}
    model = AttentiveProbe(d_in, n_tok, fields, n_hist=a.n_hist, act_dim=7 if act_len else 0, act_len=act_len, d=a.d, n_query=a.n_query, depth=a.depth).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    tag = f"{a.encoder}__{a.suite}__{a.mode}{'' if a.mode == 'readout' else f'_h{a.horizon_s:g}'}{'_noact' if a.no_action else ''}__{a.budget}__hist{a.n_hist}__s{a.seed}"
    print(f"[{tag}] train eps {len(tr_eps)} frames {stores['train'].n_frames()} samples {len(samples['train'])} | val {len(samples['val'])} dev_test {len(samples['dev_test'])} | K {K} preds {len(preds)} d_in {d_in} tokens {n_tok} | params {model.n_params()/1e6:.2f}M", flush=True)
    best, log, t0 = {"val": 1e9}, [], time.time(); rng = random.Random(a.seed); S = samples["train"]
    for step in range(1, a.steps + 1):
        sm = rng.sample(S, a.batch); X, A, Y = batch_from(stores["train"], sm, a.mode, H, a.n_hist, act_len, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16): out = model(X, A)
        out = {k: v.float() for k, v in out.items()}; loss, parts = losses(out, Y, K, sym_list); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % a.log_every == 0: log.append({"step": step, "loss": float(loss), **parts}); print(f"[{tag}] step {step} loss {float(loss):.4f} {parts} ({(time.time()-t0)/60:.1f} min)", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            vs = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for b in range(0, len(samples["val"]), 128):
                    Xv, Av, Yv = batch_from(stores["val"], samples["val"][b:b + 128], a.mode, H, a.n_hist, act_len, dev); ov = {k: v.float() for k, v in model(Xv, Av).items()}; vs.append(float(losses(ov, Yv, K, sym_list)[0]))
            v = float(np.mean(vs)); log.append({"step": step, "val_loss": v})
            if v < best["val"]: best = {"val": v, "step": step}; torch.save(model.state_dict(), os.path.join(a.out_dir, "heads", tag + ".pt"))
            print(f"[{tag}] step {step} VAL {v:.4f} best {best}", flush=True)
    model.load_state_dict(torch.load(os.path.join(a.out_dir, "heads", tag + ".pt"))); rec = evaluate(model, stores["dev_test"], samples["dev_test"], a.mode, H, a.n_hist, act_len, K, sym_list, preds, meta, dev)
    summ = summarise(rec, slots, preds)
    # references: train-mean / majority (same eval frames)
    trY = {"pos": np.nanmean(np.concatenate([stores["train"].lab[e]["pos"] for e in tr_eps]), 0), "pred": np.mean(np.concatenate([stores["train"].lab[e]["pred"] for e in tr_eps]), 0)}
    ref = []
    for x in rec:
        L = stores["dev_test"].lab[x["ep"]]; gt = L["pos"][x["j"]]; pe = np.linalg.norm(np.nan_to_num(trY["pos"]) - np.nan_to_num(gt), axis=-1) * 100
        ref.append({**x, "pos_err": pe.tolist(), "rot_err": [90.0] * K, "pred_pos": (trY["pred"] > 0.5).tolist()})
    summ["reference_train_mean"] = summarise(ref, slots, preds)["all"]
    res = {"tag": tag, "encoder": a.encoder, "suite": a.suite, "mode": a.mode, "horizon_s": a.horizon_s, "budget": a.budget, "seed": a.seed, "n_hist": a.n_hist, "no_action": a.no_action, "slots": slots, "symmetry": sym_list, "preds": preds, "d_in": d_in, "n_tokens": n_tok, "probe_params": model.n_params(), "best": best, "train_log": log, "dev_test": summ, "minutes": (time.time() - t0) / 60, "config": vars(a)}
    os.makedirs(os.path.join(a.out_dir, "results"), exist_ok=True); json.dump(res, open(os.path.join(a.out_dir, "results", tag + ".json"), "w"), indent=1); json.dump(rec, open(os.path.join(a.out_dir, "results", tag + "_persample.json"), "w"))
    print(f"[{tag}] DONE dev_test {json.dumps(summ['all'])} | ref {json.dumps(summ['reference_train_mean'])} | {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True); ap.add_argument("--suite", required=True, choices=["libero_spatial", "libero_object"]); ap.add_argument("--mode", required=True, choices=["readout", "future"]); ap.add_argument("--horizon_s", type=float, default=0.0)
    ap.add_argument("--budget", default="f100", choices=["f100", "f010"]); ap.add_argument("--n_hist", type=int, default=1); ap.add_argument("--no_action", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", default=os.path.join(HERE, "splits", "libero_probe_split.json")); ap.add_argument("--labels_manifest", required=True); ap.add_argument("--labels_dir", required=True); ap.add_argument("--episodes_dir", required=True); ap.add_argument("--cache_dir", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--batch", type=int, default=128); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--d", type=int, default=256); ap.add_argument("--n_query", type=int, default=8); ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--log_every", type=int, default=250); ap.add_argument("--eval_every", type=int, default=1000)
    a = ap.parse_args(); os.makedirs(os.path.join(a.out_dir, "heads"), exist_ok=True); main(a)
