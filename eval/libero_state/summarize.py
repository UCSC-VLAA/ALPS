# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate LIBERO state-probe results: one row per (encoder, suite, mode / horizon) with references, and paired-by-episode
differences vs a reference encoder (default alps_stage1) with 95% bootstrap CIs over the dev_test episodes.
Usage: python summarize.py <out_dir of train_probe.py> [reference encoder]"""
import json, os, sys, glob, numpy as np
D = sys.argv[1]; REF = sys.argv[2] if len(sys.argv) > 2 else "alps_stage1"
R = {}
for f in sorted(glob.glob(os.path.join(D, "results", "*.json"))):
    if f.endswith("_persample.json"): continue
    r = json.load(open(f)); key = (r["suite"], r["mode"] + ("" if r["mode"] == "readout" else f"_h{r['horizon_s']:g}") + ("_noact" if r["no_action"] else "")); R.setdefault(key, {})[r["encoder"]] = r
rows = []
for (suite, cell), encs in sorted(R.items()):
    for enc, r in sorted(encs.items()):
        a = r["dev_test"]["all"]; ref = r["dev_test"].get("reference_train_mean", {})
        rows.append({"suite": suite, "cell": cell, "encoder": enc, "n": a["n_frames"], "pos_cm": a["pos_err_cm_mean"], "pos_cm_median": a["pos_err_cm_median"], "rot_deg": a["rot_err_deg_mean"], "f1": a["pred_macro_f1"], "changed_pos_cm": a.get("changed_pos_err_cm"), "hold_changed_cm": a.get("changed_hold_current_err_cm"), "hold_all_cm": a.get("hold_current_err_cm_all"), "ref_mean_pos_cm": ref.get("pos_err_cm_mean"), "ref_mean_f1": ref.get("pred_macro_f1"), "probe_params_M": r["probe_params"] / 1e6, "d_in": r["d_in"], "tokens": r["n_tokens"], "best_step": r["best"]["step"], "minutes": r["minutes"]})
print("%-15s %-16s %-16s %6s %8s %8s %7s %7s %9s %9s %9s %8s"%("suite","cell","encoder","n","pos_cm","pos_med","rot","f1","chg_pos","hold_chg","ref_pos","ref_f1"))
for x in rows: print("%-15s %-16s %-16s %6d %8.2f %8.2f %7.2f %7.3f %9s %9s %9s %8s"%(x["suite"][7:], x["cell"], x["encoder"], x["n"], x["pos_cm"], x["pos_cm_median"], x["rot_deg"] or -1, x["f1"] or -1, "%.2f"%x["changed_pos_cm"] if x["changed_pos_cm"] else "-", "%.2f"%x["hold_changed_cm"] if x["hold_changed_cm"] else "-", "%.2f"%x["ref_mean_pos_cm"] if x["ref_mean_pos_cm"] else "-", "%.3f"%x["ref_mean_f1"] if x["ref_mean_f1"] else "-"))
# paired by episode vs REF (dev_test): position error per episode
paired = {}
for (suite, cell), encs in sorted(R.items()):
    if REF not in encs: continue
    pe_ref = {e: v["pos_err_cm_mean"] for e, v in encs[REF]["dev_test"]["per_episode"].items()}
    for enc, r in encs.items():
        if enc == REF: continue
        pe = {e: v["pos_err_cm_mean"] for e, v in r["dev_test"]["per_episode"].items()}; eps = [e for e in pe if e in pe_ref and pe[e] is not None and pe_ref[e] is not None]
        d = np.array([pe[e] - pe_ref[e] for e in eps]); rng = np.random.default_rng(0); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        paired[f"{suite}|{cell}|{enc}-{REF}"] = {"mean_pos_cm_diff": round(float(d.mean()), 3), "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], "n_eps": len(eps), "eps_better": int((d < 0).sum()), "eps_worse": int((d > 0).sum())}
print("\nPAIRED position error (cm) vs %s, by dev_test episode (negative = better than ref):" % REF)
for k, v in paired.items(): print("  %-60s %+.3f [%+.3f,%+.3f] better/worse %d/%d of %d" % (k, v["mean_pos_cm_diff"], v["ci95"][0], v["ci95"][1], v["eps_better"], v["eps_worse"], v["n_eps"]))
json.dump({"rows": rows, "paired_vs_ref": paired, "ref": REF}, open(os.path.join(D, "summary_libero.json"), "w"), indent=1)
