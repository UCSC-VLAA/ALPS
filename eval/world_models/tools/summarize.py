# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""World-model tables of the ALPS paper from the evaluation outputs, next to the paper's values.

  python summarize.py --dino_wm $WM_ROOT/results --reacher $REACHER_OUT/results [--json out.json]

Inputs (written by the scripts of dino_wm/ and reacher/):
  <dino_wm>/plan/<task>_<encoder>.json          merge_plan_shards.py (pusht, wall, point_maze, granular, rope)
  <dino_wm>/plan/<task>_dinov2_vits14_dinowm_recipe.json   (optional) the DINO-WM-recipe check of Sec. S2 / Table S11
  <dino_wm>/pred/<task>_<encoder>.npz           wm_rollout_eval.py (pusht, wall, granular, rope)
  <reacher>/plan/reacher_<encoder>_cem<seed>.json  merge_reacher.py (planner seeds 42, 43, 44)
  <reacher>/pred/reacher_<encoder>.npz          reacher_rollout_eval.py
Tables: 3 (planning), 4 (prediction 5 steps ahead), S11 (Rope mean / median), S12 (Reacher per planner seed), S16
(prediction at every step and the two references) and the paired comparisons of ALPS with each
encoder (the paper states their readings, not the table). Statistics: prediction differences paired on identical windows with a bootstrap over
trajectories (Reacher: episodes), 4,000 resamples, numpy default_rng(1); planning success with an exact McNemar test over
goals (PushT, Wall, PointMaze) or an exact sign-flip test over episodes of the successes summed over the three planner seeds
(Reacher); Chamfer distances paired over instances with a bootstrap, 4,000 resamples, default_rng(0). A prediction
difference counts when its interval excludes zero and it exceeds 10% of the other encoder's error.
"""
import argparse
import json
import os
from math import comb

import numpy as np

B = 4000
ENC = [("dinov2_b14", "DINOv2-B/14"), ("dinov3_b16", "DINOv3-B/16"), ("cradio_v3_b", "C-RADIOv3-B"), ("siglip2_b16", "SigLIP 2-B/16"),
       ("alps_stage1", "ALPS (Stage 1)"), ("alps", "ALPS")]
PRED = [("PushT", "pusht", "block"), ("Wall", "wall", "agent"), ("Granular", "granular", "chamfer"), ("Rope", "rope", "chamfer"),
        ("Reacher", "reacher", "joints_deg")]

# values of the paper (tab/*.tex), for the comparison printed next to the reproduced ones
PAPER = {
    "planning": {  # PushT success, state dist., Wall success, state dist., PointMaze success, Reacher success, Granular CD, Rope CD
        "dinov2_b14": [88, 25.0, 92, 3.62, 100, 76.7, .211, .701], "dinov3_b16": [88, 24.2, 90, 3.69, 100, 86.0, .275, .681],
        "cradio_v3_b": [90, 20.2, 90, 3.31, 100, 82.0, .180, .558], "siglip2_b16": [84, 29.0, 88, 3.73, 100, 74.0, .230, .848],
        "alps_stage1": [88, 24.0, 90, 3.43, 100, 76.0, .324, .943], "alps": [90, 23.4, 92, 3.21, 100, 78.7, .176, .750]},
    "rope": {"dinov2_b14": [.701, .446], "dinov3_b16": [.681, .406], "cradio_v3_b": [.558, .289], "siglip2_b16": [.848, .511],
             "alps_stage1": [.943, .514], "alps": [.750, .448], "dinov2_vits14_dinowm_recipe": [.547, .309]},
    "reacher": {"dinov2_b14": [72, 80, 78, 76.7], "dinov3_b16": [90, 90, 78, 86.0], "cradio_v3_b": [82, 86, 78, 82.0],
                "siglip2_b16": [68, 78, 76, 74.0], "alps_stage1": [70, 78, 80, 76.0], "alps": [82, 80, 74, 78.7]},
    "pred_full": {  # k = 1..5, true tokens (k = 5), last observation (k = 5); Table 4 = the k = 5 column
        "PushT": {"dinov2_b14": [4.29, 4.47, 5.16, 6.08, 7.16, 3.24, 38.27], "dinov3_b16": [3.87, 4.38, 5.47, 7.04, 8.70, 2.93, 37.87],
                  "cradio_v3_b": [3.51, 3.66, 4.21, 4.85, 5.64, 2.66, 38.14], "siglip2_b16": [6.17, 6.87, 8.21, 9.97, 11.96, 3.52, 38.02],
                  "alps_stage1": [5.49, 5.66, 6.58, 7.77, 9.32, 3.36, 38.13], "alps": [4.47, 4.74, 5.29, 5.85, 6.73, 2.85, 37.87]},
        "Wall": {"dinov2_b14": [6.74, 7.47, 8.22, 8.94, 9.69, 9.74, 10.73], "dinov3_b16": [6.73, 7.48, 8.26, 9.00, 9.78, 9.79, 10.72],
                 "cradio_v3_b": [6.73, 7.45, 8.24, 8.98, 9.75, 9.75, 10.71], "siglip2_b16": [6.90, 7.57, 8.42, 9.16, 9.91, 9.77, 10.75],
                 "alps_stage1": [6.71, 7.46, 8.28, 9.03, 9.81, 9.75, 10.73], "alps": [6.71, 7.43, 8.21, 8.94, 9.69, 9.76, 10.71]},
        "Granular": {"dinov2_b14": [.215, .225, .233, .240, .246, .225, .599], "dinov3_b16": [.192, .202, .210, .217, .222, .199, .602],
                     "cradio_v3_b": [.192, .201, .208, .214, .218, .200, .600], "siglip2_b16": [.228, .243, .253, .261, .268, .231, .603],
                     "alps_stage1": [.229, .244, .255, .265, .272, .223, .595], "alps": [.203, .214, .223, .231, .236, .208, .597]},
        "Rope": {"dinov2_b14": [.142, .204, .265, .335, .410, .055, 1.674], "dinov3_b16": [.133, .195, .259, .330, .407, .052, 1.680],
                 "cradio_v3_b": [.126, .179, .240, .299, .372, .051, 1.678], "siglip2_b16": [.178, .253, .320, .399, .469, .066, 1.669],
                 "alps_stage1": [.161, .228, .295, .360, .424, .056, 1.674], "alps": [.135, .195, .248, .313, .377, .053, 1.678]},
        "Reacher": {"dinov2_b14": [.90, .99, 1.08, 1.14, 1.21, .93, 25.23], "dinov3_b16": [.70, .81, .88, .95, 1.03, .75, 25.23],
                    "cradio_v3_b": [.81, .89, .95, 1.01, 1.06, .80, 25.21], "siglip2_b16": [1.10, 1.39, 1.62, 1.84, 2.02, 1.00, 25.24],
                    "alps_stage1": [.90, 1.10, 1.20, 1.30, 1.42, .89, 25.22], "alps": [.83, .93, 1.02, 1.09, 1.18, .79, 25.22]}},
    "paired_pred": {  # ALPS minus the encoder at k = 5: difference [95% interval]
        "dinov2_b14": {"PushT": (-0.43, -2.16, 0.89), "Wall": (-0.00, -0.05, 0.05), "Granular": (-.010, -.012, -.008), "Rope": (-.033, -.058, -.007), "Reacher": (-0.03, -0.08, 0.03)},
        "dinov3_b16": {"PushT": (-1.97, -4.03, -0.30), "Wall": (-0.09, -0.14, -0.04), "Granular": (.014, .012, .016), "Rope": (-.030, -.056, -.003), "Reacher": (0.15, 0.10, 0.20)},
        "cradio_v3_b": {"PushT": (1.09, 0.13, 2.09), "Wall": (-0.07, -0.11, -0.02), "Granular": (.018, .016, .020), "Rope": (.006, -.017, .029), "Reacher": (0.12, 0.08, 0.17)},
        "siglip2_b16": {"PushT": (-5.23, -7.65, -3.31), "Wall": (-0.23, -0.33, -0.13), "Granular": (-.032, -.036, -.029), "Rope": (-.092, -.121, -.060), "Reacher": (-0.84, -0.90, -0.77)},
        "alps_stage1": {"PushT": (-2.59, -4.56, -1.30), "Wall": (-0.12, -0.18, -0.07), "Granular": (-.036, -.040, -.033), "Rope": (-.046, -.072, -.021), "Reacher": (-0.24, -0.28, -0.19)}},
    "paired_plan": {  # ALPS minus the encoder: success points with exact p, Chamfer difference [95% interval]
        "dinov2_b14": {"PushT": (2.0, 1), "Wall": (0.0, 1), "PointMaze": (0.0, 1), "Reacher": (2.0, .714), "Granular": (-.035, -.128, .036), "Rope": (.049, -.177, .280)},
        "dinov3_b16": {"PushT": (2.0, 1), "Wall": (2.0, 1), "PointMaze": (0.0, 1), "Reacher": (-7.3, .118), "Granular": (-.099, -.185, -.028), "Rope": (.069, -.138, .262)},
        "cradio_v3_b": {"PushT": (0.0, 1), "Wall": (2.0, 1), "PointMaze": (0.0, 1), "Reacher": (-3.3, .525), "Granular": (-.004, -.044, .037), "Rope": (.192, .017, .381)},
        "siglip2_b16": {"PushT": (6.0, .375), "Wall": (4.0, .625), "PointMaze": (0.0, 1), "Reacher": (4.7, .283), "Granular": (-.054, -.112, -.001), "Rope": (-.098, -.364, .144)},
        "alps_stage1": {"PushT": (2.0, 1), "Wall": (2.0, 1), "PointMaze": (0.0, 1), "Reacher": (2.7, .600), "Granular": (-.148, -.273, -.045), "Rope": (-.193, -.460, .065)}},
}


def reacher_name(e):
    return "dinov2_b14_196" if e == "dinov2_b14" else e


def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def load_npz(path):
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def boot(x, traj, seed):
    """mean and 95% interval of x, trajectories (clusters) resampled with replacement"""
    ts = np.unique(traj); groups = [np.nonzero(traj == t)[0] for t in ts]; g = np.random.default_rng(seed)
    m = [x[np.concatenate([groups[i] for i in g.integers(0, len(ts), len(ts))])].mean() for _ in range(B)]
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_pair(x, y, seed=0):
    d = np.asarray(x, float) - np.asarray(y, float); g = np.random.default_rng(seed)
    m = [d[g.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def mcnemar(a, b):
    """exact two-sided McNemar test of paired successes"""
    a, b = np.asarray(a, bool), np.asarray(b, bool); n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    return n10, n01, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def signflip_p(d):
    """exact two-sided sign-flip test of the mean of integer differences"""
    dist = {0: 1}
    for x in d:
        nd = {}
        for s_, c in dist.items():
            for t in (s_ + x, s_ - x):
                nd[t] = nd.get(t, 0) + c
        dist = nd
    tot = sum(dist.values()); obs = abs(sum(d))
    return sum(c for s_, c in dist.items() if abs(s_) >= obs) / tot


def fmt(v, nd):
    if v is None:
        return "-"
    s = "%.*f" % (nd, v)
    return s[1:] if s.startswith("0.") else ("-" + s[2:] if s.startswith("-0.") else s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dino_wm", required=True, help="results folder of the DINO-WM scripts ($WM_ROOT/results)")
    ap.add_argument("--reacher", required=True, help="results folder of the Reacher scripts ($REACHER_OUT/results)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = {}

    # ---------------------------------------------------------------- planning (Table 3, S11, S12)
    plan = {e: {t: load_json(os.path.join(a.dino_wm, "plan", f"{t}_{e}.json")) for t in ("pusht", "wall", "point_maze", "granular", "rope")} for e, _ in ENC}
    reach = {e: [load_json(os.path.join(a.reacher, "plan", f"reacher_{reacher_name(e)}_cem{s}.json")) for s in (42, 43, 44)] for e, _ in ENC}
    print("Table 3: planning (reproduced | paper)")
    print("%-16s %21s %21s %10s %10s %16s %16s" % ("encoder", "PushT succ / dist", "Wall succ / dist", "PointMaze", "Reacher", "Granular CD", "Rope CD"))
    T3 = {}
    for e, lab in ENC:
        p = plan[e]
        sr = lambda t: 100 * p[t]["success_rate"] if p[t] and "success_rate" in p[t] else None  # noqa: E731
        sd = lambda t: p[t]["mean_state_dist"] if p[t] and "mean_state_dist" in p[t] else None  # noqa: E731
        cd = lambda t: p[t]["mean_cd"] if p[t] and "mean_cd" in p[t] else None  # noqa: E731
        rs = [100 * r["success_rate"] for r in reach[e] if r and r.get("success_rate") is not None]
        rm = float(np.mean(rs)) if len(rs) == 3 else None
        row = [sr("pusht"), sd("pusht"), sr("wall"), sd("wall"), sr("point_maze"), rm, cd("granular"), cd("rope")]
        T3[e] = row; P = PAPER["planning"][e]
        print("%-16s %4s %5s | %2d %4.1f  %4s %5s | %2d %4.2f  %3s | %3d  %4s | %4.1f  %4s | %4s  %4s | %4s" % (
            lab, fmt(row[0], 0), fmt(row[1], 1), P[0], P[1], fmt(row[2], 0), fmt(row[3], 2), P[2], P[3], fmt(row[4], 0), P[4],
            fmt(row[5], 1), P[5], fmt(row[6], 3), fmt(P[6], 3), fmt(row[7], 3), fmt(P[7], 3)))
    out["table3"] = T3
    print("\nTable S11: Rope, 50 instances, mean / median Chamfer (reproduced | paper)")
    for e, lab in ENC + [("dinov2_vits14_dinowm_recipe", "DINOv2-S/14, DINO-WM recipe")]:
        r = plan[e]["rope"] if e in plan else load_json(os.path.join(a.dino_wm, "plan", f"rope_{e}.json"))
        P = PAPER["rope"][e]
        print("%-30s %5s %5s | %s %s" % (lab, fmt(r["mean_cd"], 3) if r else "-", fmt(r["median_cd"], 3) if r else "-", fmt(P[0], 3), fmt(P[1], 3)))
        if e == "dinov2_vits14_dinowm_recipe" and r and len(r.get("cd", [])) >= 10:
            print("%-30s first 10 instances %.3f (paper .567)" % ("", float(np.mean(r["cd"][:10]))))
    for t, paper in (("pusht", 94), ("wall", 92)):
        r = load_json(os.path.join(a.dino_wm, "plan", f"{t}_dinov2_vits14_dinowm_recipe.json"))
        if r and "success_rate" in r:
            print("DINO-WM-recipe check %s: success %.0f%% (paper %d%%)" % (t, 100 * r["success_rate"], paper))
    print("\nTable S12: Reacher success per planner seed 42 / 43 / 44, mean (reproduced | paper)")
    for e, lab in ENC:
        rs = [None if r is None else 100 * r["success_rate"] for r in reach[e]]
        P = PAPER["reacher"][e]
        print("%-16s %s  mean %s | %s mean %.1f" % (lab, " ".join(fmt(x, 0) for x in rs), fmt(np.mean(rs), 1) if None not in rs else "-",
                                                  " ".join("%d" % x for x in P[:3]), P[3]))
        same_rows = [r["rows"] for r in reach[e] if r]
        if len(same_rows) > 1 and any(x != same_rows[0] for x in same_rows):
            print("   WARNING: the planner seeds evaluated different episodes")

    # ---------------------------------------------------------------- prediction (Table 4, S16)
    pred = {}
    for tn, t, key in PRED:
        for e, _ in ENC:
            f = os.path.join(a.reacher, "pred", f"reacher_{reacher_name(e)}.npz") if t == "reacher" else os.path.join(a.dino_wm, "pred", f"{t}_{e}.npz")
            pred[(tn, e)] = load_npz(f)
    print("\nTable 4 / S16: state error read from predicted tokens, k = 1..5, true tokens (k = 5), last observation (k = 5); "
          "reproduced, then [paper]")
    S16 = {}
    for tn, t, key in PRED:
        print(f"-- {tn} ({key})")
        for e, lab in ENC:
            r = pred[(tn, e)]; P = PAPER["pred_full"][tn][e]
            nd = 3 if tn in ("Granular", "Rope") else 2
            if r is None:
                print("%-16s -  [%s]" % (lab, " ".join(fmt(x, nd) for x in P))); continue
            row = [float(r[f"pred_{key}"][:, k].mean()) for k in range(5)] + [float(r[f"real_{key}"][:, -1].mean()), float(r[f"static_tok_{key}"][:, -1].mean())]
            S16[(tn, e)] = row
            print("%-16s %s  [%s]  (%d windows)" % (lab, " ".join(fmt(x, nd) for x in row), " ".join(fmt(x, nd) for x in P), len(r["window"])))
    out["table_s16"] = {f"{tn}|{e}": v for (tn, e), v in S16.items()}

    # ---------------------------------------------------------------- paired comparisons (readings stated in the paper, no table)
    print("\nPaired comparisons (world models): ALPS minus each encoder; reproduced, then [reference]")
    PC = {}
    for tn, t, key in PRED:
        A = pred[(tn, "alps")]
        if A is None:
            continue
        for e, lab in ENC[:-1]:
            Bm = pred[(tn, e)]
            if Bm is None:
                continue
            assert np.array_equal(A["window"], Bm["window"]) and np.array_equal(A["traj"], Bm["traj"]), (tn, e, "windows differ")
            d = A[f"pred_{key}"][:, -1] - Bm[f"pred_{key}"][:, -1]
            m, lo, hi = boot(d, A["traj"], 1); ref = float(Bm[f"pred_{key}"][:, -1].mean())
            rule = ("better" if m < 0 else "worse") if ((lo > 0 or hi < 0) and abs(m) > 0.1 * ref) else "does not meet the rule"
            PC[f"pred|{tn}|{e}"] = {"diff": m, "ci": [lo, hi], "rel_pct": 100 * m / ref, "reading": rule}
            P = PAPER["paired_pred"][e][tn]
            print("prediction %-8s - %-15s %+.4f [%+.4f, %+.4f] (%+.0f%%) %-22s [ref %+g [%+g, %+g]]" % (tn, lab, m, lo, hi, 100 * m / ref, rule, *P))

    def units(task, e):
        if task == "reacher":
            x = reach[e]
            return None if any(v is None or v["successes"] is None for v in x) else np.sum([np.asarray(v["successes"], int) for v in x], 0)
        r = plan[e][task]
        if not r or r.get("missing") or not r.get("n"):
            return None
        return np.asarray(r["success"] if task in ("pusht", "wall", "point_maze") else r["cd"], float)
    for tn, task in (("PushT", "pusht"), ("Wall", "wall"), ("PointMaze", "point_maze"), ("Reacher", "reacher"), ("Granular", "granular"), ("Rope", "rope")):
        A_ = units(task, "alps")
        if A_ is None:
            continue
        for e, lab in ENC[:-1]:
            B_ = units(task, e)
            if B_ is None:
                continue
            P = PAPER["paired_plan"][e][tn]
            if task == "reacher":
                p = signflip_p([int(v) for v in A_ - B_]); m = 100 * float(np.mean(A_ - B_)) / 3
                PC[f"plan|{tn}|{e}"] = {"diff_points": m, "p": p}
                print("planning   %-9s - %-15s %+.1f points, sign-flip p = %.3f   [ref %+g, p = %g]" % (tn, lab, m, p, *P))
            elif task in ("pusht", "wall", "point_maze"):
                n10, n01, p = mcnemar(A_, B_); m = 100 * float(np.mean(A_ - B_))
                PC[f"plan|{tn}|{e}"] = {"diff_points": m, "p": p, "alps_only": n10, "other_only": n01}
                print("planning   %-9s - %-15s %+.1f points (ALPS only %d, other only %d), McNemar p = %.3f   [ref %+g, p = %g]" % (tn, lab, m, n10, n01, p, *P))
            else:
                m, lo, hi = boot_pair(A_, B_)
                PC[f"plan|{tn}|{e}"] = {"diff": m, "ci": [lo, hi]}
                print("planning   %-9s - %-15s %+.3f [%+.3f, %+.3f]   [ref %+g [%+g, %+g]]" % (tn, lab, m, lo, hi, *P))
    out["paired"] = PC
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
