# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""Tables S15 (rendered predictions on PushT) and S17 (readout, one-step prediction and rollout) from the outputs of
dino_wm render_tokens.py (<enc>_diag.npz) and tools/render_decode.py (render_<enc>.npz), plus the windows of the motion rule
of Fig. 6 and Figs. S7 to S9.

  python render_stats.py --diag_dir <dir with <enc>_diag.npz> --render_dir <dir with render_<enc>.npz> [--boot 10000]

Encoders: ALPS, ALPS (Stage 1), C-RADIOv3-B, DINOv3-B/16, SigLIP 2-B/16 (DINOv2 is not rendered: its world model sees
196-pixel frames, which its evaluation decoder does not match). Intervals: the 21 validation trajectories resampled with
replacement (every window weighted by the count of its trajectory), 10,000 resamples drawn once with numpy default_rng(0)
and shared by all statistics; percentile intervals of the difference of the window means.
Conditions of Table S17 (block error read by the probe 5 world-model steps ahead): true future tokens (real), one-step
prediction with the true history (m3), five-step open-loop rollout (m1), rollout with the recorded proprioception (m2),
last observation repeated (static).
Motion rule (fixed on the true states before any rendering): block displacement of at least 40 units or a rotation of at
least 20 degrees from k = 0 to k = 5, ranked by displacement + 2 x |rotation in degrees|.
"""
import argparse
import math
import os

import numpy as np

ENC = [("alps", "ALPS"), ("alps_stage1", "ALPS (Stage 1)"), ("cradio_v3_b", "C-RADIOv3-B"), ("dinov3_b16", "DINOv3-B/16"), ("siglip2_b16", "SigLIP 2-B/16")]
PAPER_S15 = {"siglip2_b16": [1.000, .0081, .0107, .0150, .0308], "cradio_v3_b": [1.000, .0070, .0088, .0103, .0176],
             "dinov3_b16": [1.000, .0074, .0096, .0128, .0242], "alps_stage1": [1.000, .0066, .0093, .0126, .0248], "alps": [1.000, .0068, .0090, .0114, .0203]}
# paired k = 5 LPIPS of ALPS against each encoder: our reference values (the paper does not print these intervals)
PAPER_S15_PAIRS = {"cradio_v3_b": (.0028, .0013, .0042, 16), "dinov3_b16": (-.0038, -.0068, -.0014, -16), "alps_stage1": (-.0044, -.0061, -.0027, -18),
                   "siglip2_b16": (-.0104, -.0135, -.0079, -34)}
PAPER_S17 = {"alps": [3.1, 3.6, 6.6, 6.7, 42.0], "alps_stage1": [3.8, 4.3, 9.9, 9.2, 42.3], "cradio_v3_b": [2.4, 2.8, 5.8, 5.1, 41.9],
             "dinov3_b16": [3.0, 3.2, 9.1, 8.4, 41.7], "siglip2_b16": [3.7, 5.2, 14.1, 12.1, 42.0]}
PAPER_S17_PAIR = [(.68, .26, 1.22), (.73, .38, 1.09), (.76, -.04, 1.49), (1.60, .42, 2.97), (.13, -.60, .80)]
COND = [("real", "y_real"), ("onestep", "y_m3"), ("rollout", "y_m1"), ("rollout_recprop", "y_m2"), ("static", "y_static")]


def wrap_deg(x):
    return np.abs((x + math.pi) % (2 * math.pi) - math.pi) * 180 / math.pi


def block_err(y, s):
    """probe outputs (N, T, 6) and true states (N, T, 7) -> block position error (N, T)"""
    return np.linalg.norm(y[..., 2:4] - s[..., 2:4], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag_dir", required=True); ap.add_argument("--render_dir", required=True); ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    D, R = {}, {}
    for e, _ in ENC:
        D[e] = np.load(os.path.join(a.diag_dir, f"{e}_diag.npz"), allow_pickle=True)
        f = os.path.join(a.render_dir, f"render_{e}.npz")
        R[e] = np.load(f, allow_pickle=True) if os.path.exists(f) else None
    W0, T0, S0 = D["alps"]["window"], D["alps"]["traj"], D["alps"]["state"]
    for e, _ in ENC:
        assert np.array_equal(D[e]["window"], W0) and np.allclose(D[e]["state"], S0), e
        if R[e] is not None:
            assert np.array_equal(R[e]["window"], W0) and np.array_equal(R[e]["traj"], T0), e
    ut, ti = np.unique(T0, return_inverse=True)
    rng = np.random.default_rng(0)
    CNT = np.stack([np.bincount(rng.integers(0, len(ut), len(ut)), minlength=len(ut)) for _ in range(a.boot)]).astype(np.float64)
    WB = CNT[:, ti]

    def ci(d):
        bs = (WB @ d) / WB.sum(1)
        return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    # ---- motion rule
    disp = np.linalg.norm(S0[:, 7, 2:4] - S0[:, 2, 2:4], axis=-1)
    rot = ((S0[:, 7, 4] - S0[:, 2, 4] + math.pi) % (2 * math.pi) - math.pi) * 180 / math.pi
    moving = (disp >= 40) | (np.abs(rot) >= 20)
    order = [i for i in np.argsort(-(disp + 2 * np.abs(rot))) if moving[i]][:3]
    print(f"{len(W0)} windows, {len(ut)} trajectories; median block displacement {np.median(disp):.0f} units; {int(moving.sum())} windows meet the motion rule")
    print("top 3 of the motion rule: " + ", ".join(f"window {int(W0[i])} (traj {int(T0[i])}, {disp[i]:.0f} units, {rot[i]:+.0f} deg)" for i in order)
          + "   [paper: 834 (117, -92), 138 (115, -91), 991 (103, +95)]")

    # ---- Table S15
    print("\nTable S15: token cosine, decoded real k=5 LPIPS, rendered prediction k=1 LPIPS, k=5 l1, k=5 LPIPS (reproduced | paper)")
    for e, lab in ENC:
        r = R[e]
        if r is None:
            print("%-16s not decoded" % lab); continue
        cos = float(r["tokcos"].mean()) if "tokcos" in r.files else float("nan")
        row = [cos, float(r["lpips_real"][:, 5].mean()), float(r["lpips_m1"][:, 0].mean()), float(r["l1_m1"][:, 4].mean()), float(r["lpips_m1"][:, 4].mean())]
        print("%-16s %.3f %.4f %.4f %.4f %.4f | %.3f %.4f %.4f %.4f %.4f" % (lab, *row, *PAPER_S15[e]))
    if R["alps"] is not None:
        for e, lab in ENC[1:]:
            if R[e] is None:
                continue
            m, lo, hi = ci(R["alps"]["lpips_m1"][:, 4] - R[e]["lpips_m1"][:, 4]); other = float(R[e]["lpips_m1"][:, 4].mean())
            print("ALPS - %-15s LPIPS k=5 %+.4f [%+.4f, %+.4f] %+.0f%%   [ref %+.4f [%+.4f, %+.4f] %+d%%]" % (lab, m, lo, hi, 100 * m / other, *PAPER_S15_PAIRS[e]))

    # ---- Table S17
    print("\nTable S17: block error 5 world-model steps ahead: true tokens, one-step, rollout, rollout with recorded proprio, last obs. (reproduced | paper)")
    E5 = {}
    for e, lab in ENC:
        s = D[e]["state"]
        E5[e] = []
        for c, key in COND:
            err = block_err(D[e]["y_real"], s[:, 2:8])[:, 5] if c == "real" else block_err(D[e][key], s[:, 3:8])[:, 4]
            E5[e].append(err)
        print("%-16s %s | %s" % (lab, " ".join("%5.1f" % x.mean() for x in E5[e]), " ".join("%5.1f" % x for x in PAPER_S17[e])))
    print("ALPS - C-RADIOv3 " + "  ".join("%+.2f [%+.2f, %+.2f]" % ci(E5["alps"][j] - E5["cradio_v3_b"][j]) for j in range(5)))
    print("  paper          " + "  ".join("%+.2f [%+.2f, %+.2f]" % p for p in PAPER_S17_PAIR))


if __name__ == "__main__":
    main()
