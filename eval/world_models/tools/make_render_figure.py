# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""Fig. 6 of the paper (rendered world-model predictions on PushT) from the PNGs of render_decode.py.

  python make_render_figure.py --png_dir <render_decode out dir> [<more dirs>] --out render_main.pdf [--window 834]
                               [--rows ALPS C-RADIOv3-B]

Window 834 is rank 1 of the motion rule (render_stats.py). Columns: the current frame (k = 0: decoded REAL tokens) and the
predictions 1, 3 and 5 world-model steps ahead (+5, +15, +25 simulator steps: decoded tokens of the open-loop rollout);
top row: the true frames. Frame colours: grey = true frame, green = decoded real tokens, blue = decoded predicted tokens.
"""
import argparse
import glob
import os

import numpy as np

KS = (0, 1, 3, 5)
GREEN, BLUE, GREY = "#1a9850", "#2c7bb6", "0.55"
SHORT = {"C-RADIOv3-B": "C-RADIOv3", "ALPS-Stage1": "ALPS (Stage 1)", "DINOv3-B16": "DINOv3", "SigLIP2-B16": "SigLIP 2"}


def find(dirs, pattern):
    hits = sorted({h for d in dirs for h in glob.glob(os.path.join(d, pattern))})
    if len(hits) != 1:
        raise FileNotFoundError(f"{pattern}: {len(hits)} matches in {dirs}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png_dir", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=834); ap.add_argument("--rows", nargs="+", default=["ALPS", "C-RADIOv3-B"])
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    pre = "window%03d_traj*_" % a.window

    def tile(row, k):
        if row is None:
            return find(a.png_dir, pre + "true_k%d.png" % k), GREY
        if k == 0:
            return find(a.png_dir, pre + "%s_real_k0.png" % row), GREEN
        return find(a.png_dir, pre + "%s_m1_k%d.png" % (row, k)), BLUE

    rows = [("True", None)] + [(SHORT.get(r, r), r) for r in a.rows]
    W = 3.3; lab = 0.2; top = 0.24; n = len(KS); cw = (W - lab) / n; Hf = top + len(rows) * cw
    fig = plt.figure(figsize=(W, Hf), dpi=300)
    for j, k in enumerate(KS):
        if k:
            fig.text((lab + (j + 0.5) * cw) / W, 1 - top / Hf * 0.45, "+%d sim. steps" % (5 * k) if j == 1 else "+%d" % (5 * k), fontsize=7.5, ha="center", va="center")
    fig.text((lab + 0.5 * cw) / W, 1 - top / Hf * 0.45, "current", fontsize=7.5, ha="center", va="center")
    for i, (name, row) in enumerate(rows):
        y = 1 - (top + (i + 1) * cw) / Hf
        fig.text(lab * 0.55 / W, y + 0.5 * cw / Hf, name, fontsize=7.5, ha="center", va="center", rotation=90)
        for j, k in enumerate(KS):
            path, col = tile(row, k); img = np.asarray(Image.open(path).convert("RGB"))
            ax = fig.add_axes([(lab + j * cw) / W + 0.004, y + 0.006 * W / Hf, cw / W - 0.008, cw / Hf - 0.012 * W / Hf])
            ax.imshow(img, interpolation="lanczos"); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(1.1 if row else 0.5); s.set_color(col)
    fig.savefig(a.out, dpi=300); plt.close(fig)
    print("saved", a.out)


if __name__ == "__main__":
    main()
