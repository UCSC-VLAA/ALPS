#!/usr/bin/env python
# Copyright 2026 The ALPS Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The rollout figure of the supplementary material (closed-loop rollouts of the ALPS policies) from record_rollouts.py files.

    python make_rollout_figure.py --suite LIBERO-Spatial rollouts_spatial.npz --suite LIBERO-Goal rollouts_goal.npz \
        --rows-per-suite 2 --out policy_rollouts.pdf

Rows: for every suite the first --rows-per-suite tasks in task order whose recorded rollout succeeded; 7 frames per row,
evenly spaced over the rollout, with the time at 20 Hz (the control frequency of LIBERO). Frames are drawn at their
recorded 128 px without interpolation; text is vector (TrueType) in the PDF.
"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--suite", nargs=2, action="append", metavar=("LABEL", "NPZ"), required=True)
    ap.add_argument("--rows-per-suite", type=int, default=2)
    ap.add_argument("--frames", type=int, default=7)
    ap.add_argument("--out", default="policy_rollouts.pdf")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    rows = []
    for label, path in a.suite:
        d = np.load(path)
        tasks = sorted(int(k.split("_")[1]) for k in d.files if k.startswith("meta_"))
        ok = 0
        for ti in tasks:
            m = json.loads(str(d["meta_%d" % ti]))
            ok += int(m["success"])
            if m["success"] and sum(r[0] == label for r in rows) < a.rows_per_suite:
                rows.append((label, m, d["frames_%d" % ti]))
        print(label, "recorded rollouts: %d of %d succeeded" % (ok, len(tasks)))
    nf = a.frames
    fig = plt.figure(figsize=(7.1, len(rows) * 1.13 + 0.05), dpi=300)
    w, h = 1 / nf * 0.995, 1 / len(rows)
    for r, (label, m, frames) in enumerate(rows):
        idx = np.round(np.linspace(0, len(frames) - 1, nf)).astype(int)
        y = 1 - (r + 1) * h
        fig.text(0.004, y + h - 0.025 * 4 / len(rows), "%s: \u201c%s\u201d" % (label, m["language"]), fontsize=7.6, va="center", ha="left")
        for j, f in enumerate(idx):
            ax = fig.add_axes([j * (1 / nf) + 0.002, y + 0.004, w - 0.002, h * 0.80])
            ax.imshow(frames[f], interpolation="none")
            ax.set_xticks([])
            ax.set_yticks([])
            t = f * m["stride"] / 20.0 if f < len(frames) - 1 else m["steps"] / 20.0
            ax.text(0.03, 0.04, "%.1f s" % t, transform=ax.transAxes, fontsize=7, color="white", va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.55))
            for s in ax.spines.values():
                s.set_linewidth(0.4)
        print("row", r, label, "|", m["language"], "| steps", m["steps"], "| frames", frames.shape)
    fig.savefig(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
