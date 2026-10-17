# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""LIBERO demonstrations -> one episode npz per demo (the episodes of the LIBERO state probe).
Per demo: frames [T, res, res, 3] uint8 (agentview_rgb, stored upside-down in the hdf5 -> flipped upright, PIL bicubic
to --res), actions [T, 7] float32 (3 end-effector position deltas, 3 rotation deltas, gripper), ts [T].
episodes.json lists suite / task / demo; episode ids follow the build order (suites in the given order, tasks sorted by
file name, demos by index), which the split files in splits/ refer to.
The paper's probe episodes: --suites libero_spatial libero_object libero_goal --stride 2 --res 128
(ep00000-ep00499 = LIBERO-Spatial, ep00500-ep00999 = LIBERO-Object, ep01000-ep01499 = LIBERO-Goal; 10 Hz frames).
    python build_episodes.py --root <libero_datasets> --suites libero_spatial libero_object libero_goal --out <episodes> \
        --stride 2 --res 128
"""
import argparse, glob, json, os
import h5py, numpy as np
from PIL import Image

HZ = 20.0


def prep(x, res=224):
    out = np.empty((len(x), res, res, 3), np.uint8)
    for i, im in enumerate(x):
        out[i] = np.asarray(Image.fromarray(im[::-1]).resize((res, res), Image.BICUBIC), np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--suites", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--stride", type=int, default=1); ap.add_argument("--max_demos", type=int, default=0)
    ap.add_argument("--res", type=int, default=224); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); index = []; ei = 0
    for suite in a.suites:
        for f in sorted(glob.glob(os.path.join(a.root, suite, "*.hdf5"))):
            task = os.path.basename(f).replace("_demo.hdf5", "")
            with h5py.File(f, "r") as h:
                demos = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[1]))
                if a.max_demos: demos = demos[: a.max_demos]
                for d in demos:
                    g = h["data"][d]
                    rgb = g["obs"]["agentview_rgb"][::a.stride]; act = g["actions"][::a.stride].astype(np.float32)
                    T = min(len(rgb), len(act)); rgb, act = rgb[:T], act[:T]
                    frames = prep(rgb, a.res); ts = np.arange(T, dtype=np.float64) * a.stride / HZ
                    eid = f"ep{ei:05d}"
                    np.savez_compressed(os.path.join(a.out, f"{eid}.npz"), frames=frames, actions=act, ts=ts)
                    index.append({"episode": eid, "n_steps": int(T), "duration_s": float(ts[-1]), "camera": "agentview_rgb",
                                  "action_field": "actions", "ts_field": f"assumed_{HZ:g}Hz",
                                  "meta": {"scene_id": f"{suite}/{task}", "suite": suite, "task": task, "demo": int(d.split("_")[1])}})
                    ei += 1
            print(f"[build] {suite}/{task}: {len(demos)} demos (total {ei})", flush=True)
    with open(os.path.join(a.out, "episodes.json"), "w") as fo:
        json.dump({"data_dir": a.root, "version": "libero-hdf5", "camera": "agentview_rgb", "action_field": "actions", "episodes": index}, fo, indent=1)
    tot = sum(e["duration_s"] for e in index)
    print(f"[build] {len(index)} episodes, {tot/3600:.2f} trajectory-hours"); print("LIBERO_DYN_BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
