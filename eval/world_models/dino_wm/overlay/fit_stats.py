# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""Token statistics of a frozen encoder on the training frames of a DINO-WM task: the per-channel mean / standard deviation
that standardise the world model's tokens (models/encoder/frozen_encoder.py).

  python fit_stats.py --task <pusht|wall|point_maze|granular|rope> --encoder <name> --subsets <dir> --out <npz>
                      [--data $DATASET_DIR] [--weights <safetensors>] [--n_frames 20000 --max_trajs 4000 --seed 0]

Frames: the train split exactly as a training run sees it (the task's loader with the trajectory order file
<subsets>/<task>_100.json), through the DINO-WM image transform, encoded by FrozenTokenEncoder.forward_raw (the pixel
path and the TF32 setting of training and planning). Sampling with numpy RandomState(seed): min(max_trajs, n_train)
trajectories uniformly without replacement, then n_frames frame ids uniformly without replacement over all frames of those
trajectories. The statistics run over all 196 tokens of every sampled frame in float64: mu = mean, sd = sqrt(var + 1e-8)
with var the diagonal of the token covariance (clamped at 0). Output npz: mu, sd [D] float32, the sample list, and `meta`
(json: encoder, task, subset file + sha256, counts, ...).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def base_and_index(ds, i):
    from datasets.traj_dset import TrajSubset
    while isinstance(ds, TrajSubset):
        i = ds.indices[i]
        ds = ds.dataset
    return ds, i


def sample_frames(train, n_frames, max_trajs, seed):
    """(trajectory position, sorted frame ids) of the sampled frames, in trajectory order."""
    rng = np.random.RandomState(seed)
    n_train = len(train)
    n_trajs = min(max_trajs, n_train)
    trajs = np.sort(rng.choice(n_train, size=n_trajs, replace=False))
    lens = np.array([int(train.get_seq_length(int(t))) for t in trajs])
    cum = np.concatenate([[0], np.cumsum(lens)])
    total = int(cum[-1])
    n_frames = min(n_frames, total)
    gid = np.sort(rng.choice(total, size=n_frames, replace=False))
    tpos = np.searchsorted(cum, gid, side="right") - 1
    frame = gid - cum[tpos]
    samples = {}
    for tp, fr in zip(tpos, frame):
        samples.setdefault(int(trajs[tp]), []).append(int(fr))
    return sorted(samples.items()), n_trajs, total


class _FrameSet(torch.utils.data.Dataset):
    """One item = the sampled frames of one trajectory, decoded through the dataset's get_frames + transform."""
    def __init__(self, train, items):
        self.train, self.items = train, items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        tp, frs = self.items[i]
        base, bi = base_and_index(self.train, tp)
        obs, _, _, _ = base.get_frames(bi, sorted(frs))
        return obs["visual"].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["pusht", "wall", "point_maze", "granular", "rope"])
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--subsets", required=True, help="folder of make_subsets.py")
    ap.add_argument("--out", required=True, help="output npz")
    ap.add_argument("--data", default=os.environ.get("DATASET_DIR"))
    ap.add_argument("--weights", default=None, help="safetensors of alps / alps_stage1 (default: env ALPS_WEIGHTS / ALPS_STAGE1_WEIGHTS)")
    ap.add_argument("--n_frames", type=int, default=20000)
    ap.add_argument("--max_trajs", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8, help="DataLoader workers for frame decoding")
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"exists: {a.out}")
    os.chdir(HERE)
    import hydra
    from omegaconf import OmegaConf
    from make_run_config import build_config
    from models.encoder.frozen_encoder import FrozenTokenEncoder

    t_start = time.time()
    dev = torch.device("cuda:0")
    subset_json = os.path.join(a.subsets, f"{a.task}_100.json")
    # the task's loader with the run configuration (statistics path not used here)
    cfg = build_config(a.task, a.encoder, stats=a.out, subsets=a.subsets, data=a.data, weights=a.weights)
    _, traj = hydra.utils.call(cfg.env.dataset, num_hist=int(cfg.num_hist), num_pred=1, frameskip=int(cfg.frameskip))
    train = traj["train"]
    items, n_trajs, total = sample_frames(train, a.n_frames, a.max_trajs, a.seed)
    n_frames = sum(len(f) for _, f in items)
    print(f"[fit_stats] {a.task}: train trajectories {len(train)}, sampled {n_trajs} trajectories / {total} frames -> "
          f"{n_frames} frames from {len(items)} trajectories", flush=True)

    enc_cfg = OmegaConf.to_container(cfg.encoder, resolve=True)
    enc = FrozenTokenEncoder(enc_cfg["encoder"], stats=None, res=enc_cfg["res"], tf32=enc_cfg["tf32"], weights=enc_cfg.get("weights")).to(dev).eval()
    D = enc.raw_dim
    S = torch.zeros(D, dtype=torch.float64, device=dev)
    Q = torch.zeros(D, D, dtype=torch.float64, device=dev)
    n_tok = 0
    buf = []

    def flush():
        nonlocal n_tok, buf
        if not buf:
            return
        x = torch.cat(buf, 0).to(dev)
        buf = []
        with torch.no_grad():
            for s in range(0, x.shape[0], a.batch):
                f = enc.forward_raw(x[s:s + a.batch]).double().flatten(0, 1)          # [b * 196, D]
                S.add_(f.sum(0)); Q.add_(f.T @ f); n_tok += f.shape[0]

    loader = torch.utils.data.DataLoader(_FrameSet(train, items), batch_size=None, shuffle=False, num_workers=a.workers,
                                         prefetch_factor=4 if a.workers > 0 else None)
    done = 0
    for k, v in enumerate(loader):
        assert v.shape[-1] == 224 and v.min() >= -1.0001 and v.max() <= 1.0001, v.shape
        buf.append(v.float()); done += v.shape[0]
        if sum(b.shape[0] for b in buf) >= 4 * a.batch:
            flush()
        if k % 500 == 0:
            print(f"  {k}/{len(items)} trajectories, {done} frames, {time.time() - t_start:.0f} s", flush=True)
    flush()

    mu = S / n_tok
    cov_diag = torch.diag(Q / n_tok - torch.outer(mu, mu))
    sd = torch.sqrt(torch.clamp(cov_diag, min=0) + 1e-8)
    meta = dict(task=a.task, encoder=a.encoder, seed=a.seed, n_frames=n_frames, n_trajs=len(items), n_tokens=n_tok, raw_dim=D,
                res=enc.res, native_grid=enc.native_grid, subset_json=os.path.abspath(subset_json), subset_sha256=sha256(subset_json),
                n_train_trajs=len(train), standardisation="per-channel (t - mu) / sd over all 196 tokens of the sampled frames",
                encoder_tf32=enc.tf32, torch=torch.__version__, wallclock_s=time.time() - t_start)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, mu=mu.cpu().numpy().astype(np.float32), sd=sd.cpu().numpy().astype(np.float32),
             sample_traj=np.array([t for t, frs in items for _ in frs], dtype=np.int32),
             sample_frame=np.array([f for t, frs in items for f in sorted(frs)], dtype=np.int32), meta=json.dumps(meta))
    print(f"[fit_stats] saved {a.out}: {n_tok} tokens, D {D}; |mu| mean {mu.abs().mean():.3f}, sd mean {sd.mean():.3f} "
          f"(min {sd.min():.3e}, max {sd.max():.3f}); {time.time() - t_start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
