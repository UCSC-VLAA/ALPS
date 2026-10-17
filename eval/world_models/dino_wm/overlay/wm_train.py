# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""World-model training of the ALPS paper on the DINO-WM tasks (PushT, Wall, PointMaze, Granular, Rope): one recipe for
every task and encoder.

  torchrun --nproc_per_node 8 wm_train.py --run <run dir with hydra.yaml (make_run_config.py)> [--seed 1]
          [--total T --cos_end C --save s1,s2,...] [--stop_after N] [--resume <checkpoint>]

Unchanged from DINO-WM: the frozen encoder with its token statistics, the predictor (ProjViTPredictor: the official ViT
predictor, 6 layers, with the learned projections D -> 384 -> D), the one-step teacher-forced MSE on the visual and
proprioceptive channels (VWorldModel.forward), AdamW (betas .9 / .999, eps 1e-8, weight decay .01), no gradient clipping,
float32 predictor, on-the-fly encoding of the frames, the training slices of the official loader.
The recipe: torch.distributed, one process per GPU, 32 samples per GPU, global batch 256 on 8 GPUs (the gradient of the mean
over the 256 samples); peak learning rate = DINO-WM's rate x sqrt(256 / 32) = 1.4142e-3, for every optimizer; for update
s = 1 .. T:
   s <= W            f = s / W                                                (linear warm-up, W = 200)
   W < s <= C        f = 0.01 + 0.99 * 0.5 * (1 + cos(pi * (s - W) / (C - W)))  (cosine to 1% of the peak)
   C < s             f = 0.01                                                 (hold)
with (T, C) = (31,000, 29,500) on PushT, (15,500, 14,700) on Wall and PointMaze, (7,750, 7,350) on Granular and Rope.
Data order: the training slices in the order of the official loader (TrajSlicerDataset permutation with numpy seed(seed)
set right before the dataset call), read as one continuous stream over repeated epochs; update u takes the stream positions
[256u, 256u + 256) and GPU r the positions [256u + 32r, 256u + 32r + 32). Initialisation of the trainable modules and the
data order are the same for every encoder (seed 1 in the paper).
Every save is a full resumable state (modules, optimizers, schedulers, python / numpy / torch RNG and the CUDA RNG of every
GPU, stream position): checkpoints/model_s<update>.pth, which plan.py loads with model_epoch=s<update>.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# (total updates, end of the cosine, save points) per task, the settings of the paper's runs
SCHEDULES = {
    "pusht": (31000, 29500, list(range(1000, 29001, 1000)) + [29500, 29800, 30100, 30400, 30700, 31000]),
    "wall": (15500, 14700, list(range(1000, 14001, 1000)) + [14700, 14860, 15020, 15180, 15340, 15500]),
    "point_maze": (15500, 14700, list(range(1000, 14001, 1000)) + [14700, 14860, 15020, 15180, 15340, 15500]),
    "granular": (7750, 7350, list(range(500, 7001, 500)) + [7350, 7430, 7510, 7590, 7670, 7750]),
    "rope": (7750, 7350, list(range(500, 7001, 500)) + [7350, 7430, 7510, 7590, 7670, 7750]),
}
REFERENCE_BATCH = 32          # batch of the DINO-WM recipe, from which the learning rate is scaled


def lr_factor(u, warmup, cos_end, floor):
    """Learning-rate factor of the 0-based update index u (update s = u + 1)."""
    s = u + 1
    if s <= warmup:
        return s / warmup
    if s <= cos_end:
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * (s - warmup) / (cos_end - warmup)))
    return floor


def check_schedule(total, warmup, cos_end, floor):
    f = lambda u: lr_factor(u, warmup, cos_end, floor)  # noqa: E731
    assert abs(f(0) - 1 / warmup) < 1e-15 and f(warmup - 1) == 1.0 and abs(f(warmup) - 1.0) < 1e-6
    assert abs(f(cos_end - 1) - floor) < 1e-12 and f(cos_end) == floor and f(total - 1) == floor


def stream_positions(u, r, global_bs, per_gpu_bs, n):
    """Dataset indices that GPU r reads at update u: stream positions [global_bs * u + per_gpu_bs * r, + per_gpu_bs) mod n."""
    p0 = global_bs * u + per_gpu_bs * r
    return [(p0 + j) % n for j in range(per_gpu_bs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory with hydra.yaml; checkpoints and logs are written there")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--total", type=int, default=None); ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--cos_end", type=int, default=None); ap.add_argument("--floor", type=float, default=0.01)
    ap.add_argument("--per_gpu_bs", type=int, default=32)
    ap.add_argument("--target_global_bs", type=int, default=256, help="global batch the learning rate is scaled for (fixed by the recipe)")
    ap.add_argument("--save", default=None, help="comma-separated updates to save (default: the task's save points)")
    ap.add_argument("--workers", type=int, default=7); ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--resume", default=None); ap.add_argument("--stop_after", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import wm_predeval as W   # load_cfg; imports plan.py (and with it the DINO-WM environments), as the original trainer did
    import plan
    import hydra

    dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size(); lr_ = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr_); dev = torch.device("cuda", lr_)
    os.makedirs(os.path.join(a.run, "checkpoints"), exist_ok=True)
    LOG = open(os.path.join(a.run, "train_log.jsonl"), "a") if rank == 0 else None

    def log(**kw):
        if rank != 0:
            return
        kw["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); LOG.write(json.dumps(kw) + "\n"); LOG.flush()
        print(json.dumps(kw), flush=True)

    cfg = W.load_cfg(a.run)
    task = str(cfg.task)
    assert int(cfg.num_pred) == 1 and int(cfg.concat_dim) == 1 and not cfg.has_decoder and int(cfg.num_hist) in (1, 3) and int(cfg.frameskip) in (1, 5)
    assert int(cfg.training.batch_size) == REFERENCE_BATCH, cfg.training.batch_size
    T0, CE0, SAVE0 = SCHEDULES[task]
    T = a.total if a.total is not None else T0
    CE = a.cos_end if a.cos_end is not None else CE0
    WU, FL, PB = a.warmup, a.floor, a.per_gpu_bs
    GB = PB * world
    SCALE = math.sqrt(a.target_global_bs / REFERENCE_BATCH)
    LR_P, LR_A = float(cfg.training.predictor_lr) * SCALE, float(cfg.training.action_encoder_lr) * SCALE
    FACTOR = lambda u: lr_factor(u, WU, CE, FL)  # noqa: E731
    check_schedule(T, WU, CE, FL)
    save_list = [int(x) for x in a.save.split(",") if x] if a.save else [s for s in SAVE0 if s <= T]
    SAVE = sorted(set(save_list) | ({a.stop_after} if a.stop_after else set()))

    def seed_all(s):
        random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    seed_all(a.seed)
    dsets, _ = hydra.utils.call(cfg.env.dataset, num_hist=int(cfg.num_hist), num_pred=int(cfg.num_pred), frameskip=int(cfg.frameskip))
    train = dsets["train"]; N = len(train)
    perm_sha = hashlib.sha256(np.asarray([tuple(int(x) for x in train.slices[i]) for i in range(0, N, 97)], dtype=np.int64).tobytes()).hexdigest()[:16]

    def shard(u, r=None):
        return stream_positions(u, rank if r is None else r, GB, PB, N)

    if a.resume:
        model = plan.load_model(Path(a.resume), cfg, cfg.num_action_repeat, dev)               # modules from the checkpoint
        CK = torch.load(a.resume, map_location="cpu", weights_only=False)
        for k in ("predictor", "action_encoder", "proprio_encoder"):
            s1, s2 = getattr(model, k).state_dict(), CK[k].state_dict()
            assert list(s1) == list(s2) and all(torch.equal(s1[t].cpu(), s2[t].cpu()) for t in s1), k
    else:
        encoder = hydra.utils.instantiate(cfg.encoder)
        for p in encoder.parameters():
            p.requires_grad = False
        torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)                         # identical initialisation on every GPU and for every encoder
        proprio_encoder = hydra.utils.instantiate(cfg.proprio_encoder, in_chans=train.proprio_dim, emb_dim=cfg.proprio_emb_dim)
        action_encoder = hydra.utils.instantiate(cfg.action_encoder, in_chans=train.action_dim, emb_dim=cfg.action_emb_dim)
        num_patches = (int(cfg.img_size) // 16) ** 2 if encoder.latent_ndim != 1 else 1
        predictor = hydra.utils.instantiate(cfg.predictor, num_patches=num_patches, num_frames=int(cfg.num_hist),
                                            dim=encoder.emb_dim + (cfg.proprio_emb_dim * cfg.num_proprio_repeat + cfg.action_emb_dim * cfg.num_action_repeat) * cfg.concat_dim)
        model = hydra.utils.instantiate(cfg.model, encoder=encoder, proprio_encoder=proprio_encoder, action_encoder=action_encoder, predictor=predictor,
                                        decoder=None, proprio_dim=cfg.proprio_emb_dim, action_dim=cfg.action_emb_dim, concat_dim=cfg.concat_dim,
                                        num_action_repeat=cfg.num_action_repeat, num_proprio_repeat=cfg.num_proprio_repeat)
        model.to(dev)

    def h_params(m):
        h = hashlib.sha256()
        for k in ("predictor", "action_encoder", "proprio_encoder"):
            for t, p in getattr(m, k).state_dict().items():
                h.update(p.detach().float().cpu().numpy().tobytes())
        return h.hexdigest()[:16]
    sha_all = [None] * world; dist.all_gather_object(sha_all, h_params(model))
    assert len(set(sha_all)) == 1, "trainable weights differ between GPUs before DDP: %s" % sha_all
    TR = list(model.predictor.parameters()) + list(model.action_encoder.parameters()) + list(model.proprio_encoder.parameters())
    ddp = DDP(model, device_ids=[lr_], output_device=lr_, broadcast_buffers=False, find_unused_parameters=False)
    M = ddp.module
    opt_p = torch.optim.AdamW(M.predictor.parameters(), lr=LR_P)
    opt_a = torch.optim.AdamW(itertools.chain(M.action_encoder.parameters(), M.proprio_encoder.parameters()), lr=LR_A)
    if a.resume:
        opt_p.load_state_dict(CK["predictor_optimizer"]); opt_a.load_state_dict(CK["action_encoder_optimizer"])
    sched_p = torch.optim.lr_scheduler.LambdaLR(opt_p, FACTOR); sched_a = torch.optim.lr_scheduler.LambdaLR(opt_a, FACTOR)
    u0 = 0
    if a.resume:
        sched_p.load_state_dict(CK["sched_p"]); sched_a.load_state_dict(CK["sched_a"])
        for o, s in ((opt_p, sched_p), (opt_a, sched_a)):
            for grp, lrv in zip(o.param_groups, s.get_last_lr()):
                grp["lr"] = lrv
        assert CK["data"]["world"] == world and CK["data"]["per_gpu_bs"] == PB and CK["data"]["seed"] == a.seed and CK["data"]["perm_sha"] == perm_sha
        r_ = CK["rng"]; random.setstate(r_["python"]); np.random.set_state(r_["numpy"]); torch.set_rng_state(r_["torch"]); torch.cuda.set_rng_state(r_["cuda_per_rank"][rank])
        u0 = int(CK["data"]["next_update"])
    else:
        torch.cuda.manual_seed(a.seed * 1000 + 17 + rank)                                     # dropout stream of this GPU

    # start-up checks: disjoint shards that tile the global batch; learning-rate plan
    first = [None] * world; dist.all_gather_object(first, (rank, shard(u0)))
    allpos = sorted(i for _, s in first for i in s); assert len(allpos) == len(set(allpos)) == GB, "GPUs read overlapping samples"
    if rank == 0 and (not a.resume or not os.path.exists(os.path.join(a.run, "lr_plan.csv"))):
        with open(os.path.join(a.run, "lr_plan.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["update", "factor", "lr_predictor", "lr_action_proprio"])
            for u in range(T):
                w.writerow([u + 1, "%.10f" % FACTOR(u), "%.10e" % (LR_P * FACTOR(u)), "%.10e" % (LR_A * FACTOR(u))])
    log(event="start", task=task, num_hist=int(cfg.num_hist), frameskip=int(cfg.frameskip), encoder=str(cfg.encoder.encoder), stats=str(cfg.encoder.stats),
        resume=bool(a.resume), src=a.resume, seed=a.seed, world=world, per_gpu_bs=PB, global_bs=GB, reference_bs=REFERENCE_BATCH,
        lr_scale=SCALE, peak_lr_predictor=LR_P, peak_lr_action_proprio=LR_A, start_update=u0, N_train=N, epoch_updates=N / GB, perm_sha=perm_sha,
        first_shards={r: [tuple(int(x) for x in train.slices[i]) for i in s[:2]] for r, s in first}, params_sha=sha_all[0],
        trainable_params=sum(p.numel() for p in TR), encoder_trainable=sum(p.numel() for p in M.encoder.parameters() if p.requires_grad),
        schedule={"warmup": WU, "cos_end": CE, "floor": FL, "total": T}, saves=SAVE, torch=torch.__version__, nccl=list(torch.cuda.nccl.version()))

    def save(u):
        rng_c = [None] * world; dist.all_gather_object(rng_c, torch.cuda.get_rng_state().cpu())
        if rank == 0:
            path = os.path.join(a.run, "checkpoints", "model_s%d.pth" % u)
            ck = {"predictor": M.predictor, "proprio_encoder": M.proprio_encoder, "action_encoder": M.action_encoder, "epoch": int(GB * u // N) + 1, "global_step": u,
                  "predictor_optimizer": opt_p.state_dict(), "action_encoder_optimizer": opt_a.state_dict(), "sched_p": sched_p.state_dict(), "sched_a": sched_a.state_dict(),
                  "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda_per_rank": rng_c},
                  "data": {"seed": a.seed, "N": N, "perm_sha": perm_sha, "next_update": u, "world": world, "per_gpu_bs": PB, "global_bs": GB,
                           "order": "train.py slicer permutation (seed), continuous stream over repeated epochs, GPU r takes [GB*u + PB*r, +PB)"},
                  "schedule": {"warmup": WU, "cos_end": CE, "floor": FL, "total": T, "peak_lr_predictor": LR_P, "peak_lr_action_proprio": LR_A,
                               "lr_factor_next": FACTOR(u) if u < T else FL}}
            torch.save(ck, path + ".tmp"); os.replace(path + ".tmp", path)
            log(event="save", global_step=u, path=path, params_sha=h_params(M), lr_next_p=opt_p.param_groups[0]["lr"])
        dist.barrier()

    class Shards(torch.utils.data.Sampler):
        def __init__(self, lo, hi):
            self.lo, self.hi = lo, hi

        def __len__(self):
            return self.hi - self.lo

        def __iter__(self):
            return (shard(u) for u in range(self.lo, self.hi))

    stop = T if not a.stop_after else min(T, a.stop_after)
    g = torch.Generator(); g.manual_seed(a.seed * 1000 + rank)
    dl = torch.utils.data.DataLoader(train, batch_sampler=Shards(u0, stop), num_workers=a.workers, generator=g, pin_memory=True, prefetch_factor=4)
    t0 = time.time(); acc = torch.zeros(4, device=dev); gmax = 0.0; gsum = 0.0
    for j, (obs, act, state) in enumerate(dl):
        u = u0 + j
        exp_p, exp_a = LR_P * FACTOR(u), LR_A * FACTOR(u)
        assert all(abs(gr["lr"] - exp_p) <= 1e-12 * exp_p + 1e-18 for gr in opt_p.param_groups) and \
            all(abs(gr["lr"] - exp_a) <= 1e-12 * exp_a + 1e-18 for gr in opt_a.param_groups), (u, opt_p.param_groups[0]["lr"], exp_p)
        obs = {k: v.to(dev, non_blocking=True) for k, v in obs.items()}; act = act.to(dev, non_blocking=True)
        ddp.train()
        _, _, _, loss, comps = ddp(obs, act)
        opt_p.zero_grad(set_to_none=True); opt_a.zero_grad(set_to_none=True)
        loss.backward()                                                                       # DDP: all-reduce, average over GPUs
        gn = torch.norm(torch.stack([p.grad.detach().norm() for p in TR if p.grad is not None])).item()
        if not math.isfinite(gn) or not math.isfinite(float(loss)):
            log(event="nonfinite", update=u + 1, loss=float(loss), grad_norm=gn); raise SystemExit("non-finite loss or gradient at update %d" % (u + 1))
        opt_p.step(); opt_a.step(); sched_p.step(); sched_a.step()
        acc += torch.tensor([float(loss), float(comps["z_visual_loss"]), float(comps["z_proprio_loss"]), 1.0], device=dev); gmax = max(gmax, gn); gsum += gn
        if (u + 1) % a.log_every == 0 or u + 1 == stop:
            tot = acc.clone(); dist.all_reduce(tot)                                           # mean over GPUs and steps of the window
            n = int(acc[3].item()); dt = time.time() - t0
            log(event="train", global_step=u + 1, loss=(tot[0] / tot[3]).item(), z_visual_loss=(tot[1] / tot[3]).item(), z_proprio_loss=(tot[2] / tot[3]).item(),
                grad_norm_mean=gsum / n, grad_norm_max=gmax, lr_used_p=exp_p, lr_used_a=exp_a, lr_next_p=opt_p.param_groups[0]["lr"], it_s=n / dt,
                samples_s=n * GB / dt, max_mem_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
            t0 = time.time(); acc.zero_(); gmax = 0.0; gsum = 0.0
        if u + 1 in SAVE:
            save(u + 1)
    log(event="done", last_update=stop, params_sha=h_params(M))
    if rank == 0:
        print("WM_TRAIN_DONE", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
