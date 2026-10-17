# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""Reacher world model of the ALPS paper (Table 3, Tables S12 and S16): LeWorldModel's PreJEPA on a frozen encoder, trained
with the recipe of the DINO-WM tasks.

  torchrun --nproc_per_node 8 reacher_wm_train.py --encoder <name> --stats <reacher_<name>.npz> --out <dir>
           --run_name <name> [--config configs/reacher_wm.yaml] [--seed 1] [--weights <safetensors>]

Unchanged from the PreJEPA baseline of stable-worldmodel (configs/reacher_wm.yaml = the config of the paper's runs): the
model (CausalPredictor depth 6 / heads 16 / MLP 2048 / width 768 + 10, history 3, one prediction step, frame skip 5, no
proprioception, action Embedder 10 -> 10), the loss (prejepa.py dinowm_forward, imported from the stable-worldmodel
scripts), the dataset and its transforms (prejepa.py get_img_preprocessor: ImageNet normalisation, 224 pixels; z-score of
the actions), the train / validation split (spt random_split 0.9 / 0.1 with generator seed 42), the frozen encoder
(frozen_backbone.FrozenTokens, float32, 196 unit-length tokens, per-channel statistics of make_stats.py), AdamW without
weight decay, fp16 autocast for the predictor with a GradScaler, gradient-norm clipping at 1.0.
The recipe: torch.distributed with one process per GPU and 32 clips per GPU (global batch 256 on 8 GPUs; the gradients are
averaged over the GPUs after every backward), peak learning rate 5e-4 x sqrt(256 / 32), 200 warm-up updates, cosine to 1% at
--cos_end (24,700), hold to --total (26,000); training seed --seed for the initialisation, the data order (per epoch e:
numpy RandomState(seed * 100003 + e).permutation of the training clips, read as one stream) and dropout; the clipping acts on
the gradients after the loss scale is removed.
Every save is a full resumable state (<out>/checkpoints/model_s<u>.pth). --export updates are also written in LeWM's
save_pretrained format to $STABLEWM_HOME/checkpoints/<run_name>/weights_s<u>.pt (+ config.json), the file that eval.py and
reacher_rollout_eval.py load.
Needs the stable-worldmodel clone (env SWM_ROOT, for scripts/train/prejepa.py) and this folder on PYTHONPATH.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
SAVE_DEFAULT = ",".join([str(2000 * i) for i in range(1, 13)] + ["24700", "24960", "25220", "25480", "25740", "26000"])
EXPORT_DEFAULT = "24700,24960,25220,25480,25740,26000"
REFERENCE_BATCH = 32


def lr_factor(u, warmup, cos_end, floor):
    """Learning-rate factor of the 0-based update index u (update s = u + 1); the schedule of wm_train.py."""
    s = u + 1
    if s <= warmup:
        return s / warmup
    if s <= cos_end:
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * (s - warmup) / (cos_end - warmup)))
    return floor


def epoch_stream(seed, n, u, r, global_bs, per_gpu_bs, _cache={}):
    """train_set positions that GPU r reads at update u."""
    out = []
    p0 = global_bs * u + per_gpu_bs * r
    for j in range(per_gpu_bs):
        e, i = divmod(p0 + j, n)
        if _cache.get("e") != (seed, n, e):
            _cache.clear(); _cache["e"] = (seed, n, e); _cache["p"] = np.random.RandomState(seed * 100003 + e).permutation(n)
        out.append(int(_cache["p"][i]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True); ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--run_name", required=True)
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "reacher_wm.yaml")); ap.add_argument("--weights", default=None)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--total", type=int, default=26000); ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--cos_end", type=int, default=24700); ap.add_argument("--floor", type=float, default=0.01); ap.add_argument("--per_gpu_bs", type=int, default=32)
    ap.add_argument("--target_global_bs", type=int, default=256, help="global batch the learning rate is scaled for (fixed by the recipe)")
    ap.add_argument("--save", default=SAVE_DEFAULT); ap.add_argument("--export", default=EXPORT_DEFAULT)
    ap.add_argument("--val_every", type=int, default=2000); ap.add_argument("--val_clips", type=int, default=1600)
    ap.add_argument("--workers", type=int, default=7); ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--resume", default=None); ap.add_argument("--stop_after", type=int, default=0)
    a = ap.parse_args()

    swm_root = os.environ.get("SWM_ROOT")
    if not swm_root:
        raise RuntimeError("set SWM_ROOT to the stable-worldmodel clone (scripts/train/prejepa.py)")
    train_dir = os.path.join(swm_root, "scripts", "train")
    sys.path.insert(0, HERE); sys.path.insert(0, train_dir)
    import hdf5plugin  # noqa: F401  (Blosc filter of the pixels of reacher.h5; the data-loader workers inherit it)
    import stable_pretraining as spt                       # registers the ${eval:...} resolver
    import stable_worldmodel as swm
    import hydra
    from omegaconf import OmegaConf, open_dict
    import prejepa as PJ                                   # the original training file: get_img_preprocessor, dinowm_forward
    from stable_worldmodel.data import column_normalizer as get_column_normalizer
    from stable_worldmodel.wm.utils import save_pretrained

    dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size(); lr_ = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr_); dev = torch.device("cuda", lr_)
    os.makedirs(os.path.join(a.out, "checkpoints"), exist_ok=True)
    LOG = open(os.path.join(a.out, "train_log.jsonl"), "a") if rank == 0 else None

    def log(**kw):
        if rank != 0:
            return
        kw["time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); LOG.write(json.dumps(kw) + "\n"); LOG.flush(); print(json.dumps(kw), flush=True)

    def md5(p):
        h = hashlib.md5(); h.update(open(p, "rb").read()); return h.hexdigest()

    # ---- configuration of the paper's runs; the encoder and its statistics from the command line, the recipe from the arguments
    cfg = OmegaConf.load(a.config)
    with open_dict(cfg):
        cfg.backbone.name = a.encoder
        cfg.model.encoder.stats = os.path.abspath(a.stats)
        cfg.model.encoder.weights = os.path.abspath(a.weights) if a.weights else None
    assert cfg.dataset_name == "reacher" and int(cfg.frameskip) == 5 and int(cfg.wm.history_size) == 3 and int(cfg.wm.num_preds) == 1
    assert list(cfg.wm.encoding.keys()) == ["action"] and int(cfg.batch_size) == REFERENCE_BATCH and float(cfg.optimizer.lr) == 5e-4
    assert float(cfg.optimizer.weight_decay) == 0.0 and cfg.trainer.precision == "16-mixed" and float(cfg.trainer.gradient_clip_val) == 1.0
    assert os.path.exists(a.stats), a.stats
    SPLIT_SEED = int(cfg.seed)                                                          # 42
    OLD_GB = int(cfg.batch_size) * int(cfg.get("n_gpus", 1)); assert OLD_GB == REFERENCE_BATCH
    SCALE = math.sqrt(a.target_global_bs / OLD_GB); LR = float(cfg.optimizer.lr) * SCALE
    T, WU, CE, FL, PB = a.total, a.warmup, a.cos_end, a.floor, a.per_gpu_bs; GB = PB * world
    FACTOR = lambda u: lr_factor(u, WU, CE, FL)  # noqa: E731
    assert abs(FACTOR(0) - 1 / WU) < 1e-15 and FACTOR(WU - 1) == 1.0 and abs(FACTOR(CE - 1) - FL) < 1e-12 and FACTOR(CE) == FL and FACTOR(T - 1) == FL
    SAVE = sorted({int(x) for x in a.save.split(",") if x} | ({a.stop_after} if a.stop_after else set()))
    EXPORT = sorted({int(x) for x in a.export.split(",") if x})

    # ---- data: exactly prejepa.py run()
    encoding_keys = list(cfg.wm.encoding.keys())
    dataset = swm.data.load_dataset(cfg.dataset_name, num_steps=int(cfg.n_steps), frameskip=int(cfg.frameskip), transform=None,
                                    cache_dir=os.environ.get("LOCAL_DATASET_DIR", None), keys_to_load=["pixels"] + encoding_keys, keys_to_cache=encoding_keys)
    normalizers = [get_column_normalizer(dataset, col, col) for col in cfg.wm.encoding]
    dataset.transform = spt.data.transforms.Compose(PJ.get_img_preprocessor("pixels", "pixels", int(cfg.image_size)), *normalizers)
    for k, v in cfg.extra_dims.items():
        assert (dataset.get_dim(k) * (int(cfg.frameskip) if k == "action" else 1)) == int(v), (k, v)
    train_set, val_set = spt.data.random_split(dataset, [float(cfg.train_split), 1 - float(cfg.train_split)], generator=torch.Generator().manual_seed(SPLIT_SEED))
    N, NV = len(train_set), len(val_set)
    split_sha = hashlib.sha256(np.asarray(list(train_set.indices[:2000]) + list(val_set.indices[:2000]), dtype=np.int64).tobytes()).hexdigest()[:16]

    def shard(u, r=None):
        return epoch_stream(a.seed, N, u, rank if r is None else r, GB, PB)
    perm_sha = hashlib.sha256(np.asarray([shard(0, r) for r in range(world)] + [shard(1, r) for r in range(world)], dtype=np.int64).tobytes()).hexdigest()[:16]

    # ---- model: exactly prejepa.py run(); the predictor / embedder initialisation is seeded right before it is built (same for every encoder)
    encoder = hydra.utils.instantiate(cfg.model.encoder); encoder.eval(); encoder.requires_grad_(False)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    world_model = hydra.utils.instantiate(cfg.model, encoder=encoder).to(dev)
    TR = [p for p in world_model.parameters() if p.requires_grad]
    TRN = [n for n, p in world_model.named_parameters() if p.requires_grad]
    assert all(n.startswith("predictor.") or n.startswith("extra_encoders.") for n in TRN), [n for n in TRN if not (n.startswith("predictor.") or n.startswith("extra_encoders."))][:5]

    def h_params():
        h = hashlib.sha256()
        for n, p in world_model.named_parameters():
            if p.requires_grad:
                h.update(p.detach().float().cpu().numpy().tobytes())
        return h.hexdigest()[:16]

    class Shim:                                                                          # what dinowm_forward expects of the Lightning module
        model = world_model

        def log_dict(self, *args, **kw):
            pass
    SHIM = Shim()

    opt = torch.optim.AdamW(TR, lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, FACTOR)
    scaler = torch.cuda.amp.GradScaler()                                                 # Lightning 16-mixed default: init 2**16, growth 2 every 2000
    u0 = 0
    if a.resume:
        CK = torch.load(a.resume, map_location="cpu", weights_only=False)
        world_model.predictor.load_state_dict(CK["predictor"]); world_model.extra_encoders.load_state_dict(CK["extra_encoders"])
        opt.load_state_dict(CK["optimizer"]); scaler.load_state_dict(CK["scaler"]); sched.load_state_dict(CK["sched"])
        for grp, lrv in zip(opt.param_groups, sched.get_last_lr()):
            grp["lr"] = lrv
        assert CK["data"]["world"] == world and CK["data"]["per_gpu_bs"] == PB and CK["data"]["seed"] == a.seed and CK["data"]["perm_sha"] == perm_sha
        r_ = CK["rng"]; random.setstate(r_["python"]); np.random.set_state(r_["numpy"]); torch.set_rng_state(r_["torch"]); torch.cuda.set_rng_state(r_["cuda_per_rank"][rank])
        u0 = int(CK["data"]["next_update"])
    else:
        torch.cuda.manual_seed(a.seed * 1000 + 17 + rank)                                 # dropout stream of this GPU
    sha_all = [None] * world; dist.all_gather_object(sha_all, h_params())
    assert len(set(sha_all)) == 1, "trainable weights differ between GPUs: %s" % sha_all
    first = [None] * world; dist.all_gather_object(first, (rank, shard(u0)))
    allpos = sorted(i for _, s in first for i in s); assert len(allpos) == len(set(allpos)) == GB, "GPUs read overlapping samples"

    if rank == 0:
        OmegaConf.save(cfg, os.path.join(a.out, "template_config.yaml"))
        if not a.resume or not os.path.exists(os.path.join(a.out, "lr_plan.csv")):
            with open(os.path.join(a.out, "lr_plan.csv"), "w", newline="") as f:
                w = csv.writer(f); w.writerow(["update", "factor", "lr"])
                for u in range(T):
                    w.writerow([u + 1, "%.10f" % FACTOR(u), "%.10e" % (LR * FACTOR(u))])
    log(event="start", task="reacher", encoder=str(cfg.backbone.name), config=a.config, run_name=a.run_name, resume=bool(a.resume), src=a.resume,
        seed=a.seed, split_seed=SPLIT_SEED, world=world, per_gpu_bs=PB, global_bs=GB, reference_bs=OLD_GB, lr_scale=SCALE, peak_lr=LR,
        start_update=u0, N_train=N, N_val=NV, epoch_updates=N / GB, split_sha=split_sha, perm_sha=perm_sha, params_sha=sha_all[0],
        trainable_params=sum(p.numel() for p in TR), encoder_trainable=sum(p.numel() for p in encoder.parameters() if p.requires_grad),
        schedule={"warmup": WU, "cos_end": CE, "floor": FL, "total": T}, saves=SAVE, exports=EXPORT, stats_file=a.stats, stats_md5=md5(a.stats),
        torch=torch.__version__, nccl=list(torch.cuda.nccl.version()),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32)

    def allreduce_grads():
        gs = [p.grad for p in TR]
        flat = torch.cat([g.reshape(-1) for g in gs]); dist.all_reduce(flat); flat.div_(world)
        o = 0
        for g in gs:
            n = g.numel(); g.copy_(flat[o:o + n].view_as(g)); o += n

    VAL_IDX = list(range(min(a.val_clips, NV)))

    def val_loss():
        world_model.eval(); mine = VAL_IDX[rank::world]; tot = torch.zeros(2, device=dev)
        dl = torch.utils.data.DataLoader(torch.utils.data.Subset(val_set, mine), batch_size=PB, shuffle=False, num_workers=2, pin_memory=True)
        with torch.no_grad():
            for b in dl:
                b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
                with torch.autocast("cuda", dtype=torch.float16):
                    out = PJ.dinowm_forward(SHIM, b, "validate", cfg)
                n = b["pixels"].shape[0]; tot += torch.tensor([float(out["loss"]) * n, n], device=dev)
        dist.all_reduce(tot); world_model.train(); encoder.eval()                        # the frozen encoder always stays in eval mode
        return (tot[0] / tot[1]).item()

    def save(u):
        rng_c = [None] * world; dist.all_gather_object(rng_c, torch.cuda.get_rng_state().cpu())
        if rank == 0:
            path = os.path.join(a.out, "checkpoints", "model_s%d.pth" % u)
            ck = {"predictor": world_model.predictor.state_dict(), "extra_encoders": world_model.extra_encoders.state_dict(), "global_step": u,
                  "optimizer": opt.state_dict(), "scaler": scaler.state_dict(), "sched": sched.state_dict(),
                  "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda_per_rank": rng_c},
                  "data": {"seed": a.seed, "split_seed": SPLIT_SEED, "N": N, "perm_sha": perm_sha, "split_sha": split_sha, "next_update": u, "world": world,
                           "per_gpu_bs": PB, "global_bs": GB, "order": "per epoch e: RandomState(seed*100003+e).permutation(N_train) over train_set; continuous stream; GPU r takes [GB*u + PB*r, +PB)"},
                  "schedule": {"warmup": WU, "cos_end": CE, "floor": FL, "total": T, "peak_lr": LR, "lr_factor_next": FACTOR(u) if u < T else FL}}
            torch.save(ck, path + ".tmp"); os.replace(path + ".tmp", path)
            if u in EXPORT:
                world_model.eval()
                save_pretrained(world_model, run_name=a.run_name, config=cfg.model, filename="weights_s%d.pt" % u)
                world_model.train(); encoder.eval()
            log(event="save", global_step=u, path=path, params_sha=h_params(), exported=u in EXPORT, lr_next=opt.param_groups[0]["lr"], scale=scaler.get_scale())
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
    dl = torch.utils.data.DataLoader(train_set, batch_sampler=Shards(u0, stop), num_workers=a.workers, generator=g, pin_memory=True,
                                     prefetch_factor=4, persistent_workers=False)
    world_model.train(); encoder.eval()
    t0 = time.time(); acc = torch.zeros(3, device=dev); gmax = 0.0; gsum = 0.0; nclip = 0; nskip = 0
    for j, batch in enumerate(dl):
        u = u0 + j
        exp = LR * FACTOR(u)
        assert all(abs(gr["lr"] - exp) <= 1e-12 * exp + 1e-18 for gr in opt.param_groups), (u, opt.param_groups[0]["lr"], exp)
        batch = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            out = PJ.dinowm_forward(SHIM, batch, "fit", cfg)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        allreduce_grads()                                                                # mean over the GPUs = the gradient of the mean over 256
        scaler.unscale_(opt)
        gn = float(torch.nn.utils.clip_grad_norm_(TR, 1.0))                              # norm of the unscaled, averaged gradient, before clipping
        scale_before = scaler.get_scale()
        scaler.step(opt); scaler.update(); sched.step()
        if not math.isfinite(float(loss)):
            log(event="nonfinite", update=u + 1, loss=float(loss), grad_norm=gn); raise SystemExit("non-finite loss at update %d" % (u + 1))
        if math.isfinite(gn):
            gmax = max(gmax, gn); gsum += gn; nclip += int(gn > 1.0)
        if scaler.get_scale() < scale_before:
            nskip += 1
        acc += torch.tensor([float(loss), float(out["pixels_loss"]), 1.0], device=dev)
        if (u + 1) % a.log_every == 0 or u + 1 == stop:
            tot = acc.clone(); dist.all_reduce(tot); n = int(acc[2].item()); dt = time.time() - t0
            log(event="train", global_step=u + 1, loss=(tot[0] / tot[2]).item(), pixels_loss=(tot[1] / tot[2]).item(), grad_norm_mean=gsum / max(1, n - nskip),
                grad_norm_max=gmax, clipped=nclip, skipped=nskip, scale=scaler.get_scale(), lr_used=exp, lr_next=opt.param_groups[0]["lr"],
                it_s=n / dt, samples_s=n * GB / dt, max_mem_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
            t0 = time.time(); acc.zero_(); gmax = 0.0; gsum = 0.0; nclip = 0; nskip = 0
        if (u + 1) % a.val_every == 0 or (u + 1) in SAVE:
            vl = val_loss(); log(event="val", global_step=u + 1, val_loss=vl, val_clips=len(VAL_IDX))
        if u + 1 in SAVE:
            save(u + 1)
    log(event="done", last_update=stop, params_sha=h_params())
    if rank == 0:
        print("REACHER_WM_TRAIN_DONE", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
