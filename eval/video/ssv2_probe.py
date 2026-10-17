# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Something-Something v2 action recognition with an attentive probe on frozen tokens.

    export VJEPA2_ROOT=<checkout of github.com/facebookresearch/vjepa2 at 204698b45b3712590f06245fbfba32d3be539812>
    torchrun --nproc_per_node=8 eval/video/ssv2_probe.py --enc alps --weights alps_vitb16.safetensors \
        --data <dir with train.csv, holdout.csv, val.csv> --out runs/ssv2/alps --init_from <probe_init.pt>
    (or: bash eval/video/run_ssv2.sh <enc> <gpus> <accum> [arguments])

The frozen-encoder video classification evaluation of V-JEPA 2, with one fixed configuration for every encoder. The data
pipeline (VideoDataset, make_transforms) and the AttentiveClassifier are imported from the vjepa2 repository.
  data      16 frames with frame step 4 (decord), one clip per video and epoch. Training: RandAugment, RandomResizedCrop
            to 224 px (scale 0.08 to 1, aspect ratio 3/4 to 4/3), random erasing (probability 0.25), no flip. Evaluation:
            one temporal segment x 3 spatial views at a short side of 224 px, softmax-averaged. Clips are normalized with
            the ImageNet mean / std for every encoder; video_encoders.py converts them to each encoder's convention.
  probe     non-affine LayerNorm + linear adapter D -> 768 + learnable temporal embedding (zeros), then the
            AttentiveClassifier (depth 4 = 3 self-attention blocks + the cross-attention query block, 12 heads, width
            768) over all tokens of the clip, 174 classes. The encoder is frozen and runs under bfloat16 autocast.
  training  AdamW, learning rate 1e-3, weight decay 0.1 (constant), cosine schedule to 0 without warm-up over the
            optimizer steps, global batch 256 = GPUs x per-GPU batch x --accum, 20 epochs, bfloat16 autocast without a
            gradient scaler, torch seed 0.
  splits    train = official train minus a 2,000-video holdout (make_ssv2_csv.py; evaluated every epoch for
            monitoring only); reported = official validation (24,777 videos) at epoch 20.
  init      without --init_from the probe is drawn under torch seed 0; rank 0 saves the initialization as
            <out>/probe_init.pt (if not present). The runs of the paper started every encoder from one saved
            initialization (the probe of the first run, not released), adapted for V-JEPA 2 as in make_probe_init.py,
            which draws such a shared initialization under torch seed 0.
Deviations from the official V-JEPA 2 configuration are listed in NOTES (written to meta.json).
Outputs in --out: meta.json, log.jsonl, probe_init.pt, latest.pt (resume), probe_ep<N>.pt, val_preds_ep<N>.npz,
results_partial.json, results.json (val_top1_final = the reported number).
"""
import argparse
import contextlib
import datetime
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
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
if os.environ.get("VJEPA2_ROOT"):
    sys.path.insert(0, os.environ["VJEPA2_ROOT"])
try:
    from evals.video_classification_frozen.utils import make_transforms
    from src.datasets.video_dataset import VideoDataset
    from src.models.attentive_pooler import AttentiveClassifier
except ImportError as e:                           # pragma: no cover
    raise ImportError("set VJEPA2_ROOT to a checkout of github.com/facebookresearch/vjepa2 "
                      "(commit 204698b45b3712590f06245fbfba32d3be539812)") from e
from eval.video.video_encoders import IMAGENET, build_video_encoder  # noqa: E402


def notes(vjepa2_rope="interp"):
    rope = ("height / width RoPE positions of its 14 x 14 grid scaled to the 16 x 16 pretraining grid" if vjepa2_rope == "interp"
            else "RoPE position ids recomputed for the 14 x 14 grid")
    return [f"resolution 224 for every encoder (the official V-JEPA 2 SSv2 configuration uses 256); V-JEPA 2: {rope}",
            "the probe runs at width 768 after a linear adapter for every encoder (official: the encoder's own width; V-JEPA 2-L = 1024)",
            "probe heads 12 (head width 64, as the official 16 heads at 1024); learnable temporal embedding added (image encoders have no temporal position otherwise)",
            "probe activation checkpointing off unless --act_ckpt (numerically identical, only memory / speed)",
            "one fixed learning rate / weight decay for every encoder (official: a sweep of 5 x 4 on the validation set); no decision uses the validation set",
            "train = official train minus a 2,000-video holdout used for monitoring only",
            "train / eval use 1 temporal segment (official: 2 segments concatenated); on SSv2 the two official segments are identical clips for every video of at most 64 frames",
            "bfloat16 autocast without a gradient scaler (official: float16 autocast + GradScaler)",
            "evaluation clip sampling deterministic (random_clip_sampling=False); the official code leaves the default True"]


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--enc", required=True, help="encoder name (video_encoders.PAPER_VIDEO_ROWS)")
    p.add_argument("--weights", default=None, help="ALPS safetensors (default env ALPS_WEIGHTS / ALPS_STAGE1_WEIGHTS)")
    p.add_argument("--state_dict", default=None, help="weights loaded (strict) into a timm encoder, e.g. SigLIP 2 refined with Stage 2")
    p.add_argument("--vjepa2_rope", default="interp", choices=["interp", "recompute"])
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=os.environ.get("SSV2_DATA"), help="directory with train.csv, holdout.csv, val.csv (default env SSV2_DATA)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--global_batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--res", type=int, default=224)
    p.add_argument("--fpc", type=int, default=16, help="frames per clip")
    p.add_argument("--frame_step", type=int, default=4)
    p.add_argument("--views", type=int, default=3, help="spatial views at evaluation")
    p.add_argument("--probe_dim", type=int, default=768)
    p.add_argument("--probe_heads", type=int, default=12)
    p.add_argument("--probe_depth", type=int, default=4)
    p.add_argument("--num_classes", type=int, default=174)
    p.add_argument("--workers", type=int, default=6, help="data loader workers per GPU (evaluation: half, at least 2)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_epochs", default="10,20", help="epochs evaluated on the official validation set (plus the last)")
    p.add_argument("--save_epochs", default="10,20", help="epochs whose probe is saved (plus the last)")
    p.add_argument("--holdout_every", type=int, default=1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--limit_train", type=int, default=0, help="first N training videos only (smoke tests)")
    p.add_argument("--limit_eval", type=int, default=0, help="first N holdout / validation videos only (smoke tests)")
    p.add_argument("--max_steps", type=int, default=0, help="stop after N optimizer steps (the cosine then spans N steps)")
    p.add_argument("--accum", type=int, default=1, help="gradient accumulation micro-steps (global batch = GPUs x per-GPU batch x accum)")
    p.add_argument("--act_ckpt", type=int, default=0, help="activation checkpointing in the probe (memory only)")
    p.add_argument("--init_from", default="", help="probe_init.pt of a reference run: start from exactly that initialization")
    p.add_argument("--eval_only", default="", help="probe checkpoint to evaluate on the validation set (no training)")
    p.add_argument("--no_resume", action="store_true", help="ignore <out>/latest.pt")
    args = p.parse_args(argv)
    if not args.data:
        p.error("--data (or env SSV2_DATA) is required")
    return args


def cosine_lr(lr, total):
    """No warm-up, cosine from lr to 0 over `total` optimizer steps (held at 0 afterwards); weight decay constant."""
    return lambda t: 0.5 * lr * (1 + math.cos(math.pi * min(t, total) / total))


def total_steps(args, spe):
    return args.epochs * spe if not args.max_steps else min(args.max_steps, args.epochs * spe)


class TemporalProbe(nn.Module):
    def __init__(self, in_dim, t_tokens, num_classes, d=768, heads=12, depth=4, act_ckpt=False):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, elementwise_affine=False)
        self.adapter = nn.Linear(in_dim, d)
        self.tpos = nn.Parameter(torch.zeros(1, t_tokens, 1, d))
        self.clf = AttentiveClassifier(embed_dim=d, num_heads=heads, depth=depth, num_classes=num_classes,
                                       use_activation_checkpointing=bool(act_ckpt))
        nn.init.trunc_normal_(self.adapter.weight, std=0.02)
        nn.init.zeros_(self.adapter.bias)

    def forward(self, tok):                                  # [B, T', N, D_in]
        x = self.adapter(self.norm(tok)) + self.tpos[:, : tok.shape[1]]
        return self.clf(x.flatten(1, 2))


class VD(VideoDataset):
    strict = False

    def __getitem__(self, index):
        item = self.get_item_video(index)
        if item is None:
            if self.strict:
                raise RuntimeError(f"failed to load {self.samples[index]}")
            return super().__getitem__(index)                # official behavior: another random video (training only)
        return item


class Indexed(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        b, y, _ = self.ds[i]
        return i, b, int(y)


def make_dataset(csv, training, args, views=1, limit=0):
    tf = make_transforms(training=training, num_views_per_clip=views, random_horizontal_flip=False,
                         random_resize_aspect_ratio=(0.75, 4 / 3), random_resize_scale=(0.08, 1.0), reprob=0.25,
                         auto_augment=True, motion_shift=False, crop_size=args.res, normalize=IMAGENET)
    ds = VD(data_paths=[csv], datasets_weights=None, frames_per_clip=args.fpc, dataset_fpcs=None, duration=None, fps=None,
            frame_step=args.frame_step, num_clips=1, random_clip_sampling=training, allow_clip_overlap=True,
            filter_short_videos=False, filter_long_videos=int(1e9), shared_transform=None, transform=tf)
    ds.strict = not training
    if limit:                                                # valid for a single csv (one dataset)
        ds.samples, ds.labels = ds.samples[:limit], ds.labels[:limit]
    return Indexed(ds)


def worker_init(wid):
    s = (torch.initial_seed() + wid) % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def make_loader(ds, bs, shuffle, drop_last, workers, world, rank, seed, persistent=True, prefetch=2):
    """Evaluation loaders are created per evaluation, without persistent workers (a prefetched 3-view batch is ~1 GB of
    host memory, so persistent evaluation workers can exhaust host RAM)."""
    sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=shuffle, seed=seed, drop_last=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=workers, pin_memory=True, drop_last=drop_last,
                                     persistent_workers=(workers > 0 and persistent), worker_init_fn=worker_init,
                                     prefetch_factor=prefetch if workers > 0 else None)
    return dl, sampler


def to_clip(v, dev):
    return v.to(dev, non_blocking=True).permute(0, 2, 1, 3, 4)        # [B, C, T, H, W] -> [B, T, C, H, W]


@torch.no_grad()
def evaluate(loader, enc, probe, dev, n_total, world, rank):
    probe.eval()
    I, Y, P, PT = [], [], [], []
    for i, buf, y in loader:
        views = buf[0]
        probs = 0
        for v in views:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = probe(enc.forward_clip(to_clip(v, dev)))
            probs = probs + F.softmax(logit.float(), 1)
        probs = probs / len(views)
        pred = probs.argmax(1)
        y = y.to(dev)
        I.append(i.cpu().numpy()); Y.append(y.cpu().numpy()); P.append(pred.cpu().numpy())
        PT.append(probs.gather(1, y[:, None])[:, 0].cpu().numpy())
    I, Y, P, PT = (np.concatenate(a) if a else np.zeros(0, dtype=np.int64) for a in (I, Y, P, PT))
    if world > 1:
        g = [None] * world
        dist.all_gather_object(g, (I, Y, P, PT))
        I, Y, P, PT = (np.concatenate([x[k] for x in g]) for k in range(4))
    _, first = np.unique(I, return_index=True)               # the distributed sampler pads with repeated videos
    I, Y, P, PT = I[first], Y[first], P[first], PT[first]
    assert len(I) == n_total, (len(I), n_total)
    del loader
    probe.train()
    return {"top1": float((Y == P).mean()), "n": int(n_total)}, {"idx": I, "label": Y, "pred": P, "p_true": PT}


def _sha16(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    args = get_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))
        rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    dev = torch.device("cuda", local)
    torch.cuda.set_device(dev)
    is0 = rank == 0
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "log.jsonl"), "a") if is0 else None

    def P(*a):
        if is0:
            print(time.strftime("%H:%M:%S"), *a, flush=True)

    assert args.global_batch % (world * args.accum) == 0, "global batch must be a multiple of GPUs x accum"
    bs = args.global_batch // (world * args.accum)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    enc = build_video_encoder(args.enc, device=dev, weights=args.weights, state_dict=args.state_dict, res=args.res,
                              vjepa2_rope=args.vjepa2_rope)
    T_tok = enc.t_out(args.fpc)
    torch.manual_seed(args.seed)
    probe = TemporalProbe(enc.dim, T_tok, args.num_classes, args.probe_dim, args.probe_heads, args.probe_depth, args.act_ckpt).to(dev)
    if args.init_from:
        probe.load_state_dict(torch.load(args.init_from, map_location="cpu", weights_only=True), strict=True)
        P(f"probe initialized from {args.init_from}")
    init_path = os.path.join(args.out, "probe_init.pt")
    if is0 and not os.path.exists(init_path):
        torch.save(probe.state_dict(), init_path)
    pm = torch.nn.parallel.DistributedDataParallel(probe, device_ids=[local]) if ddp else probe
    s = args.seed * 1000 + rank + 1                           # per-rank augmentation streams
    torch.manual_seed(s); np.random.seed(s); random.seed(s)
    ds_tr = make_dataset(os.path.join(args.data, "train.csv"), True, args, views=1, limit=args.limit_train)
    ds_ho = make_dataset(os.path.join(args.data, "holdout.csv"), False, args, views=args.views, limit=args.limit_eval)
    ds_va = make_dataset(os.path.join(args.data, "val.csv"), False, args, views=args.views, limit=args.limit_eval)
    dl_tr, sm_tr = make_loader(ds_tr, bs, True, True, args.workers, world, rank, args.seed, persistent=True, prefetch=2)
    ew = max(2, args.workers // 2)

    def eval_loader(ds):                                      # built per evaluation; its workers exit afterwards
        return make_loader(ds, bs, False, False, ew, world, rank, args.seed, persistent=False, prefetch=2)[0]

    spe = len(dl_tr) // args.accum
    total = total_steps(args, spe)
    opt = torch.optim.AdamW(pm.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_at = cosine_lr(args.lr, total)
    step, start_ep = 0, 0
    if args.eval_only:
        sd = torch.load(args.eval_only, map_location="cpu", weights_only=True)
        probe.load_state_dict(sd["probe"] if "probe" in sd else sd)
        r, arr = evaluate(eval_loader(ds_va), enc, probe, dev, len(ds_va), world, rank)
        P("eval_only val", r)
        if is0:
            np.savez(os.path.join(args.out, "val_preds_evalonly.npz"), **arr)
            with open(os.path.join(args.out, "eval_only.json"), "w") as f:
                json.dump(r, f)
        return
    latest = os.path.join(args.out, "latest.pt")
    if not args.no_resume and os.path.exists(latest):
        ck = torch.load(latest, map_location="cpu", weights_only=True)
        probe.load_state_dict(ck["probe"]); opt.load_state_dict(ck["opt"]); step, start_ep = ck["step"], ck["epoch"]
        P(f"resumed from epoch {start_ep} step {step}")
        s = args.seed * 1000 + rank + 1 + 7919 * start_ep     # fresh augmentation streams after a resume
        torch.manual_seed(s); np.random.seed(s); random.seed(s)
    splits = os.path.join(args.data, "splits.json")
    meta = {"enc": enc.meta(), "args": vars(args), "world": world, "per_gpu_batch": bs, "steps_per_epoch": spe, "total_steps": total,
            "n_train": len(ds_tr), "n_holdout": len(ds_ho), "n_val": len(ds_va), "probe_params": sum(p.numel() for p in probe.parameters()),
            "tokens_per_clip": T_tok * enc.grid(args.res, args.res)[0] * enc.grid(args.res, args.res)[1],
            "gpu": torch.cuda.get_device_name(dev), "torch": torch.__version__,
            "code_sha256_16": {"ssv2_probe.py": _sha16(os.path.join(HERE, "ssv2_probe.py")),
                               "video_encoders.py": _sha16(os.path.join(HERE, "video_encoders.py")),
                               "encoders.py": _sha16(os.path.join(HERE, "..", "encoders.py"))},
            "splits": json.load(open(splits)) if os.path.exists(splits) else None, "notes": notes(args.vjepa2_rope),
            "start": time.strftime("%Y-%m-%d %H:%M:%S")}
    if is0:
        with open(os.path.join(args.out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        P(json.dumps({k: meta[k] for k in ["world", "per_gpu_batch", "steps_per_epoch", "total_steps", "n_train", "n_val",
                                           "probe_params", "tokens_per_clip"]}))
    eval_eps = {int(e) for e in args.eval_epochs.split(",") if e}
    save_eps = {int(e) for e in args.save_epochs.split(",") if e}
    t_train = t_eval = 0.0
    results = {"val": {}, "holdout": {}}
    partial = os.path.join(args.out, "results_partial.json")
    if os.path.exists(partial):
        results = json.load(open(partial))
    for ep in range(start_ep, args.epochs):
        sm_tr.set_epoch(ep)
        pm.train()
        t0 = time.time()
        nseen = ncorr = 0
        lsum = 0.0
        for k, (_, buf, y) in enumerate(dl_tr):
            if step >= total:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            x = to_clip(buf[0][0], dev)
            y = y.to(dev, non_blocking=True)
            last_micro = (k % args.accum) == args.accum - 1
            with (pm.no_sync() if (ddp and not last_micro) else contextlib.nullcontext()):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    with torch.no_grad():
                        tok = enc.forward_clip(x)
                    logit = pm(tok)
                    loss = F.cross_entropy(logit.float(), y)
                (loss / args.accum).backward()
            if last_micro:
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            nseen += len(y)
            ncorr += (logit.argmax(1) == y).sum().item()
            lsum += loss.item() * len(y)
            if is0 and (k % (args.log_every * args.accum) == 0):
                P(f"ep {ep + 1} it {k}/{spe} step {step} loss {lsum / max(1, nseen):.3f} acc {ncorr / max(1, nseen):.3f} "
                  f"lr {lr_at(step):.2e} {nseen * world / (time.time() - t0):.1f} clips/s mem {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G")
        t_ep = time.time() - t0
        t_train += t_ep
        st = torch.tensor([nseen, ncorr, lsum], device=dev, dtype=torch.float64)
        if ddp:
            dist.all_reduce(st)
        rec = {"epoch": ep + 1, "step": step, "train_loss": float(st[2] / st[0]), "train_acc": float(st[1] / st[0]),
               "train_time_s": t_ep, "clips_per_s": float(st[0] / t_ep)}
        if args.holdout_every and ((ep + 1) % args.holdout_every == 0 or ep + 1 == args.epochs):
            t1 = time.time()
            r, _ = evaluate(eval_loader(ds_ho), enc, probe, dev, len(ds_ho), world, rank)
            rec["holdout_top1"] = r["top1"]; rec["holdout_time_s"] = time.time() - t1
            t_eval += rec["holdout_time_s"]; results["holdout"][str(ep + 1)] = r["top1"]
        if (ep + 1) in eval_eps or ep + 1 == args.epochs:
            t1 = time.time()
            r, arr = evaluate(eval_loader(ds_va), enc, probe, dev, len(ds_va), world, rank)
            rec["val_top1"] = r["top1"]; rec["val_time_s"] = time.time() - t1
            t_eval += rec["val_time_s"]; results["val"][str(ep + 1)] = r["top1"]
            if is0:
                np.savez(os.path.join(args.out, f"val_preds_ep{ep + 1}.npz"), **arr,
                         video=np.array([os.path.basename(ds_va.ds.samples[i]) for i in arr["idx"]]))
        if is0:
            P(json.dumps(rec)); log.write(json.dumps(rec) + "\n"); log.flush()
            torch.save({"probe": probe.state_dict(), "opt": opt.state_dict(), "step": step, "epoch": ep + 1}, latest)
            if (ep + 1) in save_eps or ep + 1 == args.epochs:
                torch.save(probe.state_dict(), os.path.join(args.out, f"probe_ep{ep + 1}.pt"))
            results.update({"train_time_s": t_train, "eval_time_s": t_eval})
            with open(partial, "w") as f:
                json.dump(results, f, indent=1)
        if ddp:
            dist.barrier()
        if step >= total:
            break
    if is0:
        fin = {"enc": args.enc, "val_top1_final": results["val"].get(str(args.epochs)), "val_by_epoch": results["val"],
               "holdout_by_epoch": results["holdout"], "train_time_s": t_train, "eval_time_s": t_eval,
               "gpu_hours": (t_train + t_eval) * world / 3600, "world": world, "end": time.strftime("%Y-%m-%d %H:%M:%S"), "meta": meta}
        with open(os.path.join(args.out, "results.json"), "w") as f:
            json.dump(fin, f, indent=1)
        P("DONE", json.dumps({k: fin[k] for k in ["enc", "val_top1_final", "gpu_hours"]}))
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
