# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Stage 2 of ALPS (PyTorch): cross-view self-distillation on single images.

    torchrun --nproc_per_node=8 train_stage2.py --config=configs/stage2.py --workdir=runs/stage2 \
        --config.stage1_ckpt=runs/stage1b --config.data.sources="(('/data/images.jsonl', 1.0),)"

The student encoder starts from the Stage-1 encoder; a frozen copy of it is the teacher. Each update averages the
gradients of `accum_steps` micro-batches (ALPS: 4 x 256 = 1,024 images), then clips the global norm at 1. The
language interface of Stage 1 is not touched: <workdir>/alps.safetensors combines the Stage-2 encoder with the
Stage-1 text encoder, projections and logit scale (the released ALPS model); <workdir>/alps_encoder.safetensors
holds the encoder alone. `stage1_ckpt` may be a train_stage1.py workdir / checkpoint, an `alps` export, or a flat
.npz of JAX Stage-1 params.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alps_torch import convert  # noqa: E402
from alps_torch.checkpoint import CheckpointManager, encoder_state_dict, load_stage1_state_dict  # noqa: E402
from alps_torch.config import dump, parse_args  # noqa: E402
from alps_torch.data.datasets import Stage2Dataset, make_loader  # noqa: E402
from alps_torch.distributed import device_for_rank, init_from_env  # noqa: E402
from alps_torch.engine import stage2_train_step  # noqa: E402
from alps_torch.models import CrossViewFusion, FeatHead, ViT  # noqa: E402
from alps_torch.models.stage2 import Stage2Student  # noqa: E402
from alps_torch.optim import stage2_optimizer  # noqa: E402
from alps_torch.runtime import MetricLogger, compute_dtype, set_matmul_precision, to_device  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _asset(path: str) -> str:
    """Relative asset paths are looked up in train_torch/ and then in train_jax/ (where the ALPS constants live)."""
    for base in ("", HERE, os.path.join(os.path.dirname(HERE), "train_jax")):
        p = os.path.join(base, path) if base else path
        if os.path.exists(p):
            return p
    raise FileNotFoundError(path)


def build(cfg, device):
    dt = compute_dtype(cfg.precision)
    ec, pc = cfg.encoder, cfg.predictor
    mk = lambda: ViT(ec.width, ec.depth, ec.mlp_dim, ec.num_heads, ec.patch_size, img_size=cfg.res, dtype=dt,
                     remat=cfg.remat, attn_impl=cfg.attn_impl)
    s1 = load_stage1_state_dict(cfg.stage1_ckpt)
    enc_sd = encoder_state_dict(s1)
    teacher, encoder = mk(), mk()
    teacher.load_state_dict(enc_sd, strict=True)
    encoder.load_state_dict(enc_sd, strict=True)
    teacher.requires_grad_(False).eval()
    torch.manual_seed(cfg.seed)
    grid = cfg.res // ec.patch_size
    pred = CrossViewFusion(pc.width, pc.depth, pc.num_heads, grid, ec.width, pc.text_dim, pc.mlp_ratio, dtype=dt)
    head = FeatHead(ec.width, ec.width, dt)
    student = Stage2Student(encoder, pred, head)
    return student.to(device), teacher.to(device), s1


def main(argv=None):
    cfg, workdir = parse_args(argv)
    dist = init_from_env()
    device = device_for_rank()
    set_matmul_precision(cfg.allow_tf32)
    os.makedirs(workdir, exist_ok=True)
    micro = cfg.global_batch // cfg.accum_steps
    if cfg.global_batch % cfg.accum_steps or micro % dist.world:
        raise ValueError("global_batch / accum_steps must be a multiple of the number of processes")
    student, teacher, s1 = build(cfg, device)
    zs = np.load(_asset(cfg.token_stats))
    zstats = (torch.tensor(zs["mean"], dtype=torch.float32, device=device)[None, None],
              torch.tensor(zs["std"], dtype=torch.float32, device=device)[None, None])
    u0 = torch.tensor(np.load(_asset(cfg.empty_token))[cfg.empty_token_key], dtype=torch.float32, device=device)
    if zstats[0].shape[-1] != cfg.encoder.width or u0.shape[-1] != cfg.predictor.text_dim:
        raise ValueError("token statistics / constant token do not match the encoder / predictor widths")
    opt = stage2_optimizer(student, cfg.base_lr, cfg.total_steps, cfg.warmup_steps, cfg.encoder_lr_ratio,
                           cfg.weight_decay, cfg.b1, cfg.b2, cfg.clip_norm)
    mngr = CheckpointManager(os.path.join(workdir, "checkpoints"), cfg.ckpt_keep)
    start = 0
    if mngr.latest_step() is not None:
        st = mngr.load(map_location="cpu")
        student.load_state_dict(st["student"])
        opt.load_state_dict(st["optimizer"])
        start = int(st["step"])
        print(f"[stage2] resumed from step {start}", flush=True)
    if dist.rank == 0:
        dump(cfg, os.path.join(workdir, "config.json"))
    ddp = student
    if dist.distributed:
        ddp = torch.nn.parallel.DistributedDataParallel(student, device_ids=[device.index] if device.type == "cuda" else None)
    ds = Stage2Dataset([tuple(s) for s in cfg.data.sources], cfg.res, seed=cfg.seed * 1_000_003 + start,
                       rank=dist.rank, world=dist.world, crossview_cfg=cfg.crossview.to_dict(),
                       shuffle_buffer=cfg.data.shuffle_buffer)
    it = iter(make_loader(ds, micro // dist.world, cfg.data.num_workers))
    log = MetricLogger(workdir, dist.rank == 0)
    t0 = time.time()
    for step in range(start, cfg.total_steps):
        batches = [to_device(next(it), device) for _ in range(cfg.accum_steps)]
        m = stage2_train_step(ddp, teacher, opt, batches, cfg, dist, zstats=zstats, u0=u0)
        if (step + 1) % cfg.log_steps == 0 or step == start:
            log.log(step + 1, {**m, "sec_per_step": (time.time() - t0) / (step + 1 - start)})
        if dist.rank == 0 and ((step + 1) % cfg.ckpt_steps == 0 or (cfg.ckpt_final and step + 1 == cfg.total_steps)):
            mngr.save(step + 1, {"step": step + 1, "student": student.state_dict(), "optimizer": opt.state_dict(),
                                 "config": cfg.to_dict()})
    if dist.rank == 0:
        enc = student.encoder.state_dict()
        out = os.path.join(workdir, "alps.safetensors")
        meta = {"name": "alps", "steps": str(cfg.total_steps)}
        if any(k.startswith("text.") for k in s1):
            convert.export_alps(out, s1, encoder_sd=enc, metadata=meta)
            print(f"[stage2] wrote {out} (Stage-2 encoder + Stage-1 language interface)", flush=True)
        convert.export_alps(os.path.join(workdir, "alps_encoder.safetensors"), encoder_sd=enc, encoder_only=True,
                            metadata=meta)
    dist.barrier()


if __name__ == "__main__":
    main()
