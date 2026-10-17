# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Stage 1 of ALPS (PyTorch): image-text pretraining with contrastive, caption, reconstruction and dense objectives.

    torchrun --nproc_per_node=8 train_stage1.py --config=configs/stage1a.py --workdir=runs/stage1a \
        --config.data.source='/data/shards/{000000..004095}.tar'
    torchrun --nproc_per_node=8 train_stage1.py --config=configs/stage1b.py --workdir=runs/stage1b \
        --config.data.source=... --config.init_from=runs/stage1a

The losses are computed with the semantics of one global batch (alps_torch.distributed). The global batch must be
divisible by the number of processes; scale the learning rate linearly when you change the batch. A run resumes
from the newest checkpoint in <workdir>/checkpoints. At the end, <workdir>/alps_stage1.safetensors holds the
encoder and the language interface in the format of the `alps` inference package.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alps_torch import convert  # noqa: E402
from alps_torch.checkpoint import CheckpointManager, load_stage1_state_dict  # noqa: E402
from alps_torch.config import dump, parse_args  # noqa: E402
from alps_torch.data.datasets import Stage1Dataset, make_loader  # noqa: E402
from alps_torch.distributed import device_for_rank, init_from_env  # noqa: E402
from alps_torch.engine import stage1_train_step  # noqa: E402
from alps_torch.models.stage1 import project_logit_scale, stage1_model_from_config  # noqa: E402
from alps_torch.optim import stage1_optimizer  # noqa: E402
from alps_torch.runtime import MetricLogger, compute_dtype, set_matmul_precision, to_device  # noqa: E402


def build_model(cfg, device):
    torch.manual_seed(cfg.seed)
    return stage1_model_from_config(cfg.model, compute_dtype(cfg.precision), cfg.attn_impl, cfg.res).to(device)


def main(argv=None):
    cfg, workdir = parse_args(argv)
    dist = init_from_env()
    device = device_for_rank()
    set_matmul_precision(cfg.allow_tf32)
    os.makedirs(workdir, exist_ok=True)
    if cfg.global_batch % dist.world:
        raise ValueError(f"global batch {cfg.global_batch} is not divisible by {dist.world} processes")
    model = build_model(cfg, device)
    if cfg.init_from:                                     # Stage 1b: every parameter of Stage 1a, a fresh optimizer
        sd = load_stage1_state_dict(cfg.init_from)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise ValueError(f"init_from does not match the model: missing {missing[:5]}, unexpected {unexpected[:5]}")
        print(f"[stage1] initialized from {cfg.init_from}; optimizer and schedule start fresh", flush=True)
    project_logit_scale(model)
    opt = stage1_optimizer(model, cfg.peak_lr, cfg.total_steps, cfg.warmup_steps, cfg.weight_decay, cfg.b1, cfg.b2)
    mngr = CheckpointManager(os.path.join(workdir, "checkpoints"), cfg.ckpt_keep)
    start = 0
    if mngr.latest_step() is not None:
        st = mngr.load(map_location="cpu")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        start = int(st["step"])
        print(f"[stage1] resumed from step {start}", flush=True)
    if dist.rank == 0:
        dump(cfg, os.path.join(workdir, "config.json"))
    ddp = model
    if dist.distributed:
        ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    ds = Stage1Dataset(cfg.data.source, cfg.res, seed=cfg.seed * 1_000_003 + start, rank=dist.rank, world=dist.world,
                       image_key=cfg.data.image_key, web_key=cfg.data.web_key, synthetic_key=cfg.data.synthetic_key,
                       itc_len=cfg.model.itc_len, caption_len=cfg.model.caption_len,
                       shuffle_buffer=cfg.data.shuffle_buffer)
    it = iter(make_loader(ds, cfg.global_batch // dist.world, cfg.data.num_workers))
    log = MetricLogger(workdir, dist.rank == 0)
    t0 = time.time()
    for step in range(start, cfg.total_steps):
        batch = to_device(next(it), device)
        m = stage1_train_step(ddp, opt, batch, cfg, step, dist)
        if (step + 1) % cfg.log_steps == 0 or step == start:
            log.log(step + 1, {**m, "sec_per_step": (time.time() - t0) / (step + 1 - start)})
        if dist.rank == 0 and ((step + 1) % cfg.ckpt_steps == 0 or (cfg.ckpt_final and step + 1 == cfg.total_steps)):
            mngr.save(step + 1, {"step": step + 1, "model": model.state_dict(), "optimizer": opt.state_dict(),
                                 "config": cfg.to_dict()})
    if dist.rank == 0:
        out = os.path.join(workdir, "alps_stage1.safetensors")
        convert.export_alps(out, model.state_dict(), metadata={"name": "alps-stage1", "steps": str(cfg.total_steps)})
        print(f"[stage1] wrote {out}", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
