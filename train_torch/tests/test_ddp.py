# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Global-batch semantics under DistributedDataParallel: 2 and 4 CPU processes (gloo) vs one process on the whole
batch, with the same random draws. Stage 1: loss value and every gradient (ITC over the gathered batch, caption
loss over the global token count, cross-image negatives gathered from their owners) and the parameters after one
optimizer update. Stage 2: accumulation over micro-batches, clipping and one update."""
import os
import socket
import tempfile

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from alps_torch.config import Config
from alps_torch.distributed import Dist
from alps_torch.engine import stage1_forward_loss, stage2_accumulate
from alps_torch.models import CrossViewFusion, FeatHead, Stage1Model, ViT, project_logit_scale
from alps_torch.models.stage2 import Stage2Student
from alps_torch.optim import stage1_optimizer, stage2_optimizer

TINY = dict(width=64, depth=2, mlp_dim=128, num_heads=4, remat=False, vocab_size=120, text_width=64, text_depth=2,
            text_heads=4, text_mlp_dim=128, cap_depth=2, cap_heads=4, cap_mlp_dim=128, itc_proj_dim=64,
            dense_proj_dim=32)
CFG = Config(seed=0, cap_w=1.0, rec_w=1.0, dense_w=0.25, dense_n_cross=12, dense_radius=1, dense_n_same=5,
             dense_split=(0.75, 0.25), peak_lr=1e-2)
B, RES = 8, 64


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ids(rng, b, length, vocab=120):
    out = np.zeros((b, length), np.int64)
    for i in range(b):
        n = int(rng.integers(3, length + 6))
        seq = ([1] + list(rng.integers(3, vocab, size=min(n, length - 3))) + [2] + [0] * length)[:length - 1]
        if n >= length - 3:
            seq[-1] = 2
        out[i] = seq + [101]
    return torch.from_numpy(out)


def stage1_data():
    rng = np.random.default_rng(0)
    g = RES // 16
    batch = {"x_rgb": torch.from_numpy(rng.random((B, 3, RES, RES)).astype(np.float32)),
             "gx_rgb": torch.from_numpy(rng.random((B, 3, RES, RES)).astype(np.float32)),
             "crop_box": torch.from_numpy(np.stack([[rng.random() * 0.3, rng.random() * 0.3, 0.7, 0.7] for _ in range(B)]).astype(np.float32)),
             "flip": torch.from_numpy(rng.random(B) < 0.5),
             "itc_labels1": _ids(rng, B, 32), "itc_labels2": _ids(rng, B, 32), "labels": _ids(rng, B, 80)}
    draws = {"noise": torch.from_numpy(rng.normal(size=(B, g * g, TINY["width"])).astype(np.float32)),
             "gumbel_u": torch.from_numpy(rng.uniform(1e-9, 1.0, size=(B, g * g, g * g)).astype(np.float32)),
             "cross_idx": torch.from_numpy(rng.integers(0, B * g * g, size=CFG.dense_n_cross))}
    return batch, draws


def _shard(d, rank, world):
    out = {}
    for k, v in d.items():
        out[k] = v if k == "cross_idx" else v[rank * (B // world):(rank + 1) * (B // world)]
    return out


def _stage1_run(model, batch, draws, d):
    opt = stage1_optimizer(model.module if hasattr(model, "module") else model, CFG.peak_lr, 10, 1)
    loss, m = stage1_forward_loss(model, batch, CFG, 0, d, draws)
    loss.backward()
    mod = model.module if hasattr(model, "module") else model
    grads = {n: p.grad.detach().clone() for n, p in mod.named_parameters() if p.grad is not None}
    metrics = {k: float(d.mean_nograd(v)) for k, v in m.items()}
    opt.step()        # update 1 has lr 0: take a second step to move the parameters
    opt.zero_grad(set_to_none=True)
    loss, _ = stage1_forward_loss(model, batch, CFG, 1, d, draws)
    loss.backward()
    opt.step()
    project_logit_scale(mod)
    return metrics, grads, {n: p.detach().clone() for n, p in mod.state_dict().items()}


def _worker(rank, world, port, init_path, out_dir, stage):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.set_num_threads(1)
    try:
        d = Dist()
        if stage == 1:
            model = Stage1Model(**TINY)
            model.load_state_dict(torch.load(init_path, weights_only=True))
            ddp = torch.nn.parallel.DistributedDataParallel(model)
            batch, draws = stage1_data()
            res = _stage1_run(ddp, _shard(batch, rank, world), _shard(draws, rank, world), d)
        else:
            student, teacher, zstats, u0 = _stage2_models()
            student.load_state_dict(torch.load(init_path, weights_only=True))
            ddp = torch.nn.parallel.DistributedDataParallel(student)
            res = _stage2_run(ddp, teacher, [_shard(b, rank, world) for b in stage2_data()], zstats, u0, d)
        torch.save(res, os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _leaves(d):
    """Split the fused qkv biases so that the key part (exact gradient zero) can be treated separately."""
    out = {}
    for k, v in d.items():
        if k.endswith("attn.qkv.bias"):
            q, kk, vv = v.chunk(3)
            out.update({k + ".q": q, k + ".k": kk, k + ".v": vv})
        else:
            out[k] = v
    return out


def _exact_zero_grad(name):
    return name.endswith("qkv.bias.k") or name.endswith(".k.bias")     # attention key biases


def _compare(single, multi, tol=2e-5, lr=1e-2):
    """Loss terms and gradients to float rounding. Parameters after the updates: Adam turns last-bit gradient
    differences (the summation order of the all-reduce) into visible differences for a handful of entries: entries
    whose gradient is a near-total cancellation (|g| ~ 1e-8 while typical entries are ~1e-4) get an update
    m / sqrt(v) of essentially random sign, and the bfloat16 first moment occasionally rounds to the neighbouring
    value. So: 99 % of the entries agree to 1e-4 of the step size and none differs by half a step.
    Leaves whose exact gradient is zero (attention key biases) are pure rounding noise and are skipped."""
    m1, g1, p1 = single
    mm, gm, pm = multi
    for k in m1:
        assert mm[k] == pytest.approx(m1[k], rel=2e-6, abs=1e-6), (k, mm[k], m1[k])
    assert g1.keys() == gm.keys()
    g1, gm, p1, pm = _leaves(g1), _leaves(gm), _leaves(p1), _leaves(pm)
    scale = max(float(v.abs().max()) for v in g1.values())
    worst = 0.0
    for k in g1:
        if _exact_zero_grad(k):
            assert float(gm[k].abs().max()) <= 1e-6 * scale and float(g1[k].abs().max()) <= 1e-6 * scale, k
            continue
        err = float((gm[k] - g1[k]).abs().max()) / max(float(g1[k].abs().max()), 1e-3 * scale)
        worst = max(worst, err)
        assert err < tol, (k, err)
    diffs = []
    for k in p1:
        if _exact_zero_grad(k):
            continue
        diffs.append((pm[k].float() - p1[k].float()).abs().flatten() / lr)       # in units of the step size
    d = torch.cat(diffs)
    q99, dmax = float(torch.quantile(d[:1_000_000], 0.99)), float(d.max())
    print(f"worst gradient error {worst:.2e}; parameter difference / lr: 99% quantile {q99:.1e}, max {dmax:.1e}")
    assert q99 < 1e-4 and dmax < 0.5


def _ranks_identical(paths):
    p0 = torch.load(paths[0], weights_only=False)[2]
    for p in paths[1:]:
        pr = torch.load(p, weights_only=False)[2]
        assert all(torch.equal(p0[k], pr[k]) for k in p0)


@pytest.mark.parametrize("world", [2, 4])
def test_stage1_ddp_equals_single_process(world, tmp_path):
    torch.manual_seed(0)
    model = Stage1Model(**TINY)
    init = str(tmp_path / "init.pt")
    torch.save(model.state_dict(), init)
    batch, draws = stage1_data()
    single = _stage1_run(model, batch, draws, Dist())
    mp.spawn(_worker, args=(world, _free_port(), init, str(tmp_path), 1), nprocs=world, join=True)
    _ranks_identical([tmp_path / f"rank{r}.pt" for r in range(world)])
    _compare(single, torch.load(tmp_path / "rank0.pt", weights_only=False))


# ------------------------------------------------------------------------------------------------------- Stage 2
def _stage2_models():
    torch.manual_seed(1)
    g, D = RES // 16, 64
    teacher = ViT(D, 2, 128, 4).requires_grad_(False)
    student = Stage2Student(ViT(D, 2, 128, 4), CrossViewFusion(32, 2, 4, g, D, 48), FeatHead(D, D))
    student.encoder.load_state_dict(teacher.state_dict())
    with torch.no_grad():
        for p in student.encoder.parameters():
            p.add_(0.01 * torch.randn_like(p))
    rng = np.random.default_rng(3)
    zstats = (torch.from_numpy(rng.normal(size=(1, 1, D)).astype(np.float32)),
              torch.from_numpy((rng.random((1, 1, D)) + 0.5).astype(np.float32)))
    return student, teacher, zstats, torch.from_numpy(rng.normal(size=48).astype(np.float32))


def stage2_data(n_micro=2):
    rng = np.random.default_rng(5)
    out = []
    for _ in range(n_micro):
        sa = rng.random((B, 1)) * 0.5 + 0.3
        sb = rng.random((B, 1)) * 0.5 + 0.4
        out.append({"x_s_rgb": torch.from_numpy(rng.random((B, 3, RES, RES)).astype(np.float32)),
                    "x_t_rgb": torch.from_numpy(rng.random((B, 3, RES, RES)).astype(np.float32)),
                    "box_a": torch.from_numpy(np.concatenate([rng.random((B, 2)) * (1 - sa), sa], 1).astype(np.float32)),
                    "box_b": torch.from_numpy(np.concatenate([rng.random((B, 2)) * (1 - sb), sb], 1).astype(np.float32))})
    return out


def _stage2_run(student, teacher, batches, zstats, u0, d):
    mod = student.module if hasattr(student, "module") else student
    opt = stage2_optimizer(mod, 1e-2, 10, 1, clip_norm=0.5)
    metrics = {}
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        m = stage2_accumulate(student, teacher, batches, zstats=zstats, u0=u0, dist=d)
        if step == 0:
            grads = {n: p.grad.detach().clone() for n, p in mod.named_parameters() if p.grad is not None}
            metrics = {k: float(d.mean_nograd(v)) for k, v in m.items()}
        opt.step()
    return metrics, grads, {n: p.detach().clone() for n, p in mod.state_dict().items()}


@pytest.mark.parametrize("world", [2, 4])
def test_stage2_ddp_equals_single_process(world, tmp_path):
    student, teacher, zstats, u0 = _stage2_models()
    init = str(tmp_path / "init.pt")
    torch.save(student.state_dict(), init)
    single = _stage2_run(student, teacher, stage2_data(), zstats, u0, Dist())
    mp.spawn(_worker, args=(world, _free_port(), init, str(tmp_path), 2), nprocs=world, join=True)
    _ranks_identical([tmp_path / f"rank{r}.pt" for r in range(world)])
    _compare(single, torch.load(tmp_path / "rank0.pt", weights_only=False))
