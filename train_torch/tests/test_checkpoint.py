# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint round trip: an interrupted run (save after 3 updates, restore into fresh objects, 2 more) equals the
uninterrupted run bit for bit, with the bfloat16 first moment restored as bfloat16."""
import torch

from alps_torch.checkpoint import CheckpointManager
from alps_torch.models import CrossViewFusion, FeatHead, ViT
from alps_torch.models.stage2 import Stage2Student
from alps_torch.optim import stage2_optimizer


def _make():
    torch.manual_seed(0)
    return Stage2Student(ViT(32, 1, 64, 4, img_size=32), CrossViewFusion(16, 1, 4, 2, 32, 8), FeatHead(32, 32))


def _step(student, opt, i):
    g = torch.Generator().manual_seed(100 + i)
    opt.zero_grad(set_to_none=True)
    for p in student.parameters():
        if p.requires_grad:
            p.grad = torch.randn(p.shape, generator=g) * (5.0 if i % 2 else 0.01)
    opt.step()


def test_resume_is_exact(tmp_path):
    ref = _make()
    opt = stage2_optimizer(ref, 1e-2, 10, 2)
    for i in range(5):
        _step(ref, opt, i)
    s = _make()
    o = stage2_optimizer(s, 1e-2, 10, 2)
    for i in range(3):
        _step(s, o, i)
    mngr = CheckpointManager(str(tmp_path), keep=2)
    mngr.save(3, {"step": 3, "student": s.state_dict(), "optimizer": o.state_dict()})
    st = mngr.load()
    s2 = _make()
    o2 = stage2_optimizer(s2, 1e-2, 10, 2)
    s2.load_state_dict(st["student"])
    o2.load_state_dict(st["optimizer"])
    assert o2.count == 3 and all(v["exp_avg"].dtype == torch.bfloat16 for v in o2.state.values())
    for i in range(3, 5):
        _step(s2, o2, i)
    for (k, a), b in zip(ref.state_dict().items(), s2.state_dict().values()):
        assert torch.equal(a, b), k
    assert mngr.steps() == [3]
