# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the SSv2 probe (ssv2_probe.py, make_probe_init.py), with the vjepa2 modules replaced by stubs.

    pytest eval/video/tests/test_ssv2_probe.py

* learning-rate schedule, protocol defaults and the V-JEPA 2 probe initialization;
* the probe module (shapes, forward) and the construction of the vjepa2 data pipeline.
"""
import argparse
import os
import sys
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common  # noqa: E402,F401

CALLS = []


class AttentiveClassifier(nn.Module):
    """Stand-in for the vjepa2 AttentiveClassifier: draws parameters from the torch generator like a real module."""

    def __init__(self, embed_dim=768, num_heads=12, depth=1, num_classes=1000, use_activation_checkpointing=False, **kw):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(depth - 1)])
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        for b in self.blocks:
            x = x + torch.tanh(b(x))
        a = torch.softmax((x @ self.query.transpose(1, 2)).squeeze(-1), dim=1)
        return self.linear((a[..., None] * x).sum(1))


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, **kw):
        CALLS.append(("VideoDataset", kw))
        self.samples, self.labels = [f"v{i}.webm" for i in range(10)], list(range(10))

    def __len__(self):
        return len(self.samples)


def make_transforms(**kw):
    CALLS.append(("make_transforms", kw))
    return kw


def _install_stubs():
    mods = {"src": None, "src.models": None, "src.models.attentive_pooler": {"AttentiveClassifier": AttentiveClassifier},
            "src.datasets": None, "src.datasets.video_dataset": {"VideoDataset": VideoDataset}, "evals": None,
            "evals.video_classification_frozen": None, "evals.video_classification_frozen.utils": {"make_transforms": make_transforms}}
    for name, attrs in mods.items():
        m = types.ModuleType(name)
        for k, v in (attrs or {}).items():
            setattr(m, k, v)
        sys.modules[name] = m


_install_stubs()
from eval.video import make_probe_init, ssv2_probe  # noqa: E402


def test_cosine_schedule():
    lr = ssv2_probe.cosine_lr(1e-3, 100)
    assert lr(0) == 1e-3 and abs(lr(50) - 5e-4) < 1e-12 and lr(100) == 0 and lr(250) == 0
    a = ssv2_probe.get_args(["--enc", "alps", "--out", "o", "--data", "d"])
    assert ssv2_probe.total_steps(a, 651) == 20 * 651
    a = ssv2_probe.get_args(["--enc", "alps", "--out", "o", "--data", "d", "--max_steps", "70"])
    assert ssv2_probe.total_steps(a, 651) == 70


def test_protocol_defaults():
    a = ssv2_probe.get_args(["--enc", "alps", "--out", "o", "--data", "d"])
    assert (a.epochs, a.global_batch, a.lr, a.wd, a.res, a.fpc, a.frame_step, a.views) == (20, 256, 1e-3, 0.1, 224, 16, 4, 3)
    assert (a.probe_dim, a.probe_heads, a.probe_depth, a.num_classes, a.seed) == (768, 12, 4, 174, 0)
    assert (a.eval_epochs, a.save_epochs, a.holdout_every, a.vjepa2_rope) == ("10,20", "10,20", 1, "interp")


def test_probe_init_for_vjepa2():
    base = make_probe_init.fresh()
    assert tuple(base["adapter.weight"].shape) == (768, 768) and tuple(base["tpos"].shape) == (1, 16, 1, 768)
    sd = make_probe_init.adapt(base, 1024, 8)
    assert tuple(sd["adapter.weight"].shape) == (768, 1024) and not sd["adapter.bias"].any()
    assert tuple(sd["tpos"].shape) == (1, 8, 1, 768) and not sd["tpos"].any()
    assert abs(float(sd["adapter.weight"].std()) - 0.02) < 1e-3
    assert all(torch.equal(sd[k], base[k]) for k in base if k.startswith("clf."))
    torch.manual_seed(0)
    w = torch.empty(768, 1024)
    nn.init.trunc_normal_(w, std=0.02)
    assert torch.equal(sd["adapter.weight"], w)
    p = ssv2_probe.TemporalProbe(1024, 8, 174)
    p.load_state_dict(sd, strict=True)
    assert p(torch.randn(2, 8, 5, 1024)).shape == (2, 174)


def test_probe_forward():
    for in_dim, t in [(768, 16), (1024, 8)]:
        torch.manual_seed(0)
        p = ssv2_probe.TemporalProbe(in_dim, t, 174, 768, 12, 4, False)
        y = p(torch.randn(3, t, 7, in_dim))
        assert y.shape == (3, 174) and torch.isfinite(y).all()


def test_data_pipeline_construction():
    args = argparse.Namespace(res=224, fpc=16, frame_step=4)
    for training, views in [(True, 1), (False, 3)]:
        CALLS.clear()
        ds = ssv2_probe.make_dataset("x.csv", training, args, views=views, limit=4)
        assert len(ds) == 4 and ds.ds.strict == (not training)
        assert [c[0] for c in CALLS].count("VideoDataset") == 1 and any(c[0] == "make_transforms" for c in CALLS)


if __name__ == "__main__":
    sys.exit(_common.run_all(dict(globals())))
