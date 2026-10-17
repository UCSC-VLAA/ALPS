# Copyright 2026 The ALPS Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Policy weights as safetensors: scripts/export_policy_weights.py and the checkpoint loading of evaluate_policy.py
(CPU, no LIBERO)."""
import os
import sys
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import evaluate_policy as ev  # noqa: E402
import export_policy_weights as ex  # noqa: E402

pytest.importorskip("safetensors")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _state_dict(frozen=True):
    """A small policy state dict with the key layout of the paper's checkpoints (two cameras)."""
    g = torch.Generator().manual_seed(0)
    sd = {f"encoders.{i}.{k}": torch.randn(*s, generator=g) for i in (0, 1)
          for k, s in (("query_proj.weight", (8, 4)), ("pool.in_proj_weight", (24, 8)), ("out_proj.weight", (3, 8)))}
    sd.update({"policy_head.share.0.weight": torch.randn(5, 3, generator=g), "temporal_position_encoding_fn.inv_freq": torch.arange(4.0),
               "language_encoder.projection.0.weight": torch.randn(3, 6, generator=g).half()})
    if frozen:
        vit = torch.randn(6, 6, generator=g)
        for i in (0, 1):
            sd[f"encoders.{i}.vit.blocks.0.q.weight"] = vit
            sd[f"encoders.{i}.im_mean"] = torch.tensor(MEAN).view(1, 3, 1, 1)
            sd[f"encoders.{i}.im_std"] = torch.tensor(STD).view(1, 3, 1, 1)
    return sd


def test_export_keeps_exactly_the_trainable_tensors(tmp_path):
    sd = _state_dict()
    src = tmp_path / "multitask_model_ep50.pth"
    torch.save({"state_dict": sd, "cfg": {"seed": 0}, "previous_masks": None}, src)
    out = tmp_path / "run" / "multitask_model_ep50.safetensors"
    keep, dropped, norm = ex.export(str(src), str(out))
    trainable = sorted(k for k in sd if ".vit." not in k and not k.endswith(("im_mean", "im_std")))
    assert sorted(keep) == trainable and len(dropped) == 6
    assert norm["im_mean"] == pytest.approx(MEAN) and norm["im_std"] == pytest.approx(STD)
    back = ev.load_policy_weights(str(out))
    assert sorted(back) == trainable
    assert all(torch.equal(back[k], sd[k]) and back[k].dtype == sd[k].dtype for k in back)
    from safetensors import safe_open
    with safe_open(str(out), "pt") as f:
        meta = f.metadata()
    assert meta["format"] == "alps-libero-policy" and meta["dropped_frozen_tensors"] == "6" and len(meta["source_sha256"]) == 64


def test_export_of_a_checkpoint_without_frozen_weights(tmp_path):
    sd = _state_dict(frozen=False)
    src = tmp_path / "multitask_model_ep25.pth"
    torch.save({"state_dict": sd, "cfg": None, "previous_masks": None}, src)
    keep, dropped, norm = ex.export(str(src), str(tmp_path / "multitask_model_ep25.safetensors"))
    assert sorted(keep) == sorted(sd) and dropped == [] and norm == {}


def test_export_of_a_policy_with_a_module_under_two_names(tmp_path):
    """LIBERO's policy registers extra_encoder.encoders.1 also as extra_encoder.proprio_mlp: both names, one storage."""
    class Extra(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoders = torch.nn.ModuleList([torch.nn.Linear(2, 3), torch.nn.Linear(7, 3)])
            self.proprio_mlp = self.encoders[1]

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.extra_encoder = Extra()
            self.head = torch.nn.Linear(3, 2)

    torch.manual_seed(0)
    pol = Policy()
    sd = pol.state_dict()
    assert sd["extra_encoder.proprio_mlp.weight"].data_ptr() == sd["extra_encoder.encoders.1.weight"].data_ptr()
    src = tmp_path / "multitask_model_ep50.pth"
    torch.save({"state_dict": sd, "cfg": None, "previous_masks": None}, src)
    out = tmp_path / "multitask_model_ep50.safetensors"
    keep, dropped, norm = ex.export(str(src), str(out))
    back = ev.load_policy_weights(str(out))
    assert sorted(back) == sorted(sd) and dropped == []
    fresh = Policy()
    fresh.load_state_dict(back, strict=True)
    assert all(torch.equal(a, b) for a, b in zip(fresh.state_dict().values(), pol.state_dict().values()))
    assert fresh.extra_encoder.proprio_mlp is fresh.extra_encoder.encoders[1]


def test_cameras_must_agree():
    sd = _state_dict()
    sd["encoders.1.im_std"] = sd["encoders.1.im_std"] * 2
    with pytest.raises(ValueError, match="im_std"):
        ex.trainable_state_dict(sd)


def test_default_checkpoint_prefers_the_released_file(tmp_path):
    assert ev.default_checkpoint(str(tmp_path), 50) == str(tmp_path / "multitask_model_ep50.pth")
    (tmp_path / "multitask_model_ep50.pth").write_bytes(b"")
    assert ev.default_checkpoint(str(tmp_path), 50) == str(tmp_path / "multitask_model_ep50.pth")
    (tmp_path / "multitask_model_ep50.safetensors").write_bytes(b"")
    assert ev.default_checkpoint(str(tmp_path), 50) == str(tmp_path / "multitask_model_ep50.safetensors")


def test_torch_checkpoints_load_through_libero(monkeypatch, tmp_path):
    calls = []
    utils = types.ModuleType("libero.lifelong.utils")
    utils.torch_load_model = lambda path, map_location=None: calls.append((path, map_location)) or ({"w": torch.ones(2)}, None, None)
    for name in ("libero", "libero.lifelong"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "libero.lifelong.utils", utils)
    sd = ev.load_policy_weights(str(tmp_path / "multitask_model_ep50.pth"), "cpu")
    assert calls == [(str(tmp_path / "multitask_model_ep50.pth"), "cpu")] and torch.equal(sd["w"], torch.ones(2))
