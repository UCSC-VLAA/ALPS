# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Shape, loading and compatibility tests of the inference package on random weights."""
import os
import tempfile

import pytest
import torch

import alps
from alps.convert import legacy_port_to_torch


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = alps.ALPS().eval()
    with torch.no_grad():                       # move LayerNorm parameters away from 1 / 0
        for n, p in m.named_parameters():
            p.add_(0.02 * torch.randn_like(p))
    return m


def test_shapes(model):
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        o = model.encode_image(x)
        assert o["tokens"].shape == (2, 196, 768) and o["pooled"].shape == (2, 768)
        assert model.visual(torch.randn(1, 3, 320, 448))["tokens"].shape == (1, 20 * 28, 768)
        e = model.text_embedding(["a photo of a dog.", "open the drawer"])
        assert e.shape == (2, 768) and torch.allclose(e.norm(dim=-1), torch.ones(2), atol=1e-5)
        assert model(x, ["a", "b", "c"]).shape == (2, 3)


def test_interpolated_table_is_exact_on_training_grid(model):
    enc = model.visual
    assert torch.equal(enc.position_table(14, 14), enc.pos_embed)


def test_save_and_load_roundtrip(model):
    x = torch.randn(1, 3, 224, 224)
    with tempfile.TemporaryDirectory() as d:
        for name in ("m.safetensors", "m.pt"):
            path = os.path.join(d, name)
            alps.save(model.state_dict(), path, {"name": "test"})
            m2 = alps.load(path)
            with torch.no_grad():
                assert torch.equal(m2.encode_image(x)["tokens"], model.encode_image(x)["tokens"])
                assert torch.equal(m2.text_embedding("a dog"), model.text_embedding("a dog"))
        vis_only = {k: v for k, v in model.state_dict().items() if k.startswith("visual.")}
        path = os.path.join(d, "enc.safetensors")
        alps.save(vis_only, path)
        enc = alps.load_encoder(path)
        with torch.no_grad():
            assert torch.equal(enc(x)["tokens"], model.encode_image(x)["tokens"])


def test_legacy_port_mapping(model):
    """The earlier torch port used separate q/k/v layers; its state dict maps onto the release module."""
    v = model.visual
    old = {"conv.weight": v.patch_embed.weight, "cls": v.cls_token, "enc_norm.weight": v.norm.weight,
           "enc_norm.bias": v.norm.bias}
    for i, b in enumerate(v.blocks):
        q, k, w = b.attn.qkv.weight.chunk(3, 0)
        qb, kb, wb = b.attn.qkv.bias.chunk(3, 0)
        old.update({f"blocks.{i}.ln1.weight": b.norm1.weight, f"blocks.{i}.ln1.bias": b.norm1.bias,
                    f"blocks.{i}.ln2.weight": b.norm2.weight, f"blocks.{i}.ln2.bias": b.norm2.bias,
                    f"blocks.{i}.q.weight": q, f"blocks.{i}.k.weight": k, f"blocks.{i}.v.weight": w,
                    f"blocks.{i}.q.bias": qb, f"blocks.{i}.k.bias": kb, f"blocks.{i}.v.bias": wb,
                    f"blocks.{i}.proj.weight": b.attn.proj.weight, f"blocks.{i}.proj.bias": b.attn.proj.bias,
                    f"blocks.{i}.fc1.weight": b.mlp.fc1.weight, f"blocks.{i}.fc1.bias": b.mlp.fc1.bias,
                    f"blocks.{i}.fc2.weight": b.mlp.fc2.weight, f"blocks.{i}.fc2.bias": b.mlp.fc2.bias})
    new = legacy_port_to_torch(old, dst="")
    for k, t in v.state_dict().items():
        assert torch.equal(new[k], t), k


def test_timm_export(model):
    timm = pytest.importorskip("timm")
    from alps.timm_compat import to_timm_state_dict
    vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0, global_pool="avg", fc_norm=True)
    missing, unexpected = vit.load_state_dict(to_timm_state_dict(model.state_dict()), strict=False)
    assert not missing and not unexpected
    vit.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        f = vit.forward_features(x)
        o = model.encode_image(x)
        assert torch.allclose(f[:, 1:], o["tokens"], atol=1e-5, rtol=1e-5)
        assert torch.allclose(vit.forward_head(f, pre_logits=True), o["pooled"], atol=1e-5, rtol=1e-5)


def test_interpolate_mode_is_the_rescaled_sincos(model):
    """At other grids the table is the sin-cos code at indices rescaled into the 14 x 14 training range (the protocol of
    the paper's evaluations at other resolutions), not a resampled table."""
    import numpy as np
    from alps.vit import sincos_2d

    def ref(h, w, dim, temp=10000.0):
        omega = 1.0 / (temp ** (np.arange(dim // 4) / (dim // 4 - 1)))
        y, x = np.meshgrid(np.arange(h) * 14.0 / h, np.arange(w) * 14.0 / w, indexing="ij")
        y = y.flatten()[:, None] * omega[None]; x = x.flatten()[:, None] * omega[None]
        pe = np.concatenate([np.sin(x), np.cos(x), np.sin(y), np.cos(y)], 1)
        return torch.tensor(np.concatenate([np.zeros((1, dim)), pe])[None], dtype=torch.float32)
    assert torch.equal(sincos_2d(30, 53, 768, scale_to=14), ref(30, 53, 768))
    assert torch.equal(model.visual.position_table(30, 53), ref(30, 53, 768))


def test_eval_decoder_shapes():
    from alps.decoder import EvalDecoder
    dec = EvalDecoder(dim=768).eval()
    with torch.no_grad():
        assert dec(torch.randn(2, 196, 768)).shape == (2, 4, 28, 28)
        assert dec(torch.randn(1, 256, 768)).shape == (1, 4, 28, 28)       # 16 x 16 grids are resized to 14 x 14


def test_hub_names_resolve(monkeypatch, tmp_path):
    """hf-hub:<org>/<repo>[/<file>] and PRETRAINED names map to huggingface_hub downloads (no network here)."""
    import sys
    import types

    from alps import hub
    calls = []
    fake = types.ModuleType("huggingface_hub")
    fake.hf_hub_download = lambda repo, filename: calls.append((repo, filename)) or str(tmp_path / filename)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    assert hub._resolve("hf-hub:org/alps-vitb16/alps_vitb16.safetensors").endswith("alps_vitb16.safetensors")
    assert hub._resolve("hf-hub:org/alps-vitb16").endswith("model.safetensors")
    monkeypatch.setitem(hub.PRETRAINED, "alps-vitb16", "hf-hub:org/alps/alps_vitb16.safetensors")
    hub._resolve("alps-vitb16")
    assert calls == [("org/alps-vitb16", "alps_vitb16.safetensors"), ("org/alps-vitb16", "model.safetensors"),
                     ("org/alps", "alps_vitb16.safetensors")]
    monkeypatch.setitem(hub.PRETRAINED, "alps-stage1-vitb16", None)
    with pytest.raises(FileNotFoundError, match="not published yet"):
        hub._resolve("alps-stage1-vitb16")

    class EntryNotFoundError(Exception):                # the name huggingface_hub uses for a missing file
        pass

    def missing(repo, filename):
        raise EntryNotFoundError(f"{repo}/{filename}")

    fake.hf_hub_download = missing
    with pytest.raises(FileNotFoundError, match="not available on Hugging Face"):
        hub._resolve("hf-hub:UCSC-VLAA/ALPS/alps_vitb16.safetensors")


def test_released_names_point_to_the_hub_repository():
    from alps import hub
    assert hub.PRETRAINED == {"alps-vitb16": "hf-hub:UCSC-VLAA/ALPS/alps_vitb16.safetensors",
                              "alps-stage1-vitb16": "hf-hub:UCSC-VLAA/ALPS/alps_stage1_vitb16.safetensors"}
