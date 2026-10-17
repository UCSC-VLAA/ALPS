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
"""CPU tests of the frozen-encoder module of the LIBERO policies (no LIBERO, no downloads).

The stand-in encoders below replace eval/encoders.py so that preprocessing, pooling, construction order, train/eval mode and
checkpoint loading can be checked exactly; the ALPS tests compare the release ViT with the structure of the earlier port."""
import math
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from eval.encoders import FrozenEncoder  # noqa: E402
from eval.libero_policy import frozen_encoder_bridge as fb  # noqa: E402

DIM = 32


class StandIn(FrozenEncoder):
    """Tiny frozen encoder: 16 x 16 patches -> DIM, optional dropout (a train-time random op, like RADIO's CPE)."""
    name, patch, dim = "stand_in", 16, DIM

    def __init__(self, dropout=0.0, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__(mean, std)
        self.model = nn.Sequential(nn.Conv2d(3, DIM, 16, 16), nn.Dropout(dropout))

    def forward(self, x):
        return self.model[1](self.model[0](x).flatten(2).transpose(1, 2))


@pytest.fixture(autouse=True)
def _fresh_cache():
    fb.clear_cache()
    yield
    fb.clear_cache()


def use_stand_in(monkeypatch, **kw):
    monkeypatch.setattr(fb, "build_frozen_encoder", lambda encoder, weights="", alps_port_numerics=False: StandIn(**kw).freeze())


def bridge(**kw):
    args = dict(input_shape=[3, 128, 128], output_size=64, language_dim=768, encoder="stand_in", flip_ud=True)
    args.update(kw)
    return fb.FrozenEncoderPool(**args)


def frames(b=4, seed=0):
    return torch.rand(b, 3, 128, 128, generator=torch.Generator().manual_seed(seed))


# ----------------------------------------------------------------------------------------------------------- preprocessing
def test_preprocessing_is_flip_resize_normalize(monkeypatch):
    use_stand_in(monkeypatch)
    m = bridge()
    x = frames()
    ref_in = F.interpolate(torch.flip(x, dims=[-2]), size=(224, 224), mode="bilinear", align_corners=False)
    with torch.no_grad():
        ref = m.frozen(m.frozen.normalize(ref_in))
    tok = m.tokens(x)
    assert tok.dtype == torch.float32 and tok.shape == (4, 196, DIM)
    assert torch.equal(tok, ref)
    # flip_ud flips rows (upside down), not columns
    y = torch.zeros(1, 3, 128, 128)
    y[..., 0, :] = 1.0
    assert torch.equal(torch.flip(y, dims=[-2])[..., -1, :], torch.ones(1, 3, 128))


def test_no_flip(monkeypatch):
    use_stand_in(monkeypatch)
    m = bridge(flip_ud=False)
    x = frames()
    ref_in = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    with torch.no_grad():
        assert torch.equal(m.tokens(x), m.frozen(m.frozen.normalize(ref_in)))


def test_range_check(monkeypatch):
    use_stand_in(monkeypatch)
    m = bridge()
    with pytest.raises(AssertionError):
        m.tokens(frames() * 255.0)


def test_radio_input_is_clamped(monkeypatch):
    monkeypatch.setattr(fb, "build_frozen_encoder",
                        lambda encoder, weights="", alps_port_numerics=False: StandIn(mean=(0.0,) * 3, std=(1.0,) * 3).freeze())
    m = bridge(encoder="cradio_v3_b", weights="")
    assert m._clamp01
    x = frames()
    ref_in = F.interpolate(torch.flip(x, dims=[-2]), size=(224, 224), mode="bilinear", align_corners=False).clamp(0, 1)
    with torch.no_grad():
        assert torch.equal(m.tokens(x), m.frozen(ref_in))


# ------------------------------------------------------------------------------------------------------------- pooling
def test_attention_pool_output_and_parameters(monkeypatch):
    use_stand_in(monkeypatch)
    m = bridge()
    x, lang = frames(), torch.randn(4, 768)
    y = m(x, langs=lang)
    assert y.shape == (4, 64) and m.output_shape((3, 128, 128)) == (64,)
    z = m.tokens(x)
    q = m.query_proj(lang).view(4, 1, -1)
    ref = m.out_proj(m.pool(q, z, z)[0].reshape(4, -1))
    assert torch.allclose(y, ref, atol=0, rtol=0)
    n_train = sum(p.numel() for p in m.parameters())
    assert n_train == (768 * DIM + DIM) + (3 * DIM * DIM + 3 * DIM + DIM * DIM + DIM) + (DIM * 64 + 64)
    assert all(p.requires_grad for p in m.parameters())
    assert not any(p.requires_grad for p in m.frozen.parameters())
    assert all(k.split(".")[0] in ("query_proj", "pool", "out_proj") for k in m.state_dict())


def test_frozen_network_is_shared_and_moves_lazily(monkeypatch):
    use_stand_in(monkeypatch)
    a, b = bridge(), bridge()
    assert a.frozen is b.frozen
    a.to(torch.float32)                                           # module moves do not touch the frozen network
    assert a.frozen.device.type == "cpu"


def test_deepcopy_like_libero_compute_flops(monkeypatch):
    """LIBERO's compute_flops deep-copies the policy and runs it once: the copy has its own frozen network and works."""
    import copy
    use_stand_in(monkeypatch)
    m = bridge()
    c = copy.deepcopy(m)
    assert c.frozen is not m.frozen
    x, lang = frames(2), torch.randn(2, 768)
    assert torch.equal(c(x, langs=lang), m(x, langs=lang))
    c.load_state_dict(m.state_dict(), strict=True)


def test_construction_order_matches_the_research_module(monkeypatch):
    """Frozen network first (its random init consumes the generator), then query_proj, pool, out_proj; the second camera
    reuses the frozen network. Same seed -> the same trainable initial weights as the module of the runs."""
    use_stand_in(monkeypatch)
    torch.manual_seed(123)
    cams = [bridge(), bridge()]
    torch.manual_seed(123)
    StandIn()
    ref = []
    for _ in range(2):
        ref.append((nn.Linear(768, DIM), nn.MultiheadAttention(DIM, 4, batch_first=True), nn.Linear(DIM, 64)))
    for m, (q, p, o) in zip(cams, ref):
        for a, b in zip(list(m.query_proj.parameters()) + list(m.pool.parameters()) + list(m.out_proj.parameters()),
                        list(q.parameters()) + list(p.parameters()) + list(o.parameters())):
            assert torch.equal(a, b)


# ----------------------------------------------------------------------------------------------------- train / eval mode
def test_defaults_for_new_users(monkeypatch):
    """A truly frozen encoder and the released model's arithmetic by default (the paper's runs: "follow", port numerics)."""
    seen = {}

    def build(encoder, weights="", alps_port_numerics=False):
        seen["port"] = alps_port_numerics
        return StandIn(dropout=0.5).freeze()

    monkeypatch.setattr(fb, "build_frozen_encoder", build)
    m = bridge(encoder="alps", weights="alps_vitb16.safetensors")
    assert m.encoder_train_mode == "eval" and m.alps_port_numerics is False and seen["port"] is False
    m.train()
    assert not m.frozen.model.training
    fb.clear_cache()
    m = bridge(encoder="alps", weights="alps_vitb16.safetensors", alps_port_numerics=True)
    assert m.alps_port_numerics is True and seen["port"] is True
    fb.clear_cache()
    assert bridge(encoder="dinov3_b16", alps_port_numerics=True).alps_port_numerics is False     # ALPS encoders only


def test_frozen_network_follows_the_policy_mode(monkeypatch):
    use_stand_in(monkeypatch, dropout=0.5)
    m = bridge(encoder_train_mode="follow")                       # the setting of the paper's runs
    x = frames()
    assert not m.frozen.model.training                            # built in eval mode
    m.train()
    assert m.frozen.model.training                                # as in the runs: policy.train() reached the frozen network
    assert not torch.equal(m.tokens(x), m.tokens(x))              # train-time randomness of the frozen network is active
    m.eval()
    assert not m.frozen.model.training and torch.equal(m.tokens(x), m.tokens(x))


def test_encoder_train_mode_eval_keeps_eval(monkeypatch):
    use_stand_in(monkeypatch, dropout=0.5)
    m = bridge()                                                  # default: eval
    m.train()
    assert m.training and not m.frozen.model.training
    x = frames()
    assert torch.equal(m.tokens(x), m.tokens(x))


# ---------------------------------------------------------------------------------------------------------- checkpoints
def test_research_checkpoint_keys_load(monkeypatch):
    use_stand_in(monkeypatch)
    m = bridge()
    sd = dict(m.state_dict())
    sd.update({"vit.blocks.0.q.weight": torch.zeros(3), "vit.cls": torch.zeros(1),
               "im_mean": m.frozen.mean.clone(), "im_std": m.frozen.std.clone()})
    m.load_state_dict(sd, strict=True)                            # research keys are dropped, the buffers checked
    sd["im_mean"] = torch.full((1, 3, 1, 1), 0.5)
    with pytest.raises(RuntimeError):
        m.load_state_dict(sd, strict=True)


def test_research_kwargs_are_rejected(monkeypatch):
    use_stand_in(monkeypatch)
    with pytest.raises(TypeError, match="encoder_ckpt"):
        bridge(encoder_ckpt="encoder.pt")
    with pytest.raises(ValueError):
        bridge(encoder="")


# ---------------------------------------------------------------------------------------------------- ALPS and the port
class PortBlock(nn.Module):                 # structure of the earlier torch port (SDPA version used by the policy runs)
    def __init__(self, dim, heads, mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim); self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, mlp); self.fc2 = nn.Linear(mlp, dim)
        self.heads, self.hd = heads, dim // heads

    def forward(self, x):
        h = self.ln1(x)
        B, L, D = h.shape
        q = self.q(h).view(B, L, self.heads, self.hd).transpose(1, 2)
        k = self.k(h).view(B, L, self.heads, self.hd).transpose(1, 2)
        v = self.v(h).view(B, L, self.heads, self.hd).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, L, D)
        x = x + self.proj(o)
        h = self.ln2(x)
        return x + self.fc2(F.gelu(self.fc1(h)))


class PortViT(nn.Module):
    def __init__(self, dim=768, depth=12, heads=12, mlp=3072, patch=16, res=224):
        super().__init__()
        self.conv = nn.Conv2d(3, dim, patch, patch, bias=False)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_buffer("pos", fb._port_sincos(res // patch, res // patch, dim))
        self.blocks = nn.ModuleList([PortBlock(dim, heads, mlp) for _ in range(depth)])
        self.enc_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.conv(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1) + self.pos
        for b in self.blocks:
            x = b(x)
        tok = x[:, 1:]
        return tok, self.enc_norm(tok.mean(1))


PORT_MEAN = torch.tensor(np.array([0.485, 0.456, 0.406]) * 255.0, dtype=torch.float32).view(1, 3, 1, 1) / 255.0
PORT_STD = torch.tensor(np.array([0.229, 0.224, 0.225]) * 255.0, dtype=torch.float32).view(1, 3, 1, 1) / 255.0


def tiny_pair(dim=64, depth=2, heads=4, mlp=128, seed=0):
    """A port-structured ViT with random weights and the same weights in the release ViT (alps.vit), wrapped like
    eval/encoders.py's ALPSEncoder (normalize + .model)."""
    alps = pytest.importorskip("alps")
    from alps.convert import legacy_port_to_torch
    from alps.vit import VisionTransformer
    torch.manual_seed(seed)
    port = PortViT(dim, depth, heads, mlp).eval()
    with torch.no_grad():                                         # move LayerNorms / class token off their trivial init
        for p in port.parameters():
            p.add_(0.05 * torch.randn_like(p))

    class TinyALPS(FrozenEncoder):
        name, patch = "tiny_alps", 16

        def __init__(self):
            super().__init__()
            self.dim = dim
            self.model = VisionTransformer(img_size=224, patch_size=16, dim=dim, depth=depth, heads=heads, mlp_dim=mlp)

        def forward(self, x):
            return self.model(x)["tokens"]

    enc = TinyALPS()
    sd = {k[len("visual."):]: v for k, v in legacy_port_to_torch(port.state_dict()).items()}
    enc.model.load_state_dict(sd, strict=True)
    return port, enc.freeze(), alps


def port_tokens(port, x01, amp):
    x = (x01 - PORT_MEAN) / PORT_STD
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        return port(x)[0].float()


def test_port_numerics_reproduce_the_port_fp32():
    port, enc, _ = tiny_pair()
    x01 = F.interpolate(frames(2), size=(224, 224), mode="bilinear", align_corners=False)
    ref = port_tokens(port, x01, amp=False)
    with torch.no_grad():
        release = enc(enc.normalize(x01))
    fb.apply_port_numerics(enc)
    assert torch.equal(enc.std, PORT_STD) and torch.equal(enc.mean, PORT_MEAN)
    assert all(m.eps == 1e-5 for m in enc.model.modules() if isinstance(m, nn.LayerNorm))
    with torch.no_grad():
        port_like = enc(enc.normalize(x01))
    rel = lambda a: ((a - ref).norm() / ref.norm()).item()
    assert rel(port_like) < 1e-6, rel(port_like)                 # the port's arithmetic (bit for bit on the test machine)
    assert 0 < rel(release) < 1e-4, rel(release)                 # the release encoder as is: tiny float32 difference


def test_port_numerics_reproduce_the_port_under_bf16_autocast(monkeypatch):
    port, enc, _ = tiny_pair()
    try:
        F.conv2d(torch.zeros(1, 3, 16, 16, dtype=torch.bfloat16), torch.zeros(4, 3, 16, 16, dtype=torch.bfloat16))
    except RuntimeError:
        pytest.skip("no bfloat16 convolution on this CPU")
    x = frames(2)
    x01 = F.interpolate(torch.flip(x, dims=[-2]), size=(224, 224), mode="bilinear", align_corners=False)
    ref = port_tokens(port, x01, amp=True)
    noise = ((ref - port_tokens(port, x01, amp=False)).norm() / ref.norm()).item()    # bfloat16 vs float32 of the port
    toks = {}
    for flag in (False, True):
        fb.clear_cache()
        e = enc if not flag else fb.apply_port_numerics(enc)
        monkeypatch.setattr(fb, "build_frozen_encoder", lambda encoder, weights="", alps_port_numerics=True, e=e: e)
        m = fb.FrozenEncoderPool([3, 128, 128], 64, encoder="alps", weights="unused.safetensors", flip_ud=True,
                                alps_port_numerics=flag)
        m._autocast_on_cpu = True                                 # the bridge's CUDA autocast, emulated on the CPU
        toks[flag] = m.tokens(x)
    rel = {f: ((t - ref).norm() / ref.norm()).item() for f, t in toks.items()}
    assert rel[True] < 0.01 * noise, (rel, noise)                 # the port's arithmetic under autocast
    assert rel[False] > 0.1 * noise, (rel, noise)                 # the release ViT as is: a bfloat16-sized difference


def test_release_vit_draws_as_many_random_numbers_as_the_port():
    """The trainable pool is initialized after the frozen network: equal generator states after building the release ViT
    and the port ViT (both ViT-B/16) mean equal initial pool weights for a given training seed."""
    alps = pytest.importorskip("alps")
    torch.manual_seed(7)
    PortViT()
    after_port = torch.get_rng_state()
    torch.manual_seed(7)
    alps.vit_b16()
    assert torch.equal(torch.get_rng_state(), after_port)


# --------------------------------------------------------------------------------------------- real timm architectures
@pytest.mark.parametrize("name,n_tokens", [("dinov3_b16", 196), ("dinov2_b14_timm", 256), ("clip_b16_qg", 196)])
def test_timm_encoders_through_eval_encoders(monkeypatch, name, n_tokens):
    pytest.importorskip("timm")
    enc_mod = fb._encoders()

    def build(encoder, weights="", alps_port_numerics=True):
        try:
            return enc_mod.build(encoder, device="cpu", pretrained=False)
        except TypeError:
            pytest.skip("eval/encoders.build has no pretrained= switch")

    monkeypatch.setattr(fb, "build_frozen_encoder", build)
    m = bridge(encoder=name)
    x = frames(1)
    tok = m.tokens(x)
    assert tok.shape == (1, n_tokens, 768)
    x01 = F.interpolate(torch.flip(x, dims=[-2]), size=(224, 224), mode="bilinear", align_corners=False)
    with torch.no_grad():
        ref = m.frozen.model.forward_features(m.frozen.normalize(x01))[:, m.frozen.model.num_prefix_tokens:]
    assert torch.equal(tok, ref.float())


# -------------------------------------------------------------------------------------------- the LIBERO-side shim
def _shim_source():
    patch = open(os.path.join(ROOT, "eval", "libero_policy", "patches", "libero_8f1084e3.patch")).read()
    part = patch.split("+++ b/libero/lifelong/models/modules/frozen_encoder_pool.py\n")[1].split("\ndiff --git")[0]
    lines = part.split("\n")[1:]                                  # drop the @@ header
    return "\n".join(l[1:] for l in lines if l.startswith("+")) + "\n"


@pytest.mark.parametrize("with_repo", [True, False])
def test_shim_of_the_patch(with_repo):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "frozen_encoder_pool.py")
        open(path, "w").write(_shim_source())
        env = {k: v for k, v in os.environ.items() if k not in ("ALPS_REPO", "PYTHONPATH")}
        if with_repo:
            env["ALPS_REPO"] = ROOT
        code = ("import importlib.util, sys\n"
                "spec = importlib.util.spec_from_file_location('frozen_encoder_pool', sys.argv[1]); m = importlib.util.module_from_spec(spec)\n"
                "spec.loader.exec_module(m)\n"
                "print(m.FrozenEncoderPool.__module__)\n"
                "try:\n    m.FrozenEncoderPool([3, 128, 128], 64)\nexcept ImportError:\n    print('placeholder')\n"
                "except Exception as e:\n    print(type(e).__name__)\n")
        out = subprocess.run([sys.executable, "-I", "-B", "-c", code, path], cwd=d, env=env, capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr
        if with_repo:
            assert "eval.libero_policy.frozen_encoder_bridge" in out.stdout and "ValueError" in out.stdout
        else:
            assert "placeholder" in out.stdout
