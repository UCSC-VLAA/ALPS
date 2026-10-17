# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end CPU runs of the trainers on tiny synthetic data: Stage 1a (JSONL input) -> resume -> Stage 1b from
all Stage-1a parameters (tar-shard input) -> Stage 2 (mixed JSONL pair + image-directory sources, gradient
accumulation) -> exports, which the `alps` inference modules load and run with the trained encoder's tokens.

test_full_size_alps_load repeats Stage 1 and Stage 2 with the real ViT-B/16 + text tower (32-pixel images, a few
updates) and checks alps.load on the exports (needs ~8 GB of memory; marked slow)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

import synthetic_data as D

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY = ["--config.model.width=64", "--config.model.depth=1", "--config.model.mlp_dim=128",
        "--config.model.num_heads=4", "--config.model.text_width=64", "--config.model.text_depth=1",
        "--config.model.text_heads=4", "--config.model.text_mlp_dim=128", "--config.model.cap_depth=1",
        "--config.model.cap_heads=4", "--config.model.cap_mlp_dim=128", "--config.model.remat=False"]


def _run(args):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([HERE, os.path.dirname(HERE)]), OMP_NUM_THREADS="4")
    r = subprocess.run([sys.executable] + args, cwd=HERE, env=env, capture_output=True, text=True, timeout=3600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-6000:]
    return r.stdout


def _metrics(workdir):
    return [json.loads(line) for line in open(os.path.join(workdir, "metrics.jsonl"))]


def _stage1(workdir, cfg, extra):
    return _run(["train_stage1.py", f"--config=configs/{cfg}.py", f"--workdir={workdir}", "--config.res=32",
                 "--config.warmup_steps=1", "--config.log_steps=1", "--config.dense_n_cross=4",
                 "--config.data.num_workers=0"] + extra)


def _tiny_vit_tokens(sd_visual, x, width=64, depth=1, heads=4, mlp=128):
    from alps_torch.models import ViT
    m = ViT(width, depth, mlp, heads, img_size=32).eval()
    m.load_state_dict(sd_visual, strict=True)
    with torch.no_grad():
        return m(x)["tokens"]


def test_tiny_pipeline(tmp_path):
    t = str(tmp_path)
    s1_jsonl, s1_tar, s2_pairs = D.stage1_jsonl(t), D.stage1_tar(t), D.stage2_jsonl(t)
    w1a, w1b, w2 = f"{t}/s1a", f"{t}/s1b", f"{t}/s2"
    _stage1(w1a, "stage1a", ["--config.global_batch=8", "--config.total_steps=3", "--config.ckpt_steps=2",
                             f"--config.data.source={s1_jsonl}"] + TINY)
    m = _metrics(w1a)
    assert [r["step"] for r in m] == [1, 2, 3] and all(np.isfinite(r["loss"]) for r in m)
    assert m[0]["lr"] == 0.0 and m[1]["lr"] > 0
    assert abs(m[0]["L_itc"] - np.log(8)) < 0.5 and abs(m[0]["L_cap"] - np.log(32000)) < 1.5
    # resume: drop the final checkpoint, run again -> continues from step 2
    os.remove(os.path.join(w1a, "checkpoints", "ckpt_00000003.pt"))
    out = _stage1(w1a, "stage1a", ["--config.global_batch=8", "--config.total_steps=3", "--config.ckpt_steps=2",
                                   f"--config.data.source={s1_jsonl}"] + TINY)
    assert "resumed from step 2" in out and _metrics(w1a)[-1]["step"] == 3
    # Stage 1b: all Stage-1a parameters, fresh optimizer; its first update has lr 0, so after one update the
    # parameters still equal Stage 1a's
    _stage1(w1b, "stage1b", ["--config.global_batch=8", "--config.total_steps=1", "--config.ckpt_steps=1",
                             "--config.dense_n_same=2", f"--config.data.source={s1_tar}", f"--config.init_from={w1a}",
                             "--config.data.num_workers=2"] + TINY)
    a = torch.load(f"{w1a}/checkpoints/ckpt_00000003.pt", weights_only=False)
    b = torch.load(f"{w1b}/checkpoints/ckpt_00000001.pt", weights_only=False)
    assert b["optimizer"]["count"] == 1
    for k, v in a["model"].items():
        assert torch.equal(v, b["model"][k]), k
    # Stage 2 from Stage 1b
    np.savez(f"{t}/zs.npz", mean=np.zeros(64, np.float32), std=np.full(64, 2.0, np.float32), n_images=np.int64(1))
    np.savez(f"{t}/u0.npz", u0_bf16=np.random.default_rng(0).normal(size=768).astype(np.float32))
    src = f"(('{s2_pairs}', 0.8), ('{t}/img', 0.2))"
    _run(["train_stage2.py", "--config=configs/stage2.py", f"--workdir={w2}", "--config.res=32",
          "--config.global_batch=8", "--config.accum_steps=2", "--config.total_steps=3", "--config.warmup_steps=1",
          "--config.log_steps=1", "--config.ckpt_steps=3", f"--config.stage1_ckpt={w1b}",
          f"--config.token_stats={t}/zs.npz", f"--config.empty_token={t}/u0.npz", "--config.crossview.min_valid=2",
          "--config.encoder.width=64", "--config.encoder.depth=1", "--config.encoder.mlp_dim=128",
          "--config.encoder.num_heads=4", "--config.predictor.width=32", "--config.predictor.depth=1",
          "--config.predictor.num_heads=4", f"--config.data.sources={src}", "--config.data.num_workers=0"])
    m2 = _metrics(w2)
    assert [r["step"] for r in m2] == [1, 2, 3] and all(np.isfinite(r["loss"]) for r in m2)
    assert any(r["log_r"] != 0 for r in m2)                        # A != B for some pairs
    # exports: Stage-2 encoder + Stage-1 language interface, read by the alps modules
    from alps.hub import read_state_dict
    from alps.text import TextEncoder as AlpsText
    from alps.vit import VisionTransformer
    st2 = torch.load(f"{w2}/checkpoints/ckpt_00000003.pt", weights_only=False)
    enc = {k[len("encoder."):]: v for k, v in st2["student"].items() if k.startswith("encoder.")}
    sd = read_state_dict(f"{w2}/alps.safetensors")
    assert {k.split(".")[0] for k in sd} == {"visual", "text", "image_projection", "text_projection", "logit_scale"}
    vis = VisionTransformer(img_size=32, patch_size=16, dim=64, depth=1, heads=4, mlp_dim=128).eval()
    vis.load_state_dict({k[len("visual."):]: v for k, v in sd.items() if k.startswith("visual.")}, strict=True)
    txt = AlpsText(vocab_size=32000, context_length=32, dim=64, depth=1, heads=4, mlp_dim=128).eval()
    txt.load_state_dict({k[len("text."):]: v for k, v in sd.items() if k.startswith("text.")}, strict=True)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        z = vis(x)["tokens"]
    assert torch.allclose(z, _tiny_vit_tokens(enc, x), atol=1e-5, rtol=1e-5)
    for k in ("text.token_embedding.weight", "image_projection.weight", "logit_scale"):
        assert torch.equal(sd[k], b["model"][k].reshape(sd[k].shape)), k        # language side = Stage 1
    assert not torch.equal(sd["visual.patch_embed.weight"], b["model"]["visual.patch_embed.weight"])
    enc_only = read_state_dict(f"{w2}/alps_encoder.safetensors")
    assert all(k.startswith("visual.") for k in enc_only)
    # the export tool gives the same file content
    _run(["export.py", f"--stage1={w1b}", f"--stage2={w2}", f"--out={t}/cli.safetensors"])
    cli = read_state_dict(f"{t}/cli.safetensors")
    assert cli.keys() == sd.keys() and all(torch.equal(cli[k], sd[k]) for k in sd)


@pytest.mark.slow
def test_full_size_alps_load(tmp_path):
    """ViT-B/16 and the full text tower: a few updates of each stage, then alps.load on the exports."""
    import alps
    t = str(tmp_path)
    s1, s2 = D.stage1_jsonl(t, n=8), D.stage2_jsonl(t, n=8, pairs=False)
    _stage1(f"{t}/s1", "stage1a", ["--config.global_batch=2", "--config.total_steps=2", "--config.ckpt_steps=1000",
                                   "--config.ckpt_final=False", "--config.dense_n_cross=8", f"--config.data.source={s1}",
                                   "--config.model.remat=False"])
    assert all(np.isfinite(r["loss"]) for r in _metrics(f"{t}/s1"))
    m1 = alps.load(f"{t}/s1/alps_stage1.safetensors")
    _run(["train_stage2.py", "--config=configs/stage2.py", f"--workdir={t}/s2", "--config.res=64",
          "--config.global_batch=2", "--config.accum_steps=1", "--config.total_steps=2", "--config.warmup_steps=1",
          "--config.log_steps=1", "--config.ckpt_steps=1000", "--config.ckpt_final=False",
          f"--config.stage1_ckpt={t}/s1/alps_stage1.safetensors", f"--config.data.sources=(('{s2}', 1.0),)",
          "--config.data.num_workers=0"])
    assert all(np.isfinite(r["loss"]) for r in _metrics(f"{t}/s2"))
    m2 = alps.load(f"{t}/s2/alps.safetensors")
    enc = alps.load_encoder(f"{t}/s2/alps_encoder.safetensors")
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        z = m2.encode_image(x)["tokens"]
        assert z.shape == (1, 196, 768) and torch.isfinite(z).all()
        assert torch.equal(z, enc(x)["tokens"])
        assert torch.equal(m2.text_embedding("a robot arm"), m1.text_embedding("a robot arm"))
    assert not torch.equal(m2.visual.patch_embed.weight, m1.visual.patch_embed.weight)
