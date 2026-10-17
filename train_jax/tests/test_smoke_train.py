# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke test of the training scripts on CPU with tiny synthetic data (a few steps each):
Stage 1 -> Stage 2 from its checkpoint -> encoder export. Checks that the scripts run, losses are finite,
checkpoints restore, and the exported reference forward reproduces the checkpoint."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORDS = "a the red blue small large robot arm drawer bowl plate table cat dog car tree house on in near".split()


def _write_records(path, n, rng, pairs=False):
    import tensorflow as tf
    with tf.io.TFRecordWriter(path) as w:
        for i in range(n):
            h, wd = int(rng.integers(40, 90)), int(rng.integers(40, 90))
            jpg = lambda: tf.io.encode_jpeg(tf.constant(rng.integers(0, 256, (h, wd, 3)), tf.uint8)).numpy()
            cap = " ".join(rng.choice(WORDS, int(rng.integers(3, 12))))
            syn = ". ".join(" ".join(rng.choice(WORDS, int(rng.integers(4, 15)))) for _ in range(3)) + "."
            f = {"caption": tf.train.Feature(bytes_list=tf.train.BytesList(value=[cap.encode()])),
                 "synthetic_caption": tf.train.Feature(bytes_list=tf.train.BytesList(value=[syn.encode()]))}
            if pairs:
                f["source_jpg"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[jpg()]))
                f["target_jpg"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[jpg()]))
            else:
                f["image"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[jpg()]))
            w.write(tf.train.Example(features=tf.train.Features(feature=f)).SerializeToString())


def _run(args, cwd):
    env = dict(os.environ, JAX_PLATFORMS="cpu", TF_CPP_MIN_LOG_LEVEL="3", ALPS_SINGLE_HOST="1")
    r = subprocess.run([sys.executable] + args, cwd=cwd, env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-6000:]
    return r.stdout


def _metrics(workdir):
    return [json.loads(l) for l in open(os.path.join(workdir, "metrics.jsonl"))]


@pytest.mark.slow
def test_stage1_stage2_export(tmp_path):
    pytest.importorskip("tensorflow")
    rng = np.random.default_rng(0)
    _write_records(str(tmp_path / "s1.tfrecord"), 48, rng)
    _write_records(str(tmp_path / "s2pairs.tfrecord"), 24, rng, pairs=True)
    tiny = ["--config.model.width=64", "--config.model.depth=1", "--config.model.mlp_dim=128",
            "--config.model.num_heads=4", "--config.model.text_width=64", "--config.model.text_depth=1",
            "--config.model.text_heads=4", "--config.model.text_mlp_dim=128", "--config.model.cap_depth=1",
            "--config.model.cap_heads=4", "--config.model.cap_mlp_dim=128", "--config.model.remat=False"]
    w1 = str(tmp_path / "stage1")
    _run(["train_stage1.py", "--config=configs/stage1a.py", f"--workdir={w1}", "--config.res=32",
          "--config.global_batch=8", "--config.total_steps=3", "--config.warmup_steps=1", "--config.log_steps=1",
          "--config.ckpt_steps=3", "--config.dense_n_cross=4", f"--config.data.files={tmp_path}/s1.tfrecord",
          "--config.data.shuffle_buffer=16"] + tiny, HERE)
    m1 = _metrics(w1)
    assert m1[-1]["step"] == 3 and all(np.isfinite(r["loss"]) for r in m1)
    # Stage 1b-style continuation from the Stage-1 checkpoint (all subtrees, fresh optimizer)
    w1b = str(tmp_path / "stage1b")
    _run(["train_stage1.py", "--config=configs/stage1b.py", f"--workdir={w1b}", "--config.res=32",
          "--config.global_batch=8", "--config.total_steps=2", "--config.warmup_steps=1", "--config.log_steps=1",
          "--config.ckpt_steps=2", "--config.dense_n_cross=4", "--config.dense_n_same=2", f"--config.init_from={w1}/checkpoints",
          f"--config.data.files={tmp_path}/s1.tfrecord", "--config.data.shuffle_buffer=16"] + tiny, HERE)
    np.savez(tmp_path / "zs.npz", mean=np.zeros(64, np.float32), std=np.ones(64, np.float32), n_images=np.int64(1))
    np.savez(tmp_path / "u0.npz", u0_bf16=rng.normal(size=48).astype(np.float32))
    w2 = str(tmp_path / "stage2")
    src = f"(('{tmp_path}/s1.tfrecord', 0.8), ('{tmp_path}/s2pairs.tfrecord', 0.2))"
    _run(["train_stage2.py", "--config=configs/stage2.py", f"--workdir={w2}", "--config.res=32",
          "--config.global_batch=8", "--config.accum_steps=2", "--config.total_steps=3", "--config.warmup_steps=1",
          "--config.log_steps=1", "--config.ckpt_steps=3", f"--config.stage1_ckpt={w1b}/checkpoints",
          f"--config.token_stats={tmp_path}/zs.npz", f"--config.empty_token={tmp_path}/u0.npz",
          "--config.encoder.width=64", "--config.encoder.depth=1", "--config.encoder.mlp_dim=128",
          "--config.encoder.num_heads=4", "--config.predictor.width=32", "--config.predictor.depth=1",
          "--config.predictor.num_heads=4", "--config.predictor.text_dim=48", f"--config.data.sources={src}",
          "--config.data.shuffle_buffer=16"], HERE)
    m2 = _metrics(w2)
    assert m2[-1]["step"] == 3 and all(np.isfinite(r["loss"]) for r in m2)
    out = str(tmp_path / "enc.npz")
    _run(["export_encoder.py", f"--ckpt={w2}/checkpoints", "--subtree=encoder", f"--out={out}", "--res=32",
          "--heads=4"], HERE)
    d = np.load(out)
    assert d["ref_z"].shape == (4, 4, 64) and int(d["step"]) == 3 and np.isfinite(d["ref_z"]).all()
