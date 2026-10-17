# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Per-channel mean and standard deviation of the frozen Stage-1 tokens (the Stage-2 target standardization).

    python compute_token_stats.py --stage1_ckpt runs/stage1b/checkpoints --sources "<data>/*.tfrecord" \
        --out token_stats.npz [--images 32768] [--crop alps|teacher]

Tokens are computed as in Stage 2 (bfloat16, inference mode), accumulated per channel over all tokens of all images
(sums in float32 per batch, float64 across batches), std = sqrt(max(E[z^2] - E[z]^2, 1e-8)).
--crop alps (default): the recipe of the released statistics file, one random square crop of area U(0.8, 1) of the
central square per image, bilinear without antialias; --crop teacher: the teacher view B of the training pairs.
The statistics of ALPS itself are assets/alps_stage1b_token_stats.npz (32,768 images).
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from alps_jax import runtime
from alps_jax.data.pipelines import _decode, _host_files, _parse_stage2
from alps_jax.data.transforms import make_crossview_pair
from alps_jax.losses.stage2 import normalize_image
from alps_jax.models.vit import vit


def alps_crop(img_uint8, res, seed):
    """Square crop of area U(0.8, 1) of the central square: side = int(sqrt(U) * m), offsets int(U * (m - side))."""
    img = tf.image.convert_image_dtype(img_uint8, tf.float32)
    h = tf.shape(img)[0]; w = tf.shape(img)[1]; m = tf.minimum(h, w)
    img = tf.image.resize_with_crop_or_pad(img, m, m)
    k0, k1, k2 = tf.unstack(tf.random.experimental.stateless_split(seed, 3))
    side = tf.cast(tf.sqrt(tf.random.stateless_uniform([], k0, 0.8, 1.0)) * tf.cast(m, tf.float32), tf.int32)
    y0 = tf.cast(tf.random.stateless_uniform([], k1) * tf.cast(m - side, tf.float32), tf.int32)
    x0 = tf.cast(tf.random.stateless_uniform([], k2) * tf.cast(m - side, tf.float32), tf.int32)
    return tf.image.resize(tf.image.crop_to_bounding_box(img, y0, x0, side, side), (res, res), method="bilinear",
                           antialias=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_ckpt", required=True)
    ap.add_argument("--sources", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", type=int, default=32_768)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--crop", choices=("alps", "teacher"), default="alps")
    ap.add_argument("--variant", default="B")
    a = ap.parse_args()
    runtime.init_distributed()
    _, data_sh, repl = runtime.make_mesh()
    params, _ = runtime.restore_params(a.stage1_ckpt, subtrees=("visual_encoder",), sharding=repl)
    enc = vit(a.variant, dtype=jnp.bfloat16, param_dtype=jnp.float32)

    def prep(b):
        seed = tf.random.uniform([2], 0, 2 ** 30, dtype=tf.int32)
        img = _decode(b)
        return alps_crop(img, a.res, seed) if a.crop == "alps" else make_crossview_pair(img, a.res, seed=seed)["x_t_rgb"]

    ds = _host_files(a.sources, jax.process_index(), jax.process_count()).interleave(
        tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    ds = ds.map(_parse_stage2).shuffle(50_000).map(prep, num_parallel_calls=tf.data.AUTOTUNE).ignore_errors()
    ds = ds.batch(a.batch // jax.process_count(), drop_remainder=True).prefetch(2)

    @jax.jit
    def sums(p, x):
        _, z = enc.apply({"params": p}, normalize_image(x), train=False)
        z = z.astype(jnp.float32).reshape(-1, z.shape[-1])
        return jnp.sum(z, 0), jnp.sum(z * z, 0), z.shape[0]

    s1 = s2 = 0.0
    n = 0
    for i, batch in enumerate(ds.as_numpy_iterator()):
        if i * a.batch >= a.images:
            break
        x = runtime.global_batch({"x": batch}, data_sh)["x"]
        b1, b2, cnt = sums(params["visual_encoder"], x)
        s1 = s1 + np.asarray(b1, np.float64); s2 = s2 + np.asarray(b2, np.float64); n += int(cnt)
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-8))
    if jax.process_index() == 0:
        np.savez(a.out, mean=mean.astype(np.float32), std=std.astype(np.float32),
                 n_images=np.int64(n // ((a.res // 16) ** 2)))
        print(f"wrote {a.out}: {n} tokens; std range {std.min():.3f} .. {std.max():.3f}")


if __name__ == "__main__":
    main()
