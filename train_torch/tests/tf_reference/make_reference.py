# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Reference outputs of the TensorFlow transforms of train_jax for tests/test_transforms_tf.py.

Run in an environment with TensorFlow (no torch or jax needed):
    python tests/tf_reference/make_reference.py --out /tmp/alps_tf_reference.npz
then (torch environment):
    ALPS_TF_REF=/tmp/alps_tf_reference.npz pytest tests/test_transforms_tf.py

For every function the uniform draws are recomputed with the same stateless key derivation as the TF code and saved
next to the outputs, so the numpy implementations can be fed the identical draws.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "train_jax"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf  # noqa: E402

from alps_jax.data import tokenize as TK  # noqa: E402
from alps_jax.data import transforms as T  # noqa: E402


def images(rng):
    """Random-noise and smooth images of several sizes (uint8)."""
    out = []
    for (h, w) in [(173, 251), (64, 64), (300, 200), (97, 131), (240, 320), (50, 37)]:
        out.append(rng.integers(0, 256, size=(h, w, 3)).astype(np.uint8))
        yy, xx = np.mgrid[:h, :w]
        sm = np.stack([127.5 + 127 * np.sin(xx / 7.0 + c) * np.cos(yy / 11.0 - c) for c in range(3)], -1)
        out.append(np.clip(sm + rng.normal(0, 3, sm.shape), 0, 255).astype(np.uint8))
    return out


def U(key, shape=()):
    return tf.random.stateless_uniform(shape, key).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40, help="seeds per image")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    imgs = images(rng)
    R = {"n_images": np.int64(len(imgs))}
    for i, im in enumerate(imgs):
        R[f"img{i}"] = im

    # ---- plain resizes (float and uint8 inputs, up and down)
    k = 0
    for i in (0, 1, 3, 5, 8, 9):
        for (oh, ow) in [(64, 64), (224, 224), (31, 47), (128, 96)]:
            im = imgs[i]
            f = tf.image.convert_image_dtype(im, tf.float32)
            R[f"rs{k}_src"] = np.int64(i)
            R[f"rs{k}_size"] = np.asarray([oh, ow])
            R[f"rs{k}_bil_f"] = tf.image.resize(f, (oh, ow), method="bilinear", antialias=False).numpy()
            R[f"rs{k}_bil_u8"] = tf.image.resize(im, (oh, ow), method="bilinear", antialias=False).numpy()
            R[f"rs{k}_aa_u8"] = tf.image.resize(im, (oh, ow), method="bilinear", antialias=True).numpy()
            R[f"rs{k}_conv"] = f.numpy()
            k += 1
    R["n_resize"] = np.int64(k)

    # ---- Stage 1: inception crop (stateless box sampler, then the ops of T.inception_crop) and two-view
    boxes, seeds_b, sizes_b, crops = [], [], [], []
    for i, im in enumerate(imgs):
        for s in range(a.n):
            seed = tf.constant([s, 1000 * i + 7], tf.int32)
            begin, size, _ = tf.image.stateless_sample_distorted_bounding_box(
                tf.shape(im), tf.zeros([0, 0, 4], tf.float32), seed=seed, min_object_covered=0,
                area_range=(0.40, 1.0), use_image_if_no_bounding_boxes=True)
            crop = tf.slice(im, begin, size)
            crop = tf.cast(tf.image.resize(crop, (64, 64), method="bilinear", antialias=True), tf.uint8)
            boxes.append(np.concatenate([begin.numpy()[:2], size.numpy()[:2]]))
            seeds_b.append(seed.numpy()); sizes_b.append(im.shape[:2]); crops.append(crop.numpy())
    R["ic_box"], R["ic_seed"], R["ic_size"] = np.asarray(boxes), np.asarray(seeds_b), np.asarray(sizes_b)
    R["ic_out"] = np.asarray(crops)
    # many boxes for distribution checks
    for j, (h, w) in enumerate([(173, 251), (480, 640), (100, 100)]):
        bb = []
        for s in range(4000):
            begin, size, _ = tf.image.stateless_sample_distorted_bounding_box(
                [h, w, 3], tf.zeros([0, 0, 4], tf.float32), seed=[s, 99 + j], min_object_covered=0,
                area_range=(0.40, 1.0), use_image_if_no_bounding_boxes=True)
            bb.append(np.concatenate([begin.numpy()[:2], size.numpy()[:2]]))
        R[f"icd{j}_box"], R[f"icd{j}_size"] = np.asarray(bb), np.asarray([h, w])

    tv = {"draws": [], "x": [], "gx": [], "box": [], "flip": [], "img": []}
    for i, im in enumerate(imgs):
        sq = T.inception_crop(tf.constant(im), 64)                      # a res x res uint8 input, as in the pipeline
        for s in range(a.n // 4):
            seed = tf.constant([s, 31 * i + 5], tf.int32)
            v = T.make_two_view(sq, 64, seed=seed)
            ks = tf.unstack(tf.random.experimental.stateless_split(seed, 4))
            tv["draws"].append([U(kk) for kk in ks]); tv["img"].append(sq.numpy())
            tv["x"].append(v["x_rgb"].numpy()); tv["gx"].append(v["gx_rgb"].numpy())
            tv["box"].append(v["crop_box"].numpy()); tv["flip"].append(v["flip"].numpy())
        # non-square input (central square path)
        seed = tf.constant([999, i], tf.int32)
        v = T.make_two_view(tf.constant(im), 48, seed=seed)
        ks = tf.unstack(tf.random.experimental.stateless_split(seed, 4))
        R[f"tvns{i}_draws"] = np.asarray([U(kk) for kk in ks])
        for kk in ("x_rgb", "gx_rgb", "crop_box", "flip"):
            R[f"tvns{i}_{kk}"] = v[kk].numpy()
    for kk, vv in tv.items():
        R[f"tv_{kk}"] = np.asarray(vv)

    # ---- Stage 2: cross-view pairs (default config; forced fallback; photometric alone; blur alone)
    def cv_draws(seed, tries):
        kb, ka, kp, kl = tf.unstack(tf.random.experimental.stateless_split(tf.convert_to_tensor(seed, tf.int32), 4))
        b = tf.unstack(tf.random.experimental.stateless_split(kb, 3))
        aa = tf.unstack(tf.random.experimental.stateless_split(ka, 3))
        kph = tf.unstack(tf.random.experimental.stateless_split(kp, 8))
        return {"b_area": U(b[0], [1]), "b_y": U(b[1], [1]), "b_x": U(b[2], [1]), "a_area": U(aa[0], [tries]),
                "a_y": U(aa[1], [tries]), "a_x": U(aa[2], [tries]), "local": U(kl), "photo": np.asarray([U(x) for x in kph])}

    cv = 0
    for i, im in enumerate(imgs):
        for s in range(a.n // 2):
            for variant, cfg in (("std", None), ("fallback", {"min_valid": 100_000}), ("nophoto", {"photometric": False})):
                if variant != "std" and s % 4:
                    continue
                seed = [s, 17 * i + 3]
                v = T.make_crossview_pair(tf.constant(im), 64, seed=seed, cfg=cfg)
                d = cv_draws(seed, 10)
                R[f"cv{cv}_img"] = np.int64(i)
                R[f"cv{cv}_variant"] = np.asarray(variant)
                for kk, vv in d.items():
                    R[f"cv{cv}_d_{kk}"] = np.asarray(vv)
                for kk in ("x_s_rgb", "x_t_rgb", "box_a", "box_b", "n_valid"):
                    R[f"cv{cv}_{kk}"] = v[kk].numpy()
                cv += 1
    R["n_cv"] = np.int64(cv)

    ph = 0
    cfg = dict(T.CROSSVIEW_DEFAULTS)
    for s in range(60):
        x = tf.image.convert_image_dtype(imgs[s % len(imgs)], tf.float32)
        x = tf.image.resize(x, (40, 52))
        key = tf.constant([s, 4242], tf.int32)
        out = T.crossview_photometric(x, key, cfg)
        kk = tf.unstack(tf.random.experimental.stateless_split(key, 8))
        R[f"ph{ph}_x"], R[f"ph{ph}_u"], R[f"ph{ph}_out"] = x.numpy(), np.asarray([U(z) for z in kk]), out.numpy()
        ph += 1
    R["n_ph"] = np.int64(ph)
    for j, sig in enumerate([0.1, 0.37, 1.0, 2.0]):
        x = tf.image.convert_image_dtype(imgs[j], tf.float32)[:40, :50]
        R[f"blur{j}_x"], R[f"blur{j}_sigma"] = x.numpy(), np.float32(sig)
        R[f"blur{j}_out"] = T._gaussian_blur(x, tf.constant(sig, tf.float32), 7).numpy()
    R["n_blur"] = np.int64(4)

    # ---- token-statistics crop of the released statistics (train_jax/compute_token_stats.py)
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "train_jax"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("cts", os.path.join(HERE, "..", "..", "..", "train_jax",
                                                                        "compute_token_stats.py"))
    try:
        cts = importlib.util.module_from_spec(spec); spec.loader.exec_module(cts)
        for s in range(12):
            im = imgs[s % len(imgs)]
            seed = tf.constant([s, 77], tf.int32)
            kk = tf.unstack(tf.random.experimental.stateless_split(seed, 3))
            R[f"ts{s}_img"], R[f"ts{s}_u"] = np.int64(s % len(imgs)), np.asarray([U(z) for z in kk])
            R[f"ts{s}_out"] = cts.alps_crop(tf.constant(im), 64, seed).numpy()
        R["n_ts"] = np.int64(12)
    except ImportError as e:                                      # compute_token_stats imports jax
        print(f"[skip] token-statistics crop: {e}")
        R["n_ts"] = np.int64(0)

    # ---- text: the deterministic part of random_sentence (the candidate sentences)
    texts = ["A dog. A cat!", "one sentence only", "  a.. b !! c...   ", "", "x. y. z.", "Hello world! It is a red car."]
    for j, t in enumerate(texts):
        dense = tf.strings.reduce_join(tf.reshape(tf.constant(t), [-1]), separator=" ")
        parts = tf.strings.split(tf.reshape(tf.strings.regex_replace(dense, "[.!]+", "|"), [1]), sep="|").values
        parts = tf.strings.strip(parts)
        parts = tf.boolean_mask(parts, tf.strings.length(parts) > 0)
        R[f"txt{j}_in"] = np.asarray(t)
        R[f"txt{j}_parts"] = np.asarray([p.decode() for p in parts.numpy()] or [""])
        R[f"txt{j}_n"] = np.int64(parts.shape[0])
    R["n_txt"] = np.int64(len(texts))
    _ = TK  # tokenizer: TF Text is needed for a token-level comparison (see the inference package's test)
    np.savez_compressed(a.out, **R)
    print(f"wrote {a.out}: {len(R)} arrays, tf {tf.__version__}")


if __name__ == "__main__":
    main()
