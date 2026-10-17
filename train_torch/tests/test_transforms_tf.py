# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""numpy transforms (alps_torch.data) vs the TensorFlow transforms of train_jax, for identical inputs and draws.

Needs the reference file written by tests/tf_reference/make_reference.py in a TensorFlow environment:
    ALPS_TF_REF=/path/alps_tf_reference.npz pytest tests/test_transforms_tf.py

TF kernels may fuse multiply-adds (FMA contraction, compiler dependent). With alps_torch.data.tf_ops.FMA = True the
resize kernels are emulated with fused multiply-adds and must match a TF build that contracts bit for bit (checked
when the reference was written by such a build, e.g. the arm64 wheels); the default (no fusion) must agree to float
rounding.
"""
import os

import numpy as np
import pytest

from alps_torch.data import tf_ops as T
from alps_torch.data import transforms as X
from alps_torch.data.text import sentence_candidates

REF = os.environ.get("ALPS_TF_REF", "")
pytestmark = pytest.mark.skipif(not os.path.exists(REF), reason="set ALPS_TF_REF to the TF reference npz")


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(REF, allow_pickle=False))


@pytest.fixture(params=[False, True], ids=["plain", "fma"])
def fma(request):
    old = T.FMA
    T.FMA = request.param
    yield request.param
    T.FMA = old


def _tf_contracts(ref):
    """Whether the reference build fuses multiply-adds (decided on one resize)."""
    old = T.FMA
    T.FMA = True
    try:
        k = 0
        im = _img(ref, ref[f"rs{k}_src"])
        oh, ow = [int(v) for v in ref[f"rs{k}_size"]]
        return bool(np.array_equal(T.resize_bilinear(T.convert_image_dtype(im), oh, ow), ref[f"rs{k}_bil_f"]))
    finally:
        T.FMA = old


def _img(ref, i):
    return ref[f"img{int(i)}"]


def test_resize(ref, fma):
    worst = {"conv": 0.0, "bil_f": 0.0, "bil_u8": 0.0, "aa_u8": 0.0}
    for k in range(int(ref["n_resize"])):
        im = _img(ref, ref[f"rs{k}_src"])
        oh, ow = [int(v) for v in ref[f"rs{k}_size"]]
        f = T.convert_image_dtype(im)
        worst["conv"] = max(worst["conv"], np.abs(f - ref[f"rs{k}_conv"]).max())
        worst["bil_f"] = max(worst["bil_f"], np.abs(T.resize_bilinear(f, oh, ow) - ref[f"rs{k}_bil_f"]).max())
        worst["bil_u8"] = max(worst["bil_u8"], np.abs(T.resize_bilinear(im, oh, ow) - ref[f"rs{k}_bil_u8"]).max())
        worst["aa_u8"] = max(worst["aa_u8"], np.abs(T.resize_bilinear_antialias(im, oh, ow) - ref[f"rs{k}_aa_u8"]).max())
    print(f"resize max |diff| (fma={fma}):", worst)
    assert worst["conv"] == 0.0                         # x * float32(1/255), bit-exact
    if fma and _tf_contracts(ref):
        assert worst["bil_f"] == 0 and worst["bil_u8"] == 0 and worst["aa_u8"] == 0
    assert worst["bil_f"] <= 2e-5 and worst["bil_u8"] <= 5e-3 and worst["aa_u8"] <= 1e-3   # [0, 1] / [0, 255] scales


def test_inception_crop_given_box(ref, fma):
    n_diff = n_tot = 0
    for j in range(len(ref["ic_box"])):
        i = j // (len(ref["ic_box"]) // int(ref["n_images"]))
        im = _img(ref, i)
        assert tuple(im.shape[:2]) == tuple(ref["ic_size"][j])
        y, x, h, w = [int(v) for v in ref["ic_box"][j]]
        out = X.inception_crop(im, 64, box=(y, x, h, w))
        d = np.abs(out.astype(int) - ref["ic_out"][j].astype(int))
        assert d.max() <= 1
        n_diff += int((d > 0).sum()); n_tot += d.size
    print(f"inception crop (fma={fma}): {n_diff} of {n_tot} uint8 values differ by 1")
    if fma and _tf_contracts(ref):
        assert n_diff == 0
    assert n_diff / n_tot < 1e-4


def test_box_sampler_exact(ref):
    """With TF's stateless Philox stream re-implemented (tf_ops.TFPhiloxRNG), the box sampler reproduces
    tf.image.stateless_sample_distorted_bounding_box exactly, seed for seed."""
    n = 0
    for j in range(len(ref["ic_box"])):
        h, w = [int(v) for v in ref["ic_size"][j]]
        b = T.sample_distorted_bounding_box(h, w, T.TFPhiloxRNG(ref["ic_seed"][j]))
        assert tuple(b) == tuple(int(v) for v in ref["ic_box"][j]), j
        n += 1
    for j in range(3):
        h, w = [int(v) for v in ref[f"icd{j}_size"]]
        for s, tb in enumerate(ref[f"icd{j}_box"]):
            assert tuple(T.sample_distorted_bounding_box(h, w, T.TFPhiloxRNG([s, 99 + j]))) == tuple(int(v) for v in tb)
            n += 1
    print(f"box sampler: {n} of {n} TF boxes reproduced exactly")


def test_box_sampler_distribution(ref):
    """Same algorithm as tf.image.sample_distorted_bounding_box: the box statistics match (the random streams are
    different, so the comparison is statistical)."""
    rng = T.NumpyBoxRNG(np.random.default_rng(0))
    for j in range(3):
        h, w = [int(v) for v in ref[f"icd{j}_size"]]
        tb = ref[f"icd{j}_box"].astype(np.float64)
        nb = np.asarray([T.sample_distorted_bounding_box(h, w, rng) for _ in range(len(tb))], np.float64)
        for name, f in (("area", lambda b: b[:, 2] * b[:, 3] / (h * w)), ("aspect", lambda b: b[:, 3] / b[:, 2]),
                        ("y0", lambda b: b[:, 0] / h), ("x0", lambda b: b[:, 1] / w)):
            a, b = np.sort(f(tb)), np.sort(f(nb))
            grid = np.unique(np.concatenate([a, b]))
            ks = np.abs(np.searchsorted(a, grid, side="right") / len(a) - np.searchsorted(b, grid, side="right") / len(b)).max()
            assert ks < 0.05, (h, w, name, ks)                     # two-sample KS, n = 4000 each
        full_t = np.mean((tb[:, 2] == h) & (tb[:, 3] == w))
        full_n = np.mean((nb[:, 2] == h) & (nb[:, 3] == w))
        assert abs(full_t - full_n) < 0.03, (full_t, full_n)
        assert (nb[:, 2] * nb[:, 3] >= np.floor(0.4 * h * w) - 1).all() or full_n > 0


def test_two_view(ref, fma):
    tol = 0.0 if fma and _tf_contracts(ref) else 2e-6
    for j in range(len(ref["tv_draws"])):
        v = X.make_two_view(ref["tv_img"][j], 64, ref["tv_draws"][j])
        np.testing.assert_array_equal(v["crop_box"], ref["tv_box"][j])
        assert bool(v["flip"]) == bool(ref["tv_flip"][j])
        np.testing.assert_allclose(v["x_rgb"], ref["tv_x"][j], atol=tol, rtol=0)
        np.testing.assert_allclose(v["gx_rgb"], ref["tv_gx"][j], atol=tol, rtol=0)
    for i in range(int(ref["n_images"])):                      # non-square images: the central-square path
        v = X.make_two_view(_img(ref, i), 48, ref[f"tvns{i}_draws"])
        np.testing.assert_array_equal(v["crop_box"], ref[f"tvns{i}_crop_box"])
        assert bool(v["flip"]) == bool(ref[f"tvns{i}_flip"])
        np.testing.assert_allclose(v["x_rgb"], ref[f"tvns{i}_x_rgb"], atol=tol, rtol=0)
        np.testing.assert_allclose(v["gx_rgb"], ref[f"tvns{i}_gx_rgb"], atol=tol, rtol=0)


def test_crossview_pairs(ref, fma):
    seen = {"local": 0, "global": 0, "fallback": 0, "blur": 0, "gray": 0, "jitter": 0}
    worst_s = 0.0
    for c in range(int(ref["n_cv"])):
        variant = str(ref[f"cv{c}_variant"])
        cfg = {"std": None, "fallback": {"min_valid": 100_000}, "nophoto": {"photometric": False}}[variant]
        d = {k: ref[f"cv{c}_d_{k}"] for k in ("b_area", "b_y", "b_x", "a_area", "a_y", "a_x", "local", "photo")}
        v = X.make_crossview_pair(_img(ref, ref[f"cv{c}_img"]), 64, d, cfg)
        for k in ("box_a", "box_b", "n_valid"):
            np.testing.assert_array_equal(v[k], ref[f"cv{c}_{k}"], err_msg=f"pair {c} {k}")
        np.testing.assert_allclose(v["x_t_rgb"], ref[f"cv{c}_x_t_rgb"], atol=0.0 if fma and _tf_contracts(ref) else 2e-6, rtol=0)
        worst_s = max(worst_s, float(np.abs(v["x_s_rgb"] - ref[f"cv{c}_x_s_rgb"]).max()))
        seen["local" if d["local"] < 0.5 else "global"] += 1
        seen["fallback"] += int(np.array_equal(v["box_a"], v["box_b"]))
        if variant == "std":
            seen["blur"] += int(d["photo"][5] < 0.5); seen["gray"] += int(d["photo"][4] < 0.1)
            seen["jitter"] += int(d["photo"][0] < 0.8)
    print("cross-view coverage:", seen, "student-view max |diff|:", worst_s)
    assert worst_s <= 2e-6
    assert all(v > 0 for v in seen.values()), seen


def test_photometric_and_blur(ref, fma):
    worst = 0.0
    for j in range(int(ref["n_ph"])):
        out = X.photometric(ref[f"ph{j}_x"], ref[f"ph{j}_u"])
        worst = max(worst, float(np.abs(out - ref[f"ph{j}_out"]).max()))
    assert worst <= 2e-6, worst
    for j in range(int(ref["n_blur"])):
        out = T.gaussian_blur(ref[f"blur{j}_x"], ref[f"blur{j}_sigma"], 7)
        np.testing.assert_allclose(out, ref[f"blur{j}_out"], atol=1e-6, rtol=0)


def test_token_stats_crop(ref, fma):
    if int(ref["n_ts"]) == 0:
        pytest.skip("reference written without the token-statistics crop")
    for s in range(int(ref["n_ts"])):
        out = X.token_stats_crop(_img(ref, ref[f"ts{s}_img"]), 64, ref[f"ts{s}_u"])
        np.testing.assert_allclose(out, ref[f"ts{s}_out"], atol=0.0 if fma and _tf_contracts(ref) else 2e-6, rtol=0)


def test_sentence_candidates(ref):
    for j in range(int(ref["n_txt"])):
        got = sentence_candidates(str(ref[f"txt{j}_in"]))
        want = [str(p) for p in ref[f"txt{j}_parts"]] if int(ref[f"txt{j}_n"]) else []
        assert got == want, (str(ref[f"txt{j}_in"]), got, want)
