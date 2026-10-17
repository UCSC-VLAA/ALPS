# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Properties of the transforms, the caption fields and the input pipelines (no TensorFlow needed; the comparison
with the TF functions is tests/test_transforms_tf.py)."""
import numpy as np
import pytest
import torch

import synthetic_data as D
from alps_torch.data import datasets as DS
from alps_torch.data import tf_ops as T
from alps_torch.data import transforms as X
from alps_torch.data.text import Stage1Text, random_element, random_sentence, sentence_candidates
from alps_torch.models.pixel_decoder import pixel_shuffle_jax


def test_pixel_shuffle_order():
    """output (2h + i, 2w + j) channel k = input channel (2i + j) C + k (the JAX order, not torch's)."""
    c, h, w = 3, 2, 5
    x = torch.arange(4 * c * h * w, dtype=torch.float32).view(1, 4 * c, h, w)
    y = pixel_shuffle_jax(x, 2)
    for i in range(2):
        for j in range(2):
            for k in range(c):
                assert torch.equal(y[0, k, i::2, j::2], x[0, (2 * i + j) * c + k])
    assert not torch.equal(y, torch.nn.functional.pixel_shuffle(x, 2))


def test_crossview_full_frame_without_augmentation_is_identity():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (70, 90, 3)).astype(np.uint8)
    cfg = dict(b_scale=(1.0, 1.0), a_global=(1.0, 1.0), p_local=0.0, photometric=False)
    v = X.make_crossview_pair(img, 64, X.crossview_draws(rng), cfg)
    assert np.array_equal(v["box_a"], [0, 0, 1]) and np.array_equal(v["box_b"], [0, 0, 1])
    assert np.array_equal(v["x_s_rgb"], v["x_t_rgb"]) and int(v["n_valid"]) == 16


def test_crossview_geometry_properties():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (300, 400, 3)).astype(np.uint8)
    n_fallback = 0
    for _ in range(200):
        v = X.make_crossview_pair(img, 224, X.crossview_draws(rng))
        a, b = v["box_a"], v["box_b"]
        assert 0 <= a[0] and a[0] + a[2] <= 1 + 1e-6 and 0 <= b[1] and b[1] + b[2] <= 1 + 1e-6
        assert b[2] ** 2 >= 0.4 - 0.01                     # teacher area >= 0.4 of the central square
        assert int(v["n_valid"]) >= 16
        n_fallback += int(np.array_equal(a, b))
        assert v["x_s_rgb"].min() >= 0 and v["x_s_rgb"].max() <= 1 and v["x_s_rgb"].dtype == np.float32
    assert n_fallback < 5                                   # practically never at 224 px


def test_two_view_box_matches_pixels():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, (64, 64, 3)).astype(np.uint8)
    for _ in range(20):
        d = X.two_view_draws(rng)
        v = X.make_two_view(img, 64, d)
        y0, x0, s = [int(round(float(t) * 64)) for t in v["crop_box"][:3]]
        ref = T.resize_bilinear(T.convert_image_dtype(img)[y0:y0 + s, x0:x0 + s], 64, 64)
        np.testing.assert_array_equal(v["gx_rgb"], ref[:, ::-1] if v["flip"] else ref)
        assert np.array_equal(v["x_rgb"], T.convert_image_dtype(img))        # identity resize at equal size
        assert 0.70 <= v["crop_box"][2] < 1.0


def test_inception_crop_shapes_and_bounds():
    rng = T.NumpyBoxRNG(np.random.default_rng(3))
    img = np.random.default_rng(3).integers(0, 256, (120, 200, 3)).astype(np.uint8)
    for _ in range(50):
        y, x, h, w = T.sample_distorted_bounding_box(120, 200, rng)
        assert 0 <= y and y + h <= 120 and 0 <= x and x + w <= 200
        assert (h * w >= 0.4 * 120 * 200 - 1 and 0.75 - 0.02 <= w / h <= 1.33 + 0.02) or (h, w) == (120, 200)
    out = X.inception_crop(img, 32, rng=rng)
    assert out.shape == (32, 32, 3) and out.dtype == np.uint8


def test_tf_philox_stream_is_deterministic():
    a, b = T.TFPhiloxRNG([3, 4]), T.TFPhiloxRNG([3, 4])
    assert [a.uniform(97) for _ in range(20)] == [b.uniform(97) for _ in range(20)]
    assert 0.0 <= float(a.rand_float()) < 1.0


def test_blur_and_photometric_ranges():
    rng = np.random.default_rng(4)
    flat = np.full((20, 30, 3), 0.3, np.float32)
    np.testing.assert_allclose(T.gaussian_blur(flat, 1.3, 7), flat, atol=1e-6)        # normalized kernel
    x = rng.random((20, 30, 3)).astype(np.float32)
    for _ in range(20):
        y = X.photometric(x, rng.random(8))
        assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1
    gray = X.photometric(x, [0.9, 0, 0, 0, 0.0, 0.9, 0, 0])            # no jitter, grayscale, no blur
    assert np.allclose(gray[..., 0], gray[..., 1]) and np.allclose(gray[..., 1], gray[..., 2])


def test_caption_fields():
    tok = Stage1Text()
    rng = np.random.default_rng(5)
    out = tok("A photo of a dog.", ["A dog on grass. It runs fast!", "x"], rng)
    assert out["labels"].shape == (80,) and out["itc_labels1"].shape == (32,) and out["itc_labels2"].shape == (32,)
    for f in out.values():
        assert f[0] == 1 and f[-1] == 101                   # [bos] ... [CLS]
        assert 2 in f.tolist()                              # [eos]
    long = " ".join(["word"] * 200)
    f = tok(long, long, rng)["labels"]
    assert f[-2] == 2 and f[-1] == 101 and 0 not in f.tolist()        # truncated: [eos] in slot L - 2
    assert sentence_candidates("a. b!! c... ") == ["a", "b", "c"] and sentence_candidates("...") == []
    assert random_sentence("...", rng) == "..."             # no sentence: the whole text
    assert random_element([], rng) == "" and random_element("s", rng) == "s"
    picks = {random_element(["a", "b", "c"], rng) for _ in range(100)}
    assert picks == {"a", "b", "c"}


def test_stage1_dataset_jsonl_and_tar(tmp_path):
    for src in (D.stage1_jsonl(str(tmp_path)), D.stage1_tar(str(tmp_path))):
        ds = DS.Stage1Dataset(src, 32, seed=0, shuffle_buffer=4)
        batch = DS.collate([e for _, e in zip(range(6), iter(ds))])
        assert batch["x_rgb"].shape == (6, 3, 32, 32) and batch["gx_rgb"].dtype == torch.float32
        assert batch["labels"].shape == (6, 80) and batch["labels"].dtype == torch.int64
        assert batch["itc_labels1"].shape == (6, 32) and batch["crop_box"].shape == (6, 4)
        assert batch["flip"].dtype == torch.bool and ds.n_failed == 0


def test_stage2_dataset_mixed_sources_and_sharding(tmp_path):
    pairs = D.stage2_jsonl(str(tmp_path), n=12)
    D.stage1_jsonl(str(tmp_path), n=6)                      # writes images to tmp/img
    ds = DS.Stage2Dataset([(pairs, 0.7), (str(tmp_path / "img"), 0.3)], 32, seed=1,
                          crossview_cfg={"min_valid": 2})
    ex = [e for _, e in zip(range(10), iter(ds))]
    b = DS.collate(ex)
    assert b["x_s_rgb"].shape == (10, 3, 32, 32) and b["box_a"].shape == (10, 3)
    # two readers of one list source see disjoint records within a pass
    src = DS.make_source(pairs)
    src.seed = 3
    r0 = [r["source"] for _, r in zip(range(6), src.records(0, 2, np.random.default_rng(0)))]
    r1 = [r["source"] for _, r in zip(range(6), src.records(1, 2, np.random.default_rng(1)))]
    assert not set(r0) & set(r1) and len(set(r0) | set(r1)) == 12
    assert DS.expand_pattern("a/{008..011}.tar") == ["a/008.tar", "a/009.tar", "a/010.tar", "a/011.tar"]


def test_bad_sources_fail_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        DS.make_source(str(tmp_path / "missing.png"))
    with pytest.raises(FileNotFoundError):
        DS.make_source(str(tmp_path / "*.jpg"))
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"image": "nope.jpg", "caption": "a", "synthetic_caption": "b"}\n')
    old = DS.MAX_CONSECUTIVE_FAILURES
    DS.MAX_CONSECUTIVE_FAILURES = 20
    try:
        with pytest.raises(RuntimeError, match="consecutive"):
            next(iter(DS.Stage1Dataset(str(bad), 32)))
    finally:
        DS.MAX_CONSECUTIVE_FAILURES = old
