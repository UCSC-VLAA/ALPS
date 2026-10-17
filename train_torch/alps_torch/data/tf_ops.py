# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""numpy re-implementations of the TensorFlow image ops of the JAX input pipelines, with TF's float32 arithmetic
(same formulas, same order of operations; no TensorFlow needed).

    convert_image_dtype         uint8 -> float32 as x * float32(1/255) (not x / 255)
    resize_with_crop_or_pad     central crop / pad, offsets (d // 2)
    resize_bilinear             tf.image.resize(method="bilinear", antialias=False): half-pixel centres, the
                                ResizeBilinear kernel (lower = max(floor(in), 0), upper = min(ceil(in), n - 1))
    resize_bilinear_antialias   tf.image.resize(method="bilinear", antialias=True) = ScaleAndTranslate with the
                                triangle kernel, widened by the downscaling factor; rows first, then columns
    sample_distorted_bounding_box
                                the box sampler of tf.image.sample_distorted_bounding_box (area range, aspect ratio
                                range (0.75, 1.33), 100 attempts, whole image if none succeeds), drawing from an RNG
                                object with rand_float() and uniform(n), so that any random source can drive it
    gaussian_blur               REFLECT padding + separable depthwise convolution (horizontal, then vertical)

Images are [H, W, C] numpy arrays. Tests compare these functions with TensorFlow on the same inputs and draws
(tests/test_transforms_tf.py).
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

f32 = np.float32

# Fused multiply-adds inside the TF kernels (resize, depthwise convolution) round once where numpy rounds twice.
# TF builds compiled with FMA contraction (e.g. the arm64 wheels) match FMA = True bit for bit; the default (False)
# differs from them by float rounding only (~1e-7 of the value range; after the uint8 truncation of the Stage-1
# crop about 1.5e-5 of the values differ by one level). FMA = True is emulated in float64 and is slower.
FMA = False


def _fma(fma):
    return FMA if fma is None else fma


def convert_image_dtype(img: np.ndarray) -> np.ndarray:
    """tf.image.convert_image_dtype(uint8 -> float32)."""
    if img.dtype == np.float32:
        return img
    return img.astype(f32) * f32(1.0 / 255.0)


def resize_with_crop_or_pad(img: np.ndarray, th: int, tw: int) -> np.ndarray:
    h, w = img.shape[:2]
    oy, ox = max((h - th) // 2, 0), max((w - tw) // 2, 0)
    img = img[oy:oy + min(h, th), ox:ox + min(w, tw)]
    h, w = img.shape[:2]
    py, px = max((th - h) // 2, 0), max((tw - w) // 2, 0)
    if py or px or h < th or w < tw:
        out = np.zeros((th, tw) + img.shape[2:], img.dtype)
        out[py:py + h, px:px + w] = img
        img = out
    return img


def _madd(a, b, c, fma):
    """a * b + c in float32: two roundings, or one (a fused multiply-add, emulated in float64) when `fma`."""
    if _fma(fma):
        return (np.asarray(a, np.float64) * np.asarray(b, np.float64) + np.asarray(c, np.float64)).astype(f32)
    return (a * b + c).astype(f32)


def _bilinear_weights(out_size: int, in_size: int, fma=None):
    scale = f32(in_size) / f32(out_size)
    i = np.arange(out_size, dtype=f32)
    fin = _madd(i + f32(0.5), scale, f32(-0.5), fma)          # (x + 0.5f) * scale - 0.5f
    fl = np.floor(fin)
    lower = np.maximum(fl.astype(np.int64), 0)
    upper = np.minimum(np.ceil(fin).astype(np.int64), in_size - 1)
    return lower, upper, (fin - fl).astype(f32)


def resize_bilinear(img: np.ndarray, out_h: int, out_w: int, fma=None) -> np.ndarray:
    """tf.image.resize(..., method="bilinear", antialias=False) -> float32.

    fma=True contracts the multiply-adds of the kernel like a TF build compiled with FMA contraction; None = the
    module default FMA."""
    img = np.asarray(img)
    h, w = img.shape[:2]
    yl, yu, ylerp = _bilinear_weights(out_h, h, fma)
    xl, xu, xlerp = _bilinear_weights(out_w, w, fma)
    src = img.astype(f32)
    top_rows, bot_rows = src[yl], src[yu]
    tl, tr = top_rows[:, xl], top_rows[:, xu]
    bl, br = bot_rows[:, xl], bot_rows[:, xu]
    xlerp = xlerp[None, :, None]
    ylerp = ylerp[:, None, None]
    top = _madd(tr - tl, xlerp, tl, fma)                      # top_left + (top_right - top_left) * x_lerp
    bot = _madd(br - bl, xlerp, bl, fma)
    return _madd(bot - top, ylerp, top, fma)


def _triangle_spans(out_size: int, in_size: int, scale: np.float32, fma=None):
    """ComputeSpansCore of TF's ScaleAndTranslate (translation 0, antialias, triangle kernel of radius 1)."""
    inv_scale = f32(1.0 / float(scale))                   # 1.0 / scale in double, stored as float
    inv_translate = f32(-0.0)
    kernel_scale = max(inv_scale, f32(1.0))
    radius = f32(1.0)
    span_size = min(2 * int(math.ceil(float(radius * kernel_scale))) + 1, in_size)
    one_over_ks = f32(1.0) / kernel_scale
    x = np.arange(out_size, dtype=f32)
    sample = _madd(x + f32(0.5), inv_scale, inv_translate, fma)
    inside = ~((sample < 0) | (sample > f32(in_size)))
    start = np.ceil(sample - radius * kernel_scale - f32(0.5)).astype(np.int64)
    end = np.floor(sample + radius * kernel_scale - f32(0.5)).astype(np.int64)
    start = np.clip(start, 0, in_size - 1)
    end = np.clip(end, 0, in_size - 1) + 1
    this = end - start
    weights = np.zeros((out_size, span_size), f32)
    total = np.zeros(out_size, f32)
    for k in range(span_size):
        src = (start + k).astype(f32)
        pos = src + f32(0.5) - sample
        a = np.abs(pos * one_over_ks)
        wk = np.where(a < f32(1.0), f32(1.0) - a, f32(0.0)).astype(f32)
        wk = np.where(k < this, wk, f32(0.0))
        weights[:, k] = wk
        total = (total + wk).astype(f32)
    ok = np.abs(total) >= f32(1000.0) * np.finfo(f32).tiny
    inv_total = np.where(ok, f32(1.0) / np.where(ok, total, f32(1.0)), f32(0.0)).astype(f32)
    weights = (weights * inv_total[:, None]).astype(f32)
    weights[~inside] = 0.0
    start = np.where(inside, start, 0)
    return start, weights, span_size


def resize_bilinear_antialias(img: np.ndarray, out_h: int, out_w: int, fma=None) -> np.ndarray:
    """tf.image.resize(..., method="bilinear", antialias=True) -> float32 (rows filtered first, then columns,
    each output accumulated sequentially over the kernel taps in float32; fma as in resize_bilinear)."""
    img = np.asarray(img)
    h, w = img.shape[:2]
    sy = f32(out_h) / f32(h)
    sx = f32(out_w) / f32(w)
    ys, yw, yspan = _triangle_spans(out_h, h, sy, fma)
    xs, xw, xspan = _triangle_spans(out_w, w, sx, fma)
    src = img.astype(f32)
    inter = np.zeros((out_h, w) + img.shape[2:], f32)
    for k in range(yspan):
        rows = np.minimum(ys + k, h - 1)
        inter = _madd(yw[:, k][:, None, None], src[rows], inter, fma)
    out = np.zeros((out_h, out_w) + img.shape[2:], f32)
    for k in range(xspan):
        cols = np.minimum(xs + k, w - 1)
        out = _madd(xw[:, k][None, :, None], inter[:, cols], out, fma)
    return out


def truncate_to_uint8(x: np.ndarray) -> np.ndarray:
    """tf.cast(float32 -> uint8) of values inside [0, 255]: truncation toward zero."""
    return np.clip(x, 0.0, 255.0).astype(np.uint8)


# ------------------------------------------------------------------------------------------- bounding-box sampler
def _lrintf(x) -> int:
    """lrintf: round to nearest, ties to even (float32 argument)."""
    return int(np.rint(f32(x)))


class NumpyBoxRNG:
    """rand_float() in [0, 1) and uniform(n) in [0, n) from a numpy Generator (the training-time random source)."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def rand_float(self) -> np.float32:
        return f32(self.rng.integers(0, 1 << 23) * f32(2.0 ** -23))

    def uniform(self, n: int) -> int:
        return int(self.rng.integers(0, n))


def _generate_random_crop(width: int, height: int, min_rel: np.float32, max_rel: np.float32,
                          aspect: np.float32, rng):
    if max_rel <= 0.0 or aspect <= 0.0 or width <= 0 or height <= 0 or min_rel > max_rel:
        return None
    min_area = f32(f32(min_rel * f32(width)) * f32(height))
    max_area = f32(f32(max_rel * f32(width)) * f32(height))
    h = _lrintf(np.sqrt(f32(min_area / aspect)))
    max_h = _lrintf(np.sqrt(f32(max_area / aspect)))
    if _lrintf(f32(f32(max_h) * aspect)) > width:
        k_eps = float(f32(0.0000001))
        max_h = int((width + 0.5 - k_eps) / float(aspect))
        if _lrintf(f32(f32(max_h) * aspect)) > width:
            max_h -= 1
    if max_h > height:
        max_h = height
    if h >= max_h:
        h = max_h
    if h < max_h:
        h += rng.uniform(max_h - h + 1)
    w = _lrintf(f32(f32(h) * aspect))
    area = f32(w * h)
    for _ in range(2):
        if area < min_area:
            h += 1
            w = _lrintf(f32(f32(h) * aspect))
            area = f32(w * h)
    if area < min_area or area > max_area or w > width or h > height or w <= 0 or h <= 0:
        return None
    y = rng.uniform(height - h) if h < height else 0
    x = rng.uniform(width - w) if w < width else 0
    return y, x, h, w


def sample_distorted_bounding_box(height: int, width: int, rng, area_range=(0.4, 1.0),
                                  aspect_ratio_range=(0.75, 1.33), max_attempts: int = 100) -> Tuple[int, int, int, int]:
    """(offset_y, offset_x, crop_height, crop_width) of an Inception-style crop, as the TF op with no bounding
    boxes, use_image_if_no_bounding_boxes=True and min_object_covered=0."""
    lo_ar, hi_ar = f32(aspect_ratio_range[0]), f32(aspect_ratio_range[1])
    min_a, max_a = f32(area_range[0]), f32(area_range[1])
    for _ in range(int(max_attempts)):
        aspect = f32(f32(rng.rand_float() * f32(hi_ar - lo_ar)) + lo_ar)
        r = _generate_random_crop(width, height, min_a, max_a, aspect, rng)
        if r is not None and r[2] * r[3] >= 1:
            return r
    return 0, 0, height, width


# ---------------------------------------------------------------------------------------------------------- blur
def gaussian_blur(x: np.ndarray, sigma, radius: int, fma=None) -> np.ndarray:
    t = np.arange(-radius, radius + 1, dtype=f32)
    k = np.exp(f32(-0.5) * np.square(t / f32(sigma))).astype(f32)
    k = (k / k.sum(dtype=f32)).astype(f32)
    img = np.pad(x.astype(f32), ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    hh, ww = x.shape[:2]
    acc = np.zeros((hh + 2 * radius, ww) + x.shape[2:], f32)
    for j in range(2 * radius + 1):                      # horizontal pass (along W)
        acc = _madd(k[j], img[:, j:j + ww], acc, fma)
    out = np.zeros((hh, ww) + x.shape[2:], f32)
    for i in range(2 * radius + 1):                      # vertical pass (along H)
        out = _madd(k[i], acc[i:i + hh], out, fma)
    return out


# ------------------------------------------------------------------------------- TF's stateless Philox stream
_M0, _M1, _W0, _W1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85
_MASK = 0xFFFFFFFF


def _philox4x32(counter, key):
    c = list(counter)
    k0, k1 = key
    for r in range(10):
        p0, p1 = _M0 * c[0], _M1 * c[2]
        c = [((p1 >> 32) ^ c[1] ^ k0) & _MASK, p1 & _MASK, ((p0 >> 32) ^ c[3] ^ k1) & _MASK, p0 & _MASK]
        if r < 9:
            k0, k1 = (k0 + _W0) & _MASK, (k1 + _W1) & _MASK
    return c


class TFPhiloxRNG:
    """The random stream of TF's stateless ops for a seed [s0, s1] (GenerateKey + PhiloxRandom + SimplePhilox):
    reproduces e.g. tf.image.stateless_sample_distorted_bounding_box exactly. rand_float() / uniform(n) as the
    C++ SimplePhilox (uniform: rejection sampling of ExactUniformInt)."""

    def __init__(self, seed):
        s0, s1 = (int(v) & 0xFFFFFFFFFFFFFFFF for v in seed)
        ctr = [s0 & _MASK, (s0 >> 32) & _MASK, s1 & _MASK, (s1 >> 32) & _MASK]
        mix = _philox4x32(ctr, (0x3EC8F720, 0x02461E29))
        self.key = (mix[0], mix[1])
        self.counter = [0, 0, mix[2], mix[3]]
        self._buf, self._i = [], 4

    def _next(self) -> int:
        if self._i == 4:
            self._buf = _philox4x32(self.counter, self.key)
            c = self.counter
            for j in range(4):                                  # 128-bit counter increment
                c[j] = (c[j] + 1) & _MASK
                if c[j]:
                    break
            self._i = 0
        v = self._buf[self._i]
        self._i += 1
        return v

    def rand_float(self) -> np.float32:
        bits = np.uint32((127 << 23) | (self._next() & 0x7FFFFF))
        return f32(bits.view(np.float32) - f32(1.0))

    def uniform(self, n: int) -> int:
        n = int(n)
        if n == 0:
            self._next()
            return 0
        if n & (n - 1) == 0:
            return self._next() & (n - 1)
        rem = (_MASK % n + 1) % n
        while True:
            r = self._next()
            if r >= rem:
                return r % n
