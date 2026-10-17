# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Image transforms of both stages, as in the JAX input pipelines (train_jax/alps_jax/data/transforms.py).

Every random quantity is an explicit argument (uniform draws in [0, 1) as the TF code draws them), so the geometry
can be checked against the TensorFlow functions for identical draws; `*_draws(rng)` produce the draws from a numpy
Generator for training. Images are [H, W, 3] numpy arrays (uint8 input, float32 [0, 1] output).

Stage 1
    inception_crop    box from sample_distorted_bounding_box (area 40-100 %, aspect 3/4-4/3, whole image if 100
                      attempts fail), antialiased bilinear resize to res x res (aspect not preserved), TRUNCATED to
                      uint8
    make_two_view     base view x = the crop in [0, 1] (central square, bilinear resize to res: the identity in the
                      pipeline); geometric view g(x) = square crop of x with side int(sqrt(U(0.5, 1)) s), offsets
                      int(U (s - side)), bilinear resize back to res (no antialias), flipped if U < 0.5; crop_box =
                      (y0, x0, side, side) / s from the integer box
Stage 2
    make_crossview_pair
                      central square of side m; teacher box B: area U(0.4, 1), side = clip(round(sqrt(area) m), 1, m)
                      (round half to even), y0 = min(floor(U (m - side + 1)), m - side), same for x0; student box A:
                      local (area U(0.1, 0.4)) with probability 0.5 else global (U(0.4, 1)), 10 candidates, the first
                      with >= 16 of B's patch centres inside A, else A = B; both cut and resized to res (bilinear, no
                      antialias); photometric augmentation on A only
    photometric       p 0.8 colour jitter (brightness U(0.6, 1.4) x, contrast U(0.6, 1.4) around the mean
                      luminance, saturation U(0.8, 1.2) around the per-pixel luminance, each clipped to [0, 1]);
                      p 0.1 grayscale; sigma U(0.1, 2) always drawn, p 0.5 Gaussian blur of radius 7 (REFLECT);
                      a final clip
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from . import tf_ops as T

f32 = np.float32


def _u(u, a, b):
    """tf.random.stateless_uniform(..., minval=a, maxval=b) from its [0, 1) draw: u * (b - a) + a in float32."""
    return f32(f32(u) * f32(f32(b) - f32(a)) + f32(a))


# ------------------------------------------------------------------------------------------------------- Stage 1
def inception_crop_box(h: int, w: int, rng, area_min: int = 40, area_max: int = 100):
    return T.sample_distorted_bounding_box(h, w, rng, area_range=(area_min / 100, area_max / 100))


def inception_crop(image_u8: np.ndarray, res: int, box=None, rng=None, area_min: int = 40,
                   area_max: int = 100) -> np.ndarray:
    """uint8 [H, W, 3] -> uint8 [res, res, 3]. `box` = (y, x, h, w), or drawn with `rng` (a tf_ops RNG object)."""
    if box is None:
        box = inception_crop_box(image_u8.shape[0], image_u8.shape[1], rng, area_min, area_max)
    y, x, ch, cw = box
    crop = image_u8[y:y + ch, x:x + cw]
    return T.truncate_to_uint8(T.resize_bilinear_antialias(crop, res, res))


def two_view_draws(rng: np.random.Generator) -> np.ndarray:
    """[u_side, u_y, u_x, u_flip]."""
    return rng.random(4, dtype=f32)


def make_two_view(image_u8: np.ndarray, res: int, draws, dense_scale=(0.5, 1.0)) -> Dict[str, np.ndarray]:
    u_side, u_y, u_x, u_flip = [f32(v) for v in draws]
    img = T.convert_image_dtype(image_u8)
    h, w = img.shape[:2]
    s = min(h, w)
    img = T.resize_with_crop_or_pad(img, s, s)
    x_rgb = T.resize_bilinear(img, res, res)
    sf = f32(s)
    side = np.sqrt(_u(u_side, dense_scale[0], dense_scale[1]))
    side_px = int(f32(side * sf))
    y0 = int(f32(u_y * f32(s - side_px)))
    x0 = int(f32(u_x * f32(s - side_px)))
    gx_rgb = T.resize_bilinear(img[y0:y0 + side_px, x0:x0 + side_px], res, res)
    flip = bool(u_flip < f32(0.5))
    if flip:
        gx_rgb = gx_rgb[:, ::-1]
    crop_box = np.asarray([f32(y0) / sf, f32(x0) / sf, f32(side_px) / sf, f32(side_px) / sf], f32)
    return {"x_rgb": x_rgb, "gx_rgb": np.ascontiguousarray(gx_rgb), "crop_box": crop_box, "flip": np.bool_(flip)}


# ------------------------------------------------------------------------------------------------------- Stage 2
CROSSVIEW_DEFAULTS = dict(
    b_scale=(0.4, 1.0), a_local=(0.1, 0.4), a_global=(0.4, 1.0), p_local=0.5, min_valid=16, tries=10, patch=16,
    photometric=True, p_jitter=0.8, brightness=0.4, contrast=0.4, saturation=0.2, p_gray=0.1, p_blur=0.5,
    sigma=(0.1, 2.0), blur_radius=7,
)


def crossview_draws(rng: np.random.Generator, tries: int = 10) -> Dict[str, np.ndarray]:
    """Uniform [0, 1) draws of one cross-view pair (keys as in make_crossview_pair)."""
    r = lambda n: rng.random(n, dtype=f32)
    return {"b_area": r(1), "b_y": r(1), "b_x": r(1), "a_area": r(tries), "a_y": r(tries), "a_x": r(tries),
            "local": r(1)[0], "photo": r(8)}


def _cv_square(image_u8):
    img = T.convert_image_dtype(image_u8)
    m = min(img.shape[0], img.shape[1])
    return T.resize_with_crop_or_pad(img, m, m), m


def _cv_boxes(u_area, u_y, u_x, lo, hi, m: int, hi_minus_lo=None):
    mf = f32(m)
    d = f32(hi_minus_lo) if hi_minus_lo is not None else f32(hi - lo)
    area = (f32(lo) + np.asarray(u_area, f32) * d).astype(f32)
    side = np.clip(np.round(np.sqrt(area) * mf).astype(np.int64), 1, m)
    room = (m - side).astype(f32)
    y0 = np.floor(np.asarray(u_y, f32) * (room + f32(1.0))).astype(np.int64)
    x0 = np.floor(np.asarray(u_x, f32) * (room + f32(1.0))).astype(np.int64)
    y0 = np.minimum(y0, m - side); x0 = np.minimum(x0, m - side)
    box = (np.stack([y0.astype(f32), x0.astype(f32), side.astype(f32)], -1) / mf).astype(f32)
    return y0, x0, side, box


def crossview_valid_count(box_a: np.ndarray, box_b: np.ndarray, grid: int) -> np.ndarray:
    c = ((np.arange(grid, dtype=f32) + f32(0.5)) / f32(grid)).astype(f32)
    yc = box_b[0] + c * box_b[2]
    xc = box_b[1] + c * box_b[2]
    box_a = np.asarray(box_a, f32)
    v = (yc - box_a[..., 0:1]) / box_a[..., 2:3]
    u = (xc - box_a[..., 1:2]) / box_a[..., 2:3]
    rows = np.sum((v >= 0) & (v < 1), axis=-1)
    cols = np.sum((u >= 0) & (u < 1), axis=-1)
    return (rows * cols).astype(np.int32)


_LUM = np.asarray([0.299, 0.587, 0.114], f32)


def _lum(im):
    return np.sum(im * _LUM, axis=-1, keepdims=True, dtype=f32)


def photometric(x: np.ndarray, u, cfg: Optional[dict] = None) -> np.ndarray:
    """Colour jitter / grayscale / blur of the student view, from the 8 uniform draws k[0..7] of the TF code."""
    cfg = dict(CROSSVIEW_DEFAULTS, **(cfg or {}))
    u = [f32(v) for v in u]
    clip = lambda im: np.clip(im, f32(0.0), f32(1.0))
    x = x.astype(f32)
    if u[0] < f32(cfg["p_jitter"]):
        b, c, s = float(cfg["brightness"]), float(cfg["contrast"]), float(cfg["saturation"])
        x = clip(x * _u(u[1], 1.0 - b, 1.0 + b))
        mu = f32(np.mean(_lum(x), dtype=np.float64))
        x = clip((x - mu) * _u(u[2], 1.0 - c, 1.0 + c) + mu)
        g = _lum(x)
        x = clip((x - g) * _u(u[3], 1.0 - s, 1.0 + s) + g)
    if u[4] < f32(cfg["p_gray"]):
        x = np.repeat(_lum(x), 3, axis=-1)
    sigma = _u(u[6], float(cfg["sigma"][0]), float(cfg["sigma"][1]))
    if u[5] < f32(cfg["p_blur"]):
        x = T.gaussian_blur(x, sigma, int(cfg["blur_radius"]))
    return clip(x).astype(f32)


def make_crossview_pair(image_u8: np.ndarray, res: int, draws: Dict[str, np.ndarray],
                        cfg: Optional[dict] = None) -> Dict[str, np.ndarray]:
    """Teacher view B (clean) and student view A (augmented) of one image, with their normalized boxes."""
    cfg = dict(CROSSVIEW_DEFAULTS, **(cfg or {}))
    grid = int(res) // int(cfg["patch"])
    src, ms = _cv_square(image_u8)
    by0, bx0, bsd, bbox = _cv_boxes(draws["b_area"], draws["b_y"], draws["b_x"], float(cfg["b_scale"][0]),
                                    float(cfg["b_scale"][1]), ms)
    box_b = bbox[0]
    local = f32(draws["local"]) < f32(cfg["p_local"])
    lo = f32(cfg["a_local"][0] if local else cfg["a_global"][0])
    hi = f32(cfg["a_local"][1] if local else cfg["a_global"][1])
    ay0, ax0, asd, abox = _cv_boxes(draws["a_area"], draws["a_y"], draws["a_x"], lo, hi, ms, hi_minus_lo=f32(hi - lo))
    nv = crossview_valid_count(abox, box_b, grid)
    good = nv >= int(cfg["min_valid"])
    if good.any():
        i = int(np.argmax(good))
        a_y0, a_x0, a_sd = int(ay0[i]), int(ax0[i]), int(asd[i])
    else:                                             # A = B, re-rounded in the source frame
        msf = f32(ms)
        a_sd = int(np.clip(int(np.round(f32(box_b[2] * msf))), 1, ms))
        a_y0 = min(int(np.round(f32(box_b[0] * msf))), ms - a_sd)
        a_x0 = min(int(np.round(f32(box_b[1] * msf))), ms - a_sd)
    box_a = (np.asarray([a_y0, a_x0, a_sd], f32) / f32(ms)).astype(f32)
    y, x, s = int(by0[0]), int(bx0[0]), int(bsd[0])
    x_t = T.resize_bilinear(src[y:y + s, x:x + s], res, res)
    x_s = T.resize_bilinear(src[a_y0:a_y0 + a_sd, a_x0:a_x0 + a_sd], res, res)
    if cfg["photometric"]:
        x_s = photometric(x_s, draws["photo"], cfg)
    return {"x_s_rgb": x_s, "x_t_rgb": x_t, "box_a": box_a, "box_b": box_b.astype(f32),
            "n_valid": crossview_valid_count(box_a, box_b, grid)}


def token_stats_crop(image_u8: np.ndarray, res: int, draws) -> np.ndarray:
    """The crop of the released token statistics (train_jax/compute_token_stats.py --crop alps): a square of area
    U(0.8, 1) of the central square, side int(sqrt(U) m), offsets int(U (m - side)), bilinear without antialias."""
    img, m = _cv_square(image_u8)
    side = int(f32(np.sqrt(_u(draws[0], 0.8, 1.0)) * f32(m)))
    y0 = int(f32(f32(draws[1]) * f32(m - side)))
    x0 = int(f32(f32(draws[2]) * f32(m - side)))
    return T.resize_bilinear(img[y0:y0 + side, x0:x0 + side], res, res)
