# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Image transforms of both stages (TensorFlow), as in the original training code.

Stage 1: an Inception-style crop (area 40-100%, aspect 3/4-4/3) resized to the training resolution with antialiased
bilinear interpolation and TRUNCATED to uint8; then the base view x (the crop itself, in [0, 1]) and a geometric view
g(x): a square crop of area U(0.5, 1) of the base view, bilinear resize back to the resolution, flipped with
probability 0.5, with its exact normalized box.

Stage 2: the cross-view pair of a single image (see `make_crossview_pair`).
"""
import tensorflow as tf


def inception_crop(image, size, area_min=40, area_max=100, method="bilinear", antialias=True):
    begin, crop_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), tf.zeros([0, 0, 4], tf.float32), area_range=(area_min / 100, area_max / 100),
        min_object_covered=0, use_image_if_no_bounding_boxes=True)
    crop = tf.slice(image, begin, crop_size)
    crop.set_shape([None, None, image.shape[-1]])
    dtype = crop.dtype
    crop = tf.image.resize(crop, (size, size), method=method, antialias=antialias)
    return tf.cast(crop, dtype)                     # uint8: truncation toward zero, as in the original pipeline


def make_two_view(image_uint8, res, dense_scale=(0.5, 1.0), seed=(0, 0)):
    s0, s1, s2, s3 = tf.unstack(tf.random.experimental.stateless_split(seed, 4))
    img = tf.image.convert_image_dtype(image_uint8, tf.float32)
    h = tf.shape(img)[0]; w = tf.shape(img)[1]; s = tf.minimum(h, w)
    img = tf.image.resize_with_crop_or_pad(img, s, s)
    x_rgb = tf.image.resize(img, (res, res))
    sf = tf.cast(s, tf.float32)
    side = tf.sqrt(tf.random.stateless_uniform([], s0, dense_scale[0], dense_scale[1]))
    side_px = tf.cast(side * sf, tf.int32)
    y0 = tf.cast(tf.random.stateless_uniform([], s1, 0., 1.) * tf.cast(s - side_px, tf.float32), tf.int32)
    x0 = tf.cast(tf.random.stateless_uniform([], s2, 0., 1.) * tf.cast(s - side_px, tf.float32), tf.int32)
    gx_rgb = tf.image.resize(tf.image.crop_to_bounding_box(img, y0, x0, side_px, side_px), (res, res))
    flip = tf.random.stateless_uniform([], s3) < 0.5
    gx_rgb = tf.cond(flip, lambda: tf.image.flip_left_right(gx_rgb), lambda: gx_rgb)
    crop_box = tf.stack([tf.cast(y0, tf.float32) / sf, tf.cast(x0, tf.float32) / sf,
                         tf.cast(side_px, tf.float32) / sf, tf.cast(side_px, tf.float32) / sf])
    return {"x_rgb": x_rgb, "gx_rgb": gx_rgb, "crop_box": crop_box, "flip": flip}


# ---------------------------------------------------------------------------------------------------------------------
# Stage 2
CROSSVIEW_DEFAULTS = dict(
    b_scale=(0.4, 1.0),          # teacher crop B: area fraction of the central square
    a_local=(0.1, 0.4),          # student crop A when local
    a_global=(0.4, 1.0),         # student crop A when global
    p_local=0.5,                 # probability that A is local
    min_valid=16,                # at least this many of B's patch centres inside A (else the next candidate)
    tries=10,                    # A candidates per example; if none qualifies, A = B
    patch=16,
    photometric=True,            # on the student view only
    p_jitter=0.8, brightness=0.4, contrast=0.4, saturation=0.2,
    p_gray=0.1, p_blur=0.5, sigma=(0.1, 2.0), blur_radius=7,
)


def _cv_square(image_uint8):
    img = tf.image.convert_image_dtype(image_uint8, tf.float32)
    h = tf.shape(img)[0]; w = tf.shape(img)[1]; m = tf.minimum(h, w)
    return tf.image.resize_with_crop_or_pad(img, m, m), m


def _cv_boxes(key, lo, hi, m, n):
    k0, k1, k2 = tf.unstack(tf.random.experimental.stateless_split(key, 3))
    mf = tf.cast(m, tf.float32)
    area = lo + tf.random.stateless_uniform([n], k0) * (hi - lo)
    side = tf.clip_by_value(tf.cast(tf.round(tf.sqrt(area) * mf), tf.int32), 1, m)
    room = tf.cast(m - side, tf.float32)
    y0 = tf.cast(tf.floor(tf.random.stateless_uniform([n], k1) * (room + 1.0)), tf.int32)
    x0 = tf.cast(tf.floor(tf.random.stateless_uniform([n], k2) * (room + 1.0)), tf.int32)
    y0 = tf.minimum(y0, m - side); x0 = tf.minimum(x0, m - side)
    box = tf.stack([tf.cast(y0, tf.float32), tf.cast(x0, tf.float32), tf.cast(side, tf.float32)], axis=-1) / mf
    return y0, x0, side, box


def crossview_valid_count(box_a, box_b, grid):
    c = (tf.range(grid, dtype=tf.float32) + 0.5) / float(grid)
    yc = box_b[0] + c * box_b[2]; xc = box_b[1] + c * box_b[2]
    ay0 = box_a[..., 0:1]; ax0 = box_a[..., 1:2]; asd = box_a[..., 2:3]
    v = (yc - ay0) / asd; u = (xc - ax0) / asd
    rows = tf.reduce_sum(tf.cast((v >= 0.) & (v < 1.), tf.int32), axis=-1)
    cols = tf.reduce_sum(tf.cast((u >= 0.) & (u < 1.), tf.int32), axis=-1)
    return rows * cols


def _gaussian_blur(x, sigma, radius):
    t = tf.range(-radius, radius + 1, dtype=tf.float32)
    k = tf.exp(-0.5 * tf.square(t / sigma)); k = k / tf.reduce_sum(k)
    img = tf.pad(x[None], [[0, 0], [radius, radius], [radius, radius], [0, 0]], mode="REFLECT")
    kh = tf.tile(tf.reshape(k, [1, -1, 1, 1]), [1, 1, 3, 1])
    kv = tf.tile(tf.reshape(k, [-1, 1, 1, 1]), [1, 1, 3, 1])
    img = tf.nn.depthwise_conv2d(img, kh, [1, 1, 1, 1], "VALID")
    img = tf.nn.depthwise_conv2d(img, kv, [1, 1, 1, 1], "VALID")
    return img[0]


def crossview_photometric(x, key, cfg):
    """Colour jitter without hue (p 0.8; brightness, contrast, saturation in this order), grayscale (p 0.1), Gaussian
    blur (p 0.5, sigma U(0.1, 2)), on an [H, W, 3] image in [0, 1]."""
    k = tf.unstack(tf.random.experimental.stateless_split(key, 8))
    U = lambda kk, a=0., b=1.: tf.random.stateless_uniform([], kk, a, b)
    lum = lambda im: tf.reduce_sum(im * tf.constant([0.299, 0.587, 0.114], tf.float32), axis=-1, keepdims=True)
    clip = lambda im: tf.clip_by_value(im, 0., 1.)

    def jitter(im):
        im = clip(im * U(k[1], 1. - cfg["brightness"], 1. + cfg["brightness"]))
        mu = tf.reduce_mean(lum(im))
        im = clip((im - mu) * U(k[2], 1. - cfg["contrast"], 1. + cfg["contrast"]) + mu)
        g = lum(im)
        return clip((im - g) * U(k[3], 1. - cfg["saturation"], 1. + cfg["saturation"]) + g)

    x = tf.cond(U(k[0]) < cfg["p_jitter"], lambda: jitter(x), lambda: x)
    x = tf.cond(U(k[4]) < cfg["p_gray"], lambda: tf.tile(lum(x), [1, 1, 3]), lambda: x)
    sig = U(k[6], float(cfg["sigma"][0]), float(cfg["sigma"][1]))
    x = tf.cond(U(k[5]) < cfg["p_blur"], lambda: _gaussian_blur(x, sig, int(cfg["blur_radius"])), lambda: x)
    return clip(x)


def make_crossview_pair(image_uint8, res, seed=(0, 0), cfg=None, method="bilinear", antialias=False):
    """Teacher view B (clean) and student view A (augmented) of one image, with their normalized boxes.

    B: square crop of the image's central square, area U(b_scale). A: local (U(a_local)) with probability p_local,
    else global (U(a_global)); `tries` candidates of that kind, the first with >= min_valid of B's patch centres inside
    it is kept (else A = B). Both resized to `res` (bilinear, no antialias); photometric augmentation on A only.
    """
    cfg = dict(CROSSVIEW_DEFAULTS, **(cfg or {}))
    grid = int(res) // int(cfg["patch"])
    kb, ka, kp, kl = tf.unstack(tf.random.experimental.stateless_split(tf.convert_to_tensor(seed, tf.int32), 4))
    src, ms = _cv_square(image_uint8)
    tgt, mt = src, ms
    by0, bx0, bsd, bbox = _cv_boxes(kb, float(cfg["b_scale"][0]), float(cfg["b_scale"][1]), mt, 1)
    box_b = bbox[0]
    n = int(cfg["tries"])
    local = tf.random.stateless_uniform([], kl) < float(cfg["p_local"])
    lo = tf.where(local, float(cfg["a_local"][0]), float(cfg["a_global"][0]))
    hi = tf.where(local, float(cfg["a_local"][1]), float(cfg["a_global"][1]))
    ay0, ax0, asd, abox = _cv_boxes(ka, lo, hi, ms, n)
    nv = crossview_valid_count(abox, box_b, grid)
    good = nv >= int(cfg["min_valid"])
    first = tf.argmax(tf.cast(good, tf.int32), output_type=tf.int32)
    any_good = tf.reduce_any(good)
    msf = tf.cast(ms, tf.float32)
    f_sd = tf.clip_by_value(tf.cast(tf.round(box_b[2] * msf), tf.int32), 1, ms)
    f_y0 = tf.minimum(tf.cast(tf.round(box_b[0] * msf), tf.int32), ms - f_sd)
    f_x0 = tf.minimum(tf.cast(tf.round(box_b[1] * msf), tf.int32), ms - f_sd)
    pick = lambda cand, fb: tf.where(any_good, tf.gather(cand, first), fb)
    a_y0, a_x0, a_sd = pick(ay0, f_y0), pick(ax0, f_x0), pick(asd, f_sd)
    box_a = tf.stack([tf.cast(a_y0, tf.float32), tf.cast(a_x0, tf.float32), tf.cast(a_sd, tf.float32)]) / msf

    def cut(img, y0, x0, sd):
        return tf.image.resize(tf.image.crop_to_bounding_box(img, y0, x0, sd, sd), (res, res), method=method,
                               antialias=antialias)
    x_t = cut(tgt, by0[0], bx0[0], bsd[0])
    x_s = cut(src, a_y0, a_x0, a_sd)
    if cfg["photometric"]:
        x_s = crossview_photometric(x_s, kp, cfg)
    return {"x_s_rgb": x_s, "x_t_rgb": x_t, "box_a": box_a, "box_b": box_b,
            "n_valid": crossview_valid_count(box_a, box_b, grid)}
