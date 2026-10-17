# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch modules vs the Flax modules of train_jax (float32, CPU): forward outputs and gradients (parameters and
inputs) on perturbed random weights, through the conversion tables of alps_torch.convert."""
import numpy as np
import pytest
import torch

from parity_utils import (assert_close, assert_flat_close, flat_np, jax, jnp, nhwc_to_nchw, perturb, to_torch,
                          torch_grads)

from alps_torch import convert as C
from alps_torch.models import (CaptionDecoder, CrossViewFusion, DenseProjector, FeatHead, RGBDecoder, TextEncoder,
                               ViT, crossview_geometry)

f32 = jnp.float32
FWD, GRAD = 1e-5, 1e-5


def _grad_check(module, rules, gj, tol=GRAD, what=""):
    gt = C.torch_to_flax(torch_grads(module), rules)
    return assert_flat_close(gt, flat_np(gj), tol, what=what, grads=True)


@pytest.mark.parametrize("train", [False, True])
def test_vit(train):
    from alps_jax.models.vit import ViT as FViT
    rng = np.random.default_rng(0)
    fm = FViT(width=64, depth=2, mlp_dim=128, num_heads=4, dtype=f32)
    x = rng.normal(size=(2, 64, 96, 3)).astype(np.float32)                 # 4 x 6 grid: row-major order matters
    params = perturb(fm.init(jax.random.PRNGKey(0), jnp.asarray(x))["params"], 1)
    R, Rg = rng.normal(size=(2, 24, 64)).astype(np.float32), rng.normal(size=(2, 64)).astype(np.float32)

    def f(p, xx):
        pooled, tok = fm.apply({"params": p}, xx, train=train)
        return jnp.sum(tok * R) + jnp.sum(pooled * Rg), (pooled, tok)
    (lj, (pj, tj)), (gj, gxj) = jax.value_and_grad(f, argnums=(0, 1), has_aux=True)(params, jnp.asarray(x))

    rules = C.vit_rules(depth=2)
    tm = ViT(width=64, depth=2, mlp_dim=128, num_heads=4)
    tm.load_state_dict(C.flax_to_torch(flat_np(params), rules), strict=True)
    tm.norm.requires_grad_(True)
    xt = nhwc_to_nchw(x).requires_grad_(True)
    out = tm(xt, train=train, pooled=True)
    loss = (out["tokens"] * to_torch(R)).sum() + (out["pooled"] * to_torch(Rg)).sum()
    loss.backward()
    assert_close(out["tokens"], np.asarray(tj), FWD, "tokens")
    assert_close(out["pooled"], np.asarray(pj), FWD, "pooled")
    _grad_check(tm, rules, gj, what="vit param grads")
    assert_close(xt.grad.permute(0, 2, 3, 1), np.asarray(gxj), GRAD, "vit input grad")


@pytest.mark.parametrize("train", [False, True])
def test_text_encoder(train):
    from alps_jax.models.text import TextEncoder as FText
    rng = np.random.default_rng(1)
    fm = FText(vocab_size=120, width=64, depth=2, num_heads=4, mlp_dim=128, max_len=32, dtype=f32)
    ids = rng.integers(0, 120, size=(3, 32)).astype(np.int32)
    params = perturb(fm.init(jax.random.PRNGKey(0), jnp.asarray(ids))["params"], 2)
    R = rng.normal(size=(3, 64)).astype(np.float32)
    f = lambda p: (lambda o: (jnp.sum(o * R), o))(fm.apply({"params": p}, jnp.asarray(ids), train=train))
    (_, oj), gj = jax.value_and_grad(f, has_aux=True)(params)
    rules = C.text_rules(depth=2)
    tm = TextEncoder(vocab_size=120, width=64, depth=2, num_heads=4, mlp_dim=128, max_len=32)
    tm.load_state_dict(C.flax_to_torch(flat_np(params), rules), strict=True)
    o = tm(torch.from_numpy(ids).long(), train=train)
    (o * to_torch(R)).sum().backward()
    assert_close(o, np.asarray(oj), FWD, "text pooled")
    _grad_check(tm, rules, gj, what="text grads")


def test_caption_decoder():
    from alps_jax.models.text import CaptionDecoder as FCap
    rng = np.random.default_rng(2)
    fm = FCap(vocab_size=120, width=64, depth=2, num_heads=4, mlp_dim=128, max_len=80, dtype=f32)
    z = rng.normal(size=(2, 16, 64)).astype(np.float32)
    ids = rng.integers(0, 120, size=(2, 80)).astype(np.int32)
    params = perturb(fm.init(jax.random.PRNGKey(0), jnp.asarray(z), jnp.asarray(ids))["params"], 3)
    R = rng.normal(size=(2, 80, 120)).astype(np.float32)

    def f(p, zz):
        o = fm.apply({"params": p}, zz, jnp.asarray(ids), train=True)
        return jnp.sum(o * R), o
    (_, oj), (gj, gzj) = jax.value_and_grad(f, argnums=(0, 1), has_aux=True)(params, jnp.asarray(z))
    rules = C.caption_decoder_rules(depth=2)
    tm = CaptionDecoder(vocab_size=120, width=64, depth=2, num_heads=4, mlp_dim=128, max_len=80)
    tm.load_state_dict(C.flax_to_torch(flat_np(params), rules), strict=True)
    zt = to_torch(z).requires_grad_(True)
    o = tm(zt, torch.from_numpy(ids).long(), train=True)
    (o * to_torch(R)).sum().backward()
    assert_close(o, np.asarray(oj), FWD, "caption logits")
    _grad_check(tm, rules, gj, what="caption grads")
    assert_close(zt.grad, np.asarray(gzj), GRAD, "caption image-token grad")


def test_caption_mask_is_prefix_lm():
    """Changing a later caption token must not change earlier logits; the image prefix never sees the text."""
    tm = CaptionDecoder(vocab_size=50, width=32, depth=2, num_heads=4, mlp_dim=64, max_len=12).eval()
    z = torch.randn(1, 5, 32)
    a = torch.randint(0, 50, (1, 12))
    b = a.clone(); b[0, 7] = (a[0, 7] + 1) % 50
    with torch.no_grad():
        la, lb = tm(z, a), tm(z, b)
    assert torch.equal(la[:, :7], lb[:, :7]) and not torch.equal(la[:, 7:], lb[:, 7:])


def test_rgb_decoder():
    from alps_jax.models.pixel_decoder import RGBDecoder as FDec
    rng = np.random.default_rng(3)
    fm = FDec(dtype=f32)
    z = rng.normal(size=(2, 3, 4, 64)).astype(np.float32)                 # a 3 x 4 grid
    params = perturb(fm.init(jax.random.PRNGKey(0), jnp.asarray(z))["params"], 4)
    R = rng.normal(size=(2, 48, 64, 3)).astype(np.float32)

    def f(p, zz):
        o = fm.apply({"params": p}, zz)
        return jnp.sum(o * R), o
    (_, oj), (gj, gzj) = jax.value_and_grad(f, argnums=(0, 1), has_aux=True)(params, jnp.asarray(z))
    rules = C.rgb_decoder_rules()
    tm = RGBDecoder(64)
    tm.load_state_dict(C.flax_to_torch(flat_np(params), rules), strict=True)
    zt = to_torch(z.reshape(2, 12, 64)).requires_grad_(True)
    o = tm(zt, (3, 4))
    (o * nhwc_to_nchw(R)).sum().backward()
    assert_close(o.permute(0, 2, 3, 1), np.asarray(oj), FWD, "rgb")
    _grad_check(tm, rules, gj, what="rgb grads")
    assert_close(zt.grad.reshape(2, 3, 4, 64), np.asarray(gzj), GRAD, "rgb input grad")


def test_dense_projector():
    from alps_jax.models.dense import DenseProjector as FProj
    rng = np.random.default_rng(4)
    fm = FProj(dim=32, dtype=f32)
    z = (3.0 + rng.normal(size=(2, 16, 64))).astype(np.float32)
    params = perturb(fm.init(jax.random.PRNGKey(0), jnp.asarray(z))["params"], 5)
    R = rng.normal(size=(2, 16, 32)).astype(np.float32)
    f = lambda p: (lambda o: (jnp.sum(o * R), o))(fm.apply({"params": p}, jnp.asarray(z)))
    (_, oj), gj = jax.value_and_grad(f, has_aux=True)(params)
    rules = C._lin("Dense_0", "proj")
    tm = DenseProjector(64, 32)
    tm.load_state_dict(C.flax_to_torch(flat_np(params), rules), strict=True)
    o = tm(to_torch(z))
    (o * to_torch(R)).sum().backward()
    assert_close(o, np.asarray(oj), FWD, "dense projector")
    _grad_check(tm, rules, gj, what="dense projector grads")


def test_predictor_and_head():
    from alps_jax.models.predictor import CrossViewFusion as FCVF, FeatHead as FHead, crossview_geometry as fgeo
    rng = np.random.default_rng(5)
    b, g, D, T = 3, 4, 64, 48
    box_a = np.asarray([[0.1, 0.2, 0.5], [0., 0., 1.], [0.3, 0.1, 0.4]], np.float32)
    box_b = np.asarray([[0., 0., 0.8], [0., 0., 1.], [0.2, 0.2, 0.7]], np.float32)
    q_pos, valid, log_r = fgeo(jnp.asarray(box_a), jnp.asarray(box_b), g)
    qt, vt, lt = crossview_geometry(to_torch(box_a), to_torch(box_b), g)
    assert_close(qt, np.asarray(q_pos), 1e-7, "q_pos")
    assert np.array_equal(vt.numpy(), np.asarray(valid)) and valid.sum() > 0
    assert_close(lt, np.asarray(log_r), 1e-7, "log_r")
    za = rng.normal(size=(b, g * g, D)).astype(np.float32)
    u = rng.normal(size=(b, 32, T)).astype(np.float32)
    m = np.zeros((b, 32), bool); m[:, 0] = True
    fp = FCVF(width=32, depth=2, num_heads=4, grid=g, token_dim=D, text_dim=T, dtype=f32)
    fh = FHead(D, dtype=f32)
    pp = perturb(fp.init(jax.random.PRNGKey(0), q_pos, log_r, jnp.asarray(za), jnp.asarray(u), jnp.asarray(m))["params"], 6)
    ph = perturb(fh.init(jax.random.PRNGKey(1), jnp.asarray(za))["params"], 7)
    R = rng.normal(size=(b, g * g, D)).astype(np.float32)

    def f(p, zz):
        o = fh.apply({"params": p["h"]}, fp.apply({"params": p["p"]}, q_pos, log_r, zz, jnp.asarray(u), jnp.asarray(m)))
        return jnp.sum(o * R), o
    (_, oj), (gj, gzj) = jax.value_and_grad(f, argnums=(0, 1), has_aux=True)({"p": pp, "h": ph}, jnp.asarray(za))
    rp, rh = C.predictor_rules(depth=2), C.feat_head_rules()
    tp = CrossViewFusion(width=32, depth=2, num_heads=4, grid=g, token_dim=D, text_dim=T)
    th = FeatHead(D, D)
    tp.load_state_dict(C.flax_to_torch(flat_np(pp), rp), strict=True)
    th.load_state_dict(C.flax_to_torch(flat_np(ph), rh), strict=True)
    zt = to_torch(za).requires_grad_(True)
    o = th(tp(qt, lt, zt, to_torch(u), torch.from_numpy(m)))
    (o * to_torch(R)).sum().backward()
    assert_close(o, np.asarray(oj), FWD, "predictor output")
    _grad_check(tp, rp, gj["p"], what="predictor grads")
    _grad_check(th, rh, gj["h"], what="head grads")
    assert_close(zt.grad, np.asarray(gzj), GRAD, "predictor input grad")
    # masked text slots contribute nothing: changing their values leaves the output unchanged
    u2 = u.copy(); u2[:, 1:] = 100.0 * rng.normal(size=u2[:, 1:].shape)
    with torch.no_grad():
        o2 = th(tp(qt, lt, zt, to_torch(u2), torch.from_numpy(m)))
    assert torch.equal(o.detach(), o2)


def test_conversion_matches_inference_package():
    """The ViT / text tables give the same state dicts as alps.convert (the inference package's converter)."""
    from alps.convert import text_to_torch, vit_to_torch
    from alps_jax.models.text import TextEncoder as FText
    from alps_jax.models.vit import ViT as FViT
    pv = flat_np(FViT(width=64, depth=2, mlp_dim=128, num_heads=4).init(jax.random.PRNGKey(0), jnp.zeros((1, 32, 32, 3)))["params"])
    pt = flat_np(FText(vocab_size=50, width=64, depth=2, num_heads=4, mlp_dim=128).init(
        jax.random.PRNGKey(0), jnp.zeros((1, 32), jnp.int32))["params"])
    for mine, ref in ((C.flax_to_torch(pv, C.vit_rules(depth=2)), vit_to_torch(pv, dst="")),
                      (C.flax_to_torch(pt, C.text_rules(depth=2)), text_to_torch(pt, dst=""))):
        assert mine.keys() == ref.keys()
        for k in ref:
            assert torch.equal(mine[k], ref[k]), k
    # and the round trip torch -> Flax is the identity
    back = C.torch_to_flax(C.flax_to_torch(pv, C.vit_rules(depth=2)), C.vit_rules(depth=2), flax_shapes={k: v.shape for k, v in pv.items()})
    for k in pv:
        assert np.array_equal(back[k], pv[k]), k
