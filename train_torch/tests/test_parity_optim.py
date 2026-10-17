# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Optimizer, schedule and logit-scale projection vs the optax chains of train_jax: N updates of the real (tiny)
Stage-1 and Stage-2 parameter trees with the same random gradients, compared leaf by leaf (parameters, the bfloat16
first moment and the second moment)."""
import numpy as np
import pytest
import torch

from parity_utils import assert_flat_close, flat_np, jax, jnp, perturb

from alps_torch import convert as C
from alps_torch.models import CrossViewFusion, FeatHead, Stage1Model, ViT, project_logit_scale
from alps_torch.models.stage2 import Stage2Student
from alps_torch.optim import named_state, stage1_optimizer, stage2_optimizer, warmup_cosine

optax = pytest.importorskip("optax")
TINY = dict(width=64, depth=2, mlp_dim=128, num_heads=4, remat=False, vocab_size=120, text_width=64, text_depth=2,
            text_heads=4, text_mlp_dim=128, cap_depth=2, cap_heads=4, cap_mlp_dim=128, itc_proj_dim=64,
            dense_proj_dim=32)


@pytest.mark.parametrize("total,warm,base", [(390_981, 3_909, 1.0), (62_556, 3_127, 1.0), (4_883, 244, 1e-4), (10, 0, 1.0)])
def test_schedule(total, warm, base):
    from alps_jax.optim import warmup_cosine as jwc
    fj, ft = jwc(total, warm, base), warmup_cosine(total, warm, base)
    steps = sorted({0, 1, 2, max(warm - 1, 0), warm, warm + 1, total // 3, total // 2, total - 1, total, total + 7})
    for s in steps:
        a, b = float(fj(jnp.int32(s))), ft(s)
        assert abs(a - b) <= 2 * np.spacing(np.float32(max(abs(a), 1e-30))), (s, a, b)
    assert ft(0) == 0.0 or warm == 0


def _grads_like(flat, seed, scale=1.0, zero=()):
    rng = np.random.default_rng(seed)
    return {k: (np.zeros_like(v) if any(k.startswith(z) for z in zero)
                else (scale * rng.normal(size=v.shape)).astype(np.float32)) for k, v in flat.items()}


def _set_torch_grads(model, flat_grads, rules, none_prefixes=()):
    sd = C.flax_to_torch(flat_grads, rules)
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        p.grad = None if n.startswith(none_prefixes) else sd[n].clone()


def test_stage1_optimizer():
    from alps_jax.models.stage1 import Stage1Model as FStage1
    from alps_jax.optim import project_logit_scale as jproj, stage1_optimizer as jopt
    fm = FStage1(**TINY)
    x = jnp.zeros((2, 32, 32, 3))
    i32, i80 = jnp.zeros((2, 32), jnp.int32), jnp.zeros((2, 80), jnp.int32)
    k = jax.random.PRNGKey(0)
    params = perturb(fm.init({n: k for n in ("params", "dropout", "drop_path", "random_mask")}, x_enc=x, gx_enc=x,
                             x_rgb=x, crop_box=jnp.zeros((2, 4)), flip=jnp.zeros((2,), bool), text=i32, itc_text2=i32,
                             cap_text=i80, train=True)["params"], 1)
    params["logit_scale"] = jnp.asarray([4.55], jnp.float32)          # near the upper bound ln 100 = 4.605
    peak, total, warm = 3e-2, 12, 3
    tx, _ = jopt(peak, total, warm)
    opt_j = tx.init(params)
    rules = C.stage1_rules(2, 2, 2)
    tm = Stage1Model(**TINY)
    tm.load_state_dict(C.flax_to_torch(flat_np(params), rules), strict=True)
    opt_t = stage1_optimizer(tm, peak, total, warm)
    pj = params
    # encoder_norm gets no gradient in training (frozen in torch); the image projection gets a zero gradient here
    # (None in torch), so only weight decay moves it
    zero = ("visual_encoder/encoder_norm", "image_projection_head")
    for step in range(8):
        g = _grads_like(flat_np(pj), step, scale=0.5 + step, zero=zero)
        gtree = jax.tree_util.tree_map(jnp.asarray, C.unflatten(g))
        upd, opt_j = tx.update(gtree, opt_j, pj)
        pj = jproj(optax.apply_updates(pj, upd))
        _set_torch_grads(tm, g, rules, none_prefixes=("image_projection",))
        opt_t.step()
        project_logit_scale(tm)
        got = C.torch_to_flax(tm.state_dict(), rules)
        assert_flat_close(got, flat_np(pj), 2e-6, what=f"params after update {step + 1}", floor=0.0)
    assert float(tm.logit_scale) == pytest.approx(float(pj["logit_scale"][0]), abs=1e-7)
    # moments: the bf16 first moment and the f32 second moment, leaf by leaf
    mu = flat_np(opt_j[0].mu)
    nu = flat_np(opt_j[0].nu)
    st = {n: s for n, s in named_state(opt_t).items()}
    m_t = C.torch_to_flax({n: s["exp_avg"].float() for n, s in st.items()}, rules, strict=False)
    v_t = C.torch_to_flax({n: s["exp_avg_sq"] for n, s in st.items()}, rules, strict=False)
    for kk in m_t:
        np.testing.assert_allclose(m_t[kk], mu[kk], rtol=2 ** -7, atol=1e-30, err_msg=kk)      # bf16 resolution
        np.testing.assert_allclose(v_t[kk], nu[kk], rtol=2e-6, atol=1e-30, err_msg=kk)
    frac_equal = np.mean(np.concatenate([(m_t[kk] == mu[kk]).ravel() for kk in m_t]))
    assert frac_equal > 0.999, frac_equal                               # bf16 moments agree bit for bit


def _stage2_setup(base, total, warm):
    from alps_jax.models.predictor import CrossViewFusion as FCVF, FeatHead as FHead
    from alps_jax.models.vit import ViT as FViT
    from alps_jax.optim import stage2_optimizer as jopt
    D, g = 64, 4
    k = jax.random.PRNGKey(0)
    pe = FViT(width=D, depth=2, mlp_dim=128, num_heads=4).init(k, jnp.zeros((1, 64, 64, 3)))["params"]
    pt = FCVF(width=32, depth=2, num_heads=4, grid=g, token_dim=D, text_dim=48).init(
        k, jnp.zeros((1, g * g, 2)), jnp.zeros((1,)), jnp.zeros((1, g * g, D)), jnp.zeros((1, 32, 48)),
        jnp.ones((1, 32), bool))["params"]
    ph = FHead(D).init(k, jnp.zeros((1, g * g, D)))["params"]
    params = perturb({"encoder": pe, "transition": pt, "feat_head": ph}, 3)
    tx, _ = jopt(base, total, warm)
    rules = C.stage2_rules(depth=2, pred_depth=2)
    st = Stage2Student(ViT(D, 2, 128, 4), CrossViewFusion(32, 2, 4, g, D, 48), FeatHead(D, D))
    st.load_state_dict(C.flax_to_torch(flat_np(params), rules), strict=True)
    return params, tx, rules, st, stage2_optimizer(st, base, total, warm)


@pytest.mark.parametrize("clipped", [False, True])
def test_stage2_optimizer(clipped):
    """Encoder at 0.1 x, predictor and head at 1 x, global-norm clipping at 1. Without clipping the updates agree
    to float rounding. With clipping the global norm itself differs in the last bit between the two frameworks
    (summation order), which flips the bfloat16 rounding of a few first-moment entries (1 bf16 ulp = 2^-8 relative);
    those entries' updates then differ by up to ~0.4 %, everything else agrees to float rounding."""
    base, total, warm = 1e-2, 30, 4
    pj, tx, rules, st, opt_t = _stage2_setup(base, total, warm)
    opt_j = tx.init(pj)
    for step in range(24):
        if clipped:
            gscale = 3.0 if step % 2 else 0.01                       # norms ~1200 and ~4: both clipped
        else:
            gscale = 1e-5 * (1 + step % 3)                           # norms < 1: never clipped
        gr = _grads_like(flat_np(pj), 100 + step, scale=gscale, zero=("encoder/encoder_norm",))
        gtree = jax.tree_util.tree_map(jnp.asarray, C.unflatten(gr))
        gn = float(optax.global_norm(gtree))
        upd, opt_j = tx.update(gtree, opt_j, pj)
        pj = optax.apply_updates(pj, upd)
        _set_torch_grads(st, gr, rules)
        opt_t.step()
        assert float(opt_t.last_grad_norm) == pytest.approx(gn, rel=1e-6)
        got, ref = C.torch_to_flax(st.state_dict(), rules), flat_np(pj)
        if not clipped:
            assert_flat_close(got, ref, 2e-6, what=f"stage2 params after update {step + 1}", floor=0.0)
        else:
            assert_flat_close(got, ref, 1e-3, what=f"stage2 params after update {step + 1}", floor=0.0)
            close = np.concatenate([(np.abs(got[k] - ref[k]) <= 2e-6 * np.abs(ref[k]).max()).ravel() for k in ref])
            assert close.mean() > 0.999, close.mean()


def test_projection_and_first_update():
    m = Stage1Model(**TINY)
    with torch.no_grad():
        m.logit_scale.fill_(9.0)
    project_logit_scale(m)
    assert float(m.logit_scale) == pytest.approx(float(np.float32(np.log(100.0))))
    assert float(m.logit_scale.exp()) == pytest.approx(100.0, rel=1e-6)
    with torch.no_grad():
        m.logit_scale.fill_(-1.0)
    project_logit_scale(m)
    assert float(m.logit_scale) == 0.0
    # update 1 has learning rate 0: parameters unchanged, moments updated
    opt = stage1_optimizer(m, 1e-3, 100, 10)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    for p in m.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p)
    opt.step()
    assert all(torch.equal(before[n], p) for n, p in m.named_parameters())
    assert opt.count == 1 and all(s["exp_avg"].dtype == torch.bfloat16 for s in opt.state.values())


def test_decay_mask_matches_flax_kernel_mask():
    """The torch decay groups are exactly the leaves the JAX code decays (names ending in /kernel)."""
    from alps_jax.models.stage1 import Stage1Model as FStage1
    fm = FStage1(**TINY)
    x = jnp.zeros((1, 32, 32, 3))
    i32, i80 = jnp.zeros((1, 32), jnp.int32), jnp.zeros((1, 80), jnp.int32)
    k = jax.random.PRNGKey(0)
    flat = flat_np(fm.init({n: k for n in ("params", "dropout", "drop_path", "random_mask")}, x_enc=x, gx_enc=x, x_rgb=x,
                           crop_box=jnp.zeros((1, 4)), flip=jnp.zeros((1,), bool), text=i32, itc_text2=i32,
                           cap_text=i80, train=True)["params"])
    rules = C.stage1_rules(2, 2, 2)
    tm = Stage1Model(**TINY)
    opt = stage1_optimizer(tm, 1e-3, 10, 1)
    decayed = {n for g in opt.param_groups if g["weight_decay"] for n in g["names"]}
    flax_kernels = {k for k in flat if k.endswith("/kernel")}
    mapped = set()
    for src, dst, kind in rules:
        srcs = (src,) if isinstance(src, str) else src
        if dst in decayed:
            mapped.update(srcs)
        else:
            assert not any(s.endswith("/kernel") for s in srcs), (dst, srcs)
    assert mapped == flax_kernels
