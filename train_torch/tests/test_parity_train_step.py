# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Complete training steps vs train_jax (float32, CPU): N updates of Stage 1 (model + loss + optimizer + logit-scale
projection, with the JAX per-step random draws) and of Stage 2 (2 micro-batches accumulated, clipping, the two
learning-rate groups), compared on the losses of every step and the parameters after the last one.

Adam divides every entry by its own gradient scale, so entries whose gradient is a near-total cancellation (|g| far
below the typical size) turn last-bit differences into visible update differences, and the bfloat16 first moment
occasionally rounds to the neighbouring value; the parameters are therefore compared in units of the step size
(median entry within 2e-4 of a step, 99 % within 1e-2, none beyond half a step: float32 gradients of two frameworks
agree to ~1e-6 of each leaf's largest entry, i.e. to ~1e-5 relative for a median entry, which Adam passes on to the
update), the losses to float rounding."""
import numpy as np
import pytest
import torch

from parity_utils import flat_np, jax, jnp, nhwc_to_nchw, perturb, to_torch

from alps_torch import convert as C
from alps_torch.config import Config
from alps_torch.distributed import Dist
from alps_torch.engine import stage1_train_step, stage2_train_step
from alps_torch.models import CrossViewFusion, FeatHead, Stage1Model, ViT
from alps_torch.models.stage2 import Stage2Student
from alps_torch.optim import stage1_optimizer, stage2_optimizer

optax = pytest.importorskip("optax")
from test_parity_losses import TINY, _random_mask_key, ids_batch  # noqa: E402


def _step_units(got, ref, lr, skip=()):
    d = np.concatenate([np.abs(got[k] - ref[k]).ravel() / lr for k in ref if not any(s in k for s in skip)])
    return float(np.median(d)), float(np.quantile(d, 0.99)), float(d.max())


def test_stage1_steps():
    from alps_jax.losses.stage1 import stage1_loss
    from alps_jax.losses.stage2 import normalize_image
    from alps_jax.models.stage1 import Stage1Model as FStage1
    from alps_jax.optim import project_logit_scale as jproj, stage1_optimizer as jopt
    n_steps, peak, total, warm = 4, 3e-3, 10, 1
    cfg = Config(seed=0, cap_w=1.0, rec_w=1.0, dense_w=0.25, dense_n_cross=8, dense_radius=1, dense_n_same=5,
                 dense_split=(0.75, 0.25), peak_lr=peak)
    rng = np.random.default_rng(0)
    b, res = 4, 64
    g = res // 16
    batches = [dict(x=rng.random((b, res, res, 3)).astype(np.float32), gx=rng.random((b, res, res, 3)).astype(np.float32),
                    box=np.asarray([[0.1, 0.2, 0.7, 0.7], [0., 0., 0.8, 0.8], [0.05, 0.3, 0.6, 0.6], [0., 0., 1., 1.]], np.float32),
                    flip=rng.random(b) < 0.5, t1=ids_batch(rng, b, 32, 120), t2=ids_batch(rng, b, 32, 120),
                    cap=ids_batch(rng, b, 80, 120)) for _ in range(n_steps)]
    fm = FStage1(**TINY, dtype=jnp.float32)
    k = jax.random.PRNGKey(0)
    a0 = dict(x_enc=jnp.zeros((b, res, res, 3)), gx_enc=jnp.zeros((b, res, res, 3)), x_rgb=jnp.zeros((b, res, res, 3)),
              crop_box=jnp.zeros((b, 4)), flip=jnp.zeros((b,), bool), text=jnp.zeros((b, 32), jnp.int32),
              itc_text2=jnp.zeros((b, 32), jnp.int32), cap_text=jnp.zeros((b, 80), jnp.int32))
    params = perturb(fm.init({n: k for n in ("params", "dropout", "drop_path", "random_mask")}, **a0, train=True)["params"], 1)
    tx, _ = jopt(peak, total, warm)
    state = tx.init(params)
    rng_loop = jax.random.PRNGKey(42)
    rules = C.stage1_rules(2, 2, 2)
    tm = Stage1Model(**TINY)
    tm.load_state_dict(C.flax_to_torch(flat_np(params), rules), strict=True)
    opt = stage1_optimizer(tm, peak, total, warm)
    losses_j, losses_t = [], []
    for step, bt in enumerate(batches):
        r = jax.random.fold_in(rng_loop, step)
        a = dict(x_enc=normalize_image(bt["x"]), gx_enc=normalize_image(bt["gx"]), x_rgb=jnp.asarray(bt["x"]),
                 crop_box=jnp.asarray(bt["box"]), flip=jnp.asarray(bt["flip"]), text=jnp.asarray(bt["t1"]),
                 itc_text2=jnp.asarray(bt["t2"]), cap_text=jnp.asarray(bt["cap"]))

        def loss_fn(p):
            out = fm.apply({"params": p}, **a, train=True, rngs={"dropout": r, "drop_path": r, "random_mask": r})
            return stage1_loss(out, r, cap_w=1.0, rec_w=1.0, dense_w=0.25, dense_n_cross=8, dense_radius=1,
                               dense_n_same=5)
        (lj, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        upd, state = tx.update(grads, state, params)
        params = jproj(optax.apply_updates(params, upd))
        losses_j.append(float(lj))
        draws = {"noise": to_torch(np.asarray(jax.random.normal(_random_mask_key(r), (b, g, g, 64))).reshape(b, g * g, 64)),
                 "gumbel_u": to_torch(np.asarray(jax.random.uniform(jax.random.fold_in(jax.random.fold_in(r, 1), 101),
                                                                    (b, g * g, g * g), minval=1e-9, maxval=1.0))),
                 "cross_idx": to_torch(np.asarray(jax.random.randint(jax.random.fold_in(r, 1), (8,), 0, b * g * g))).long()}
        batch = {"x_rgb": nhwc_to_nchw(bt["x"]), "gx_rgb": nhwc_to_nchw(bt["gx"]), "crop_box": to_torch(bt["box"]),
                 "flip": torch.from_numpy(bt["flip"]), "itc_labels1": torch.from_numpy(bt["t1"]).long(),
                 "itc_labels2": torch.from_numpy(bt["t2"]).long(), "labels": torch.from_numpy(bt["cap"]).long()}
        m = stage1_train_step(tm, opt, batch, cfg, step, Dist(), draws)
        losses_t.append(m["loss"])
    np.testing.assert_allclose(losses_t, losses_j, rtol=1e-5)
    got = C.torch_to_flax(tm.state_dict(), rules)
    med, q99, dmax = _step_units(got, flat_np(params), peak, skip=("key/bias",))
    print(f"stage 1, {n_steps} updates: losses {losses_t}; parameter difference / lr: median {med:.1e}, "
          f"99% {q99:.1e}, max {dmax:.1e}")
    assert med < 2e-4 and q99 < 1e-2 and dmax < 0.5


def test_stage2_steps():
    from alps_jax.losses.stage2 import crossview_loss as jloss
    from alps_jax.models.predictor import CrossViewFusion as FCVF, FeatHead as FHead
    from alps_jax.models.vit import ViT as FViT
    from alps_jax.optim import stage2_optimizer as jopt
    D, g, res, b, T, n_steps, accum = 64, 4, 64, 4, 48, 4, 2
    base, total, warm = 1e-3, 10, 1
    rng = np.random.default_rng(1)
    enc, fus, head = FViT(width=D, depth=2, mlp_dim=128, num_heads=4), FCVF(width=32, depth=2, num_heads=4, grid=g, token_dim=D, text_dim=T), FHead(D)
    k = jax.random.PRNGKey(5)
    pe = enc.init({"params": k}, jnp.zeros((1, res, res, 3)))["params"]
    pt = fus.init({"params": k}, jnp.zeros((1, g * g, 2)), jnp.zeros((1,)), jnp.zeros((1, g * g, D)),
                  jnp.zeros((1, 32, T)), jnp.ones((1, 32), bool))["params"]
    ph = head.init({"params": k}, jnp.zeros((1, g * g, D)))["params"]
    teacher = perturb(pe, 4)
    params = {"encoder": perturb(pe, 1), "transition": perturb(pt, 2), "feat_head": perturb(ph, 3)}
    u0 = rng.normal(size=T).astype(np.float32)
    zs = (rng.normal(size=(1, 1, D)).astype(np.float32), (rng.random((1, 1, D)) + 0.5).astype(np.float32))

    def mk_batch():
        sa, sb = rng.random((b, 1)) * 0.5 + 0.3, rng.random((b, 1)) * 0.5 + 0.4
        return {"x_s_rgb": rng.random((b, res, res, 3)).astype(np.float32), "x_t_rgb": rng.random((b, res, res, 3)).astype(np.float32),
                "box_a": np.concatenate([rng.random((b, 2)) * (1 - sa), sa], 1).astype(np.float32),
                "box_b": np.concatenate([rng.random((b, 2)) * (1 - sb), sb], 1).astype(np.float32)}
    data = [[mk_batch() for _ in range(accum)] for _ in range(n_steps)]
    tx, _ = jopt(base, total, warm, clip_norm=0.5)
    state = tx.init(params)
    fn = lambda p, bt: jloss(p, teacher, {kk: jnp.asarray(v) for kk, v in bt.items()}, encoder=enc, predictor=fus,
                             head=head, zstats=tuple(map(jnp.asarray, zs)), u0=jnp.asarray(u0))
    grad_fn = jax.value_and_grad(fn, has_aux=True)
    rules = C.stage2_rules(depth=2, pred_depth=2)
    student = Stage2Student(ViT(D, 2, 128, 4), CrossViewFusion(32, 2, 4, g, D, T), FeatHead(D, D))
    student.load_state_dict(C.flax_to_torch(flat_np(params), rules), strict=True)
    tea = ViT(D, 2, 128, 4).requires_grad_(False)
    tea.load_state_dict(C.flax_to_torch(flat_np(teacher), C.vit_rules(depth=2)), strict=True)
    opt = stage2_optimizer(student, base, total, warm, clip_norm=0.5)
    lj_all, lt_all = [], []
    for step in range(n_steps):
        gsum, ls = None, []
        for bt in data[step]:
            (l, _), gr = grad_fn(params, bt)
            gsum = gr if gsum is None else jax.tree.map(jnp.add, gsum, gr)
            ls.append(float(l))
        grads = jax.tree.map(lambda x: x / accum, gsum)
        upd, state = tx.update(grads, state, params)
        params = optax.apply_updates(params, upd)
        lj_all.append(np.mean(ls))
        tb = [{"x_s_rgb": nhwc_to_nchw(bt["x_s_rgb"]), "x_t_rgb": nhwc_to_nchw(bt["x_t_rgb"]),
               "box_a": to_torch(bt["box_a"]), "box_b": to_torch(bt["box_b"])} for bt in data[step]]
        m = stage2_train_step(student, tea, opt, tb, Config(), Dist(), zstats=tuple(map(to_torch, zs)), u0=to_torch(u0))
        lt_all.append(m["loss"])
    np.testing.assert_allclose(lt_all, lj_all, rtol=1e-5)
    got = C.torch_to_flax(student.state_dict(), rules)
    med, q99, dmax = _step_units(got, flat_np(params), base, skip=("key/bias", "_k/bias"))
    print(f"stage 2, {n_steps} updates: losses {lt_all}; parameter difference / lr: median {med:.1e}, "
          f"99% {q99:.1e}, max {dmax:.1e}")
    assert med < 2e-4 and q99 < 1e-2 and dmax < 0.5
