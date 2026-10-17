# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Stage-1 and Stage-2 objectives vs train_jax (float32, CPU): model outputs, every loss term and the gradients of
the total loss w.r.t. every parameter, with the JAX random draws (reconstruction noise, Gumbel uniforms of the
same-image subsampling, cross-image indices) fed to the PyTorch side."""
import numpy as np
import pytest
import torch

from parity_utils import assert_close, assert_flat_close, flat_np, jax, jnp, nhwc_to_nchw, perturb, to_torch, torch_grads

from alps_torch import convert as C
from alps_torch.losses import stage1 as L1
from alps_torch.losses.stage2 import crossview_loss
from alps_torch.models import CrossViewFusion, FeatHead, Stage1Model, ViT
from alps_torch.models.stage2 import Stage2Student

f32 = jnp.float32
TINY = dict(width=64, depth=2, mlp_dim=128, num_heads=4, remat=False, vocab_size=120, text_width=64, text_depth=2,
            text_heads=4, text_mlp_dim=128, cap_depth=2, cap_heads=4, cap_mlp_dim=128, itc_proj_dim=64,
            dense_proj_dim=32)


def ids_batch(rng, b, length, vocab):
    """[bos] w .. [eos] [PAD] .. [CLS] layout (some captions truncated, i.e. [eos] in slot length - 2)."""
    out = np.zeros((b, length), np.int32)
    for i in range(b):
        n = int(rng.integers(3, length + 6))
        body = list(rng.integers(3, vocab, size=min(n, length - 3)))
        seq = ([1] + body + [2] + [0] * length)[:length - 1]
        if n >= length - 3:
            seq[-1] = 2
        out[i] = seq + [101 % vocab]
    return out


def stage1_inputs(seed=0, b=4, res=64):
    rng = np.random.default_rng(seed)
    x = rng.random((b, res, res, 3)).astype(np.float32)
    gx = rng.random((b, res, res, 3)).astype(np.float32)
    box = np.asarray([[0.1, 0.2, 0.7, 0.7], [0., 0., 0.8, 0.8], [0.05, 0.3, 0.6, 0.6], [0., 0., 1., 1.]][:b], np.float32)
    flip = np.asarray([True, False, True, False][:b])
    return dict(x=x, gx=gx, crop_box=box, flip=flip, text=ids_batch(rng, b, 32, 120),
                itc_text2=ids_batch(rng, b, 32, 120), cap_text=ids_batch(rng, b, 80, 120))


def _random_mask_key(r):
    """The key Stage1Model draws its reconstruction noise from (first make_rng('random_mask') at the root scope)."""
    import flax.linen as nn

    class Probe(nn.Module):
        @nn.compact
        def __call__(self):
            return self.make_rng("random_mask")
    return Probe().apply({}, rngs={"random_mask": r})


def _jax_stage1(inp, n_same, radius, n_cross, rec_w, dtype=f32, seed=0):
    from alps_jax.losses.stage1 import stage1_loss
    from alps_jax.losses.stage2 import normalize_image
    from alps_jax.models.stage1 import Stage1Model as FStage1
    fm = FStage1(**TINY, dtype=dtype)
    a = dict(x_enc=normalize_image(inp["x"]), gx_enc=normalize_image(inp["gx"]), x_rgb=jnp.asarray(inp["x"]),
             crop_box=jnp.asarray(inp["crop_box"]), flip=jnp.asarray(inp["flip"]), text=jnp.asarray(inp["text"]),
             itc_text2=jnp.asarray(inp["itc_text2"]), cap_text=jnp.asarray(inp["cap_text"]))
    k = jax.random.PRNGKey(seed)
    params = fm.init({n: k for n in ("params", "dropout", "drop_path", "random_mask")}, **a, train=True)["params"]
    params = perturb(params, seed + 1)
    r = jax.random.PRNGKey(11)

    def f(p):
        out = fm.apply({"params": p}, **a, train=True, rngs={"dropout": r, "drop_path": r, "random_mask": r})
        loss, m = stage1_loss(out, r, cap_w=1.0, rec_w=rec_w, dense_w=0.25, dense_n_cross=n_cross,
                              dense_radius=radius, dense_n_same=n_same)
        return loss, (m, out)
    (loss, (m, out)), g = jax.value_and_grad(f, has_aux=True)(params)
    b, res = inp["x"].shape[:2]
    gg = res // 16
    draws = dict(noise=np.asarray(jax.random.normal(_random_mask_key(r), (b, gg, gg, TINY["width"]), f32)).reshape(b, gg * gg, -1),
                 gumbel=np.asarray(jax.random.uniform(jax.random.fold_in(jax.random.fold_in(r, 1), 101),
                                                      (b, gg * gg, gg * gg), minval=1e-9, maxval=1.0)),
                 idx=np.asarray(jax.random.randint(jax.random.fold_in(r, 1), (n_cross,), 0, b * gg * gg)))
    return params, float(loss), {k: float(v) for k, v in m.items()}, out, g, draws


def _torch_stage1(params_flat, inp, draws, n_same, radius, rec_w, dtype=torch.float32):
    from alps.transforms import normalize
    tm = Stage1Model(**TINY, dtype=dtype)
    rules = C.stage1_rules(2, 2, 2)
    tm.load_state_dict(C.flax_to_torch(params_flat, rules), strict=True)
    x, gx = nhwc_to_nchw(inp["x"]), nhwc_to_nchw(inp["gx"])
    out = tm(normalize(x), normalize(gx), x, to_torch(inp["crop_box"]), torch.from_numpy(inp["flip"]),
             torch.from_numpy(inp["text"]).long(), torch.from_numpy(inp["itc_text2"]).long(),
             torch.from_numpy(inp["cap_text"]).long(), train=True, noise=to_torch(draws["noise"]))
    loss, m = L1.stage1_loss(out, cross_idx=to_torch(draws["idx"]).long(), cap_w=1.0, rec_w=rec_w,
                             dense_w=0.25, dense_radius=radius, dense_n_same=n_same, gumbel_u=to_torch(draws["gumbel"]))
    loss.backward()
    return tm, rules, out, float(loss), {k: float(v) for k, v in m.items()}


# (5, 1): the Stage-1b form (Gumbel top-k); (None, 0): the Stage-1a form; (8, 1): more negatives requested than allowed
# cells (7 on a 4 x 4 grid), so ties at -inf are broken by index as in lax.top_k
@pytest.mark.parametrize("n_same,radius,rec_w", [(5, 1, 1.0), (None, 0, 2.0), (8, 1, 1.0)])
def test_stage1_loss_and_grads(n_same, radius, rec_w):
    inp = stage1_inputs()
    params, lj, mj, oj, gj, draws = _jax_stage1(inp, n_same, radius, 8, rec_w)
    tm, rules, ot, lt, mt = _torch_stage1(flat_np(params), inp, draws, n_same, radius, rec_w)
    # model outputs
    for k in ("img_emb", "txt_emb", "txt_emb2", "cap_logits", "h_x", "h_gx"):
        assert_close(ot[k], np.asarray(oj[k]), 1e-5, k)
    for k in ("rec_pred", "rec_pred_noisy"):
        assert_close(ot[k].permute(0, 2, 3, 1), np.asarray(oj[k]), 1e-5, k)
    # every loss term
    for k in ("L_itc", "L_cap", "L_rec", "L_dense", "pnce_same", "pnce_cross", "loss", "logit_scale", "latent_rms"):
        assert mt[k] == pytest.approx(mj[k], rel=2e-6, abs=1e-7), (k, mt[k], mj[k])
    # gradients of the total loss w.r.t. all 8 subtrees
    gt = C.torch_to_flax(torch_grads(tm), rules)
    assert_flat_close(gt, flat_np(gj), 1e-5, what="stage1 grads", grads=True)


def test_stage1_bfloat16():
    """bfloat16 compute (the dtype of the ALPS runs): same cast points and output dtypes as Flax; values agree to
    bfloat16 accuracy (accumulation order and excess precision differ)."""
    inp = stage1_inputs(seed=2)
    params, lj, mj, oj, gj, draws = _jax_stage1(inp, 5, 1, 8, 1.0, dtype=jnp.bfloat16)
    tm, rules, ot, lt, mt = _torch_stage1(flat_np(params), inp, draws, 5, 1, 1.0, dtype=torch.bfloat16)
    for k in ("img_emb", "txt_emb", "cap_logits", "rec_pred", "h_x"):
        assert str(ot[k].dtype).split(".")[-1] == str(oj[k].dtype), (k, ot[k].dtype, oj[k].dtype)
    for k in ("L_itc", "L_cap", "L_rec", "L_dense", "loss"):
        assert mt[k] == pytest.approx(mj[k], rel=2e-2), (k, mt[k], mj[k])
    gt = C.torch_to_flax(torch_grads(tm), rules)
    gjf = flat_np(gj)
    a = np.concatenate([gt[k].ravel() for k in sorted(gt)]); b = np.concatenate([gjf[k].ravel() for k in sorted(gt)])
    assert a @ b / (np.linalg.norm(a) * np.linalg.norm(b)) > 0.99


def test_stage1_terms_individually():
    """Each Stage-1 loss function alone vs its JAX counterpart, on random inputs (values and input gradients)."""
    from alps_jax.losses import stage1 as J
    rng = np.random.default_rng(3)
    # ITC multi-positive
    e = [rng.normal(size=(6, 16)).astype(np.float32) for _ in range(3)]
    s = np.float32(7.3)
    fj = lambda a, b, c: J.itc_loss_multipos(a, b, c, jnp.asarray(s))
    lj, gj = jax.value_and_grad(fj, argnums=(0, 1, 2))(*map(jnp.asarray, e))
    et = [to_torch(a).requires_grad_(True) for a in e]
    lt = L1.itc_loss_multipos(*et, torch.tensor(s))
    lt.backward()
    assert float(lt) == pytest.approx(float(lj), rel=1e-6)
    for a, b in zip(et, gj):
        assert_close(a.grad, np.asarray(b), 1e-5, "itc grad")
    # caption NTP (some rows end in [eos] [CLS], padding in between)
    ids = ids_batch(rng, 3, 20, 50)
    lg = rng.normal(size=(3, 20, 50)).astype(np.float32)
    lj, gj = jax.value_and_grad(lambda x: J.caption_ntp_loss(x, jnp.asarray(ids)))(jnp.asarray(lg))
    xt = to_torch(lg).requires_grad_(True)
    lt = L1.caption_ntp_loss(xt, torch.from_numpy(ids))
    lt.backward()
    assert float(lt) == pytest.approx(float(lj), rel=1e-6)
    assert_close(xt.grad, np.asarray(gj), 1e-5, "caption grad")
    # reconstruction (l1 + SSIM + Sobel), non-square images
    p = rng.random((2, 20, 28, 3)).astype(np.float32) * 1.4 - 0.2
    t = rng.random((2, 20, 28, 3)).astype(np.float32)
    for jf, tf_ in ((J.ssim_loss, L1.ssim_loss), (J.sobel_gradient_loss, L1.sobel_gradient_loss),
                    (J.recon_loss, L1.recon_loss)):
        lj, gj = jax.value_and_grad(lambda x: jf(x, jnp.asarray(t)))(jnp.asarray(p))
        xt = nhwc_to_nchw(p).requires_grad_(True)
        lt = tf_(xt, nhwc_to_nchw(t))
        lt.backward()
        assert float(lt) == pytest.approx(float(lj), rel=2e-6), jf.__name__
        assert_close(xt.grad.permute(0, 2, 3, 1), np.asarray(gj), 1e-5, jf.__name__)


def test_dense_positive_geometry():
    """Bilinear positive and exclusion centre vs JAX, including flips and crops touching the border."""
    from alps_jax.models import dense as J
    from alps_torch.models import dense as T
    rng = np.random.default_rng(4)
    feat = rng.normal(size=(5, 6, 6, 8)).astype(np.float32)
    box = np.asarray([[0., 0., 1., 1.], [0.5, 0.5, 0.5, 0.5], [0.13, 0.27, 0.71, 0.71], [0., 0.3, 0.7, 0.7],
                      [0.25, 0., 0.75, 0.75]], np.float32)
    flip = np.asarray([False, True, True, False, True])
    cj = J.warp_coords(jnp.asarray(box), jnp.asarray(flip), 6)
    ct = T.warp_coords(to_torch(box), torch.from_numpy(flip), 6)
    assert_close(ct, np.asarray(cj), 1e-7, "coords")
    assert_close(T.bilinear_sample(to_torch(feat), ct), np.asarray(J.bilinear_sample(jnp.asarray(feat), cj)), 1e-6, "bilinear")
    # identity box without flip: the positive is the feature itself
    s = T.bilinear_sample(to_torch(feat[:1]), ct[:1])
    assert torch.allclose(s, to_torch(feat[:1]), atol=1e-6)
    r = torch.round(torch.tensor([0.5, 1.5, 2.5, -0.5]))
    assert r.tolist() == [0.0, 2.0, 2.0, -0.0]                         # round half to even, as jnp.round


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_stage2_loss_and_grads(dtype):
    from alps_jax.losses.stage2 import crossview_loss as jloss
    from alps_jax.models.predictor import CrossViewFusion as FCVF, FeatHead as FHead
    from alps_jax.models.vit import ViT as FViT
    jdt = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}[dtype]
    tdt = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    D, g, res, b, T = 64, 4, 64, 3, 48
    rng = np.random.default_rng(1)
    enc = FViT(width=D, depth=2, mlp_dim=128, num_heads=4, dtype=jdt)
    fus = FCVF(width=32, depth=2, num_heads=4, grid=g, token_dim=D, text_dim=T, dtype=jdt)
    head = FHead(D, dtype=jdt)
    x0 = jnp.zeros((b, res, res, 3), f32)
    k = jax.random.PRNGKey(5)
    pe = enc.init({"params": k}, x0, train=False)["params"]
    _, z = enc.apply({"params": pe}, x0, train=False)
    pt = fus.init({"params": k}, jnp.zeros((b, g * g, 2)), jnp.zeros((b,)), z.astype(f32), jnp.zeros((b, 32, T)),
                  jnp.ones((b, 32), bool))["params"]
    ph = head.init({"params": k}, z)["params"]
    params = {"encoder": perturb(pe, 1), "transition": perturb(pt, 2), "feat_head": perturb(ph, 3)}
    teacher = perturb(pe, 4)
    batch = {"x_s_rgb": rng.random((b, res, res, 3)).astype(np.float32),
             "x_t_rgb": rng.random((b, res, res, 3)).astype(np.float32),
             "box_a": np.asarray([[0.1, 0.2, 0.5], [0., 0., 1.], [0.3, 0.1, 0.4]], np.float32),
             "box_b": np.asarray([[0., 0., 0.8], [0., 0., 1.], [0.2, 0.2, 0.7]], np.float32)}
    u0 = rng.normal(size=T).astype(np.float32)
    zs = (rng.normal(size=(1, 1, D)).astype(np.float32), (rng.random((1, 1, D)) + 0.5).astype(np.float32))
    jfn = lambda p: jloss(p, teacher, {k: jnp.asarray(v) for k, v in batch.items()}, encoder=enc, predictor=fus,
                          head=head, zstats=tuple(map(jnp.asarray, zs)), u0=jnp.asarray(u0))
    (lj, mj), gj = jax.value_and_grad(jfn, has_aux=True)(params)

    rules = C.stage2_rules(depth=2, pred_depth=2)
    student = Stage2Student(ViT(D, 2, 128, 4, dtype=tdt), CrossViewFusion(32, 2, 4, g, D, T, dtype=tdt), FeatHead(D, D, tdt))
    student.load_state_dict(C.flax_to_torch(flat_np(params), rules), strict=True)
    student.encoder.norm.requires_grad_(True)
    tea = ViT(D, 2, 128, 4, dtype=tdt)
    tea.load_state_dict(C.flax_to_torch(flat_np(teacher), C.vit_rules(depth=2)), strict=True)
    tb = {"x_s_rgb": nhwc_to_nchw(batch["x_s_rgb"]), "x_t_rgb": nhwc_to_nchw(batch["x_t_rgb"]),
          "box_a": to_torch(batch["box_a"]), "box_b": to_torch(batch["box_b"])}
    lt, mt = crossview_loss(student, tea, tb, zstats=tuple(map(to_torch, zs)), u0=to_torch(u0))
    lt.backward()
    gt = C.torch_to_flax(torch_grads(student), rules)
    if dtype == "float32":
        assert float(lt) == pytest.approx(float(lj), rel=2e-6)
        assert float(mt["n_valid"]) == float(mj["n_valid"])
        assert_flat_close(gt, flat_np(gj), 1e-5, what="stage2 grads", grads=True)
    else:     # bfloat16 compute: same cast points, accumulation order and excess precision differ
        assert float(lt) == pytest.approx(float(lj), rel=2e-2)
        gtv = np.concatenate([v.ravel() for v in gt.values()])
        gjv = np.concatenate([flat_np(gj)[k].ravel() for k in gt])
        cos = gtv @ gjv / (np.linalg.norm(gtv) * np.linalg.norm(gjv))
        assert cos > 0.99, cos
