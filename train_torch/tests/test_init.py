# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""The PyTorch initializers draw from the distributions of the Flax initializers: per leaf, the standard deviation,
the maximum magnitude (truncation / uniform bounds) and the mean agree with a Flax init of the same module
(width 768, one block per tower, so every leaf has enough entries for the statistics)."""
import numpy as np
import pytest
import torch

from parity_utils import flat_np, jax, jnp

from alps_torch import convert as C
from alps_torch.models import CrossViewFusion, FeatHead, Stage1Model

CFG = dict(width=768, depth=1, mlp_dim=3072, num_heads=12, remat=False, vocab_size=32_000, text_width=768,
           text_depth=1, text_heads=12, text_mlp_dim=3072, cap_depth=1, cap_heads=12, cap_mlp_dim=3072)


def _stats(a):
    a = np.asarray(a, np.float64).ravel()
    return a.std(), np.abs(a).max(), a.mean()


def _compare(flax_flat, torch_flat, min_size=256):
    checked = 0
    for k, a in flax_flat.items():
        b = torch_flat[k]
        sa, ma, mu_a = _stats(a)
        sb, mb, mu_b = _stats(b)
        if a.size < min_size or sa == 0:                     # constants (LayerNorm 1 / 0, biases 0, scalars)
            assert np.array_equal(np.unique(a), np.unique(b)) or np.allclose(a, b, atol=1e-6), k
            continue
        assert abs(sb / sa - 1) < 0.06, (k, sa, sb)
        # family: a normal has ~2.1 % of its mass beyond 2.3 std; a truncated normal (bound 2.27 std) or a uniform
        # (1.73 std) has none; bounded families must share the bound
        tail_a = np.mean(np.abs(np.asarray(a, np.float64) - mu_a) > 2.3 * sa)
        tail_b = np.mean(np.abs(np.asarray(b, np.float64) - mu_b) > 2.3 * sb)
        assert (tail_a > 0.004) == (tail_b > 0.004), (k, tail_a, tail_b)
        if tail_a <= 0.004:
            assert abs(mb / ma - 1) < 0.05, (k, ma, mb)
        assert abs(mu_a - mu_b) < 5 * sa / np.sqrt(a.size) + 1e-12, (k, mu_a, mu_b)
        checked += 1
    return checked


def test_stage1_init_matches_flax():
    from alps_jax.models.stage1 import Stage1Model as FStage1
    fm = FStage1(**CFG)
    x = jnp.zeros((1, 32, 32, 3))
    i32, i80 = jnp.zeros((1, 32), jnp.int32), jnp.zeros((1, 80), jnp.int32)
    k = jax.random.PRNGKey(0)
    pf = flat_np(fm.init({n: k for n in ("params", "dropout", "drop_path", "random_mask")}, x_enc=x, gx_enc=x,
                         x_rgb=x, crop_box=jnp.zeros((1, 4)), flip=jnp.zeros((1,), bool), text=i32, itc_text2=i32,
                         cap_text=i80, train=True)["params"])
    torch.manual_seed(0)
    tm = Stage1Model(**CFG)
    pt = C.torch_to_flax(tm.state_dict(), C.stage1_rules(1, 1, 1))
    n = _compare(pf, pt)
    assert n > 30
    assert float(tm.logit_scale) == pytest.approx(float(pf["logit_scale"][0]), abs=1e-7)
    # spot checks of the documented magnitudes
    assert np.abs(pt["visual_encoder/embedding/kernel"]).max() <= np.sqrt(6 / 768) + 1e-7         # kaiming uniform
    assert np.abs(pt["visual_encoder/Transformer/encoderblock_0/MlpBlock_0/Dense_1/kernel"]).max() <= 2 * 0.02 / 0.8796 + 1e-7
    assert abs(pt["visual_encoder/cls"].std() - 1e-6) < 2e-7


def test_predictor_init_matches_flax():
    from alps_jax.models.predictor import CrossViewFusion as FCVF, FeatHead as FHead
    g = 14
    fp = FCVF()
    args = (jnp.zeros((1, g * g, 2)), jnp.zeros((1,)), jnp.zeros((1, g * g, 768)), jnp.zeros((1, 32, 768)),
            jnp.ones((1, 32), bool))
    pf = flat_np(fp.init(jax.random.PRNGKey(0), *args)["params"])
    hf = flat_np(FHead().init(jax.random.PRNGKey(1), jnp.zeros((1, g * g, 768)))["params"])
    torch.manual_seed(0)
    tp, th = CrossViewFusion(), FeatHead()
    assert _compare(pf, C.torch_to_flax(tp.state_dict(), C.predictor_rules(depth=4))) > 20
    assert _compare(hf, C.torch_to_flax(th.state_dict(), C.feat_head_rules())) >= 1
    assert sum(p.numel() for p in tp.parameters()) == 18_000_640                       # the Flax parameter counts
    assert sum(p.numel() for p in th.parameters()) == 592_128


def test_full_size_parameter_counts():
    """The parameter counts of the ALPS Stage-1 model (identical to the Flax tree, 338,429,316 in total)."""
    m = Stage1Model()
    count = lambda mod: sum(p.numel() for p in mod.parameters())
    got = {"visual": count(m.visual), "text": count(m.text), "caption_decoder": count(m.caption_decoder),
           "pixel_decoder": count(m.pixel_decoder), "dense_projector": count(m.dense_projector),
           "image_projection": count(m.image_projection), "text_projection": count(m.text_projection),
           "total": count(m)}
    assert got == {"visual": 85_646_592, "text": 109_656_576, "caption_decoder": 134_301_440,
                   "pixel_decoder": 7_448_195, "dense_projector": 196_864, "image_projection": 589_824,
                   "text_projection": 589_824, "total": 338_429_316}
