# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""tools/compute_t5_empty_token.py: the constant text token of Stage 2."""
import importlib.util
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("compute_t5_empty_token", os.path.join(HERE, "tools", "compute_t5_empty_token.py"))
t5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t5)
SIZES = dict(vocab=16, d_model=32, heads=2, d_kv=16, d_ff=64, depth=2)


def _random_params(rng):
    d, h, dk, ff = SIZES["d_model"], SIZES["heads"], SIZES["d_kv"], SIZES["d_ff"]
    p = {"embed": rng.normal(size=(SIZES["vocab"], d)), "rel_bias": rng.normal(size=(32, h)), "final_ln/weight": 1 + 0.1 * rng.normal(size=d)}
    for i in range(SIZES["depth"]):
        p.update({f"blk{i}/ln0/weight": 1 + 0.1 * rng.normal(size=d), f"blk{i}/ln1/weight": 1 + 0.1 * rng.normal(size=d)})
        p.update({f"blk{i}/{n}/kernel": rng.normal(size=(d, h * dk)) / np.sqrt(d) for n in ("q", "k", "v")})
        p.update({f"blk{i}/o/kernel": rng.normal(size=(h * dk, d)) / np.sqrt(h * dk),
                  f"blk{i}/wi_0/kernel": rng.normal(size=(d, ff)) / np.sqrt(d), f"blk{i}/wi_1/kernel": rng.normal(size=(d, ff)) / np.sqrt(d),
                  f"blk{i}/wo/kernel": rng.normal(size=(ff, d)) / np.sqrt(ff)})
    return {k: np.asarray(v, np.float32) for k, v in p.items()}


def test_padding_does_not_change_the_token():
    jnp = pytest.importorskip("jax.numpy")
    p = _random_params(np.random.default_rng(0))
    for dtype in (jnp.float32, jnp.bfloat16):
        u32 = t5.empty_instruction_token(p, dtype, length=32, **SIZES)
        u1 = t5.empty_instruction_token(p, dtype, length=1, **SIZES)
        assert u32.shape == (SIZES["d_model"],) and np.isfinite(u32).all()
        np.testing.assert_array_equal(u32, u1)


@pytest.mark.skipif(not os.environ.get("ALPS_T5_WEIGHTS"), reason="set ALPS_T5_WEIGHTS to google/flan-t5-base model.safetensors")
def test_reproduces_the_asset():
    jnp = pytest.importorskip("jax.numpy")
    path = os.environ["ALPS_T5_WEIGHTS"]
    flat = t5.params_from_hf(path) if path.endswith(".safetensors") else dict(np.load(path))
    ref = np.load(os.path.join(HERE, "assets", "flan_t5_base_empty_instruction.npz"))
    np.testing.assert_array_equal(t5.empty_instruction_token(flat, jnp.bfloat16), ref["u0_bf16"])
    np.testing.assert_array_equal(t5.empty_instruction_token(flat, jnp.float32), ref["u0_f32"])
