# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Convolutional pixel decoder of the Stage-1 reconstruction objective (Flax), as in the original code.

[B, g, g, D] tokens -> 1x1 conv to 512 -> four up-blocks (3x3 conv to 4c, pixel shuffle x2, GroupNorm(min(32, c)),
SiLU, 3x3 conv to c) for c = 256, 128, 64, 32 -> 3x3 conv to RGB (float32, no output activation).
The pixel shuffle is channel-last (r, r, C): output (2h+i, 2w+j, k) = input channel (2i + j) * C + k.
"""
from typing import Any

import flax.linen as nn
import jax.numpy as jnp


def pixel_shuffle(x, r):
    b, h, w, c = x.shape
    cout = c // (r * r)
    x = x.reshape(b, h, w, r, r, cout).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(b, h * r, w * r, cout)


class _UpBlock(nn.Module):
    cout: int
    dtype: Any = jnp.bfloat16
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(4 * self.cout, (3, 3), padding="SAME", dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = pixel_shuffle(x, 2)
        x = nn.GroupNorm(num_groups=min(32, self.cout), dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = nn.silu(x)
        return nn.Conv(self.cout, (3, 3), padding="SAME", dtype=self.dtype, param_dtype=self.param_dtype)(x)


class RGBDecoder(nn.Module):
    dtype: Any = jnp.bfloat16
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, z):
        x = nn.Conv(512, (1, 1), dtype=self.dtype, param_dtype=self.param_dtype)(z)
        for c in (256, 128, 64, 32):
            x = _UpBlock(c, self.dtype, self.param_dtype)(x)
        return nn.Conv(3, (3, 3), padding="SAME", dtype=jnp.float32, param_dtype=self.param_dtype)(x)
