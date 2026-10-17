# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""The official DINO-WM ViTPredictor wrapped in two learnable linear projections on the visual channels.

  token in  [.., D + extra]   D = native encoder width (standardised tokens), extra = proprio + action embeddings (10 + 10)
    visual [.., :D] -> down: Linear(D -> proj_dim = 384)           (learnable)
    extra  [.., D:] -> unchanged
  -> official ViTPredictor(dim = proj_dim + extra = 404, depth 6, heads 16, MLP 2048, dropout 0.1, block-causal mask)
  -> predicted visual [.., :384] -> up: Linear(384 -> D)           (learnable)
     predicted extra  [.., 384:] -> unchanged
  token out [.., D + extra]

VWorldModel, its loss (MSE of the visual + proprio channels against the detached standardised target tokens), the rollout
(predicted tokens fed back and projected again by `down` at the next step) and the planning objective (MSE in the
standardised native space) are unchanged: the world model's latent space is the encoder's native token space. The
projections belong to the predictor, are trained with it and are saved inside the checkpoint's `predictor` entry.

Initialisation: `down.weight` orthogonal (orthonormal rows: unit-variance outputs for unit-variance inputs, the scale of the
standardised targets), `up.weight = down.weight.T` (up(down(x)) is an orthogonal projection of x at initialisation), biases
zero. Parameters of the projections: 2 * D * proj_dim + proj_dim + D.
"""
import torch
import torch.nn as nn

from models.vit import ViTPredictor


class ProjViTPredictor(nn.Module):
    def __init__(self, *, num_patches, num_frames, dim, extra_dim=20, proj_dim=384, bias=True, tied_init=True, **vit_kwargs):
        super().__init__()
        assert dim > extra_dim, (dim, extra_dim)
        self.visual_dim = int(dim - extra_dim)       # native D
        self.extra_dim = int(extra_dim)
        self.proj_dim = int(proj_dim)
        self.down = nn.Linear(self.visual_dim, self.proj_dim, bias=bias)
        self.up = nn.Linear(self.proj_dim, self.visual_dim, bias=bias)
        with torch.no_grad():
            nn.init.orthogonal_(self.down.weight)
            if tied_init:
                self.up.weight.copy_(self.down.weight.T)
            else:
                nn.init.orthogonal_(self.up.weight)
            if bias:
                self.down.bias.zero_()
                self.up.bias.zero_()
        self.vit = ViTPredictor(num_patches=num_patches, num_frames=num_frames, dim=self.proj_dim + self.extra_dim, **vit_kwargs)
        n_proj = sum(p.numel() for p in self.down.parameters()) + sum(p.numel() for p in self.up.parameters())
        n_vit = sum(p.numel() for p in self.vit.parameters())
        print(f"[ProjViTPredictor] native D={self.visual_dim} extra={self.extra_dim} proj_dim={self.proj_dim} "
              f"(inner ViT dim {self.proj_dim + self.extra_dim}); params: projections {n_proj:,}, ViT {n_vit:,}, "
              f"total {n_proj + n_vit:,}")

    def forward(self, x):  # x: (b, num_frames * num_patches, D + extra)
        vis, extra = x[..., : self.visual_dim], x[..., self.visual_dim :]
        h = torch.cat([self.down(vis), extra], dim=-1)
        y = self.vit(h)
        return torch.cat([self.up(y[..., : self.proj_dim]), y[..., self.proj_dim :]], dim=-1)
