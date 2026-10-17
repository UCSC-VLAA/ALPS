# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""LIBERO state probe: one small ATTENTIVE PROBE for every encoder (E frozen; only this head trains).
Inputs are cached patch tokens [T_hist, N, D] (native grid, per-channel standardised with TRAIN-split statistics of the same encoder).
  (a) current-state readout:  tokens(t)                       -> object positions [K,3], orientations (6D or yaw), relations logits
  (b) future-state prediction: tokens(t-h..t) + actions(t..t+H) -> object state at t+H   (h = fixed history, H = 1 s / 3 s)
Architecture (fixed for all encoders; only the input adapter D -> d varies): Linear adapter, learned positional emb over N tokens and
history slots, action MLP (future mode), Q learned queries cross-attending to all tokens (2 layers, d=256, 4 heads), then per-field
linear heads. Losses: L1 on positions (metres), orientation loss after symmetry canonicalisation (per object type), CE on relations.
References computed by the trainer: train mean / majority class, no-action variant, hold-current-state."""
import torch, torch.nn as nn, torch.nn.functional as F


class AttentiveProbe(nn.Module):
    def __init__(self, d_in, n_tokens, fields, n_hist=1, act_dim=0, act_len=0, d=256, n_query=8, depth=2, heads=4):
        super().__init__(); self.fields = fields; self.n_hist, self.act_len = n_hist, act_len
        self.adapter = nn.Linear(d_in, d); self.pos = nn.Parameter(torch.zeros(1, n_tokens, d)); nn.init.trunc_normal_(self.pos, std=0.02)
        self.hist_pos = nn.Parameter(torch.zeros(1, n_hist, 1, d)); nn.init.trunc_normal_(self.hist_pos, std=0.02)
        self.use_action = act_len > 0
        if self.use_action:
            self.act_mlp = nn.Sequential(nn.Linear(act_dim, d), nn.GELU(), nn.Linear(d, d)); self.act_pos = nn.Parameter(torch.zeros(1, act_len, d)); nn.init.trunc_normal_(self.act_pos, std=0.02)
        self.queries = nn.Parameter(torch.zeros(1, n_query, d)); nn.init.trunc_normal_(self.queries, std=0.02)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(d); self.heads = nn.ModuleDict({name: nn.Linear(d * n_query, out_dim) for name, out_dim in fields.items()})

    def forward(self, tokens, actions=None, action_mask=None):
        """tokens [B, n_hist, N, D] (n_hist=1 for readout); actions [B, act_len, A] (future mode) -> {field: [B, out_dim]}"""
        B, Th, N, _ = tokens.shape
        x = self.adapter(tokens) + self.pos[:, None] + self.hist_pos[:, :Th]; x = x.reshape(B, Th * N, -1)
        if self.use_action and actions is not None:
            a = self.act_mlp(actions) + self.act_pos; x = torch.cat([x, a], 1)
        q = self.queries.expand(B, -1, -1)
        for layer in self.layers: q = layer(q, x)
        q = self.norm(q).reshape(B, -1)
        return {name: head(q) for name, head in self.heads.items()}

    def n_params(self): return sum(p.numel() for p in self.parameters())


def orientation_error_deg(pred_6d, gt_quat_wxyz, symmetry):
    """Geodesic angle between predicted rotation (6D -> R) and GT quaternion, minimised over the object's symmetry group
    ('none', 'z2' = 180 deg about z, 'z4' = 90 deg about z, 'cyl' = any rotation about z => compare z-axis only)."""
    R = rot6d_to_matrix(pred_6d); Rg = quat_to_matrix(gt_quat_wxyz)
    return torch.rad2deg(torch.acos(orientation_cos(pred_6d, gt_quat_wxyz, symmetry).clamp(-1 + 1e-6, 1 - 1e-6)))


def orientation_cos(pred_6d, gt_quat_wxyz, symmetry):
    """max over the symmetry group of cos(angle between predicted and GT rotation) in [-1,1]; smooth, safe to backprop (loss = 1 - cos)."""
    R = rot6d_to_matrix(pred_6d); Rg = quat_to_matrix(gt_quat_wxyz)
    if symmetry == "cyl":
        z = R[..., :, 2]; zg = Rg[..., :, 2]; return (z * zg).sum(-1).clamp(-1, 1)
    angles = {"none": [0.0], "z2": [0.0, 180.0], "z4": [0.0, 90.0, 180.0, 270.0]}[symmetry]
    coss = []
    for a in angles:
        Rz = rotz(torch.tensor(a, device=R.device)); Rc = Rg @ Rz; coss.append((((R.transpose(-1, -2) @ Rc).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
    return torch.stack(coss, -1).max(-1).values


def rot6d_to_matrix(x):
    a1, a2 = x[..., :3], x[..., 3:6]; b1 = F.normalize(a1, dim=-1); b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1); b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], -1)


def quat_to_matrix(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1), torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1), torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def rotz(deg):
    r = torch.deg2rad(deg); c, s = torch.cos(r), torch.sin(r); z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)
