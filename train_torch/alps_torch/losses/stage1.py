# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Stage-1 losses, with the global-batch semantics of the JAX code under data parallelism (alps_torch.distributed).

    L = L_itc + w_cap L_cap + w_rec L_rec + w_dense L_dense
    L_itc   = 1/2 [ITC(v, t_web) + ITC(v, t_sentence)]   multi-positive: one [B, B] softmax per caption, both
              directions; v, t L2-normalized with the epsilon outside the square root; scale = exp(logit_scale)
    L_cap   = next-token cross-entropy of the caption decoder, summed over the non-pad targets (the trailing [CLS]
              target included) and divided by their global count
    L_rec   = rho(D(z), x) + 0.25 rho(D(z + eps), x), rho = l1 + 0.2 (1 - SSIM) + 0.1 Sobel l1, where SSIM uses a
              7 x 7 box window (VALID), C1 = 0.01^2, C2 = 0.03^2, and Sobel is a zero-padded depthwise
              cross-correlation; eps = 0.05 rms(z) N(0, 1) with the per-token rms detached (in the model)
    L_dense = 0.75 L_same + 0.25 L_cross                  patch InfoNCE, temperature 0.1

B is the GLOBAL batch. Each function returns this rank's share in the form described in alps_torch.distributed:
its average over ranks is the global loss, and DDP's gradient averaging yields the global gradient.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from ..distributed import Dist
from ..models.dense import bilinear_sample, warp_coords


def _single(dist: Optional[Dist]) -> bool:
    return dist is None or not dist.distributed


# ------------------------------------------------------------------------------------------------------------ ITC
def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True)) + eps)


def itc_loss(image_emb: torch.Tensor, text_emb: torch.Tensor, logit_scale: torch.Tensor,
             dist: Optional[Dist] = None) -> torch.Tensor:
    v = _l2_normalize(image_emb.float())
    t = _l2_normalize(text_emb.float())
    s = logit_scale.float()
    if _single(dist):
        logits = (v @ t.T) * s
        labels = torch.arange(v.shape[0], device=v.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    V, T = dist.all_gather(v), dist.all_gather(t)
    labels = dist.rank * v.shape[0] + torch.arange(v.shape[0], device=v.device)
    return 0.5 * (F.cross_entropy((v @ T.T) * s, labels) + F.cross_entropy((t @ V.T) * s, labels))


def itc_loss_multipos(image_emb, text_emb1, text_emb2, logit_scale, dist: Optional[Dist] = None) -> torch.Tensor:
    return 0.5 * (itc_loss(image_emb, text_emb1, logit_scale, dist) + itc_loss(image_emb, text_emb2, logit_scale, dist))


# -------------------------------------------------------------------------------------------------------- caption
def caption_ntp_loss(logits: torch.Tensor, text: torch.Tensor, pad_id: int = 0,
                     dist: Optional[Dist] = None) -> torch.Tensor:
    """logits[:, t] predicts text[:, t + 1]; mean over the non-pad targets of the global batch."""
    tgt = text[:, 1:].long()
    lg = logits[:, :-1].float()
    valid = (tgt != pad_id).float()
    ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none").view_as(valid)
    num = (ce * valid).sum()
    if _single(dist):
        return num / torch.clamp(valid.sum(), min=1.0)
    return dist.world * num / torch.clamp(dist.sum_nograd(valid.sum()), min=1.0)


# ------------------------------------------------------------------------------------------------- reconstruction
def l1_loss(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.abs(p.float() - t.float()).mean()


def _box_mean(x: torch.Tensor, k: int = 7) -> torch.Tensor:
    """k x k box mean, stride 1, VALID, as two 1-D window sums (H then W) divided by k^2. x: [B, C, H, W]."""
    x = x.unfold(2, k, 1).sum(-1)
    x = x.unfold(3, k, 1).sum(-1)
    return x / (k * k)


def ssim_loss(p: torch.Tensor, t: torch.Tensor, k: int = 7, c1: float = 0.01 ** 2, c2: float = 0.03 ** 2):
    p = p.float(); t = t.float()
    mp, mt = _box_mean(p, k), _box_mean(t, k)
    sp = _box_mean(p * p, k) - mp * mp
    st = _box_mean(t * t, k) - mt * mt
    spt = _box_mean(p * t, k) - mp * mt
    ssim = ((2 * mp * mt + c1) * (2 * spt + c2)) / ((mp * mp + mt * mt + c1) * (sp + st + c2))
    return 1.0 - ssim.mean()


_SX = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
_SY = _SX.T.contiguous()


def _grad(img: torch.Tensor):
    c = img.shape[1]
    dw = lambda kern: F.conv2d(img, kern.to(img)[None, None].expand(c, 1, 3, 3), padding=1, groups=c)
    return dw(_SX), dw(_SY)


def sobel_gradient_loss(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    p = p.float(); t = t.float()
    px, py = _grad(p); tx, ty = _grad(t)
    return (torch.abs(px - tx) + torch.abs(py - ty)).mean()


def recon_loss(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return l1_loss(p, t) + 0.2 * ssim_loss(p, t) + 0.1 * sobel_gradient_loss(p, t)


# ---------------------------------------------------------------------------------------------- dense InfoNCE
def _l2n(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + eps)


def gumbel_uniforms(shape, generator=None, device=None) -> torch.Tensor:
    """U(1e-9, 1) draws of the Gumbel top-k subsampling (jax.random.uniform(minval=1e-9, maxval=1))."""
    u = torch.rand(shape, generator=generator, device=device, dtype=torch.float32)
    return torch.clamp(u * (1.0 - 1e-9) + 1e-9, min=1e-9)


def cross_indices(n_total: int, n_cross: int, generator=None, device=None) -> torch.Tensor:
    """The step's global set of cross-image negative indices into the flattened [B_global * N] patches. Must be
    the same on every rank: draw it from a generator seeded identically on all ranks."""
    return torch.randint(0, n_total, (n_cross,), generator=generator, device=device)


def patch_infonce_loss(h_x: torch.Tensor, h_gx: torch.Tensor, crop_box: torch.Tensor, flip: torch.Tensor, *,
                       cross_idx: torch.Tensor, gumbel_u: Optional[torch.Tensor] = None, generator=None,
                       temperature: float = 0.1, exclude_radius: int = 1, n_same: Optional[int] = None,
                       split_weights: Sequence[float] = (0.75, 0.25), dist: Optional[Dist] = None):
    """Returns (loss, diagnostics). h_x, h_gx: [b, g, g, D] projected features of x and g(x) (this rank's images).

    cross_idx [n_cross]: global patch indices (image = idx // N in the global batch). gumbel_u [b, N, N]: U(1e-9, 1)
    draws of the same-image subsampling (drawn from `generator` if None; used only when n_same < N - 1)."""
    h_x = h_x.float(); h_gx = h_gx.float()
    b, g, _, d = h_x.shape
    n = g * g
    coords = warp_coords(crop_box, flip, g)
    pos = bilinear_sample(h_x, coords)
    q = _l2n(h_gx).reshape(b, n, d)
    p = _l2n(pos).reshape(b, n, d)
    k = _l2n(h_x).reshape(b, n, d)
    pos_logit = (q * p).sum(-1, keepdim=True)
    same = torch.einsum("bnd,bmd->bnm", q, k)
    m0 = torch.round(coords[..., 0] * g - 0.5).reshape(b, n, 1, 1)
    n0 = torch.round(coords[..., 1] * g - 0.5).reshape(b, n, 1, 1)
    ar = torch.arange(g, device=h_x.device, dtype=torch.float32)
    rows, cols = ar.view(1, 1, g, 1), ar.view(1, 1, 1, g)
    near = ((torch.abs(rows - m0) <= exclude_radius) & (torch.abs(cols - n0) <= exclude_radius)).reshape(b, n, n)
    same_neg = torch.where(near, torch.tensor(-1e9, device=same.device), same)
    if n_same is not None and int(n_same) < n - 1:
        u = gumbel_u if gumbel_u is not None else gumbel_uniforms(same.shape, generator, same.device)
        score = torch.where(near, torch.tensor(float("-inf"), device=same.device), -torch.log(-torch.log(u.float())))
        # top-k with lax.top_k's tie rule (lower index first): a stable descending sort
        sel = torch.sort(score, dim=-1, descending=True, stable=True).indices[..., :int(n_same)]
        same_neg = torch.gather(same, -1, sel)
    flat = k.reshape(b * n, d)
    idx = cross_idx.to(h_x.device).long()
    rank = 0 if _single(dist) else dist.rank
    if _single(dist):
        rows_k = flat[idx]
    else:                                  # rows owned by this rank, summed over ranks with gradient
        local = idx - rank * b * n
        own = (local >= 0) & (local < b * n)
        rows_k = torch.where(own[:, None], flat[local.clamp(0, b * n - 1)], torch.zeros((), device=flat.device))
        rows_k = dist.all_reduce_sum(rows_k)
    cross = torch.einsum("bnd,kd->bnk", q, rows_k)
    img = rank * b + torch.arange(b, device=h_x.device)
    cross = torch.where(((idx[None, :] // n) == img[:, None])[:, None, :], torch.tensor(-1e9, device=cross.device),
                        cross)

    def nce(neg):
        lg = torch.cat([pos_logit, neg], -1) / temperature
        return -torch.log_softmax(lg, -1)[..., 0].mean()

    l_same, l_cross = nce(same_neg), nce(cross)
    loss = float(split_weights[0]) * l_same + float(split_weights[1]) * l_cross
    return loss, {"pnce_same": l_same.detach(), "pnce_cross": l_cross.detach()}


# ------------------------------------------------------------------------------------------------------------ total
def stage1_loss(out: Dict[str, torch.Tensor], *, cross_idx: torch.Tensor, cap_w: float = 1.0, rec_w: float = 1.0,
                dense_w: float = 0.25, dense_radius: int = 1, dense_n_same: Optional[int] = 64,
                dense_split: Sequence[float] = (0.75, 0.25), pad_id: int = 0, gumbel_u: Optional[torch.Tensor] = None,
                generator=None, dist: Optional[Dist] = None):
    """Total Stage-1 loss from the outputs of Stage1Model(train=True). Returns (loss, metrics)."""
    if "txt_emb2" in out:
        l_itc = itc_loss_multipos(out["img_emb"], out["txt_emb"], out["txt_emb2"], out["logit_scale"], dist)
    else:
        l_itc = itc_loss(out["img_emb"], out["txt_emb"], out["logit_scale"], dist)
    l_rec = recon_loss(out["rec_pred"], out["x_rgb"]) + 0.25 * recon_loss(out["rec_pred_noisy"], out["x_rgb"])
    l_dense, diag = patch_infonce_loss(out["h_x"], out["h_gx"], out["crop_box"], out["flip"], cross_idx=cross_idx,
                                       gumbel_u=gumbel_u, generator=generator, exclude_radius=dense_radius,
                                       n_same=dense_n_same, split_weights=dense_split, dist=dist)
    l_cap = torch.zeros((), device=l_itc.device)
    if "cap_logits" in out:
        l_cap = caption_ntp_loss(out["cap_logits"], out["cap_text"], pad_id=pad_id, dist=dist)
    loss = l_itc + cap_w * l_cap + rec_w * l_rec + dense_w * l_dense
    metrics = {"loss": loss, "L_itc": l_itc, "L_cap": l_cap, "L_rec": l_rec, "L_dense": l_dense,
               "logit_scale": out["logit_scale"], "latent_rms": out["latent_rms"], **diag}
    return loss, {k: v.detach().float() for k, v in metrics.items()}
