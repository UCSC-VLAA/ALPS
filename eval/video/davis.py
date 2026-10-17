# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""DAVIS-2017 semi-supervised video object segmentation by label propagation of frozen patch tokens (no training).

    python eval/video/davis.py --enc alps --weights alps_vitb16.safetensors \
        --davis DAVIS --eval_repo davis2017-evaluation --out runs/davis/alps

The protocol of the paper, identical for every encoder: the label propagation of DINO (github.com/facebookresearch/dino,
eval_video_segmentation.py) with its default hyper-parameters.
  * Frames (480p release) are resized with cv2 (bilinear) to a short side of 480 px floored to a multiple of the patch
    size (476 px for patch 14); the long side is scaled and floored to a multiple of 64 px when the patch size divides
    64, else to a multiple of the patch size. Frames are ImageNet-normalized; video_encoders.py converts them to the
    convention of each encoder and handles its positions at the frame grid.
  * Features: the native last-layer patch tokens of every frame, kept in float16 and L2-normalized. Image encoders
    encode every frame alone; V-JEPA 2 encodes the 16 frames up to frame t (padded at the start with the first frame)
    and the tokens of its last tubelet are the features of frame t. Prediction of frame t only uses frames <= t.
  * Frame t is predicted from the first frame (ground-truth mask) and the 7 previous frames (their predicted soft
    masks): affinity exp(cos / 0.1) between every target token and the tokens of these frames within a radius of 12
    token cells; per target token, the 5 largest affinities over all context tokens are kept and normalized to sum 1,
    and the soft masks are propagated with these weights.
  * The soft mask is upsampled bilinearly by the patch size, normalized per object, and its argmax is resized (nearest)
    to the frame size. J and F are computed by github.com/davisvideochallenge/davis2017-evaluation.
  * Encoder inference runs under bfloat16 autocast (--precision bf16, Tab. S19 and Sec. 5.2) or in float32 (--precision fp32,
    the Precision paragraph of the supplementary); label propagation always runs in float32.
Outputs: <out>/masks/<video>/<frame>.png and <out>/results.json (summary, per video, per object, timing, meta).
"""
import argparse
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from eval.video.video_encoders import IMAGENET, build_video_encoder  # noqa: E402

PROTOCOL = ("DINO eval_video_segmentation defaults (first frame + 7 previous frames, radius 12, top-5, temperature 0.1), "
            "native last-layer patch tokens stored in float16 and L2-normalized, soft masks propagated")


def target_size(h, w, patch, short=480):
    """DINO's read_frame rule: (height, width) of the resized frame."""
    q = 64 if 64 % patch == 0 else patch
    if h <= w:
        th = (short // patch) * patch
        tw = int((w * th / h) // q) * q
    else:
        tw = (short // patch) * patch
        th = int((h * tw / w) // q) * q
    return th, tw


def read_frames(paths, th, tw, device):
    """Frames resized to (th, tw), RGB, ImageNet-normalized: [F, 3, th, tw] float32 on `device`."""
    ims = [cv2.resize(cv2.imread(f), (tw, th))[:, :, ::-1] for f in paths]
    x = torch.from_numpy(np.stack(ims)).to(device).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(IMAGENET[0], device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET[1], device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def clip_window(t, t_hist):
    """Frame indices of the clip that ends at frame t, padded at the start with frame 0."""
    return [max(0, t - t_hist + 1 + k) for k in range(t_hist)]


@torch.no_grad()
def encode_video(enc, paths, th, tw, device, precision="bf16", batch=16, t_hist=16):
    """Tokens of every frame, [F, hw, D] float16."""
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16")
    if enc.tubelet == 1:
        out = []
        for i in range(0, len(paths), batch):
            x = read_frames(paths[i:i + batch], th, tw, device)
            with torch.autocast(**amp):
                out.append(enc.forward_images(x).half())
        return torch.cat(out)
    x_all = read_frames(paths, th, tw, device)
    out = []
    for t in range(len(paths)):
        with torch.autocast(**amp):
            tok = enc.forward_clip(x_all[clip_window(t, t_hist)][None])      # [1, T', hw, D]
        out.append(tok[0, -1].half())                                        # the last tubelet = frame t
    return torch.stack(out)


def neighborhood_mask(h, w, r, device):
    """[hw (target), hw (source)]: 1 where both cell offsets are at most r."""
    ii, jj = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    ii, jj = ii.flatten(), jj.flatten()
    m = ((ii[:, None] - ii[None]).abs() <= r) & ((jj[:, None] - jj[None]).abs() <= r)
    return m.float().to(device)


def to_one_hot(seg_small, n_labels, device):
    y = torch.from_numpy(seg_small.astype(np.int64)).to(device)
    return F.one_hot(y, n_labels).permute(2, 0, 1)[None].float()            # [1, C, h, w]


def norm_mask(mask):
    """Per label: (m - min) / max, for labels present in the mask (DINO's norm_mask)."""
    for k in range(mask.shape[0]):
        m = mask[k]
        if m.max() > 0:
            mask[k] = (m - m.min()) / m.max()
    return mask


@torch.no_grad()
def propagate(feat_tar, feats_src, segs, nb, temp=0.1, topk=5, radius=12):
    """DINO's label propagation. feat_tar [hw, D] (any scale); feats_src [n, D, hw] (L2-normalized); segs [n, C, h, w]
    soft masks of the n context frames; nb [hw, hw] neighborhood mask -> soft mask of the target frame [1, C, h, w]."""
    n, c, h, w = segs.shape
    hw = h * w
    ft = F.normalize(feat_tar.float(), dim=1)[None].expand(n, -1, -1)                  # n x hw_tar x D
    aff = torch.exp(torch.bmm(ft, feats_src) / temp)                                    # n x hw_tar x hw_src
    if radius > 0:
        aff = aff * nb[None]
    aff = aff.transpose(2, 1).reshape(-1, hw)                                           # (n * hw_src) x hw_tar
    tk = torch.topk(aff, dim=0, k=topk)[0]
    aff[aff < tk.min(dim=0, keepdim=True)[0]] = 0
    aff = aff / aff.sum(dim=0, keepdim=True).clamp_min(1e-12)
    s = segs.reshape(n, c, hw).transpose(2, 1).reshape(-1, c).T                         # C x (n * hw_src)
    return (s @ aff).reshape(1, c, h, w)


def _save_mask(lab, palette, path):
    im = Image.fromarray(lab)
    im.putpalette(palette)
    im.save(path)


def run_video(enc, davis, name, mask_dir, device, n_last=7, radius=12, topk=5, temp=0.1, short=480, t_hist=16,
              batch=16, precision="bf16"):
    """Predict and save the masks of one video; returns its timing record."""
    vdir = os.path.join(davis, "JPEGImages", "480p", name)
    frames = sorted(f for f in os.listdir(vdir) if f.endswith(".jpg"))
    paths = [os.path.join(vdir, f) for f in frames]
    ann0 = Image.open(os.path.join(davis, "Annotations", "480p", name, frames[0][:-4] + ".png"))
    palette = ann0.getpalette()
    seg0 = np.array(ann0)
    oh, ow = seg0.shape
    ps = enc.patch
    th, tw = target_size(oh, ow, ps, short)
    h, w = th // ps, tw // ps
    n_labels = int(seg0.max()) + 1
    t0 = time.time()
    feats = encode_video(enc, paths, th, tw, device, precision, batch, t_hist)          # [F, hw, D] float16
    t_enc = time.time() - t0
    fn = F.normalize(feats.float(), dim=2).transpose(1, 2).contiguous()                # [F, D, hw]
    first_seg = to_one_hot(np.array(ann0.resize((w, h), Image.NEAREST)), n_labels, device)
    nb = neighborhood_mask(h, w, radius, device)
    odir = os.path.join(mask_dir, name)
    os.makedirs(odir, exist_ok=True)
    _save_mask(seg0.astype(np.uint8), palette, os.path.join(odir, frames[0][:-4] + ".png"))
    que = []
    t0 = time.time()
    for t in range(1, len(paths)):
        used_f = torch.stack([fn[0]] + [q[0] for q in que])
        used_s = torch.cat([first_seg] + [q[1] for q in que])
        seg_t = propagate(feats[t], used_f, used_s, nb, temp, topk, radius)
        if len(que) == n_last:
            que.pop(0)
        que.append((fn[t], seg_t.clone()))
        up = norm_mask(F.interpolate(seg_t, scale_factor=ps, mode="bilinear", align_corners=False)[0])
        lab = up.argmax(0).byte().cpu().numpy()
        lab = np.array(Image.fromarray(lab).resize((ow, oh), Image.NEAREST))
        _save_mask(lab, palette, os.path.join(odir, frames[t][:-4] + ".png"))
    return {"video": name, "n_frames": len(paths), "n_objects": n_labels - 1, "size": [th, tw], "grid": [h, w],
            "t_encode_s": t_enc, "t_prop_s": time.time() - t0}


def davis_scores(davis, split, mask_dir, eval_repo=None):
    """J and F of the saved masks (davis2017-evaluation, semi-supervised task)."""
    if eval_repo:
        sys.path.insert(0, eval_repo)
    from davis2017.evaluation import DAVISEvaluation
    res = DAVISEvaluation(davis_root=davis, task="semi-supervised", gt_set=split).evaluate(mask_dir)
    J, Fm = res["J"], res["F"]
    per_obj = {k: {"J": float(J["M_per_object"][k]), "F": float(Fm["M_per_object"][k])} for k in J["M_per_object"]}
    per_video = {}
    for k, v in per_obj.items():
        per_video.setdefault(k.rsplit("_", 1)[0], []).append(v)
    per_video = {s: {"J": float(np.mean([o["J"] for o in objs])), "F": float(np.mean([o["F"] for o in objs])),
                     "JF": float(np.mean([(o["J"] + o["F"]) / 2 for o in objs])), "n_obj": len(objs)}
                 for s, objs in per_video.items()}
    summary = {"J&F_mean": float((np.mean(J["M"]) + np.mean(Fm["M"])) / 2), "J_mean": float(np.mean(J["M"])),
               "J_recall": float(np.mean(J["R"])), "J_decay": float(np.mean(J["D"])), "F_mean": float(np.mean(Fm["M"])),
               "F_recall": float(np.mean(Fm["R"])), "F_decay": float(np.mean(Fm["D"])), "n_objects": len(per_obj),
               "n_videos": len(per_video)}
    return summary, per_video, per_obj


def _sha16(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--enc", required=True, help="encoder name (video_encoders.PAPER_VIDEO_ROWS)")
    ap.add_argument("--weights", default=None, help="ALPS safetensors (default env ALPS_WEIGHTS / ALPS_STAGE1_WEIGHTS)")
    ap.add_argument("--state_dict", default=None, help="weights loaded (strict) into a timm encoder, e.g. SigLIP 2 refined with Stage 2")
    ap.add_argument("--vjepa2_rope", default="interp", choices=["interp", "recompute"],
                    help="V-JEPA 2 height / width RoPE positions scaled to its 16 x 16 pretraining grid (paper) or integer positions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--davis", required=True, help="DAVIS-2017 root: JPEGImages/480p, Annotations/480p, ImageSets/2017")
    ap.add_argument("--eval_repo", default=os.environ.get("DAVIS_EVAL_REPO"),
                    help="checkout of davis2017-evaluation (default env DAVIS_EVAL_REPO; not needed if it is installed)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n_last", type=int, default=7)
    ap.add_argument("--radius", type=int, default=12)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--short", type=int, default=480)
    ap.add_argument("--t_hist", type=int, default=16, help="clip length of video encoders")
    ap.add_argument("--batch", type=int, default=16, help="frames per forward pass of image encoders")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"], help="encoder inference")
    ap.add_argument("--videos", default="", help="comma-separated subset: masks only, no scores (results_partial.json)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    enc = build_video_encoder(args.enc, device=device, weights=args.weights, state_dict=args.state_dict,
                              res=224, vjepa2_rope=args.vjepa2_rope)
    mask_dir = os.path.join(args.out, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    with open(os.path.join(args.davis, "ImageSets", "2017", f"{args.split}.txt")) as f:
        vids = f.read().split()
    if args.videos:
        vids = [v for v in vids if v in args.videos.split(",")]
    meta = {"enc": enc.meta(), "args": vars(args), "protocol": PROTOCOL, "precision": args.precision,
            "code_sha256_16": {n: _sha16(os.path.join(HERE, *p)) for n, p in
                               [("davis.py", ["davis.py"]), ("video_encoders.py", ["video_encoders.py"]),
                                ("encoders.py", ["..", "encoders.py"])]},
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "start": time.strftime("%Y-%m-%d %H:%M:%S")}
    per = []
    t0 = time.time()
    for i, v in enumerate(vids):
        r = run_video(enc, args.davis, v, mask_dir, device, args.n_last, args.radius, args.topk, args.temp, args.short,
                      args.t_hist, args.batch, args.precision)
        per.append(r)
        print(f"[{i + 1}/{len(vids)}] {v} frames {r['n_frames']} grid {r['grid']} enc {r['t_encode_s']:.1f}s "
              f"prop {r['t_prop_s']:.1f}s", flush=True)
    meta["time_total_s"] = time.time() - t0
    if args.videos:
        with open(os.path.join(args.out, "results_partial.json"), "w") as f:
            json.dump({"per_video_timing": per, "meta": meta}, f, indent=1)
        return
    summary, per_video, per_obj = davis_scores(args.davis, args.split, mask_dir, args.eval_repo)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"summary": summary, "per_video": per_video, "per_object": per_obj, "per_video_timing": per, "meta": meta},
                  f, indent=1)
    print("RESULT", args.enc, json.dumps(summary))


if __name__ == "__main__":
    main()
