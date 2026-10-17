# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
"""PushT rendering evaluation, decoder side (Table S15, Fig. 6): decodes the tokens exported by dino_wm render_tokens.py with
the encoder's evaluation decoder and scores them against the true frames.

  python render_decode.py --enc <encoder> --tokens <enc>_tokens.npz --decoder <PushT decoder state dict> --out <dir>
                          [--vae stabilityai/sdxl-vae] [--fig_windows 834,138,991] [--check_tokens]

Decoder: the evaluation decoder of the reconstruction benchmark (tokens -> 1x1 conv to 512 -> transposed conv 14 -> 28 ->
5 residual blocks -> 4 SDXL latent channels; class Decoder below, the same module), trained on the RAW tokens of the frozen
encoder on 19,000 PushT training frames (the PushT option of the reconstruction benchmark); frozen SDXL VAE decoder;
224 x 224 image in [-1, 1]. The world model's tokens are converted back to the raw token space by render_tokens.py
(z * sd + mu). Metrics per window and k: l1 = mean |x - y| on images in [-1, 1], LPIPS-VGG (lpips package), against the
true frame. Conditions: real (k = 0..5), m1 = the open-loop rollout of Table 4 (k = 1..5), m2, m3 (render_tokens.py), and
static = the decoded real tokens of k = 0 scored against the true frames k = 1..5.
--check_tokens: cosine between the exported real tokens of k = 0 and the tokens that the decoder's own encoder
(eval/encoders.py, the encoder's own preprocessing at 224 pixels) gives for the same uint8 frame ("token cosine" of
Table S15; env ALPS_ROOT, ALPS_WEIGHTS / ALPS_STAGE1_WEIGHTS for the ALPS encoders).
Outputs: <out>/render_<enc>.npz (l1_* / lpips_* per window and k, tokcos, window, traj) and PNGs of the figure windows:
window<W>_traj<T>_<label>_<real|m1|m2|m3>_k<k>.png and window<W>_traj<T>_true_k<k>.png.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCALING = 0.13025          # SDXL VAE latent scaling
LABELS = {"alps": "ALPS", "alps_stage1": "ALPS-Stage1", "cradio_v3_b": "C-RADIOv3-B", "dinov3_b16": "DINOv3-B16", "siglip2_b16": "SigLIP2-B16",
          "dinov2_b14": "DINOv2-B14"}


def _decoder_class():
    """The evaluation decoder of alps/decoder.py (identical module and parameter names to the decoders of the paper)."""
    root = os.environ.get("ALPS_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    from alps.decoder import EvalDecoder
    return EvalDecoder


Decoder = _decoder_class()


def release_encoders():
    root = os.environ.get("ALPS_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    spec = importlib.util.spec_from_file_location("alps_eval_encoders", os.path.join(root, "eval", "encoders.py"))
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", required=True); ap.add_argument("--tokens", required=True); ap.add_argument("--decoder", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--fig_windows", default="834,138,991"); ap.add_argument("--label", default=None)
    ap.add_argument("--vae", default="stabilityai/sdxl-vae", help="HF id or local folder of the SDXL VAE")
    ap.add_argument("--bs", type=int, default=5); ap.add_argument("--check_tokens", action="store_true")
    a = ap.parse_args(); dev = os.environ.get("DEVICE", "cuda"); os.makedirs(a.out, exist_ok=True)
    import lpips as lpips_lib
    from diffusers import AutoencoderKL
    from PIL import Image
    name = a.label or LABELS.get(a.enc, a.enc)
    figw = [int(x) for x in a.fig_windows.split(",")] if a.fig_windows else []
    vae = AutoencoderKL.from_pretrained(a.vae).to(dev).eval(); lp = lpips_lib.LPIPS(net="vgg", verbose=False).to(dev).eval()

    @torch.no_grad()
    def render(dec, z, chunk=16):                     # z (B, P, D) float -> image (B, 3, 224, 224) in [-1, 1]
        return torch.cat([vae.decode(dec(z[i:i + chunk]) / SCALING).sample.clamp(-1, 1) for i in range(0, len(z), chunk)])

    @torch.no_grad()
    def score(img, true11):
        return (img - true11).abs().mean(dim=(1, 2, 3)), lp(img, true11).flatten()

    u8 = lambda x: ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()  # noqa: E731
    C = np.load(a.tokens)
    Wn, Tn = C["window"], C["traj"]; N = len(Wn); dim = C["real"].shape[-1]
    dec = Decoder(dim=dim).to(dev).eval(); dec.load_state_dict(torch.load(a.decoder, map_location=dev))
    E = None
    if a.check_tokens:
        kw = {"dinov2_norm": "imagenet"} if a.enc == "dinov2_b14" else {}
        E = release_encoders().build(a.enc, device=dev, **kw)
    M = {k: [] for k in ("l1_real", "lpips_real", "l1_static", "lpips_static", "tokcos")}
    for c in ("m1", "m2", "m3"):
        M["l1_" + c] = []; M["lpips_" + c] = []
    pos = {int(w): i for i, w in enumerate(Wn)}; figsel = {pos[w]: w for w in figw if w in pos}
    for s in range(0, N, a.bs):
        fr = torch.from_numpy(np.ascontiguousarray(C["frames"][s:s + a.bs])).to(dev).permute(0, 1, 4, 2, 3).float()   # (B, 6, 3, H, W) 0..255
        B = fr.shape[0]; true11 = fr / 127.5 - 1.0
        zr = torch.from_numpy(C["real"][s:s + a.bs].astype(np.float32)).to(dev)
        if E is not None:
            vt = E.encode_frames(np.ascontiguousarray(C["frames"][s:s + a.bs, 0])).float()                          # decoder's own token space, k = 0
            g = int(round(vt.shape[1] ** .5))
            if g != 14:
                vt = F.interpolate(vt.transpose(1, 2).reshape(B, -1, g, g), size=(14, 14), mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)
            M["tokcos"].append(F.cosine_similarity(vt, zr[:, 0], dim=-1).mean(1).cpu())
        ir = render(dec, zr.flatten(0, 1)); l1r, lpr = score(ir, true11.flatten(0, 1))
        M["l1_real"].append(l1r.view(B, 6).cpu()); M["lpips_real"].append(lpr.view(B, 6).cpu())
        ir = ir.view(B, 6, 3, 224, 224)
        st0 = ir[:, :1].expand(-1, 5, -1, -1, -1).flatten(0, 1); l1s, lps = score(st0, true11[:, 1:].flatten(0, 1))
        M["l1_static"].append(l1s.view(B, 5).cpu()); M["lpips_static"].append(lps.view(B, 5).cpu())
        imgs = {"real": ir}
        for c in ("m1", "m2", "m3"):
            ip = render(dec, torch.from_numpy(C[c][s:s + a.bs].astype(np.float32)).to(dev).flatten(0, 1)); l1p, lpp = score(ip, true11[:, 1:].flatten(0, 1))
            M["l1_" + c].append(l1p.view(B, 5).cpu()); M["lpips_" + c].append(lpp.view(B, 5).cpu()); imgs[c] = ip.view(B, 5, 3, 224, 224)
        for i in range(B):
            if s + i not in figsel:
                continue
            w, t = int(Wn[s + i]), int(Tn[s + i]); pre = os.path.join(a.out, "window%03d_traj%02d_" % (w, t))
            for k in range(6):
                Image.fromarray(C["frames"][s + i, k]).save(pre + "true_k%d.png" % k)
                Image.fromarray(u8(imgs["real"][i, k:k + 1])[0]).save(pre + "%s_real_k%d.png" % (name, k))
            for c in ("m1", "m2", "m3"):
                for k in range(1, 6):
                    Image.fromarray(u8(imgs[c][i, k - 1:k])[0]).save(pre + "%s_%s_k%d.png" % (name, c, k))
    out = {k: torch.cat(v).numpy() for k, v in M.items() if v}
    np.savez(os.path.join(a.out, "render_%s.npz" % a.enc), **out, window=Wn, traj=Tn, meta=json.dumps({"enc": a.enc, "decoder": a.decoder, "label": name}))
    print("MEANS", a.enc, json.dumps({k: [round(float(x), 5) for x in np.atleast_1d(v.mean(0))] for k, v in out.items()}), flush=True)
    print("RENDER_DECODE_DONE", a.enc, flush=True)


if __name__ == "__main__":
    main()
