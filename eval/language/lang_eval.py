# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Language evaluation of image-text encoders, each with its OWN text tower, pooled image embedding and preprocessing.

    python lang_eval.py --model alps --weights alps_vitb16.safetensors --data <clip_benchmark_datasets> --out runs/lang
    python lang_eval.py --model siglip2 --data <clip_benchmark_datasets> --out runs/lang

Tasks (Table 1 of the paper):
  flickr  Flickr30k Karpathy test, 1,000 images x 5 captions: image->text and text->image R@1
  coco    COCO Karpathy test, 5,000 images x 5 captions: image->text and text->image R@1
  in1k    ImageNet-1k validation (50,000 images): zero-shot top-1 / top-5 with the OpenAI CLIP class names and 80 templates
          (as distributed with CLIP_benchmark); class embedding = normalized mean of the normalized prompt embeddings
Data layout (the CLIP_benchmark download): <data>/flickr30k/{flickr30k_test_karpathy.txt, images}, <data>/mscoco_captions/
{coco_test_karpathy.json, val2014/}, <data>/imagenet1k/val/<wnid>/*.JPEG. Inputs at 224 px for every model.
Precision: float32 for ALPS (its activations overflow float16); fp16 autocast for the public models (bf16 for
C-RADIOv3, as in its official script), the settings of the paper (--amp overrides).
Writes lang_<tag>.json (scores) and lang_<tag>_hits.npz (per-query hits, for paired tests).
"""
import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
MODELS = ("alps", "alps_stage1", "siglip2", "clip_qg", "pe_core", "tipsv2", "cradio_clip", "cradio_siglip2", "ov1", "dinotxt")


def squash(res, mode, mean, std):
    m, s = torch.tensor(mean).view(3, 1, 1), torch.tensor(std).view(3, 1, 1)
    rs = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC}[mode]

    def f(path):
        im = Image.open(path).convert("RGB").resize((res, res), rs)
        return (torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1) - m) / s
    return f


def build_model(name, dev, res, weights):
    """-> prep(path) -> tensor, img(x) -> [B, D], txt(list) -> [N, D], note."""
    if name in ("alps", "alps_stage1"):
        import cv2
        sys.path.insert(0, ROOT)
        import alps
        model = alps.load(weights, device=dev)
        m, s = torch.tensor(alps.IMAGENET_MEAN).view(3, 1, 1), torch.tensor(alps.IMAGENET_STD).view(3, 1, 1)

        def prep(path):            # the training code's evaluation preprocessing: shorter side 256 (area), central 224 crop
            im = np.asarray(Image.open(path).convert("RGB"))
            h, w = im.shape[:2]
            r = 256.0 / min(h, w)
            nh, nw = int(round(h * r)), int(round(w * r))
            im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
            dy, dx = (nh - 224) // 2, (nw - 224) // 2
            im = im[dy:dy + 224, dx:dx + 224]
            return (torch.from_numpy(im.astype(np.float32) / 255.0).permute(2, 0, 1) - m) / s
        return (prep, lambda x: model.image_embedding(x, normalize=False), lambda t: model.text_embedding(t, normalize=False),
                "ALPS: linear projection of the mean raw patch token; native text encoder (32 tokens); resize 256 (area) + "
                "central crop 224, ImageNet mean / std; float32")
    if name == "siglip2":
        import timm
        from transformers import AutoModel, AutoTokenizer
        hf = AutoModel.from_pretrained("google/siglip2-base-patch16-224").eval().to(dev)
        tok = AutoTokenizer.from_pretrained("google/siglip2-base-patch16-224")
        net = timm.create_model("vit_base_patch16_siglip_224.v2_webli", pretrained=True, num_classes=0, img_size=res).eval().to(dev)

        def txt(texts):
            ids = tok([t.lower() for t in texts], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
            o = hf.get_text_features(input_ids=ids.input_ids.to(dev))
            return o if torch.is_tensor(o) else o.pooler_output
        return squash(res, "bilinear", (0.5,) * 3, (0.5,) * 3), net, txt, \
            "timm image tower (MAP pooling) + HF text tower, lower-cased, max_length 64; squash resize bilinear, mean = std = 0.5"
    if name in ("clip_qg", "pe_core"):
        import open_clip
        hub = {"clip_qg": "hf-hub:timm/vit_base_patch16_clip_224.openai", "pe_core": "hf-hub:timm/PE-Core-B-16"}[name]
        oc, _, pv = open_clip.create_model_and_transforms(hub)
        oc = oc.eval().to(dev)
        tokz = open_clip.get_tokenizer(hub)
        return (lambda p: pv(Image.open(p).convert("RGB"))), oc.encode_image, (lambda t: oc.encode_text(tokz(t).to(dev))), \
            "open_clip model with its own preprocessing, tokenizer and pooling"
    if name == "tipsv2":
        from transformers import AutoModel
        tm = AutoModel.from_pretrained("google/tipsv2-b14", trust_remote_code=True).eval().to(dev)
        return squash(res, "bilinear", (0.0,) * 3, (1.0,) * 3), (lambda x: tm.encode_image(x).cls_token[:, 0]), \
            (lambda t: tm.encode_text(list(t))), f"google/tipsv2-b14: class token vs text embedding; squash bilinear {res}, [0, 1]"
    if name in ("cradio_clip", "cradio_siglip2"):
        import importlib.util
        import torchvision.transforms.v2 as T2
        repo, ckpt = os.environ.get("RADIO_REPO", "NVlabs_RADIO"), os.environ.get("RADIO_CKPT", "c-radio_v3-b_half.pth.tar")
        spec = importlib.util.spec_from_file_location("radio_resize", os.path.join(repo, "examples", "common", "resize_transform.py"))
        rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
        head = "clip" if name == "cradio_clip" else "siglip2"
        rm = torch.hub.load(repo, "radio_model", source="local", version=ckpt, progress=False, skip_validation=True,
                            adaptor_names=[head]).eval().to(dev)
        ad = rm.adaptors[head]
        (ad.oc_model if head == "clip" else ad.text_model).to(dev).eval()
        step = int(getattr(rm, "min_resolution_step", rm.patch_size))
        tf = T2.Compose([rt.ResizeTransform([res, res], step), T2.ToImage(), T2.ToDtype(torch.float32, scale=True),
                         rt.PadToSquare(None), T2.CenterCrop([res, res])])
        if head == "clip":
            txt = lambda t: ad.encode_text(ad.tokenizer(list(t)).to(dev), normalize=False)
        else:
            txt = lambda t: ad.encode_text({k: v.to(dev) for k, v in ad.tokenizer(list(t)).items()}, normalize=False)
        return (lambda p: tf(Image.open(p).convert("RGB"))), (lambda x: rm(x)[head].summary), txt, \
            f"C-RADIOv3-B summary through its {head} adaptor head; text = that teacher's text tower; official center-crop preprocessing"
    if name == "ov1":
        root = os.environ.get("OV_ROOT", "OpenVision")
        sys.path.insert(0, os.path.join(root, "code", "OpenVision-main"))
        from src.convert_upload.open_clip.factory import create_model_from_pretrained, get_tokenizer
        hub = "hf-hub:UCSC-VLAA/openvision-vit-base-patch16-224"
        oc, pv = create_model_from_pretrained(hub)
        oc = oc.eval().to(dev)
        tokz = get_tokenizer(hub)

        def txt(t):
            ids = tokz(list(t), context_length=oc.context_length)
            return oc.encode_text((ids if torch.is_tensor(ids) else torch.as_tensor(np.asarray(ids))).to(dev))
        return (lambda p: pv(Image.open(p).convert("RGB"))), oc.encode_image, txt, \
            "OpenVision B/16 224 (official patched open_clip): its own preprocessing, tokenizer (context 80), pooled embeddings"
    if name == "dinotxt":
        dt = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l").eval().to(dev)
        from dinov2.data.transforms import make_classification_eval_transform
        from dinov2.hub.dinotxt import get_tokenizer
        tok, tf = get_tokenizer(), make_classification_eval_transform()
        return (lambda p: tf(Image.open(p).convert("RGB"))), dt.encode_image, (lambda t: dt.encode_text(tok.tokenize(list(t)).to(dev))), \
            "dino.txt ViT-L/14 reg4: encode_image / encode_text, its own tokenizer and eval transform"
    raise SystemExit(f"unknown model {name}; known: {', '.join(MODELS)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--weights", default=None, help="safetensors file for alps / alps_stage1")
    ap.add_argument("--data", required=True, help="CLIP_benchmark datasets root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="flickr,coco,in1k")
    ap.add_argument("--amp", default="auto", choices=["auto", "fp16", "bf16", "none"])
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--bs", type=int, default=250)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev, D = a.device, a.data
    os.makedirs(a.out, exist_ok=True)
    tag = a.model + ("" if a.res == 224 else f"_r{a.res}")
    prep, img, txt, note = build_model(a.model, dev, a.res, a.weights)
    amp_mode = a.amp if a.amp != "auto" else ("none" if a.model.startswith("alps") else "bf16" if a.model.startswith("cradio") else "fp16")
    amp = (lambda: contextlib.nullcontext()) if amp_mode == "none" else \
        (lambda: torch.autocast(device_type="cuda", dtype=torch.float16 if amp_mode == "fp16" else torch.bfloat16))
    print(f"[{tag}] {note}; precision {amp_mode}", flush=True)

    class DS(torch.utils.data.Dataset):
        def __init__(self, paths): self.p = paths
        def __len__(self): return len(self.p)
        def __getitem__(self, i): return prep(self.p[i])

    @torch.no_grad()
    def img_embs(paths):
        out = []
        for x in torch.utils.data.DataLoader(DS(paths), batch_size=a.bs, num_workers=a.workers):
            with amp():
                e = img(x.to(dev)).float()
            assert torch.isfinite(e).all(), "non-finite image embeddings"
            out.append(F.normalize(e, dim=-1).cpu())
        return torch.cat(out)

    @torch.no_grad()
    def txt_embs(texts, bs=500):
        out = []
        for i in range(0, len(texts), bs):
            with amp():
                e = txt(texts[i:i + bs]).float()
            assert torch.isfinite(e).all(), "non-finite text embeddings"
            out.append(F.normalize(e, dim=-1).cpu())
        return torch.cat(out)

    res, hits = {"model": a.model, "res": a.res, "note": note, "amp": amp_mode}, {}

    def retrieval(task, imgs, caps):
        t0 = time.time()
        E = img_embs(imgs)
        texts = [c for cs in caps for c in cs]
        owner = torch.tensor([i for i, cs in enumerate(caps) for _ in cs])
        T = txt_embs(texts)
        S = E.double() @ T.double().T
        h_i2t = (owner[S.argmax(1)] == torch.arange(len(imgs))).numpy()
        h_t2i = (S.argmax(0) == owner).numpy()
        res[f"{task}_i2t_R1"], res[f"{task}_t2i_R1"] = 100 * float(h_i2t.mean()), 100 * float(h_t2i.mean())
        hits[f"{task}_i2t_R1"], hits[f"{task}_t2i_R1"] = h_i2t, h_t2i
        print(f"[{tag}] {task}: I->T R@1 {100 * h_i2t.mean():.2f}   T->I R@1 {100 * h_t2i.mean():.2f}   "
              f"({len(imgs)} images, {len(texts)} captions, {time.time() - t0:.0f}s)", flush=True)

    for task in a.tasks.split(","):
        if task == "flickr":
            data = {}
            with open(os.path.join(D, "flickr30k", "flickr30k_test_karpathy.txt")) as f:
                f.readline()
                for line in f:
                    line = line.strip()
                    if line:
                        im, c = line.split(".jpg,", 1)
                        data.setdefault(im + ".jpg", []).append(c)
            ims = list(data)
            root = os.path.join(D, "flickr30k")
            if not os.path.exists(os.path.join(root, ims[0])):
                root = os.path.join(root, "flickr30k_images")
            retrieval("flickr", [os.path.join(root, i) for i in ims], [data[i][:5] for i in ims])
        elif task == "coco":
            J = json.load(open(os.path.join(D, "mscoco_captions", "coco_test_karpathy.json")))
            caps = {}
            for an in J["annotations"]:
                caps.setdefault(an["image_id"], []).append(an["caption"])
            ims = [im for im in J["images"] if im["id"] in caps]
            retrieval("coco", [os.path.join(D, "mscoco_captions", "val2014", im["file_name"]) for im in ims],
                      [caps[im["id"]][:5] for im in ims])
        elif task == "in1k":
            t0 = time.time()
            lists = json.load(open(os.path.join(ROOT, "alps", "assets", "imagenet_zeroshot.json")))["clip_benchmark"]
            cn, tp = lists["class_names"], lists["templates"]
            W = torch.stack([F.normalize(txt_embs([t.replace("{c}", c) for t in tp]).mean(0), dim=-1) for c in cn])
            root = os.path.join(D, "imagenet1k", "val")
            wn = sorted(os.listdir(root))
            assert len(wn) == 1000
            paths, lab = [], []
            for i, w in enumerate(wn):
                fs = sorted(os.listdir(os.path.join(root, w)))
                paths += [os.path.join(root, w, f) for f in fs]
                lab += [i] * len(fs)
            lab = torch.tensor(lab)
            E = img_embs(paths)
            top = (E.double() @ W.double().T).topk(5, dim=1).indices
            h1, h5 = (top[:, 0] == lab).numpy(), (top == lab[:, None]).any(1).numpy()
            res["in1k_top1"], res["in1k_top5"] = 100 * float(h1.mean()), 100 * float(h5.mean())
            hits["in1k_top1"], hits["in1k_top5"] = h1, h5
            print(f"[{tag}] in1k: top-1 {100 * h1.mean():.2f}   top-5 {100 * h5.mean():.2f}   ({len(paths)} images, "
                  f"{len(tp)} templates, {time.time() - t0:.0f}s)", flush=True)
        json.dump(res, open(os.path.join(a.out, f"lang_{tag}.json"), "w"), indent=1)
        np.savez(os.path.join(a.out, f"lang_{tag}_hits.npz"), **hits)
    print(f"[{tag}] done", flush=True)


if __name__ == "__main__":
    main()
