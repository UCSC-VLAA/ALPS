# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
"""Zero-shot ImageNet-1k top-1 / top-5 with the ALPS language interface.

    python examples/zero_shot_imagenet.py --weights alps_vitb16.safetensors --val /data/imagenet/val

`--val` is the validation set in class-folder layout (1,000 folders named by WordNet id); the classes are taken
in sorted-id order, which is the order of the CLIP class names. Preprocessing: shorter side 256 (area
averaging), central 224 crop, as in the paper's evaluation.
"""
import argparse
import os

import torch
from PIL import Image

import alps
from alps.zeroshot import imagenet_classifier


class Folder(torch.utils.data.Dataset):
    def __init__(self, root):
        self.classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        if len(self.classes) != 1000:
            raise SystemExit(f"expected 1000 class folders in {root}, found {len(self.classes)}")
        self.items = [(os.path.join(root, c, f), i) for i, c in enumerate(self.classes)
                      for f in sorted(os.listdir(os.path.join(root, c)))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return alps.center_crop_transform(Image.open(p).convert("RGB"))[0], y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    model = alps.load(a.weights, device=a.device)
    W = imagenet_classifier(model)                                  # [1000, 768]
    dl = torch.utils.data.DataLoader(Folder(a.val), batch_size=a.batch, num_workers=a.workers)
    top1 = top5 = n = 0
    with torch.no_grad():
        for x, y in dl:
            s = model.image_embedding(x.to(a.device)) @ W.T
            k = s.topk(5, dim=-1).indices.cpu()
            top1 += (k[:, 0] == y).sum().item()
            top5 += (k == y[:, None]).any(-1).sum().item()
            n += len(y)
    print(f"ImageNet zero-shot: top-1 {100 * top1 / n:.2f}  top-5 {100 * top5 / n:.2f}  ({n} images)")


if __name__ == "__main__":
    main()
