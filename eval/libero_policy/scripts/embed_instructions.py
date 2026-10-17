#!/usr/bin/env python
# Copyright 2026 The ALPS Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Instruction embeddings of the 40 LIBERO tasks from an encoder's own text tower, for the policies of the lower block of
Tab. 2 (task_embedding_format=file:<npz>).

    python embed_instructions.py --model alps --alps-weights alps_vitb16.safetensors \
        --instructions libero_instructions.json --bert bert_pooler_libero.npz --out alps_task_embs.npz
    python embed_instructions.py --model clip | siglip2 | cradio ...          (same arguments, no --alps-weights)

The embedding of an instruction is the encoder's projected text embedding BEFORE L2 normalization, encoded as each model's
language evaluation encodes text, times ONE global factor: the mean norm of the 40 BERT embeddings it replaces divided by
the mean norm of the 40 projected embeddings (libero_instructions.json and bert_pooler_libero.npz: dump_libero_instructions.py).

  alps     the ALPS text tower and text projection (Stage 1, unchanged by Stage 2): alps.load(<weights>).text_embedding(...,
           normalize=False), ALPSTokenizer (lower-cased WordPiece, 32 slots)
  clip     open_clip hf-hub:timm/vit_base_patch16_clip_224.openai, encode_text
  siglip2  transformers google/siglip2-base-patch16-224, get_text_features of the lower-cased text, padding to max_length 64
  cradio   the text tower that C-RADIOv3-B's "clip" adaptor head is aligned with (DFN CLIP ViT-H/14-378 through open_clip),
           AD.encode_text(..., normalize=False); RADIO code: env RADIO_REPO (a clone of NVlabs/RADIO), checkpoint: RADIO_CKPT

Texts are encoded in four batches of 10 (one per suite), on the CPU in float32. Output npz: descriptions (40, suite order
libero_10, libero_goal, libero_spatial, libero_object), embs (40, D) scaled, raw (40, D), scale, bert_norm, note.
--reference <npz> compares the result with a released file (e.g. raw/task_embeddings/<model>_task_embs.npz).
Needs a recent environment (SigLIP 2: transformers >= 4.49), not the LIBERO one.
"""
import argparse
import json
import os
import sys

import numpy as np

SUITES = ["libero_10", "libero_goal", "libero_spatial", "libero_object"]
NOTES = {
    "alps": "ALPS text tower (Stage-1 text side, unchanged by Stage 2), projected embedding before L2 normalization",
    "clip": "CLIP ViT-B/16 text tower: open_clip hf-hub:timm/vit_base_patch16_clip_224.openai encode_text before normalization",
    "siglip2": "SigLIP 2 B/16 text tower: HF google/siglip2-base-patch16-224 get_text_features, lower-cased, max_length 64",
    "cradio": "C-RADIOv3-B clip adaptor head: the teacher's text tower (DFN CLIP ViT-H/14-378) encode_text before normalization",
}


def text_fn(model, device, alps_weights=None):
    import torch
    torch.set_grad_enabled(False)
    if model == "alps":
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
        import alps
        m = alps.load(alps_weights).to(device).eval()
        return lambda ts: m.text_embedding(list(ts), normalize=False)
    if model == "clip":
        import open_clip
        hub = "hf-hub:timm/vit_base_patch16_clip_224.openai"
        oc, _, _ = open_clip.create_model_and_transforms(hub)
        oc = oc.eval().to(device)
        tok = open_clip.get_tokenizer(hub)
        return lambda ts: oc.encode_text(tok(list(ts)).to(device))
    if model == "siglip2":
        from transformers import AutoModel, AutoTokenizer
        hf = AutoModel.from_pretrained("google/siglip2-base-patch16-224").eval().to(device)
        tok = AutoTokenizer.from_pretrained("google/siglip2-base-patch16-224")

        def f(ts):
            ids = tok([t.lower() for t in ts], padding="max_length", max_length=64, truncation=True, return_tensors="pt").input_ids
            o = hf.get_text_features(input_ids=ids.to(device))
            return o if torch.is_tensor(o) else o.pooler_output      # transformers >= 5 returns an output object
        return f
    if model == "cradio":
        repo, ckpt = os.environ.get("RADIO_REPO"), os.environ.get("RADIO_CKPT")
        if not repo or not ckpt:
            raise SystemExit("cradio: set RADIO_REPO (clone of NVlabs/RADIO) and RADIO_CKPT (c-radio_v3-b_half.pth.tar)")
        rm = torch.hub.load(repo, "radio_model", source="local", version=ckpt, progress=False, skip_validation=True,
                            adaptor_names=["clip"]).eval().to(device)
        ad = rm.adaptors["clip"]
        ad.oc_model.to(device).eval()
        return lambda ts: ad.encode_text(ad.tokenizer(list(ts)).to(device), normalize=False)
    raise SystemExit(f"unknown model {model!r}")


def scale_to_bert(raw, bert_norm):
    """One global factor: mean norm of the BERT embeddings / mean norm of the raw embeddings (float32, as in the runs)."""
    norms = np.linalg.norm(raw, axis=1)
    scale = bert_norm / float(norms.mean())
    return (raw * scale).astype(np.float32), scale


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, choices=sorted(NOTES))
    ap.add_argument("--instructions", required=True, help="libero_instructions.json of dump_libero_instructions.py")
    ap.add_argument("--bert", required=True, help="bert_pooler_libero.npz of dump_libero_instructions.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alps-weights", default=os.environ.get("ALPS_WEIGHTS"), help="alps: alps_vitb16.safetensors")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reference", default=None, help="a released *_task_embs.npz to compare with")
    a = ap.parse_args()
    if a.model == "alps" and not a.alps_weights:
        ap.error("--model alps needs --alps-weights (or env ALPS_WEIGHTS)")
    inst = json.load(open(a.instructions))
    texts = [t for s in SUITES for t in inst[s]]
    assert len(texts) == 40, len(texts)
    bert = np.load(a.bert)
    bert_norm = float(np.linalg.norm(np.concatenate([bert[s] for s in SUITES]), axis=1).mean())
    import torch
    f = text_fn(a.model, a.device, a.alps_weights)
    raw = torch.cat([f(texts[i:i + 10]) for i in range(0, 40, 10)]).float().cpu().numpy()
    embs, scale = scale_to_bert(raw, bert_norm)
    norms = np.linalg.norm(raw, axis=1)
    for k, s in enumerate(SUITES):
        e = raw[10 * k:10 * k + 10]
        en = e / np.linalg.norm(e, axis=1, keepdims=True)
        cs = en @ en.T
        np.fill_diagonal(cs, -1)
        print("%-14s raw norm mean %.2f [%.2f, %.2f]  max cosine between two instructions %.3f" % (
            s, norms[10 * k:10 * k + 10].mean(), norms[10 * k:10 * k + 10].min(), norms[10 * k:10 * k + 10].max(), cs.max()))
    print("BERT mean norm %.6f | raw mean norm %.6f | scale %.7f | dim %d" % (bert_norm, norms.mean(), scale, raw.shape[1]))
    note = "%s; times one global factor %.5f = BERT pooler mean norm %.3f / raw mean norm %.3f over the 40 instructions" % (
        NOTES[a.model], scale, bert_norm, norms.mean())
    np.savez(a.out, descriptions=np.array(texts), embs=embs, raw=raw.astype(np.float32), scale=np.float32(scale),
             bert_norm=np.float32(bert_norm), note=note)
    print("saved", a.out)
    if a.reference:
        r = np.load(a.reference)
        assert [str(x) for x in r["descriptions"]] == texts, "instruction order differs from the reference"
        d = np.abs(r["embs"] - embs)
        print("vs %s: max |embs diff| %.3g (relative %.3g), scale %.7f vs %.7f" % (
            a.reference, d.max(), d.max() / np.abs(r["embs"]).max(), scale, float(r["scale"])))


if __name__ == "__main__":
    main()
