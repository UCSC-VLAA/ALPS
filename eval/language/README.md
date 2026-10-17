# Language alignment

Zero-shot ImageNet-1k (top-1 over the 50,000 validation images, OpenAI CLIP class names and 80 templates as distributed
with CLIP_benchmark) and Flickr30k / COCO Karpathy-test retrieval (R@1 both ways, 5 captions per image), each model with
its own text tower, pooled image embedding and official preprocessing (`lang_eval.py`).

```bash
python lang_eval.py --model alps --weights alps_vitb16.safetensors --data <clip_benchmark_datasets> --out runs/lang
python lang_eval.py --model alps_stage1 --weights alps_stage1_vitb16.safetensors --data <clip_benchmark_datasets> --out runs/lang
python lang_eval.py --model siglip2 --data <clip_benchmark_datasets> --out runs/lang      # clip_qg, pe_core, tipsv2, cradio_clip, ov1, dinotxt
```
Data: the CLIP_benchmark dataset layout (`clip_benchmark` download scripts; ImageNet needs its own licence):
`flickr30k/` with `flickr30k_test_karpathy.txt`, `mscoco_captions/` with `coco_test_karpathy.json` and `val2014/`,
`imagenet1k/val/<wnid>/`. ALPS: image embedding = linear projection of the mean raw patch token, text embedding = its
native text encoder, float32, preprocessing = shorter side 256 (area interpolation) and central 224 crop.
C-RADIOv3 needs github.com/NVlabs/RADIO (env RADIO_REPO, RADIO_CKPT); OpenVision needs github.com/UCSC-VLAA/OpenVision
(env OV_ROOT).

Reproduction with this script and the released weights (the paper's evaluation used an earlier PyTorch port of the
encoder and text embeddings exported from the JAX code):

| | IN-1k top-1 / top-5 | Flickr I->T | T->I | COCO I->T | T->I |
|---|---|---|---|---|---|
| ALPS (Stage 1), paper's evaluation | 74.454 / 94.352 | 93.40 | 79.38 | 70.58 | 51.948 |
| ALPS (Stage 1), this script | 74.452 / 94.354 | 93.40 | 79.38 | 70.58 | 51.948 |
| ALPS, paper's evaluation | 73.436 / 93.878 | 93.20 | 78.92 | 70.20 | 51.892 |
| ALPS, this script | 73.436 / 93.878 | 93.20 | 78.92 | 70.20 | 51.892 |
