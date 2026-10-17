# Model card: ALPS ViT-B/16

## Model
* Architecture: ViT-B/16 (width 768, 12 blocks, 12 heads, MLP 3,072), class token, fixed 2-D sine-cosine
  position embedding, no patch-embedding bias, LayerNorm epsilon 1e-6. 85.6M parameters.
* Output: 196 raw patch tokens of width 768 at 224 x 224 pixels (the visual state), without a final LayerNorm.
* Language interface (from Stage 1, unchanged by Stage 2): a 12-layer text encoder of width 768 (109.7M parameters,
  WordPiece vocabulary, 32 tokens), linear image projection of the mean token, linear text projection, learned
  contrastive scale.

## Files
Hugging Face repository `UCSC-VLAA/ALPS` (published with the paper): `alps_vitb16.safetensors` (ALPS),
`alps_stage1_vitb16.safetensors` (ALPS (Stage 1)), each with the encoder and the language interface, and `libero_policies/`,
the eight ALPS policies of Table 2 (`eval/libero_policy/README.md`). Load with `alps.load("alps-vitb16")`.

## Training
* Stage 1, from scratch: DataComp-1B images with their web captions and the synthetic captions of the LLaVA-LLaMA-3
  captioner of Recap-DataComp-1B. Objectives: image-text contrastive alignment (two positives per image), caption
  prediction, pixel reconstruction from the tokens and from noisy tokens, cross-view dense correspondence. 12.8 billion
  samples at 128 pixels, then 1.0 billion samples at 224 pixels.
* Stage 2: 5 million single-image draws from a curated collection of public image-editing datasets (images only; the
  editing instructions and edited pairs are not used). Cross-view self-distillation: a predictor reads the tokens of an
  augmented student crop and predicts the standardized tokens of the frozen Stage-1 encoder on a larger clean crop.
* No video, no actions, no teacher model other than the frozen Stage-1 encoder of ALPS itself.

## Intended use
Frozen visual state for world models, robot policies and probes; image-text retrieval and zero-shot classification
through the native text encoder; pixel reconstruction through an evaluation decoder trained on the frozen tokens.

## Limitations
* Evaluated with the ViT-B/16 backbone on simulation benchmarks (LIBERO, DINO-WM tasks, Reacher); real-robot use is
  untested.
* Pretrained on still images only; fine-grained temporal actions (e.g. SSv2) are not its strength.
* The language interface was trained on web and synthetic captions and inherits their biases.

## Citation
See the paper, "ALPS: Visual State Encoders for World Models" ([arXiv:2610.30120](https://arxiv.org/abs/2610.30120)):

```bibtex
@article{liu2026alps,
  title   = {ALPS: Visual State Encoders for World Models},
  author  = {Liu, Yanqing and Zhang, Letian and Liu, Chengzhi and Wang, Zijun and Mei, Jieru and Fei, Hongliang and Zhou, Yuyin and Xie, Cihang},
  journal = {arXiv preprint arXiv:2610.30120},
  year    = {2026}
}
```
