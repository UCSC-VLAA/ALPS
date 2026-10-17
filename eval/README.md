# Unified evaluation of frozen visual encoders

The evaluations of the ALPS paper. Every benchmark reads the frozen tokens of an encoder through one interface,
`eval/encoders.py` (`build(name).encode_frames(uint8 frames) -> native last-layer patch tokens`), which covers ALPS,
ALPS (Stage 1) and the 15 public encoders of the paper (`encoders.PAPER_ROWS` maps the table rows to registry names).

| Folder | Benchmark | Paper |
|---|---|---|
| `libero_state/` | LIBERO state probe: object positions, orientations and predicates now, 1 s and 3 s ahead (attentive probe on frozen tokens) | Tab. 1 state columns, Tab. S4-S7, Fig. 3 |
| `language/` | ImageNet zero-shot, Flickr30k and COCO retrieval with each model's own text tower | Tab. 1 language columns |
| `reconstruction/` | evaluation decoder to SDXL latents, l1 and LPIPS on 1,000 held-out ImageNet images | Tab. 1 reconstruction columns, Tab. S18, Fig. 4 |
| `video/` | DAVIS-2017 label propagation, Something-Something v2 attentive probe | Tab. S19 |
| `libero_policy/` | LIBERO policies trained on frozen tokens (patch for the LIBERO code base) | Tab. 2, S8-S10 |
| `world_models/` | DINO-WM planning and prediction accuracy (PushT, Wall, PointMaze, Granular, Rope), LeWorldModel Reacher | Tab. 3, 4, S11-S17, Fig. 6 |

Common settings: inputs at 224 px unless a benchmark says otherwise; float32; one seed (0) for every trained read-out;
token conventions per encoder in the docstring of `encoders.py`. Datasets are public and are downloaded by the user;
each README says from where.

Reference numbers (Tab. 1; state = position error relative to ALPS (Stage 1), lower is better):

| Encoder | state now | +1 s | +3 s | mean | IN-1k | Flickr I->T | T->I | COCO I->T | T->I | l1 | LPIPS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TIPSv2-B/14 | .947 | .946 | .995 | .963 | 71.1 | 86.5 | 72.2 | 63.7 | 48.2 | .204 | .467 |
| DINOv2-B/14 | .916 | .901 | 1.010 | .942 | n/a | n/a | n/a | n/a | n/a | .248 | .535 |
| PE-Spatial-B/16 | 1.000 | .892 | .917 | .936 | n/a | n/a | n/a | n/a | n/a | .160 | .386 |
| DINOv3-B/16 | .860 | .911 | .840 | .870 | n/a | n/a | n/a | n/a | n/a | .196 | .424 |
| C-RADIOv3-B | .876 | .838 | .877 | .864 | 66.5 | 85.7 | 74.3 | 56.5 | 46.0 | .161 | .359 |
| CLIP-B/16 | 1.091 | 1.110 | 1.380 | 1.194 | 68.4 | 81.9 | 62.0 | 52.5 | 33.1 | .291 | .579 |
| SigLIP 2-B/16 | 1.149 | 1.202 | 1.130 | 1.160 | 77.6 | 91.7 | 78.1 | 67.9 | 52.4 | .264 | .541 |
| PE-Core-B/16 | 1.118 | 1.099 | 1.292 | 1.170 | 78.4 | 93.9 | 80.9 | 71.1 | 50.3 | .264 | .559 |
| OpenVision-B/16 | 1.042 | 1.008 | 1.032 | 1.027 | 73.3 | 94.5 | 78.9 | 71.3 | 51.3 | .258 | .551 |
| MAE-B/16 | .935 | 1.008 | 1.027 | .990 | n/a | n/a | n/a | n/a | n/a | .109 | .260 |
| VC-1-B | 1.064 | .841 | .924 | .943 | n/a | n/a | n/a | n/a | n/a | .106 | .258 |
| OpenVision 3-B | 1.029 | .934 | 1.071 | 1.011 | n/a | n/a | n/a | n/a | n/a | .104 | .243 |
| OpenVision 2-L/14 | .969 | .881 | .987 | .945 | n/a | n/a | n/a | n/a | n/a | .311 | .602 |
| V-JEPA 2-L/16 | .741 | .841 | .904 | .829 | n/a | n/a | n/a | n/a | n/a | .173 | .413 |
| dino.txt-L/14 | 1.090 | 1.170 | 1.418 | 1.226 | 81.6 | 90.7 | 75.8 | 64.9 | 46.7 | .239 | .552 |
| ALPS (Stage 1) | 1.000 | 1.000 | 1.000 | 1.000 | 74.5 | 93.4 | 79.4 | 70.6 | 52.0 | .112 | .365 |
| **ALPS** | .842 | .834 | .813 | .829 | 73.4 | 93.2 | 78.9 | 70.2 | 51.9 | .120 | .372 |

The evaluations in the paper ran the ALPS encoders through an earlier PyTorch port with LayerNorm epsilon 1e-5; the
released model uses the training value 1e-6. The two give tokens with cosine similarity above 0.99999 and reproduce the
language numbers above at the printed precision.
