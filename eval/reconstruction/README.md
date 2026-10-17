# Pixel reconstruction from frozen tokens

One common evaluation decoder recipe for every encoder (`alps/decoder.py`): tokens -> 14 x 14 grid -> 1x1 conv to 512
-> transposed conv to 28 x 28 -> 5 residual blocks -> 4 latent channels of the frozen SDXL autoencoder
(stabilityai/sdxl-vae), which decodes the 224 x 224 image. Loss: MSE on the scaled latent + LPIPS (VGG) of the decoded
image; token-noise augmentation; AdamW 1e-4, batch 8, 20,000 steps; float32; seed 0.

```bash
python build_imagenet_split.py <imagenet>/val <caches>        # imagenet224_20k.npy, imagenet224_50k.npy
python train_decoder.py --enc alps --weights alps_vitb16.safetensors --cache <caches>/imagenet224_20k.npy --out runs/recon
python train_decoder.py --enc dinov3_b16 --cache <caches>/imagenet224_20k.npy --out runs/recon
# stronger decoder (Tab. S18): width 1,024, 10 blocks, 60,000 steps, 49,000 training images
python train_decoder.py --enc alps --weights ... --cache <caches>/imagenet224_50k.npy --ch 1024 --nblocks 10 --steps 60000 --tag _large --out runs/recon
```
Split: per ImageNet class (sorted ids and file names) the first 19 validation images train the decoder and the 20th is
held out (1,000 images, the last entries of the cache); l1 on [-1, 1] and LPIPS (VGG) on the held-out images.
Paper values (l1 / LPIPS): ALPS .120 / .372, ALPS (Stage 1) .112 / .365; stronger decoder: ALPS .107 / .302.
The trained decoders also render the world-model predictions (eval/world_models).

PushT decoders (they render the world models' predicted tokens, Tab. S14, S15, Fig. 6):
```bash
python build_pusht_cache.py <DINO-WM data>/pusht_noise <caches>/pusht224_cache.npy
python train_decoder.py --enc alps --weights alps_vitb16.safetensors --cache <caches>/pusht224_cache.npy --out runs/recon_pusht
```

The trained decoders of the paper are not released; the commands above train them. Check of this code against the
paper's run: `score_decoder.py` with the paper's ALPS decoder and the released encoder gives l1 .1202 / LPIPS .3723 on
the 1,000 held-out images (the paper's run: .12015 / .37232).
