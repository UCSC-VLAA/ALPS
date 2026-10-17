#!/usr/bin/env bash
# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
# PushT rendering evaluation, token side (Tables S15 and S17, Fig. 6), on the CPU as in the paper:
#   bash render.sh <encoder>          (needs the PushT probe of predeval.sh)
# -> $WM_ROOT/results/render/<encoder>_diag.npz and <encoder>_tokens.npz; decode the tokens with tools/render_decode.py.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ENC=${1:?encoder}
RUN=$RUNS/outputs/$(run_name pusht "$ENC")
mkdir -p "$RESULTS/render"
cd "$DINO_WM"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} python render_tokens.py --run "$RUN" --epoch s31000 \
  --probe "$RESULTS/probes/probe_pusht_$ENC.pt" --out "$RESULTS/render/${ENC}_diag.npz" --tok_out "$RESULTS/render/${ENC}_tokens.npz" \
  --n_windows 200 --bs 10 --device cpu
