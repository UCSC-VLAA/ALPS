#!/usr/bin/env bash
# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
# Reacher prediction accuracy of one world model (Table 4, Table S16): held-out episodes and probe frames (once), the state
# probe of the encoder, the open-loop evaluation at the fixed endpoint:
#   CUDA_VISIBLE_DEVICES=<gpu> bash predeval.sh <encoder>
# -> $REACHER_OUT/results/probes/probe_reacher_<encoder>.pt, $REACHER_OUT/results/pred/reacher_<encoder>.npz
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ENC=${1:?encoder}
RUN=$(run_name "$ENC")
HELD=$REACHER_OUT/reacher_heldout.h5; FR=$REACHER_OUT/probe_frames
mkdir -p "$RESULTS/probes" "$RESULTS/pred"
cd "$REACHER_DIR"
[ -f "$HELD" ] || "$PYTHON" reacher_collect_heldout.py "$HELD"
[ -f "$FR/meta.npz" ] || "$PYTHON" reacher_predeval.py frames --out "$FR"
PROBE=$RESULTS/probes/probe_reacher_$ENC.pt
[ -f "$PROBE" ] || "$PYTHON" reacher_predeval.py probe --name "$ENC" --policy "$RUN/weights_s26000.pt" --frames "$FR" --out "$PROBE"
"$PYTHON" reacher_rollout_eval.py --name "$ENC" --policy "$RUN/weights_s26000.pt" --probe "$PROBE" --heldout "$HELD" \
  --out "$RESULTS/pred/reacher_$ENC.npz"
