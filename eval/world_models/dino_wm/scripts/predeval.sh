#!/usr/bin/env bash
# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
# Prediction accuracy of one world model (Table 4, Table S16): the state probe of the encoder (trained once per encoder and
# task; it reads only the frozen encoder and its statistics) and the open-loop evaluation at the fixed endpoint:
#   CUDA_VISIBLE_DEVICES=<gpu> bash predeval.sh <pusht|wall|granular|rope> <encoder>
# -> $WM_ROOT/results/probes/probe_<task>_<encoder>.pt, $WM_ROOT/results/pred/<task>_<encoder>.npz
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
TASK=${1:?task}; ENC=${2:?encoder}
task_settings "$TASK"
RUN=$RUNS/outputs/$(run_name "$TASK" "$ENC")
mkdir -p "$RESULTS/probes" "$RESULTS/pred"
PROBE=$RESULTS/probes/probe_${TASK}_$ENC.pt
cd "$DINO_WM"
case $TASK in granular|rope) PROBER=wm_predeval_deform.py ;; *) PROBER=wm_predeval.py ;; esac
[ -f "$PROBE" ] || python "$PROBER" probe --run "$RUN" --epoch "s$T" --out "$PROBE"
python wm_rollout_eval.py --run "$RUN" --probe "$PROBE" --epoch "s$T" --out "$RESULTS/pred/${TASK}_$ENC.npz"
