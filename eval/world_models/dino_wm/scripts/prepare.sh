#!/usr/bin/env bash
# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
# Trajectory order file and token statistics of one task for the listed encoders (default: the six of the paper):
#   CUDA_VISIBLE_DEVICES=0 bash prepare.sh <task> [encoder ...]
# -> $WM_ROOT/subsets/<task>_100.json, $WM_ROOT/stats/<task>_<encoder>.npz (existing files are kept).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
TASK=${1:?task}; shift; ENCS=${*:-$ENCODERS}
mkdir -p "$SUBSETS" "$STATS"
cd "$DINO_WM"
[ -f "$SUBSETS/${TASK}_100.json" ] || python make_subsets.py --data "$DATASET_DIR" --out "$SUBSETS" --tasks "$TASK"
for e in $ENCS; do
  out=$STATS/${TASK}_$e.npz
  if [ -f "$out" ]; then echo "exists: $out"; continue; fi
  python fit_stats.py --task "$TASK" --encoder "$e" --subsets "$SUBSETS" --data "$DATASET_DIR" --out "$out"
done
