#!/bin/bash
# Copyright 2026 The ALPS Authors.
# SPDX-License-Identifier: Apache-2.0
# The six probe cells of one encoder (paper protocol): token cache, then readout and 1 s / 3 s future prediction on
# LIBERO-Object and LIBERO-Spatial, 16 history frames, budget f100, seed 0.
#   EPISODES=... LABELS=... CACHE=... OUT=... bash run_encoder.sh <encoder> [weights.safetensors]
set -euo pipefail
ENC=$1; W=${2:-}
HERE=$(cd "$(dirname "$0")" && pwd)
: "${EPISODES:?set EPISODES (build_episodes.py output)}" "${LABELS:?set LABELS (label_states.py output)}"
: "${CACHE:?set CACHE (token cache dir)}" "${OUT:?set OUT (probe output dir)}"
PY=${PY:-python}
WARGS=(); [ -n "$W" ] && WARGS=(--weights "$W")
$PY "$HERE/extract_tokens.py" --encoder "$ENC" "${WARGS[@]}" --episodes_dir "$EPISODES" --cache_dir "$CACHE"
COMMON=(--encoder "$ENC" --seed 0 --n_hist 16 --budget f100 --episodes_dir "$EPISODES" --cache_dir "$CACHE" --out_dir "$OUT"
        --labels_dir "$LABELS/labels" --labels_manifest "$LABELS/labels_manifest.json")
for suite in libero_object libero_spatial; do
  $PY "$HERE/train_probe.py" "${COMMON[@]}" --suite $suite --mode readout --horizon_s 0
done
for h in 1 3; do
  for suite in libero_object libero_spatial; do
    $PY "$HERE/train_probe.py" "${COMMON[@]}" --suite $suite --mode future --horizon_s $h
  done
done
