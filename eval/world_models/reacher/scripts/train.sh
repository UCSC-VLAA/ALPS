#!/usr/bin/env bash
# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
# Token statistics (if missing) and one Reacher world model on 8 GPUs (8 x 32 = global batch 256), training seed $SEED (default 1):
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash train.sh <encoder> [extra reacher_wm_train.py arguments]
# -> $REACHER_OUT/stats/reacher_<encoder>.npz, $REACHER_OUT/runs/<run>/ (full-state checkpoints, logs) and the exported
#    weights $STABLEWM_HOME/checkpoints/<run>/weights_s<update>.pt (+ config.json), run = reacher_wm_<encoder>_s<seed>.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ENC=${1:?encoder}; shift
NPROC=${NPROC:-8}; PORT=${PORT:-29541}
RUN=$(run_name "$ENC")
mkdir -p "$STATS" "$REACHER_OUT/runs/$RUN"
FIRST_GPU=${CUDA_VISIBLE_DEVICES:-0}; FIRST_GPU=${FIRST_GPU%%,*}
[ -f "$STATS/reacher_$ENC.npz" ] || CUDA_VISIBLE_DEVICES=$FIRST_GPU "$PYTHON" "$REACHER_DIR/make_stats.py" "$ENC" --out "$STATS"
cd "$REACHER_DIR"
echo "START $(date -u +%FT%TZ) train reacher $ENC seed $SEED -> $REACHER_OUT/runs/$RUN"
set +e
"$PYTHON" -m torch.distributed.run --nproc_per_node "$NPROC" --master_port "$PORT" reacher_wm_train.py --encoder "$ENC" \
  --stats "$STATS/reacher_$ENC.npz" --out "$REACHER_OUT/runs/$RUN" --run_name "$RUN" --seed "$SEED" "$@" 2>&1 | tee -a "$REACHER_OUT/runs/$RUN/train_stdout.log"
RC=${PIPESTATUS[0]}
echo "END $(date -u +%FT%TZ) rc=$RC"
exit "$RC"
