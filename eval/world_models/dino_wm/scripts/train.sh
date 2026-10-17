#!/usr/bin/env bash
# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
# One world model of Tables 3 and 4 on 8 GPUs (8 x 32 = global batch 256), training seed $SEED (default 1):
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash train.sh <task> <encoder> [extra wm_train.py arguments]
# Writes $WM_ROOT/runs/outputs/<task>_<encoder>_wm_s<seed>/{hydra.yaml, checkpoints/model_s<update>.pth, train_log.jsonl, lr_plan.csv}.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
TASK=${1:?task}; ENC=${2:?encoder}; shift 2
NPROC=${NPROC:-8}; PORT=${PORT:-29531}
RUN=$RUNS/outputs/$(run_name "$TASK" "$ENC")
cd "$DINO_WM"
[ -f "$RUN/hydra.yaml" ] || python make_run_config.py --task "$TASK" --encoder "$ENC" --stats "$STATS/${TASK}_$ENC.npz" \
  --subsets "$SUBSETS" --data "$DATASET_DIR" --out "$RUN"
echo "START $(date -u +%FT%TZ) train $TASK $ENC seed $SEED -> $RUN"
set +e
torchrun --nproc_per_node "$NPROC" --master_port "$PORT" wm_train.py --run "$RUN" --seed "$SEED" --workers 7 "$@" 2>&1 | tee -a "$RUN/train_stdout.log"
RC=${PIPESTATUS[0]}
echo "END $(date -u +%FT%TZ) rc=$RC"
exit "$RC"
