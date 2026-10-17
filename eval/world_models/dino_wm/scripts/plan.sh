#!/usr/bin/env bash
# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
# One planning shard [A, B) of a world model with the settings of Table 3 (common.sh task_settings: CEM 300 / 30, horizon 5,
# planner.max_iter 10, planning seed 99, the fixed endpoint checkpoint, chunked rollouts, TF32):
#   CUDA_VISIBLE_DEVICES=<gpu> bash plan.sh <task> <encoder> <A> <B> [extra hydra overrides]
# Log: $WM_ROOT/plan_logs/<task>_<encoder>_<A>_<B>.log (START / END lines, read by tools/merge_plan_shards.py);
# planning outputs (logs.json, plan_targets.pkl): $WM_ROOT/plan_outputs/<task>_<encoder>_<A>_<B>/.
# MODEL / EPOCH override the run name and the checkpoint (e.g. MODEL=pusht_dinov2_vits14_dinowm_recipe EPOCH=final).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
TASK=${1:?task}; ENC=${2:?encoder}; A=${3:?A}; B=${4:?B}; shift 4
task_settings "$TASK" || exit 2
MODEL=${MODEL:-$(run_name "$TASK" "$ENC")}
EPOCH=${EPOCH:-s$T}
TAG=${TASK}_${ENC}_${A}_${B}
mkdir -p "$WM_ROOT/plan_logs" "$WM_ROOT/plan_outputs"
LOG=$WM_ROOT/plan_logs/$TAG.log
cd "$DINO_WM" || exit 1
{
  echo "START $(date -u +%FT%TZ) cfg=$PLAN_CFG model=$MODEL epoch=$EPOCH n_evals=$NEV shard=[$A,$B) extra=$PLAN_EXTRA $*"
  T0=$(date +%s)
  # shellcheck disable=SC2086
  python plan.py --config-name "$PLAN_CFG.yaml" ckpt_base_path="$RUNS" model_name="$MODEL" model_epoch="$EPOCH" planner.max_iter=10 \
    n_evals="$NEV" seed=99 $PLAN_EXTRA +eval_start="$A" +eval_end="$B" hydra.run.dir="$WM_ROOT/plan_outputs/$TAG" "$@"
  RC=$?
  echo "END $(date -u +%FT%TZ) rc=$RC wallclock_s=$(( $(date +%s) - T0 ))"
} >> "$LOG" 2>&1
