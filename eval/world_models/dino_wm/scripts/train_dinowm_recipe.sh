#!/usr/bin/env bash
# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
# Pipeline check (Table S11, last row; Sec. S2): DINOv2-S/14 with the DINO-WM encoder, predictor and schedule, trained by the
# official train.py (conf/train_dinowm_recipe.yaml), seed 0, global batch 32 split over the listed GPUs:
#   bash train_dinowm_recipe.sh <pusht|wall|rope> <gpus, e.g. 0,1,2,3> [extra hydra overrides]
# Budgets of the paper: PushT training.max_steps=123858 (the released PushT model; 4 GPUs), Wall training.batch_size=128
# training.max_steps=36010 (the released Wall model; 2 GPUs), Rope training.max_steps=53450 (100 epochs; 4 GPUs).
# Run directory: $WM_ROOT/runs/outputs/<task>_dinov2_vits14_dinowm_recipe; plan it with
#   MODEL=<task>_dinov2_vits14_dinowm_recipe EPOCH=final bash plan.sh <task> dinov2_vits14_dinowm_recipe <A> <B>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
TASK=${1:?task}; GPUS=${2:?gpus}; shift 2
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
case $TASK in
  pusht) OVR="env=pusht" ;;
  wall)  OVR="env=wall num_hist=1" ;;
  rope)  OVR="env=deformable_env env.kwargs.object_name=rope env.dataset.object_name=rope task=rope num_hist=1 frameskip=1" ;;
  *) echo "unknown task $TASK" >&2; exit 2 ;;
esac
RUN=${TASK}_dinov2_vits14_dinowm_recipe
mkdir -p "$RUNS/outputs/$RUN"
cd "$DINO_WM" || exit 1
echo "START $(date -u +%FT%TZ) $RUN gpus=$GPUS"
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch --num_processes "$NPROC" --main_process_port $((20000 + RANDOM % 20000)) train.py \
  --config-name train_dinowm_recipe.yaml $OVR ckpt_base_path="$RUNS" subset_dir="$SUBSETS" run_name="$RUN" \
  hydra.run.dir="$RUNS/outputs/$RUN" "$@" 2>&1 | tee -a "$RUNS/outputs/$RUN/train_stdout.log"
echo "END $(date -u +%FT%TZ) rc=${PIPESTATUS[0]}"
