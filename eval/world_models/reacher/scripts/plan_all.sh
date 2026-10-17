#!/usr/bin/env bash
# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
# Reacher planning of one world model (Table 3, Table S12): LeWorldModel's eval.py with config reacher.yaml (the same 50
# episodes, top-level seed 42; goals 25 steps ahead in a recorded episode; budget 50 environment steps; CEM 300 samples,
# 30 iterations, top 30, horizon 5), planner seeds 42, 43, 44 (solver.seed), each in 5 shards of 10 episodes, with the
# fixed endpoint weights_s26000.pt; rendering pinned to the shard's GPU (eval_pinned.py). Then the merge into
# $REACHER_OUT/results/plan/reacher_<encoder>_cem<seed>.json:
#   GPUS="<uuid or index> ..." bash plan_all.sh <encoder>
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/common.sh"
ENC=${1:?encoder}
RUN=$(run_name "$ENC")
GPUS=${GPUS:-0}; read -r -a G <<< "$GPUS"
LOGS=$REACHER_OUT/plan_logs; mkdir -p "$LOGS" "$RESULTS/plan"
[ -f "$STABLEWM_HOME/checkpoints/$RUN/weights_s26000.pt" ] || { echo "missing $STABLEWM_HOME/checkpoints/$RUN/weights_s26000.pt" >&2; exit 1; }
for s in 42 43 44; do   # one checkpoint folder name per planner seed, so that every seed writes its own results folder
  [ -e "$STABLEWM_HOME/checkpoints/${RUN}_cem$s" ] || ln -s "$RUN" "$STABLEWM_HOME/checkpoints/${RUN}_cem$s"
done
k=0; pids=()
for s in 42 43 44; do
  for i in 0 1 2 3 4; do
    if [ -f "$STABLEWM_HOME/${RUN}_cem$s/shard_${i}_5/episode_results.npz" ]; then echo "done: seed $s shard $i"; continue; fi
    g=${G[$((k % ${#G[@]}))]}; k=$((k + 1))
    uuid=$g
    if [[ "$g" != GPU-* ]]; then uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$g"); fi
    ( cd "$LEWM_ROOT" && TARGET_GPU_UUID=$uuid CUDA_VISIBLE_DEVICES=$uuid "$PYTHON" -u "$REACHER_DIR/eval_pinned.py" --config-name=reacher.yaml \
        policy="${RUN}_cem$s/weights_s26000.pt" eval.num_eval=50 eval.dataset_name=reacher solver.seed=$s +eval.shard=$i/5 \
        > "$LOGS/${RUN}_cem${s}_shard$i.log" 2>&1 ) &
    pids+=($!)
    if [ "${#pids[@]}" -ge "${#G[@]}" ]; then wait "${pids[0]}"; pids=("${pids[@]:1}"); fi
  done
done
wait
for s in 42 43 44; do
  "$PYTHON" "$HERE/../../tools/merge_reacher.py" "$STABLEWM_HOME/${RUN}_cem$s" --encoder "$ENC" --seed "$s" > "$RESULTS/plan/reacher_${ENC}_cem$s.json"
  echo "seed $s: $(cut -c1-160 "$RESULTS/plan/reacher_${ENC}_cem$s.json")"
done
