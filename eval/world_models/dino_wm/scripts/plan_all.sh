#!/usr/bin/env bash
# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
# All planning shards of one world model (the shards of the paper: 5 x 10 instances; Granular 5 x 2), then the merge into
# $WM_ROOT/results/plan/<task>_<encoder>.json:
#   GPUS="1 2 3 4 5" PER_GPU=1 bash plan_all.sh <task> <encoder> [extra hydra overrides]
# GPUS: GPU indices or UUIDs; PER_GPU: shards per GPU at a time. PyFleX tasks (Granular, Rope): a shard starts only when the
# simulator scenes of the previous one exist (simultaneous initialisations crash), and PyFleX places the rendering context
# of every process on the first GPU, so keep that GPU out of GPUS (README). A shard whose log ends with rc=0 is skipped.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/common.sh"
TASK=${1:?task}; ENC=${2:?encoder}; shift 2
task_settings "$TASK" || exit 2
GPUS=${GPUS:-0}; PER_GPU=${PER_GPU:-1}
SLOTS=(); for g in $GPUS; do for _ in $(seq "$PER_GPU"); do SLOTS+=("$g"); done; done
PIDS=(); for ((i = 0; i < ${#SLOTS[@]}; i++)); do PIDS[i]=0; done
LOGS=()
for ((A = 0; A < NEV; A += SHARD)); do
  B=$(( A + SHARD < NEV ? A + SHARD : NEV ))
  LOG=$WM_ROOT/plan_logs/${TASK}_${ENC}_${A}_${B}.log; LOGS+=("$LOG")
  if [ -f "$LOG" ] && tail -n 1 "$LOG" | grep -q "^END .*rc=0"; then echo "done: [$A,$B)"; continue; fi
  slot=-1
  while [ "$slot" -lt 0 ]; do
    for ((i = 0; i < ${#SLOTS[@]}; i++)); do
      if [ "${PIDS[i]}" = 0 ] || ! kill -0 "${PIDS[i]}" 2>/dev/null; then slot=$i; break; fi
    done
    [ "$slot" -lt 0 ] && sleep 30
  done
  n0=$(grep -a -c "^START" "$LOG" 2>/dev/null); n0=${n0:-0}
  CUDA_VISIBLE_DEVICES=${SLOTS[slot]} bash "$HERE/plan.sh" "$TASK" "$ENC" "$A" "$B" "$@" &
  PIDS[slot]=$!
  echo "started [$A,$B) on GPU ${SLOTS[slot]}"
  if [ "$SIM" = 1 ]; then          # wait for this shard's scenes (the full instance set + the shard) before the next start
    t=0
    while [ "$t" -lt 900 ]; do
      sleep 10; t=$((t + 10))
      seg=$(awk -v k=$((n0 + 1)) '/^START/{n++} n==k' "$LOG" 2>/dev/null)
      echo "$seg" | grep -a -q "^END" && break
      [ "$(echo "$seg" | grep -a -c "scene initialized")" -ge $((NEV + B - A)) ] && break
    done
  else
    sleep 20
  fi
done
wait
mkdir -p "$RESULTS/plan"
OUT=$RESULTS/plan/${TASK}_${ENC}.json
python "$HERE/../../tools/merge_plan_shards.py" "${LOGS[@]}" > "$OUT"
echo "merged -> $OUT: $(cut -c1-200 "$OUT")"
