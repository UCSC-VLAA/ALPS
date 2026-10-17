#!/usr/bin/env bash
# Copyright 2026 The ALPS Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Train and evaluate the policies of the paper listed in ../paper_policies.tsv whose name or use matches a pattern, one after
# the other on one GPU (start one instance per GPU with disjoint patterns to run in parallel).
# usage: run_paper_policies.sh '<grep -E pattern on the row>' [gpu=0]
#   e.g. 'Table 2 \(BERT\)'   'LIBERO-Goal \| ALPS \|'   'Sec. S3'   '.' (all 63). One policy took 16-27 h of training
#   on one RTX A5000 in the paper's runs, plus about 1 h per evaluation.
# environment: LIBERO_ROOT; ALPS_WEIGHTS, ALPS_STAGE1_WEIGHTS, RADIO_REPO + RADIO_CKPT, SIGLIP2_STAGE2_WEIGHTS as the selected
#   rows need them; TASK_EMBEDDINGS_DIR (folder of the *_task_embs.npz files) for the own-text-tower rows; DRY_RUN=1 prints only
set -euo pipefail
[ $# -ge 1 ] || { grep '^# usage\|^#   e.g.\|^# environment\|^#   rows' "$0"; exit 2; }
PATTERN=$1; GPU=${2:-0}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LIST=$HERE/../paper_policies.tsv
: "${LIBERO_ROOT:?set LIBERO_ROOT}"
n=0
while IFS=$'\t' read -r policy used suite enc wenv sarg seed epochs instr paper success; do
  case $policy in policy|\#*) continue ;; esac
  printf '%s\t%s\n' "$policy" "$used" | grep -Eq -- "$PATTERN" || continue
  n=$((n + 1))
  W=""; [ "$wenv" = "-" ] || W=${!wenv:-}
  if [ "$wenv" != "-" ] && [ -z "$W" ]; then echo "SKIP $policy: set $wenv"; continue; fi
  EMB=""
  if [ "$instr" != "bert" ]; then EMB=${TASK_EMBEDDINGS_DIR:?set TASK_EMBEDDINGS_DIR}/$instr; fi
  echo "== $policy | $used | paper $paper"
  ARGS=("$enc" "$sarg" "$seed"); [ -z "$EMB" ] || ARGS+=("$EMB")
  # The settings of the paper's runs, which differ from the module's defaults for new users: the frozen network followed the
  # policy's train / eval mode (it was a submodule of the policy), except for C-RADIOv3, the only encoder that computes
  # differently in training mode (its cropped position embedding is randomized), whose policies were trained with the frozen
  # network in eval mode (the default); and ALPS / ALPS (Stage 1) computed with the arithmetic of the earlier port.
  PAPER="policy.image_encoder.network_kwargs.encoder_train_mode=follow"
  case $enc in
    cradio_v3_b) PAPER="policy.image_encoder.network_kwargs.encoder_train_mode=eval" ;;
    alps|alps_stage1) PAPER="$PAPER policy.image_encoder.network_kwargs.alps_port_numerics=true" ;;
  esac
  if [ -n "${DRY_RUN:-}" ]; then
    echo "   ENCODER_WEIGHTS=$W GPU=$GPU EPOCHS=$epochs EXTRA_OVERRIDES='$PAPER' bash train_policy.sh ${ARGS[*]}"; continue
  fi
  OUT=$(ENCODER_WEIGHTS=$W GPU=$GPU EPOCHS=$epochs EXTRA_OVERRIDES="$PAPER ${EXTRA_OVERRIDES:-}" \
        bash "$HERE/train_policy.sh" "${ARGS[@]}" | tee /dev/stderr | grep 'TRAIN_OK' || true)
  RUNDIR=${OUT##* -> }
  [ -n "$OUT" ] && [ -d "$RUNDIR" ] || { echo "FAILED to train $policy"; continue; }
  bash "$HERE/evaluate_policy.sh" "$RUNDIR" "$GPU"
done < "$LIST"
echo "$n policies matched '$PATTERN'"
