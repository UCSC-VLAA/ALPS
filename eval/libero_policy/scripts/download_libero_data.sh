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
# The LIBERO demonstrations of the four suites of the paper (official hdf5 files, Hugging Face dataset
# yifengzhu-hf/LIBERO-datasets, 50 demonstrations x 10 tasks per suite; LIBERO-10 is about 14 GB).
# usage: LIBERO_ROOT=<patched LIBERO> LIBERO_DATA=<datasets dir> bash download_libero_data.sh [suite ...]
#   suites: libero_10 libero_goal libero_spatial libero_object (default: all four); a suite with 10 hdf5 files is skipped
set -euo pipefail
: "${LIBERO_ROOT:?set LIBERO_ROOT}"; : "${LIBERO_DATA:?set LIBERO_DATA}"
SUITES=("$@"); [ ${#SUITES[@]} -gt 0 ] || SUITES=(libero_10 libero_goal libero_spatial libero_object)
mkdir -p "$LIBERO_DATA"; D=$(cd "$LIBERO_DATA" && pwd)
cd "$LIBERO_ROOT"; export PYTHONPATH="$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
for s in "${SUITES[@]}"; do
  n=$(ls "$D/$s"/*.hdf5 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" = 10 ]; then echo "$s: 10 files present, skipped"; continue; fi
  if [ "$s" = libero_10 ]; then
    # the official script offers libero_100 (= libero_10 + libero_90); the Hugging Face dataset keeps libero_10 apart
    python -c "import sys; from libero.libero.utils.download_utils import download_from_huggingface as d; d('libero_10', sys.argv[1], check_overwrite=False)" "$D"
  else
    python benchmark_scripts/download_libero_datasets.py --datasets "$s" --use-huggingface --download-dir "$D"
  fi
  echo "$s: $(ls "$D/$s"/*.hdf5 | wc -l | tr -d ' ') hdf5 files"
done
