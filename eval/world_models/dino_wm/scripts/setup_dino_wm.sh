#!/usr/bin/env bash
# Copyright 2026 The ALPS Authors. Licensed under the Apache License, Version 2.0.
# Prepares a DINO-WM checkout for the ALPS world-model experiments:
#   bash setup_dino_wm.sh <new target folder>
# clones github.com/gaoyuezhou/dino_wm at commit 0a9492fa, applies dino_wm.patch and copies the new files of overlay/.
set -euo pipefail
TARGET=${1:?usage: setup_dino_wm.sh <new target folder>}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
COMMIT=0a9492fa12044b852ae9e001cc74604b79c8bb0c
if [ -e "$TARGET" ]; then echo "$TARGET exists; choose a new folder" >&2; exit 1; fi
git clone https://github.com/gaoyuezhou/dino_wm.git "$TARGET"
git -C "$TARGET" checkout -q "$COMMIT"
git -C "$TARGET" apply "$HERE/dino_wm.patch"
cp -R "$HERE/overlay/." "$TARGET/"
echo "DINO-WM ready in $TARGET (commit $COMMIT + dino_wm.patch + overlay)"
