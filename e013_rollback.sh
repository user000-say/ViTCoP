#!/usr/bin/env bash
# Restore pre-E013 backup directory.
set -euo pipefail
BK="${1:-}"
if [[ -z "$BK" || ! -d "$BK" ]]; then
  echo "Usage: bash e013_rollback.sh <backup_dir>" >&2
  exit 2
fi
ROOT=/home/xiangyue/ViTCoP
cp -a "$BK/clip_encoder.py" "$ROOT/LLaVA/llava/model/multimodal_encoder/clip_encoder.py"
cp -a "$BK/vitcop_modeling_llama.py" "$ROOT/LLaVA/llava/model/language_model/vitcop_modeling_llama.py"
rm -f "$ROOT/LLaVA/llava/model/multimodal_encoder/e013_stage1_support.py"
echo "rolled back from $BK"
