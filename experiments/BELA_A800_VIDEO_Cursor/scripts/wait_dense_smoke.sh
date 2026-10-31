#!/usr/bin/env bash
# Start the official Dense smoke on the first free card among physical GPUs 0 and 1.
set -euo pipefail
ROOT=/data1/xiangyue
LOG=$ROOT/logs/wait_dense_smoke.log
OUT=$ROOT/logs/VIDEO_DENSE_SMOKE.json
PY=$ROOT/envs/vitcop-a800-video/bin/python
mkdir -p "$ROOT/logs" "$ROOT/tmp" "$ROOT/hf" "$ROOT/cache" "$ROOT/torch"
if [[ -f $OUT ]]; then
  echo "ALREADY $OUT" | tee -a "$LOG"
  exit 0
fi
export PYTHONPATH=$ROOT/src/LLaVA-NeXT
export BELA_VIDEO_MODEL=$ROOT/models/LLaVA-Video-7B-Qwen2
export BELA_SMOKE_VIDEO=$ROOT/videos/smoke_color.mp4
export BELA_SMOKE_OUT=$OUT
export HF_HOME=$ROOT/hf
export HUGGINGFACE_HUB_CACHE=$ROOT/hf/hub
export TRANSFORMERS_CACHE=$ROOT/hf/transformers
export XDG_CACHE_HOME=$ROOT/cache
export TMPDIR=$ROOT/tmp
export TORCH_HOME=$ROOT/torch
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
while true; do
  for g in 0 1; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
    if [[ ${used:-99999} -lt 800 ]]; then
      echo "START gpu=$g used=${used} $(date -Is)" | tee -a "$LOG"
      export CUDA_VISIBLE_DEVICES=$g
      "$PY" "$ROOT/ViTCoP/experiments/BELA_A800_VIDEO_Cursor/scripts/dense_official_smoke.py" \
        > "$ROOT/logs/dense_official_smoke.log" 2>&1
      echo "SMOKE_EXIT $? $(date -Is)" | tee -a "$LOG"
      exit 0
    fi
  done
  echo "WAIT gpu0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ') gpu1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ') $(date -Is)" >> "$LOG"
  sleep 30
done
