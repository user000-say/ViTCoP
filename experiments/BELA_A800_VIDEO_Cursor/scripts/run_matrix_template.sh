#!/usr/bin/env bash
set -euo pipefail
# This script only launches the REAL inference program after Cursor creates it.
# It deliberately fails closed; it does not pretend to be the LLaVA-Video model.
: "${BELA_VIDEO_RUNNER:?Cursor must create and set BELA_VIDEO_RUNNER first}"
: "${BELA_RUN_ROOT:?Choose a non-secret run output directory}"
[[ -f "$BELA_VIDEO_RUNNER" ]] || { echo "Runner does not exist"; exit 1; }
[[ $# -eq 4 ]] || {
  echo "Usage: $0 GPU_ID (DENSE|T32|T16) (DENSE|BELA_VIDEO_NATIVE|BELA_VIDEO_QACOVER) (Video-MME|LongVideoBench|MLVU|MVBench)"
  exit 2
}
GPU="$1"
TIER="$2"
ARM="$3"
DATASET="$4"
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "GPU_ID must be numeric"; exit 2; }
case "$TIER" in DENSE|T32|T16) ;; *) echo "invalid tier";exit 2;; esac
case "$ARM" in DENSE|BELA_VIDEO_NATIVE|BELA_VIDEO_QACOVER) ;; *) echo "invalid arm";exit 2;; esac
case "$DATASET" in Video-MME|LongVideoBench|MLVU|MVBench) ;; *) echo "invalid dataset";exit 2;; esac
if [[ "$TIER" == "DENSE" && "$ARM" != "DENSE" ]] || [[ "$TIER" != "DENSE" && "$ARM" == "DENSE" ]]; then
  echo "Dense and pruned configuration tags must match"
  exit 2
fi
OUT="${BELA_RUN_ROOT}/${DATASET}/${TIER}/${ARM}"
mkdir -p "$OUT"
echo "[BELA] GPU=${GPU} dataset=${DATASET} tier=${TIER} arm=${ARM}"
# No credentials are passed, and no remote SSH address appears in this script.
export CUDA_VISIBLE_DEVICES="$GPU"
exec python3 "$BELA_VIDEO_RUNNER" \
    --model-id lmms-lab/LLaVA-Video-7B-Qwen2 \
    --dataset "$DATASET" --tier "$TIER" --arm "$ARM" \
    --max-frames 64 --output-dir "$OUT"
