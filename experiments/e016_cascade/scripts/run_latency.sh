#!/usr/bin/env bash
# Full-VLM TTFT/Fixed32 for E016 cascade on one physical GPU (1024-panel).
set -euo pipefail
GPU="${1:?gpu}"
REPO=/home/xiangyue/ViTCoP
PKG=$REPO/experiments/e016_cascade
FAIR=$REPO/experiments/e014_latency_fair_256
PY=$FAIR/bin/python
OUT=$PKG/latency/runs/gpu${GPU}/CASCADE_L8_96_L10_64_L16_32_L20_16
LOG=$PKG/logs/latency_gpu${GPU}.log
mkdir -p "$(dirname "$OUT")" "$PKG/logs"
exec >>"$LOG" 2>&1
if [[ -f "$OUT/latency_summary.json" ]]; then
  echo "SKIP_EXISTS $(date -Is)"; exit 0
fi
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$REPO/LLaVA:${PYTHONPATH:-}"
# shellcheck disable=SC1091
source "$PKG/scripts/e016_env.sh"
rm -rf "$OUT"; mkdir -p "$OUT"
echo "START cascade latency gpu=$GPU $(date -Is)"
"$PY" "$FAIR/scripts/benchmark_llava_e2e_anygpu.py" \
  --repo-root "$REPO" \
  --panel "$FAIR/panel/speed_256.jsonl" \
  --model-path "$FAIR/models_llava" \
  --conv-mode llava_v1 \
  --out-dir "$OUT" \
  --warmups 8 --limit 0
echo "DONE $(date -Is)"
[[ -f "$OUT/latency_summary.json" ]] || exit 1
