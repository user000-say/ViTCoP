#!/usr/bin/env bash
# Unlabeled 128-image probe → freeze E015_THRESHOLDS. Not ACC smoke.
set -euo pipefail
REPO=/home/xiangyue/ViTCoP
PKG=$REPO/experiments/e015_dyn_cover
GPU="${1:?gpu}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO:$REPO/LLaVA:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
source "$PKG/scripts/e015_env.sh"
export E014_VARIANT=E015_PROBE
export E014_AUDIT=True
export E014_B=0
unset E015_THRESHOLDS || true

OUT=$PKG/probe/gpu${GPU}
rm -rf "$OUT"
mkdir -p "$OUT" "$PKG/logs" "$PKG/state"
LOG=$PKG/logs/probe_gpu${GPU}.log
BENCH=$REPO/experiments/e014_latency_fair_256
PY=$BENCH/bin/python
PANEL=$BENCH/panel/speed_256.jsonl
CAL_PY=/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python

{
  echo "START probe calib gpu=$GPU $(date -Is)"
  "$PY" "$BENCH/scripts/benchmark_llava_e2e_anygpu.py" \
    --repo-root "$REPO" \
    --panel "$PANEL" \
    --model-path "$BENCH/models_llava" \
    --conv-mode llava_v1 \
    --out-dir "$OUT" \
    --warmups 2 \
    --limit 128
  echo "DONE probe $(date -Is)"
} >"$LOG" 2>&1

"$CAL_PY" "$PKG/scripts/calibrate_e015.py" "$LOG" \
  --out "$PKG/state/e015_thresholds.json" --min-images 128 \
  | tee "$PKG/logs/calibrate.txt"
echo "THRESHOLDS frozen at $PKG/state/e015_thresholds.json"
