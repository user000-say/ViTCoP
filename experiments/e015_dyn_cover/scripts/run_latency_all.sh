#!/usr/bin/env bash
# Sequential fair-256 TTFT on one GPU for E015 dyn-cover arms (+ BASE/COVER/AEE refs if missing).
set -euo pipefail
REPO=/home/xiangyue/ViTCoP
PKG=$REPO/experiments/e015_dyn_cover
GPU="${1:?gpu}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO:$REPO/LLaVA:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
BENCH=$REPO/experiments/e014_latency_fair_256
PY=$BENCH/bin/python
PANEL=$BENCH/panel/speed_256.jsonl
LOG=$PKG/logs/latency_gpu${GPU}.log
PYENV=/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python
mkdir -p "$PKG/logs" "$PKG/latency/runs/gpu${GPU}"

ARMS=(
  BASE_K96
  FIXED_COVER_B16_K96
  FIXED_COVER_B32_K96
  FIXED_RISK_B16_K96
  AEE_K96
  E015_COVER_FIXED_B16_K96
  E015_COVER_FIXED_B32_K96
  E015_AEE_FIXED_B16_K96
  E015_AEE_FIXED_B32_K96
  E015_GLOBAL_COVER_FIXED_B16_K96
  E015_GLOBAL_AEE_FIXED_B16_K96
  E015_COVER_DYNAMIC_K96
  E015_AEE_DYNAMIC_K96
  E015_GLOBAL_COVER_DYNAMIC_K96
  E015_GLOBAL_AEE_DYNAMIC_K96
)

{
  echo "START e015 dyn-cover latency gpu=$GPU $(date -Is)"
  for ARM in "${ARMS[@]}"; do
    OUT=$PKG/latency/runs/gpu${GPU}/$ARM
    if [[ -f $OUT/latency_summary.json ]]; then
      echo "SKIP $ARM already complete"
      continue
    fi
    rm -rf "$OUT"
    mkdir -p "$OUT"
    # shellcheck disable=SC1091
    source "$PKG/scripts/e015_env.sh"
    export E014_AUDIT=False
    ENVFILE=$OUT/arm_env.sh
    "$PYENV" - <<PY
import os, sys
sys.path.insert(0, "$REPO")
from e014_arms import apply_arm_env
meta = apply_arm_env("$ARM")
keys = sorted(k for k in os.environ if k.startswith(("E0","USE_","STAGE","COMPUTE")))
with open("$ENVFILE","w") as f:
    for k in keys:
        f.write(f"export {k}={os.environ[k]!r}\n")
    f.write(f"export E014_ARM={meta['arm']!r}\n")
print("wrote", "$ENVFILE", "arm", meta["arm"])
PY
    # shellcheck disable=SC1090
    source "$ENVFILE"
    echo "=== BENCH $ARM $(date -Is) ==="
    "$PY" "$BENCH/scripts/benchmark_llava_e2e_anygpu.py" \
      --repo-root "$REPO" \
      --panel "$PANEL" \
      --model-path "$BENCH/models_llava" \
      --conv-mode llava_v1 \
      --out-dir "$OUT" \
      --warmups 8 \
      --limit 0
  done
  echo "DONE e015 dyn-cover latency $(date -Is)"
} >"$LOG" 2>&1
