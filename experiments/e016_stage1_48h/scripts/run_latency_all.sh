#!/usr/bin/env bash
set -euo pipefail
REPO=/home/xiangyue/ViTCoP
PKG=$REPO/experiments/e016_stage1_48h
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
  FIXED_COVER_B32_K96
  E015_GLOBAL_AEE_FIXED_B16_K96
  E016_MARGINAL_B32_K96
  E016_PROTECTED_B32_K96
  E016_SPATIAL_B32_K96
  E016_HIER_B32_K96
  E016_CENTERED_B32_K96
)
{
  echo "START e016 stage1 latency gpu=$GPU $(date -Is)"
  for ARM in "${ARMS[@]}"; do
    OUT=$PKG/latency/runs/gpu${GPU}/$ARM
    if [[ -f $OUT/latency_summary.json ]]; then
      echo "SKIP $ARM"; continue
    fi
    # reuse prior fair256 TTFT if identical arm already measured elsewhere for BASE/COVER32
    PREV=$REPO/experiments/e015_dyn_cover/latency/runs/gpu4/$ARM/latency_summary.json
    if [[ -f $PREV && "$ARM" != E016_* ]]; then
      mkdir -p "$OUT"; cp -a "$PREV" "$OUT/"; echo "COPY $ARM from e015_dyn_cover"; continue
    fi
    rm -rf "$OUT"; mkdir -p "$OUT"
    source "$PKG/scripts/e016_env.sh"
    export E014_AUDIT=False E016_AUDIT_COVERAGE=False
    ENVFILE=$OUT/arm_env.sh
    "$PYENV" - <<PY
import os,sys
sys.path.insert(0,"$REPO")
from e014_arms import apply_arm_env
meta=apply_arm_env("$ARM")
keys=sorted(k for k in os.environ if k.startswith(("E0","USE_","STAGE","COMPUTE")))
with open("$ENVFILE","w") as f:
  for k in keys: f.write(f"export {k}={os.environ[k]!r}\n")
  f.write(f"export E014_ARM={meta['arm']!r}\n")
print("arm", meta["arm"])
PY
    source "$ENVFILE"
    echo "=== BENCH $ARM $(date -Is) ==="
    "$PY" "$BENCH/scripts/benchmark_llava_e2e_anygpu.py" \
      --repo-root "$REPO" --panel "$PANEL" \
      --model-path "$BENCH/models_llava" --conv-mode llava_v1 \
      --out-dir "$OUT" --warmups 8 --limit 0
  done
  echo "DONE e016 stage1 latency $(date -Is)"
} >"$LOG" 2>&1
