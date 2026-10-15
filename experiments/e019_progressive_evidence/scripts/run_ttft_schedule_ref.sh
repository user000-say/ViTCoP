#!/usr/bin/env bash
set -euo pipefail
REPO=/home/xiangyue/ViTCoP
PKG=$REPO/experiments/e019_progressive_evidence
BENCH=$REPO/experiments/e014_latency_fair_256
PYBENCH=$BENCH/bin/python
PYENV=/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python
mkdir -p "$PKG/logs" "$PKG/latency/runs" "$PKG/state"

run_arm() {
  local ARM=$1 GPU=$2
  local OUT=$PKG/latency/runs/gpu${GPU}/$ARM
  local LOG=$PKG/logs/ttft_${ARM}_gpu${GPU}.log
  if [[ -f $OUT/latency_summary.json ]]; then
    echo "SKIP $ARM"; return 0
  fi
  rm -rf "$OUT"; mkdir -p "$OUT"
  local ENVFILE=$OUT/arm_env.sh
  "$PYENV" - <<PY
import os, sys
sys.path.insert(0, "$REPO")
from e014_arms import apply_arm_env
meta = apply_arm_env("$ARM")
keys = sorted(k for k in os.environ if k.startswith(("E0", "USE_", "STAGE", "COMPUTE")))
with open("$ENVFILE", "w") as f:
    for k in keys:
        f.write(f"export {k}={os.environ[k]!r}\n")
    f.write(f"export E014_ARM={meta['arm']!r}\n")
    f.write("export E018_TRACE=''\n")
print("arm", meta["arm"], "events", os.environ.get("E018_EVENTS"), "E014", os.environ.get("E014_ENABLE"))
PY
  (
    echo "START $ARM gpu=$GPU $(date -Is)"
    export CUDA_VISIBLE_DEVICES=$GPU
    export PYTHONPATH="$REPO:$REPO/LLaVA:${PYTHONPATH:-}"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    # shellcheck disable=SC1090
    source "$ENVFILE"
    "$PYBENCH" "$BENCH/scripts/benchmark_llava_e2e_anygpu.py" \
      --repo-root "$REPO" --panel "$BENCH/panel/speed_256.jsonl" \
      --model-path "$BENCH/models_llava" --conv-mode llava_v1 \
      --out-dir "$OUT" --warmups 8 --limit 0
    echo "DONE $ARM $(date -Is)"
  ) >"$LOG" 2>&1 &
  echo $! >"$PKG/state/ttft_${ARM}.pid"
  echo "LAUNCH $ARM gpu=$GPU pid=$(cat "$PKG/state/ttft_${ARM}.pid")"
}

# wave 1
run_arm E019_COVER_LATE_WMATCH32 0
run_arm E019_BASE_LATE_WMATCH32 2
run_arm E019_COVER_FIRST_SHIFT_WMATCH32 4
run_arm E019_BASE_FIRST_SHIFT_WMATCH32 6
wait
# wave 2
run_arm E019_COVER_THREE_WMATCH32 0
run_arm E019_BASE_THREE_WMATCH32 2
wait
echo "[$(date -Is)] TTFT schedule_ref COMPLETE" | tee -a "$PKG/logs/ttft_campaign.log"
