#!/usr/bin/env bash
# E020.3 TTFT on GPU6 only: SHIFT/K108/K100 hard-delete + first-merge anchors.
set -euo pipefail
REPO=/home/xiangyue/ViTCoP
PKG=$REPO/experiments/e020_3
BENCH=$REPO/experiments/e014_latency_fair_256
PYBENCH=$BENCH/bin/python
PYENV=/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python
GPU=6
mkdir -p "$PKG/logs" "$PKG/latency/runs" "$PKG/state"

# Resolve stable matrix arms for D hard-delete / first-merge on each profile.
mapfile -t ARMS < <("$PYENV" - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "/home/xiangyue/ViTCoP")
sys.path.insert(0, "/home/xiangyue/ViTCoP/experiments")
from e020_3.apply_matrix_arm import BY_ID, arm_name, MATRIX
want = []
# A: COVER SHIFT RDRIFT off-path anchor
for c in MATRIX["configs"]:
    if (c["block"] == "A" and c["entry"] == "COVER" and c["profile_alias"] == "SHIFT"
            and c["first_selector"] == "RDRIFT"):
        want.append(arm_name(c))
        break
# D: each profile hard-delete then first-merge (alpha=0.5)
for profile in ("SHIFT_K108", "SHIFT_K100", "FINAL11", "FINAL15_WMATCH"):
    hard = first = None
    for c in MATRIX["configs"]:
        if c["block"] != "D" or c["profile_alias"] != profile:
            continue
        ev = tuple(c["operator"].get("events") or [])
        if ev == () and hard is None:
            hard = arm_name(c)
        if ev == ("first",) and first is None:
            first = arm_name(c)
    if hard:
        want.append(hard)
    if first:
        want.append(first)
print("\n".join(want))
PY
)

run_arm() {
  local ARM=$1
  local OUT=$PKG/latency/runs/gpu${GPU}/$ARM
  local LOG=$PKG/logs/ttft_${ARM}_gpu${GPU}.log
  if [[ -f $OUT/latency_summary.json ]]; then
    echo "SKIP $ARM"
    return 0
  fi
  rm -rf "$OUT"
  mkdir -p "$OUT"
  local ENVFILE=$OUT/arm_env.sh
  "$PYENV" - <<PY
import os, sys
sys.path.insert(0, "$REPO")
sys.path.insert(0, "$REPO/experiments")
from e014_arms import apply_arm_env
meta = apply_arm_env("$ARM")
keys = sorted(k for k in os.environ if k.startswith(("E0", "USE_", "STAGE", "COMPUTE")))
with open("$ENVFILE", "w") as f:
    for k in keys:
        f.write(f"export {k}={os.environ[k]!r}\n")
    f.write(f"export E014_ARM={meta['arm']!r}\n")
    f.write("export E018_TRACE=''\n")
print("arm", meta["arm"], "events", os.environ.get("E018_EVENTS"),
      "e0203", os.environ.get("E020_3_ENABLE"), os.environ.get("E020_3_BLOCK"))
PY
  echo "START $ARM gpu=$GPU $(date -Is)" | tee -a "$PKG/logs/ttft_campaign.log"
  (
    export CUDA_VISIBLE_DEVICES=$GPU
    export PYTHONPATH="$REPO:$REPO/experiments:$REPO/LLaVA:${PYTHONPATH:-}"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    # shellcheck disable=SC1090
    source "$ENVFILE"
    "$PYBENCH" "$BENCH/scripts/benchmark_llava_e2e_anygpu.py" \
      --repo-root "$REPO" --panel "$BENCH/panel/speed_256.jsonl" \
      --model-path "$BENCH/models_llava" --conv-mode llava_v1 \
      --out-dir "$OUT" --warmups 8 --limit 0
    echo "DONE $ARM $(date -Is)"
  ) >"$LOG" 2>&1
  echo "DONE $ARM $(date -Is)" | tee -a "$PKG/logs/ttft_campaign.log"
}

echo "[$(date -Is)] TTFT queue arms=${#ARMS[@]} gpu=$GPU" | tee -a "$PKG/logs/ttft_campaign.log"
(
  for ARM in "${ARMS[@]}"; do
    run_arm "$ARM"
  done
  echo "[$(date -Is)] TTFT anchors COMPLETE" | tee -a "$PKG/logs/ttft_campaign.log"
) </dev/null >>"$PKG/logs/ttft_runner.outer.log" 2>&1 &
echo $! > "$PKG/state/ttft_runner.pid"
echo "[$(date -Is)] ttft_runner_pid=$(cat "$PKG/state/ttft_runner.pid")" | tee -a "$PKG/logs/ttft_campaign.log"
