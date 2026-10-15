#!/usr/bin/env bash
# EARLY TTFT: reuse TWO summaries; LATE TTFT measured on GPUs 0/2; BASE/COVER LATE on 0/2, remeasure EARLY on 4/6 for package completeness.
set -euo pipefail
REPO=/home/xiangyue/ViTCoP
PKG=$REPO/experiments/e018_extended_4gpu
PREV=$REPO/experiments/e018_continuation/latency/runs
BENCH=$REPO/experiments/e014_latency_fair_256
PANEL=$BENCH/panel/speed_256.jsonl
MODEL=$BENCH/models_llava
PY=$BENCH/bin/python
PYENV=/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python
mkdir -p "$PKG/logs" "$PKG/latency/runs" "$PKG/state"

# Seed EARLY TTFT from TWO (identical schedule)
python3 - <<'PY'
import json, shutil
from pathlib import Path
prev=Path("/home/xiangyue/ViTCoP/experiments/e018_continuation/latency/runs")
pkg=Path("/home/xiangyue/ViTCoP/experiments/e018_extended_4gpu/latency/runs")
pairs=[
 ("gpu2/E018_COVER_TWO32","gpu4/E018_COVER_EARLY32"),
 ("gpu6/E018_BASE_TWO32","gpu6/E018_BASE_EARLY32"),
]
for s,d in pairs:
    src,dst=prev/s, pkg/d
    if (dst/"latency_summary.json").exists():
        print("skip seed", d); continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        (dst/"SEEDED_FROM_TWO.txt").write_text(f"from {s}\n")
        print("seeded", d)
PY

declare -A ARM_GPU=(
  [E018_COVER_LATE32]=0
  [E018_BASE_LATE32]=2
  [E018_COVER_EARLY32]=4
  [E018_BASE_EARLY32]=6
)

for ARM in E018_COVER_LATE32 E018_BASE_LATE32; do
  GPU=${ARM_GPU[$ARM]}
  OUT=$PKG/latency/runs/gpu${GPU}/$ARM
  LOG=$PKG/logs/ttft_${ARM}_gpu${GPU}.log
  if [[ -f $OUT/latency_summary.json ]]; then echo "SKIP $ARM"; continue; fi
  rm -rf "$OUT"; mkdir -p "$OUT"
  ENVFILE=$OUT/arm_env.sh
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
print("arm", meta["arm"], os.environ.get("E018_EVENTS"))
PY
  (
    echo "START $ARM gpu=$GPU $(date -Is)"
    export CUDA_VISIBLE_DEVICES=$GPU
    export PYTHONPATH="$REPO:$REPO/LLaVA:${PYTHONPATH:-}"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    source "$ENVFILE"
    "$PY" "$BENCH/scripts/benchmark_llava_e2e_anygpu.py" \
      --repo-root "$REPO" --panel "$PANEL" --model-path "$MODEL" \
      --conv-mode llava_v1 --out-dir "$OUT" --warmups 8 --limit 0
    echo "DONE $ARM $(date -Is)"
  ) >"$LOG" 2>&1 &
  echo $! >"$PKG/state/ttft_${ARM}.pid"
  echo "LAUNCH $ARM gpu=$GPU pid=$(cat $PKG/state/ttft_${ARM}.pid)"
done
echo "TTFT LATE launched; EARLY seeded from TWO $(date -Is)" | tee -a "$PKG/logs/ttft_campaign.log"
