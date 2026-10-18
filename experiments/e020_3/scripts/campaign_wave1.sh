#!/usr/bin/env bash
# E020.3 wave-1: ACC on GPUs 0/2/4; GPU6 reserved for TTFT.
set -euo pipefail
REPO=/home/xiangyue/ViTCoP
PKG=$REPO/experiments/e020_3
PY=/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python
mkdir -p "$PKG/logs" "$PKG/state"

for g in 0 2 4 6; do
  echo $$ > "$REPO/experiments/e014_latency_fair_256/state/snatch/worker_gpu${g}.pid" 2>/dev/null || true
done

"$PY" - <<'PY'
from pathlib import Path
import shutil, sys
sys.path.insert(0, "/home/xiangyue/ViTCoP")
from e010_common import sha256
REPO = Path("/home/xiangyue/ViTCoP")
RUNTIME = REPO / "logs/e014_aee/code_runtime"
files = [
    "LLaVA/llava/model/e018_contract.py",
    "LLaVA/llava/model/e018_stage2_support.py",
    "LLaVA/llava/model/language_model/vitcop_modeling_llama.py",
    "LLaVA/llava/model/multimodal_encoder/clip_encoder.py",
]
for name in files:
    src, dst = REPO / name, RUNTIME / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists() and (not dst.exists() or sha256(dst) != sha256(src)):
        shutil.copy2(src, dst)
        print("synced", name)
# Mirror LLaVA → llava for non-runtime imports.
src = REPO / "LLaVA/llava/model/e018_stage2_support.py"
dst = REPO / "llava/model/e018_stage2_support.py"
if src.exists() and (not dst.exists() or sha256(dst) != sha256(src)):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print("synced llava/model/e018_stage2_support.py")
PY

echo "[$(date -Is)] LAUNCH full_e0203_wave1 ACC gpus=0,2,4" | tee -a "$PKG/logs/campaign.log"
setsid "$PY" "$REPO/e014_queue.py" \
  --phase full_e0203_wave1 --gpus 0,2,4 --mem-free-mib 2000 \
  </dev/null >>"$PKG/logs/acc_queue.log" 2>&1 &
echo $! > "$PKG/state/acc_queue.pid"
echo "[$(date -Is)] acc_queue_pid=$(cat "$PKG/state/acc_queue.pid")" | tee -a "$PKG/logs/campaign.log"
