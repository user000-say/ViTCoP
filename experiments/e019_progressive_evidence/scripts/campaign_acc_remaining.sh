#!/usr/bin/env bash
# After Block-A schedule-ref ACC (+ TTFT wave), run remaining 36 configs × 4 tasks.
set -euo pipefail
REPO=/home/xiangyue/ViTCoP
PKG=$REPO/experiments/e019_progressive_evidence
PY=/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python
mkdir -p "$PKG/logs" "$PKG/state"

# Wait for in-flight TTFT runner if present.
if [[ -f $PKG/state/ttft_runner.pid ]]; then
  pid=$(cat "$PKG/state/ttft_runner.pid")
  if kill -0 "$pid" 2>/dev/null; then
    echo "[$(date -Is)] wait TTFT runner pid=$pid" | tee -a "$PKG/logs/campaign.log"
    while kill -0 "$pid" 2>/dev/null; do sleep 60; done
  fi
fi

for g in 0 2 4 6; do echo $$ > "$REPO/experiments/e014_latency_fair_256/state/snatch/worker_gpu${g}.pid"; done
"$PY" - <<'PY'
from pathlib import Path
import shutil, sys
sys.path.insert(0, "/home/xiangyue/ViTCoP")
from e010_common import sha256
REPO = Path("/home/xiangyue/ViTCoP")
RUNTIME = REPO / "logs/e014_aee/code_runtime"
files = [
    "LLaVA/llava/model/language_model/vitcop_modeling_llama.py",
    "LLaVA/llava/model/multimodal_encoder/clip_encoder.py",
    "LLaVA/llava/model/e018_contract.py",
    "LLaVA/llava/model/e018_stage2_support.py",
    "LLaVA/llava/model/e019_evidence_torch.py",
    "LLaVA/llava/model/e019_selector_hook.py",
    "LLaVA/llava/model/e018_counterfactual.py",
    "LLaVA/llava/model/e018_qed_rs_diagnostic.py",
]
for name in files:
    src, dst = REPO / name, RUNTIME / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists() and (not dst.exists() or sha256(dst) != sha256(src)):
        shutil.copy2(src, dst)
        print("synced", name)
PY

echo "[$(date -Is)] LAUNCH full_e019_remaining (36 configs / 144 ACC jobs) gpus=0,2,4,6" | tee -a "$PKG/logs/campaign.log"
setsid "$PY" "$REPO/e014_queue.py" \
  --phase full_e019_remaining --gpus 0,2,4,6 --mem-free-mib 2000 \
  </dev/null >>"$PKG/logs/acc_remaining.log" 2>&1 &
echo $! > "$PKG/state/acc_remaining.pid"
echo "[$(date -Is)] acc_remaining_pid=$(cat $PKG/state/acc_remaining.pid)" | tee -a "$PKG/logs/campaign.log"
