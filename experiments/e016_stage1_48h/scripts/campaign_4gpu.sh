#!/usr/bin/env bash
# Skip smoke. Full ACC on 0,2,6 + TTFT on 4.
set -euo pipefail
REPO=/home/xiangyue/ViTCoP
PKG=$REPO/experiments/e016_stage1_48h
PY=/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python
mkdir -p "$PKG/logs" "$PKG/state"
log(){ echo "[$(date -Is)] $*" | tee -a "$PKG/logs/campaign.log"; }

for g in 0 2 4 6; do echo $$ > $REPO/experiments/e014_latency_fair_256/state/snatch/worker_gpu${g}.pid; done

"$PY" - <<'PY'
from pathlib import Path
import shutil,sys
sys.path.insert(0,"/home/xiangyue/ViTCoP")
from e010_common import sha256
REPO=Path("/home/xiangyue/ViTCoP"); RUNTIME=REPO/"logs/e014_aee/code_runtime"
files=[
 "LLaVA/llava/model/language_model/vitcop_modeling_llama.py",
 "LLaVA/llava/model/multimodal_encoder/clip_encoder.py",
 "LLaVA/llava/model/multimodal_encoder/e014_config.py",
 "LLaVA/llava/model/multimodal_encoder/e014_stage1_support.py",
 "LLaVA/llava/model/multimodal_encoder/e015_stage1_support.py",
 "LLaVA/llava/model/multimodal_encoder/e015_regional_config.py",
 "LLaVA/llava/model/multimodal_encoder/e015_regional_support.py",
 "LLaVA/llava/model/multimodal_encoder/e016_stage1_support.py",
]
for name in files:
  src,dst=REPO/name, RUNTIME/name
  dst.parent.mkdir(parents=True, exist_ok=True)
  if src.exists() and (not dst.exists() or sha256(dst)!=sha256(src)):
    shutil.copy2(src,dst); print("synced", name)
PY

log "LATENCY start gpu4"
setsid bash "$PKG/scripts/run_latency_all.sh" 4 </dev/null &>/dev/null &
echo $! > "$PKG/state/latency.pid"
log "latency_pid=$(cat $PKG/state/latency.pid)"

log "LAUNCH_ACC full_e016_stage1_k96 gpus=0,2,6"
setsid "$PY" "$REPO/e014_queue.py" \
  --phase full_e016_stage1_k96 --gpus 0,2,6 --mem-free-mib 2000 \
  </dev/null >>"$PKG/logs/acc_queue.log" 2>&1 &
echo $! > "$PKG/state/acc_queue.pid"
log "acc_queue_pid=$(cat $PKG/state/acc_queue.pid)"
log "campaign launched (skip smoke)"
