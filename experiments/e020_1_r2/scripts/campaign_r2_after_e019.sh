#!/usr/bin/env bash
# Wait for E019 remaining (do not kill), then R2 smoke → collect → fit → note for core ACC.
set -euo pipefail
REPO=/home/xiangyue/ViTCoP
PKG=$REPO/experiments/e020_1_r2
E019=$REPO/experiments/e019_progressive_evidence
PY=/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python
LOG=$PKG/logs/campaign_r2.log
mkdir -p "$PKG/logs" "$PKG/state" "$PKG/data" "$PKG/models"

log(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

log "R2 campaign start; wait E019 remaining if running"
if [[ -f $E019/state/acc_remaining.pid ]]; then
  pid=$(cat "$E019/state/acc_remaining.pid")
  if kill -0 "$pid" 2>/dev/null; then
    log "wait e019 pid=$pid"
    while kill -0 "$pid" 2>/dev/null; do sleep 60; done
    log "e019 remaining finished"
  fi
fi

# pick free GPU
GPU=$("$PY" - <<'PY'
import subprocess
want={0,2,4,6}
text=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],text=True)
for line in text.strip().splitlines():
    i,m=line.split(","); i=int(i); m=int(m)
    if i in want and m<2000:
        print(i); break
else:
    print(4)
PY
)
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$PKG:$REPO:$REPO/e020_lib:$REPO/LLaVA
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
log "using physical GPU=$GPU as cuda:0"

CFG=$PKG/data/replay_config.json
"$PY" - <<PY
import json
from pathlib import Path
rt=json.loads(Path("$PKG/runtime.json").read_text())
cfg={
  "backend":"project_e020_backend:make",
  "source":rt["source"],
  "schedules":["FIRST_SHIFT","FIRST_NEXT_WMATCH"],
  "arms":["COVER","BASE"],
  "max_examples":128,
  "backend_config":{
    "qa_jsonl":"$REPO/logs/sdmc_13b_r1/20260927_1345/manifest/calibration_train_qa.jsonl",
    "model":"liuhaotian/llava-v1.5-7b",
    "device":"cuda:0",
  },
}
Path("$CFG").write_text(json.dumps(cfg,indent=2))
print("wrote", "$CFG")
PY

# smoke 8 states
SMOKE_OUT=$PKG/data/e0201r2_smoke.npz
log "SMOKE pairdata states=8 actions=16"
cd "$PKG"
set +e
"$PY" pairdata.py --adapter replay_adapter:make --config "$CFG" \
  --out "$SMOKE_OUT" --states 8 --actions 16 \
  --candidate-limit 16 --shortlist 16 --group-sizes 1 --noop-every 8 \
  >>"$PKG/logs/collect_smoke.log" 2>&1
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  log "SMOKE FAILED rc=$rc; see logs/collect_smoke.log — do not claim backend integrated"
  echo "smoke_failed $(date -Is)" > "$PKG/state/smoke_failed.txt"
  exit $rc
fi
log "SMOKE OK → full collect states=256"
OUT=$PKG/data/e0201r2_pairs.npz
"$PY" pairdata.py --adapter replay_adapter:make --config "$CFG" \
  --out "$OUT" --states 256 --actions 16 \
  --candidate-limit 16 --shortlist 16 --group-sizes 1 --noop-every 8 \
  >>"$PKG/logs/collect_full.log" 2>&1
log "FIT models"
"$PY" fit_models.py "$OUT" --out "$PKG/models" --modes current,no_context,shuffled,full \
  >>"$PKG/logs/fit.log" 2>&1
log "diagnostics"
"$PY" diagnostics.py "$OUT" --models "$PKG/models" --group-size 1 --candidate-limit 16 \
  --out "$PKG/data/e0201r2_diagnostics.json" >>"$PKG/logs/diagnostics.log" 2>&1 || true
log "R2 collect+fit complete; core ACC via close_core / e014 queue next"
echo "collect_fit_ok $(date -Is)" > "$PKG/state/collect_fit_ok.txt"
