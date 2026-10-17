#!/usr/bin/env bash
# SSH-safe R2 resume: smoke → collect → fit → learned ACC.
# Logs only to stderr+file so CUDA_VISIBLE_DEVICES never captures log text.
# Smoke failures are recorded to INCIDENT_LOG.md and retried (env blips);
# hard code errors still need agent fix, but we never exit silently after one blip.
set -uo pipefail
trap '' HUP
REPO=/home/xiangyue/ViTCoP
PKG=$REPO/experiments/e020_1_r2
PY=/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python
LOG=$PKG/logs/orchestrate_e019_r2.log
INC=$PKG/INCIDENT_LOG.md
mkdir -p "$PKG/logs" "$PKG/state" "$PKG/data" "$PKG/models"
echo $$ > "$PKG/state/orchestrate_e019_r2.pid"
# CRITICAL: never write logs to stdout — command substitutions must stay clean.
log(){ echo "[$(date -Is)] $*" | tee -a "$LOG" >&2; }

record_incident() {
  local title="$1" detail="$2"
  {
    echo ""
    echo "## AUTO $(date -Is) — $title"
    echo '```'
    echo "$detail" | tail -80
    echo '```'
  } >>"$INC"
  log "INCIDENT recorded: $title"
}

pick_gpu() {
  "$PY" - <<'PY'
import subprocess
want={0,2,4,6}
text=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],text=True)
free=[]
for line in text.strip().splitlines():
    i,m=line.split(","); i=int(i); m=int(m)
    if i in want and m<2000: free.append(i)
print(free[0] if free else -1)
PY
}

GPU=$(pick_gpu)
if [[ -z "${GPU:-}" || "$GPU" -lt 0 ]]; then log "no free GPU"; exit 1; fi
# sanitize: must be a single integer
if ! [[ "$GPU" =~ ^[0-9]+$ ]]; then
  record_incident "GPU pick polluted" "GPU='$GPU'"
  exit 1
fi
log "RESUME R2 collect/fit/acc on GPU=$GPU (E019 already 144/144)"

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$PKG:$REPO:$REPO/e020_lib:$REPO/LLaVA
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

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
PY

cd "$PKG"
log "SMOKE states=8 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
"$PY" -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'dev0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)" | tee -a "$LOG" >&2

SMOKE_OK=0
for attempt in 1 2 3; do
  : > "$PKG/logs/collect_smoke.log"
  log "SMOKE attempt $attempt/3"
  set +e
  "$PY" pairdata.py --adapter replay_adapter:make --config "$CFG" \
    --out "$PKG/data/e0201r2_smoke.npz" --states 8 --actions 16 \
    --candidate-limit 16 --shortlist 16 --group-sizes 1 --noop-every 8 \
    >>"$PKG/logs/collect_smoke.log" 2>&1
  rc=$?
  set -e
  if [[ $rc -eq 0 && -f $PKG/data/e0201r2_smoke.npz ]]; then
    SMOKE_OK=1
    break
  fi
  detail=$(tr '\r' '\n' < "$PKG/logs/collect_smoke.log" | grep -vE 'Loading checkpoint|%\|' | tail -60)
  record_incident "SMOKE attempt $attempt rc=$rc" "$detail"
  # env-ish retries only; code AttributeError/ValueError still retry once after short sleep
  # in case of transient CUDA init — agent should fix code between attempts if same traceback.
  sleep 15
done

if [[ $SMOKE_OK -ne 1 ]]; then
  log "SMOKE FAILED after 3 attempts — leaving state for agent; do not mark complete"
  echo "smoke_failed $(date -Is)" > "$PKG/state/smoke_failed.txt"
  exit 1
fi

rm -f "$PKG/state/smoke_failed.txt"
echo "ok $(date -Is)" > "$PKG/state/smoke_ok.txt"
log "SMOKE OK → full collect"
set -e
"$PY" pairdata.py --adapter replay_adapter:make --config "$CFG" \
  --out "$PKG/data/e0201r2_pairs.npz" --states 256 --actions 16 \
  --candidate-limit 16 --shortlist 16 --group-sizes 1 --noop-every 8 \
  >>"$PKG/logs/collect_full.log" 2>&1
log "FIT"
"$PY" fit_models.py "$PKG/data/e0201r2_pairs.npz" --out "$PKG/models" \
  --modes current,no_context,shuffled,full >>"$PKG/logs/fit.log" 2>&1
mkdir -p "$REPO/experiments/e020_continuation/models"
for m in current no_context shuffled full; do
  [[ -f $PKG/models/$m.npz ]] && ln -sfn "$PKG/models/$m.npz" "$REPO/experiments/e020_continuation/models/$m.npz"
done
echo "collect_fit_ok $(date -Is)" > "$PKG/state/collect_fit_ok.txt"
log "LAUNCH learned ACC"
for g in 0 2 4 6; do echo $$ > "$REPO/experiments/e014_latency_fair_256/state/snatch/worker_gpu${g}.pid" 2>/dev/null || true; done
"$PY" - <<'PY'
from pathlib import Path
import shutil, sys
sys.path.insert(0,"/home/xiangyue/ViTCoP")
from e010_common import sha256
REPO=Path("/home/xiangyue/ViTCoP"); RUNTIME=REPO/"logs/e014_aee/code_runtime"
for name in ["LLaVA/llava/model/e018_stage2_support.py"]:
  src,dst=REPO/name, RUNTIME/name
  if src.exists() and (not dst.exists() or sha256(dst)!=sha256(src)):
    dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src,dst)
lib_dst=RUNTIME/"e020_lib"; lib_dst.mkdir(parents=True, exist_ok=True)
for p in (REPO/"e020_lib").glob("*.py"):
  d=lib_dst/p.name
  if not d.exists() or sha256(d)!=sha256(p): shutil.copy2(p,d)
PY
setsid "$PY" "$REPO/e014_queue.py" --phase full_e020_learned --gpus 0,2,4,6 --mem-free-mib 2000 \
  </dev/null >>"$PKG/logs/learned_acc_queue.log" 2>&1 &
echo $! > "$PKG/state/learned_acc.pid"
log "learned_acc pid=$(cat $PKG/state/learned_acc.pid)"
while kill -0 "$(cat $PKG/state/learned_acc.pid)" 2>/dev/null; do sleep 120; done
"$PY" "$REPO/experiments/e020_continuation/scripts/analyze_hypothesis.py" >"$PKG/HYPOTHESIS_AFTER_R2.txt" 2>&1 || true
echo "orchestrate_complete $(date -Is)" > "$PKG/state/orchestrate_complete.txt"
log "ALL DONE"
