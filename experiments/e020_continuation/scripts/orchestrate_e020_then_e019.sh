#!/usr/bin/env bash
# SSH-safe chain: finish E020 priority → (optional learned) → E019 B/C → verdict.
set -euo pipefail
REPO=/home/xiangyue/ViTCoP
E020=$REPO/experiments/e020_continuation
E019=$REPO/experiments/e019_progressive_evidence
PY=/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python
LOG=$E020/logs/orchestrate.log
mkdir -p "$E020/logs" "$E020/state" "$E019/logs" "$E019/state"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

wait_pid() {
  local pid=$1 name=$2
  if [[ -z "${pid:-}" ]]; then return 0; fi
  if ! kill -0 "$pid" 2>/dev/null; then return 0; fi
  log "wait $name pid=$pid"
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
  log "done $name"
}

sync_runtime() {
  "$PY" - <<'PY'
from pathlib import Path
import shutil, sys
sys.path.insert(0, "/home/xiangyue/ViTCoP")
from e010_common import sha256
REPO = Path("/home/xiangyue/ViTCoP")
RUNTIME = REPO / "logs/e014_aee/code_runtime"
files = [
    "LLaVA/llava/model/e018_stage2_support.py",
    "LLaVA/llava/model/e019_evidence_torch.py",
    "LLaVA/llava/model/e019_selector_hook.py",
]
for name in files:
    src, dst = REPO / name, RUNTIME / name
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if (not dst.exists()) or sha256(dst) != sha256(src):
            shutil.copy2(src, dst)
            print("synced", name)
# e020_lib
import pathlib
lib_src = REPO / "e020_lib"
lib_dst = RUNTIME / "e020_lib"
lib_dst.mkdir(parents=True, exist_ok=True)
for p in lib_src.glob("*.py"):
    d = lib_dst / p.name
    if (not d.exists()) or sha256(d) != sha256(p):
        shutil.copy2(p, d)
        print("synced e020_lib", p.name)
PY
}

launch_queue() {
  local phase=$1 pidfile=$2 logfile=$3
  sync_runtime
  for g in 0 2 4 6; do
    echo $$ > "$REPO/experiments/e014_latency_fair_256/state/snatch/worker_gpu${g}.pid" 2>/dev/null || true
  done
  setsid "$PY" "$REPO/e014_queue.py" \
    --phase "$phase" --gpus 0,2,4,6 --mem-free-mib 2000 \
    </dev/null >>"$logfile" 2>&1 &
  echo $! > "$pidfile"
  log "LAUNCH phase=$phase pid=$(cat "$pidfile") log=$logfile"
}

log "ORCHESTRATE start"

# --- Stage 1: E020 priority (A+D) ---
if [[ -f $E020/state/acc_queue.pid ]] && kill -0 "$(cat $E020/state/acc_queue.pid)" 2>/dev/null; then
  wait_pid "$(cat $E020/state/acc_queue.pid)" e020_priority
else
  # resume if incomplete
  done=$("$PY" - <<'PY'
from pathlib import Path
import e014_arms as m
root=Path("/home/xiangyue/ViTCoP/logs/e014_aee/full/full_e020_priority")
tasks=["gqa","pope","mme","mmbench_en_dev"]
print(sum(1 for a in m.FULL_E020_PRIORITY for t in tasks if (root/a/t/"metrics.json").exists()))
PY
)
  if [[ "$done" -lt 104 ]]; then
    log "E020 priority incomplete ($done/104); launching"
    launch_queue full_e020_priority "$E020/state/acc_queue.pid" "$E020/logs/acc_queue.log"
    wait_pid "$(cat $E020/state/acc_queue.pid)" e020_priority
  else
    log "E020 priority already complete ($done/104)"
  fi
fi

"$PY" "$E020/scripts/analyze_hypothesis.py" >>"$LOG" 2>&1 || true

# --- Stage 2a: pairdata + fit (required for learned B/C) ---
MODELS=$E020/models
if [[ ! -f $MODELS/full.npz || ! -f $MODELS/current.npz ]]; then
  if [[ -x $E020/scripts/collect_and_fit.sh ]]; then
    log "RUN collect_and_fit.sh (may take many hours on free GPUs)"
    # Prefer GPU 4 if free after priority; script picks a free GPU.
    bash "$E020/scripts/collect_and_fit.sh" >>"$E020/logs/collect_fit.log" 2>&1 \
      || log "collect_and_fit failed/skipped (see collect_fit.log)"
  else
    log "no collect_and_fit.sh; learned B/C will be skipped"
  fi
fi

# --- Stage 2b: learned B/C if models present ---
if [[ -f $MODELS/full.npz && -f $MODELS/current.npz ]]; then
  log "fitted models present → launch full_e020_learned"
  launch_queue full_e020_learned "$E020/state/learned_queue.pid" "$E020/logs/learned_queue.log"
  wait_pid "$(cat $E020/state/learned_queue.pid)" e020_learned
else
  log "SKIP learned B/C ACC: no $MODELS/{full,current}.npz (pairdata/fit not done)"
  echo "learned_skipped_no_models $(date -Is)" > "$E020/state/learned_skipped.txt"
fi

"$PY" "$E020/scripts/analyze_hypothesis.py" >>"$LOG" 2>&1 || true

# --- Stage 3: E019 B/C remaining ---
rm -f "$E019/state/paused_for_e020.txt"
# stop any stale remaining waiter claims on GPUs after E020
if [[ -f $E019/state/acc_remaining.pid ]] && kill -0 "$(cat $E019/state/acc_remaining.pid)" 2>/dev/null; then
  wait_pid "$(cat $E019/state/acc_remaining.pid)" e019_remaining
else
  done19=$("$PY" - <<'PY'
from pathlib import Path
from e014_queue import PHASE_ARMS
root=Path("/home/xiangyue/ViTCoP/logs/e014_aee/full/full_e019_remaining")
arms=PHASE_ARMS["full_e019_remaining"]
tasks=["gqa","pope","mme","mmbench_en_dev"]
print(sum(1 for a in arms for t in tasks if (root/a/t/"metrics.json").exists()))
print(len(arms)*4)
PY
)
  done_n=$(echo "$done19" | head -1)
  tot_n=$(echo "$done19" | tail -1)
  if [[ "$done_n" -lt "$tot_n" ]]; then
    log "E019 remaining incomplete ($done_n/$tot_n); launching"
    bash "$E019/scripts/campaign_acc_remaining.sh"
    wait_pid "$(cat $E019/state/acc_remaining.pid)" e019_remaining
  else
    log "E019 remaining already complete ($done_n/$tot_n)"
  fi
fi

"$PY" "$E020/scripts/analyze_hypothesis.py" | tee "$E020/HYPOTHESIS_VERDICT_LATEST.txt" | tee -a "$LOG"
log "ORCHESTRATE complete → $E020/HYPOTHESIS_VERDICT.md"
