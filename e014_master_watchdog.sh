#!/usr/bin/env bash
# SSH-safe E014 continuum: full_k96 -> full_k64 -> full_k32 -> collect -> return.
set -uo pipefail
REPO=/home/xiangyue/ViTCoP
PY=/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python
cd "$REPO"
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1
mkdir -p logs/e014_aee/logs experiments/e014_aee
log(){ echo "[$(date '+%F %T')] $*" | tee -a logs/e014_aee/logs/master_watchdog.log; }

run_phase(){
  local phase="$1"
  log "START phase=$phase"
  while true; do
    "$PY" e014_queue.py --phase "$phase" --limit 0 --once >>"logs/e014_aee/logs/queue_${phase}.watchdog.log" 2>&1 || true
    done_n=$(find "logs/e014_aee/full/$phase" -name metrics.json 2>/dev/null | wc -l)
    expect=36
    [[ "$phase" == full_k64 || "$phase" == full_k32 ]] && expect=20
    log "phase=$phase metrics=$done_n/$expect"
    if [[ "$done_n" -ge "$expect" ]]; then
      log "COMPLETE phase=$phase"
      break
    fi
    sleep 60
  done
}

run_phase full_k96
run_phase full_k64
run_phase full_k32
"$PY" e014_full_collect.py >>logs/e014_aee/logs/master_watchdog.log 2>&1 || true
"$PY" e014_write_return.py >>logs/e014_aee/logs/master_watchdog.log 2>&1 || true
log "ALL PHASES DONE"
