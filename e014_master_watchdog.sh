#!/usr/bin/env bash
# SSH-safe E014 continuum:
# full_k96 -> full_k64 -> full_k32(L8) -> full_k32_l10 -> full_k32_l12 -> collect/return
set -uo pipefail
REPO=/home/xiangyue/ViTCoP
PY=/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python
cd "$REPO"
export PYTHONPATH="$REPO" PYTHONUNBUFFERED=1
mkdir -p logs/e014_aee/logs experiments/e014_aee
log(){ echo "[$(date '+%F %T')] $*" | tee -a logs/e014_aee/logs/master_watchdog.log; }

expect_for(){
  case "$1" in
    full_k96) echo 36 ;;
    full_k64|full_k32|full_k32_l10|full_k32_l12) echo 20 ;;
    *) echo 0 ;;
  esac
}

run_phase(){
  local phase="$1"
  local expect
  expect="$(expect_for "$phase")"
  log "START phase=$phase expect=$expect"
  while true; do
    "$PY" e014_queue.py --phase "$phase" --limit 0 --once >>"logs/e014_aee/logs/queue_${phase}.watchdog.log" 2>&1 || true
    done_n=$(find "logs/e014_aee/full/$phase" -name metrics.json 2>/dev/null | wc -l)
    log "phase=$phase metrics=$done_n/$expect"
    if [[ "$done_n" -ge "$expect" ]]; then
      log "COMPLETE phase=$phase"
      break
    fi
    sleep 60
  done
}

# Resume-safe: completed phases finish immediately via metrics count.
run_phase full_k96
run_phase full_k64
run_phase full_k32
run_phase full_k32_l10
run_phase full_k32_l12
"$PY" e014_full_collect.py >>logs/e014_aee/logs/master_watchdog.log 2>&1 || true
"$PY" e014_write_return.py >>logs/e014_aee/logs/master_watchdog.log 2>&1 || true
log "ALL PHASES DONE"
