#!/bin/bash
# Wait for NeXT-7B A0–A4 30-job queue to drain, then:
#   1) NeXT-13B forward smoke (A0–A4 token traces @ L15/L30)
#   2) if PASS → NeXT-13B 30-job ACC on GPUs 4/5/6/7
set -u
PKG=/home/xiangyue/ViTCoP/experiments/bela_next_7b13b_v100_20261101
PY=/home/xiangyue/envs/vitcop-v100-4model/bin/python
LOG7=$PKG/logs/queue/queue.log
LOG=$PKG/logs/queue_next13
mkdir -p "$LOG"

echo "CHAIN13_WAIT_BEGIN $(date -Is)" | tee -a "$LOG/chain.log"

# Wait until 7B queue drained (or process gone + drained marker)
while true; do
  if grep -q 'QUEUE_DRAINED' "$LOG7" 2>/dev/null; then
    echo "CHAIN13_SEE_DRAINED $(date -Is)" | tee -a "$LOG/chain.log"
    break
  fi
  if ! pgrep -f 'run_next7_a0_a4_six.sh' >/dev/null 2>&1; then
    # launcher exited without marker — still wait for worker PIDs
    if ! pgrep -f 'lmms_bela_next_entry.py' >/dev/null 2>&1; then
      echo "CHAIN13_NO_7B_WORKERS $(date -Is)" | tee -a "$LOG/chain.log"
      break
    fi
  fi
  sleep 60
done

# Ensure GPUs 4-7 free (<800 MiB)
echo "CHAIN13_WAIT_GPU_FREE $(date -Is)" | tee -a "$LOG/chain.log"
while true; do
  busy=0
  for g in 4 5 6 7; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    if [[ ${used:-99999} -ge 800 ]]; then busy=1; fi
  done
  # also no leftover bela entry
  if pgrep -f 'lmms_bela_next_entry.py' >/dev/null 2>&1; then busy=1; fi
  [[ $busy -eq 0 ]] && break
  sleep 30
done
echo "CHAIN13_GPU_FREE $(date -Is)" | tee -a "$LOG/chain.log"

# Smoke on GPU 4
echo "CHAIN13_SMOKE_START $(date -Is)" | tee -a "$LOG/chain.log"
BELA_SMOKE_GPU=4 "$PY" -u "$PKG/scripts/smoke_next13_a0_a4.py" \
  2>&1 | tee "$LOG/smoke_next13.log"
smoke_rc=${PIPESTATUS[0]}
echo "CHAIN13_SMOKE_RC=$smoke_rc $(date -Is)" | tee -a "$LOG/chain.log"
if [[ $smoke_rc -ne 0 ]]; then
  echo "CHAIN13_BLOCKED_SMOKE_FAIL — not launching 13B ACC" | tee -a "$LOG/chain.log"
  exit 3
fi

# Full ACC
echo "CHAIN13_ACC_START $(date -Is)" | tee -a "$LOG/chain.log"
bash "$PKG/scripts/run_next13_a0_a4_six.sh" 2>&1 | tee -a "$LOG/acc_launcher.log"
acc_rc=${PIPESTATUS[0]}
echo "CHAIN13_ACC_RC=$acc_rc $(date -Is)" | tee -a "$LOG/chain.log"
exit $acc_rc
