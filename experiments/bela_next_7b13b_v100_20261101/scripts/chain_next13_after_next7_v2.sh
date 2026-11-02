#!/bin/bash
# After NeXT-7B v2 ACC drains → NeXT-13B smoke → NeXT-13B v2 ACC
set -u
PKG=/home/xiangyue/ViTCoP/experiments/bela_next_7b13b_v100_20261101
PY=/home/xiangyue/envs/vitcop-v100-4model/bin/python
LOG7=$PKG/logs/queue_v2/queue.log
LOG=$PKG/logs/queue_next13_v2
mkdir -p "$LOG"

echo "CHAIN13V2_WAIT_BEGIN $(date -Is)" | tee -a "$LOG/chain.log"
while true; do
  if grep -q 'QUEUE_DRAINED' "$LOG7" 2>/dev/null; then
    echo "CHAIN13V2_SEE_DRAINED $(date -Is)" | tee -a "$LOG/chain.log"
    break
  fi
  if ! pgrep -f 'run_next7_a0_a4_six.sh' >/dev/null 2>&1; then
    if ! pgrep -f 'lmms_bela_next_entry.py' >/dev/null 2>&1; then
      echo "CHAIN13V2_NO_7B_WORKERS $(date -Is)" | tee -a "$LOG/chain.log"
      break
    fi
  fi
  sleep 60
done

echo "CHAIN13V2_WAIT_GPU_FREE $(date -Is)" | tee -a "$LOG/chain.log"
while true; do
  busy=0
  for g in 4 5 6 7; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    [[ ${used:-99999} -ge 800 ]] && busy=1
  done
  pgrep -f 'lmms_bela_next_entry.py' >/dev/null 2>&1 && busy=1
  [[ $busy -eq 0 ]] && break
  sleep 30
done

echo "CHAIN13V2_SMOKE_START $(date -Is)" | tee -a "$LOG/chain.log"
BELA_SMOKE_GPU=4 "$PY" -u "$PKG/scripts/smoke_next13_a0_a4.py" 2>&1 | tee "$LOG/smoke_next13.log"
smoke_rc=${PIPESTATUS[0]}
echo "CHAIN13V2_SMOKE_RC=$smoke_rc $(date -Is)" | tee -a "$LOG/chain.log"
[[ $smoke_rc -ne 0 ]] && exit 3

# generate-path arm differentiation on 13B (A0 vs A4)
BELA_SMOKE_GPU=4 BELA_NEXT_MODEL=13b "$PY" -u "$PKG/scripts/diag_generate_path_prune.py" \
  2>&1 | tee "$LOG/diag_gen.log" || true

echo "CHAIN13V2_ACC_START $(date -Is)" | tee -a "$LOG/chain.log"
bash "$PKG/scripts/run_next13_a0_a4_six.sh" 2>&1 | tee -a "$LOG/acc_launcher.log"
echo "CHAIN13V2_ACC_RC=${PIPESTATUS[0]} $(date -Is)" | tee -a "$LOG/chain.log"
