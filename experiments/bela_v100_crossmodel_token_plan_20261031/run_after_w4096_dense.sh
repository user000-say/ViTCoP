#!/bin/bash
set -u
ROOT=/home/xiangyue/ViTCoP
LOG=$ROOT/experiments/bela_v100_crossmodel_token_plan_20261031/logs
mkdir -p "$LOG"
QUEUE_LOG=$ROOT/experiments/bela_stage1_decision_package/logs/w4096_s1_p3_six_queue.log
echo "WAIT_W4096 $(date -Is)" | tee -a "$LOG/chain.log"
# wait until queue process gone and drained marker
while pgrep -f 'launch_w4096_s1_p3_six.py' >/dev/null; do sleep 60; done
grep -q W4096_S1_P3_SIX_DRAINED "$QUEUE_LOG" || echo "WARN no drain marker" | tee -a "$LOG/chain.log"
echo "START_NEXT7_DENSE $(date -Is)" | tee -a "$LOG/chain.log"
bash "$ROOT/experiments/bela_v100_4model_qacover_20261028/scripts/run_next7_dense6.sh" \
  >>"$LOG/next7_dense_chain.log" 2>&1
echo "START_REMAINING_DENSE $(date -Is)" | tee -a "$LOG/chain.log"
bash "$ROOT/experiments/bela_v100_4model_qacover_20261028/scripts/run_remaining_dense.sh" \
  >>"$LOG/remaining_dense_chain.log" 2>&1
echo "CHAIN_DONE $(date -Is)" | tee -a "$LOG/chain.log"
