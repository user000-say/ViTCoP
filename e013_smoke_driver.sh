#!/usr/bin/env bash
set -uo pipefail
cd /home/xiangyue/ViTCoP
PY=/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python
export PYTHONUNBUFFERED=1 PYTHONPATH=/home/xiangyue/ViTCoP
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=4
LOG=/home/xiangyue/ViTCoP/logs/e013_p1/smoke_driver.log
mkdir -p /home/xiangyue/ViTCoP/logs/e013_p1/smoke
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "E013 smoke start gpu4"
fail=0
for arm in BASE_S8 B0_S8 REPLACE_COVER16 EXTRA_ATTN16 EXTRA_COVER16 EXTRA_NULL16 EXTRA_DUPLICATE16; do
  say "arm $arm"
  if ! "$PY" e013_smoke.py --arm "$arm" --gpu 4 >>"$LOG" 2>&1; then
    say "SMOKE_FAIL $arm"
    fail=1
    break
  fi
done
if [[ "$fail" -eq 0 ]]; then
  if "$PY" e013_smoke_compare.py >>"$LOG" 2>&1; then
    say "SMOKE_PASS"
    echo PASS > /home/xiangyue/ViTCoP/logs/e013_p1/SMOKE_STATUS
  else
    say "SMOKE_FAIL compare"
    echo FAIL > /home/xiangyue/ViTCoP/logs/e013_p1/SMOKE_STATUS
    fail=1
  fi
else
  echo FAIL > /home/xiangyue/ViTCoP/logs/e013_p1/SMOKE_STATUS
fi
exit "$fail"
