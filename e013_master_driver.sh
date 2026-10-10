#!/usr/bin/env bash
# E013-P1 continuum: smoke -> screen1 -> screen2 -> crossover -> stage1 latency.
# Detach with setsid/nohup. Survives SSH disconnect.
set -uo pipefail
cd /home/xiangyue/ViTCoP
PY=/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python
export PYTHONUNBUFFERED=1 PYTHONPATH=/home/xiangyue/ViTCoP
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
ROOT=/home/xiangyue/ViTCoP/logs/e013_p1
LOG=$ROOT/master_driver.log
mkdir -p "$ROOT/logs" "$ROOT/smoke" "$ROOT/panel" "$ROOT/latency"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "E013 master start"
chmod +x e013_smoke_driver.sh e013_rollback.sh 2>/dev/null || true

# --- Step 3: GPU4 smoke ---
if [[ ! -f $ROOT/SMOKE_STATUS ]] || [[ "$(cat $ROOT/SMOKE_STATUS 2>/dev/null)" != "PASS" ]]; then
  say "running smoke on GPU4"
  setsid bash e013_smoke_driver.sh >>"$LOG" 2>&1
  wait_ec=$?
  if [[ ! -f $ROOT/SMOKE_STATUS ]] || [[ "$(cat $ROOT/SMOKE_STATUS)" != "PASS" ]]; then
    say "SMOKE_FAIL ec=$wait_ec — HARD STOP, rollback recommended"
    echo FAIL > "$ROOT/MASTER_STATUS"
    exit 1
  fi
fi
say "smoke PASS"

# --- Step 4: screen1 512-panel ---
say "screen1 queue"
setsid "$PY" e013_queue.py --phase screen1 --limit 512 >>"$ROOT/logs/queue_screen1.log" 2>&1 &
Q1=$!
# Wait until all screen1 markers exist
while true; do
  missing=0
  for arm in BASE_S8 B0_S8 REPLACE_RANDOM16 REPLACE_ATTN16 REPLACE_COVER16 EXTRA_RANDOM16 EXTRA_ATTN16 EXTRA_COVER16; do
    for task in gqa pope mme mmbench_en_dev; do
      [[ -f $ROOT/panel/screen1/$arm/$task/metrics.json ]] || missing=$((missing+1))
    done
  done
  say "screen1 missing=$missing"
  [[ "$missing" -eq 0 ]] && break
  # keep queue alive if it died
  if ! kill -0 "$Q1" 2>/dev/null; then
    say "screen1 queue died; restart"
    setsid "$PY" e013_queue.py --phase screen1 --limit 512 >>"$ROOT/logs/queue_screen1.log" 2>&1 &
    Q1=$!
  fi
  sleep 120
done
say "screen1 complete"
kill "$Q1" 2>/dev/null || true

# --- Step 5: screen2 ---
say "screen2 queue"
setsid "$PY" e013_queue.py --phase screen2 --limit 512 >>"$ROOT/logs/queue_screen2.log" 2>&1 &
Q2=$!
while true; do
  missing=0
  for arm in EXTRA_NULL16 EXTRA_DUPLICATE16 EXTRA_ATTN8 EXTRA_ATTN32 EXTRA_RANDOM8 EXTRA_RANDOM32 EXTRA_COVER8 EXTRA_COVER32; do
    for task in gqa pope mme mmbench_en_dev; do
      [[ -f $ROOT/panel/screen2/$arm/$task/metrics.json ]] || missing=$((missing+1))
    done
  done
  say "screen2 missing=$missing"
  [[ "$missing" -eq 0 ]] && break
  if ! kill -0 "$Q2" 2>/dev/null; then
    say "screen2 queue died; restart"
    setsid "$PY" e013_queue.py --phase screen2 --limit 512 >>"$ROOT/logs/queue_screen2.log" 2>&1 &
    Q2=$!
  fi
  sleep 120
done
say "screen2 complete"
kill "$Q2" 2>/dev/null || true

# --- Step 7 I-1: ABBA crossover on GPU4 (after panels free GPU4) ---
say "ABBA crossover GPU4"
export CUDA_VISIBLE_DEVICES=4
for spec in "LEGACY_L12:A" "E012_S12D12:A" "E012_S12D12:B" "LEGACY_L12:B"; do
  arm="${spec%%:*}"
  tag="${spec##*:}"
  say "crossover $arm tag=$tag"
  if ! "$PY" e013_latency.py --arm "$arm" --gpu 4 --tag "$tag" >>"$ROOT/logs/crossover.log" 2>&1; then
    say "CROSSOVER_FAIL $arm $tag"
    echo FAIL > "$ROOT/MASTER_STATUS"
    exit 1
  fi
done
# User latency standard: relative Fixed32 |Δ| ≤ 15% (not absolute ms band).
"$PY" e013_latency_crossover_gate.py \
  --legacy-a "$ROOT/latency/LEGACY_L12_A/latency.jsonl" \
  --e012-a "$ROOT/latency/E012_S12D12_A/latency.jsonl" \
  --e012-b "$ROOT/latency/E012_S12D12_B/latency.jsonl" \
  --legacy-b "$ROOT/latency/LEGACY_L12_B/latency.jsonl" \
  --max-rel-percent 15 \
  --out "$ROOT/latency/CROSSOVER_GATE.json" >>"$LOG" 2>&1
gate_ec=$?
if [[ "$gate_ec" -ne 0 ]]; then
  say "CROSSOVER gate FAIL — skip Stage1 latency attribution; keep accuracy logs"
  echo CROSSOVER_FAIL > "$ROOT/MASTER_STATUS"
  # Still write return skeleton later; do not run Stage1 latency as algorithm claim.
  exit 0
fi
say "crossover PASS"

# --- Step 7 I-2: Stage1 latency ---
for arm in BASE_S8 REPLACE_ATTN16 REPLACE_COVER16 EXTRA_ATTN16 EXTRA_COVER16; do
  say "stage1 latency $arm"
  if ! "$PY" e013_latency.py --arm "$arm" --gpu 4 >>"$ROOT/logs/stage1_latency.log" 2>&1; then
    say "LATENCY_FAIL $arm"
    echo FAIL > "$ROOT/MASTER_STATUS"
    exit 1
  fi
done

echo PASS > "$ROOT/MASTER_STATUS"
say "E013 master continuum done (ABBA+stage1). Full ACC supervised by e013_continuum_watchdog.sh"
exit 0
