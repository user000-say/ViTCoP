#!/usr/bin/env bash
# SSH-proof supervisor: keep full ACC + (master ABBA/latency) alive; then RETURN.
# User overrides: ACC=full 4-dataset only; latency |rel|<=15%. Else follow handoff.
set -uo pipefail
cd /home/xiangyue/ViTCoP
PY=/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python
export PYTHONUNBUFFERED=1 PYTHONPATH=/home/xiangyue/ViTCoP
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
ROOT=/home/xiangyue/ViTCoP/logs/e013_p1
LOG=$ROOT/watchdog.log
mkdir -p "$ROOT/logs" "$ROOT/full"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

FULL_ARMS=(BASE_S8 B0_S8 REPLACE_RANDOM16 REPLACE_ATTN16 REPLACE_COVER16 EXTRA_RANDOM16 EXTRA_ATTN16 EXTRA_COVER16 EXTRA_NULL16 EXTRA_DUPLICATE16)
TASKS=(gqa pope mme mmbench_en_dev)

full_missing() {
  local m=0 arm task
  for arm in "${FULL_ARMS[@]}"; do
    for task in "${TASKS[@]}"; do
      if [[ ! -f $ROOT/full/full/$arm/$task/metrics.json && ! -f $ROOT/full/$arm/$task/metrics.json ]]; then
        m=$((m+1))
      fi
    done
  done
  echo "$m"
}

ensure_full_queue() {
  if pgrep -f 'e013_queue.py --phase full' >/dev/null 2>&1; then
    return 0
  fi
  local miss
  miss=$(full_missing)
  [[ "$miss" -eq 0 ]] && return 0
  # Leave GPU4 for ABBA/stage1 latency while those run.
  local gpus=0,1,2,3,5,6,7
  if ! pgrep -f 'e013_latency.py' >/dev/null 2>&1 && ! pgrep -f 'e013_master_driver.sh' >/dev/null 2>&1; then
    gpus=0,1,2,3,4,5,6,7
  fi
  say "restart full queue gpus=$gpus missing=$miss"
  setsid "$PY" e013_queue.py --phase full --gpus "$gpus" >>"$ROOT/logs/queue_full.log" 2>&1 &
}

ensure_master_or_resume_latency() {
  if pgrep -f 'e013_master_driver.sh' >/dev/null 2>&1; then
    return 0
  fi
  # Resume unfinished ABBA / stage1 on GPU4 if master died.
  if [[ ! -f $ROOT/latency/LEGACY_L12_A/latency.jsonl ]]; then
    say "resume ABBA LEGACY_L12 A"
    setsid bash -c "export CUDA_VISIBLE_DEVICES=4; cd /home/xiangyue/ViTCoP; $PY e013_latency.py --arm LEGACY_L12 --gpu 4 --tag A >>$ROOT/logs/crossover.log 2>&1" &
    return 0
  fi
  if [[ ! -f $ROOT/latency/E012_S12D12_A/latency.jsonl ]]; then
    say "resume ABBA E012_S12D12 A"
    setsid bash -c "export CUDA_VISIBLE_DEVICES=4; cd /home/xiangyue/ViTCoP; $PY e013_latency.py --arm E012_S12D12 --gpu 4 --tag A >>$ROOT/logs/crossover.log 2>&1" &
    return 0
  fi
  if [[ ! -f $ROOT/latency/E012_S12D12_B/latency.jsonl ]]; then
    say "resume ABBA E012_S12D12 B"
    setsid bash -c "export CUDA_VISIBLE_DEVICES=4; cd /home/xiangyue/ViTCoP; $PY e013_latency.py --arm E012_S12D12 --gpu 4 --tag B >>$ROOT/logs/crossover.log 2>&1" &
    return 0
  fi
  if [[ ! -f $ROOT/latency/LEGACY_L12_B/latency.jsonl ]]; then
    say "resume ABBA LEGACY_L12 B"
    setsid bash -c "export CUDA_VISIBLE_DEVICES=4; cd /home/xiangyue/ViTCoP; $PY e013_latency.py --arm LEGACY_L12 --gpu 4 --tag B >>$ROOT/logs/crossover.log 2>&1" &
    return 0
  fi
  if [[ ! -f $ROOT/latency/CROSSOVER_GATE.json ]]; then
    say "run crossover gate 15%"
    "$PY" e013_latency_crossover_gate.py \
      --legacy-a "$ROOT/latency/LEGACY_L12_A/latency.jsonl" \
      --e012-a "$ROOT/latency/E012_S12D12_A/latency.jsonl" \
      --e012-b "$ROOT/latency/E012_S12D12_B/latency.jsonl" \
      --legacy-b "$ROOT/latency/LEGACY_L12_B/latency.jsonl" \
      --max-rel-percent 15 \
      --out "$ROOT/latency/CROSSOVER_GATE.json" >>"$LOG" 2>&1 || true
  fi
  local gate_status
  gate_status=$(python3 -c "import json;print(json.load(open('$ROOT/latency/CROSSOVER_GATE.json')).get('status','?'))" 2>/dev/null || echo MISSING)
  if [[ "$gate_status" != "PASS" ]]; then
    say "crossover gate=$gate_status — skip Stage1 attribution; ACC continues"
    echo CROSSOVER_FAIL > "$ROOT/MASTER_STATUS"
    return 0
  fi
  for arm in BASE_S8 REPLACE_ATTN16 REPLACE_COVER16 EXTRA_ATTN16 EXTRA_COVER16; do
    if [[ ! -f $ROOT/latency/$arm/latency.jsonl ]]; then
      if pgrep -f 'e013_latency.py' >/dev/null 2>&1; then
        return 0
      fi
      say "resume stage1 latency $arm"
      setsid bash -c "export CUDA_VISIBLE_DEVICES=4; cd /home/xiangyue/ViTCoP; $PY e013_latency.py --arm $arm --gpu 4 >>$ROOT/logs/stage1_latency.log 2>&1" &
      return 0
    fi
  done
  echo PASS > "$ROOT/MASTER_STATUS"
}

bridge12_if_needed() {
  # Handoff optional L12 bridge — only after full ACC markers exist, so it cannot
  # steal GPUs from official N-full jobs. Diagnostic panel only; not mixed into L8 ACC.
  [[ -f $ROOT/BRIDGE12_DONE ]] && return 0
  local miss
  miss=$(full_missing)
  [[ "$miss" -eq 0 ]] || return 0
  if pgrep -f 'e013_queue.py --phase bridge12' >/dev/null 2>&1; then
    return 0
  fi
  local bmiss=0 arm task
  for arm in BASE_S12 EXTRA_RANDOM16_S12 EXTRA_ATTN16_S12 EXTRA_COVER16_S12; do
    for task in gqa pope mme mmbench_en_dev; do
      [[ -f $ROOT/panel/bridge12/$arm/$task/metrics.json ]] || bmiss=$((bmiss+1))
    done
  done
  if [[ "$bmiss" -eq 0 ]]; then
    echo DONE > "$ROOT/BRIDGE12_DONE"
    return 0
  fi
  say "launch bridge12 diagnostic panel (E008 depth check; not mixed with L8 ACC)"
  # Avoid GPU4 if stage1 latency still running.
  local gpus=0,1,2,3,5,6,7
  pgrep -f 'e013_latency.py' >/dev/null 2>&1 || gpus=0,1,2,3,4,5,6,7
  setsid "$PY" e013_queue.py --phase bridge12 --limit 512 --gpus "$gpus" >>"$ROOT/logs/queue_bridge12.log" 2>&1 &
}

finalize() {
  "$PY" e013_panel_collect.py >>"$LOG" 2>&1 || true
  "$PY" e013_full_collect.py >>"$LOG" 2>&1 || true
  "$PY" e013_delta_correct.py >>"$LOG" 2>&1 || true
  "$PY" e013_write_return.py >>"$LOG" 2>&1 || true
}

say "watchdog start"
while true; do
  ensure_full_queue
  ensure_master_or_resume_latency
  bridge12_if_needed
  miss=$(full_missing)
  abba_left=0
  for f in LEGACY_L12_A E012_S12D12_A E012_S12D12_B LEGACY_L12_B; do
    [[ -f $ROOT/latency/$f/latency.jsonl ]] || abba_left=$((abba_left+1))
  done
  stage_left=0
  if [[ -f $ROOT/latency/CROSSOVER_GATE.json ]]; then
    gst=$(python3 -c "import json;print(json.load(open('$ROOT/latency/CROSSOVER_GATE.json')).get('status',''))" 2>/dev/null || true)
    if [[ "$gst" == "PASS" ]]; then
      for arm in BASE_S8 REPLACE_ATTN16 REPLACE_COVER16 EXTRA_ATTN16 EXTRA_COVER16; do
        [[ -f $ROOT/latency/$arm/latency.jsonl ]] || stage_left=$((stage_left+1))
      done
    fi
  else
    stage_left=-1
  fi
  say "progress full_missing=$miss abba_left=$abba_left stage_left=$stage_left"
  # Refresh RETURN periodically so partial state is visible after disconnect.
  finalize
  if [[ "$miss" -eq 0 && "$abba_left" -eq 0 && ( "$stage_left" -eq 0 || "$stage_left" -eq -1 ) ]]; then
    # If gate failed, stage_left stays 0 path via CROSSOVER_FAIL; if pending gate, wait.
    if [[ ! -f $ROOT/latency/CROSSOVER_GATE.json ]]; then
      sleep 120
      continue
    fi
    gst=$(python3 -c "import json;print(json.load(open('$ROOT/latency/CROSSOVER_GATE.json')).get('status',''))")
    if [[ "$gst" == "PASS" && "$stage_left" -ne 0 ]]; then
      sleep 120
      continue
    fi
    finalize
    echo COMPLETE > "$ROOT/WATCHDOG_STATUS"
    say "ALL DONE — E013_RETURN.md written"
    exit 0
  fi
  sleep 180
done
