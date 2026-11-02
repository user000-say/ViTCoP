#!/bin/bash
# NeXT-7B Stage2 input-QK + mid9 + slots + QA-slice gated A0–A4 R4.
# GPUs 0–7: pack up to 4 free cards; NEVER release a card we hold (holder stays).
set -u
ROOT=/home/xiangyue/ViTCoP
PKG=$ROOT/experiments/bela_next_7b13b_v100_20261101
PY=/home/xiangyue/envs/vitcop-v100-4model/bin/python
ENTRY=$PKG/scripts/lmms_bela_next_entry.py
HOLD_PY=$PKG/scripts/gpu_hold.py
GATE_PY=$PKG/scripts/gate_next7_arm_r4.py
OUT_ROOT=$PKG/logs/acc_a0_a4_s2qk_20261102
LOG=$PKG/logs/queue_s2qk_20261102
mkdir -p "$OUT_ROOT" "$LOG"

export PYTHONPATH=$PKG:$PKG/BELA_NEXT_7B13B_V100_CURSOR_20261101/src:$ROOT/lmms-eval:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export BELA_NEXT_V2_ENABLE=1
export BELA_NEXT_MAX_NEW=256

ARMS=(
  NEXT7_T160_A0_PAPER
  NEXT7_T160_A1_PAPER_VAR
  NEXT7_T160_A2_LATE
  NEXT7_T160_A3_LESS_S1
  NEXT7_T160_A4_STRONG_LATE
)
R4_TASKS=(gqa pope mme mmbench_en_dev)
EXTRA_TASKS=(scienceqa_img textvqa_val_ocr)
GPUS=(0 1 2 3 4 5 6 7)
MAX_HOLD=4

JOBS=()
for arm in "${ARMS[@]}"; do
  for task in "${R4_TASKS[@]}"; do
    JOBS+=("${arm}|${task}")
  done
done

declare -A PID GPU_OF ARM_GATED HOLD_PID
idx=0
n=${#JOBS[@]}

ours_gpu_count() {
  local c=0 g
  for g in "${GPUS[@]}"; do
    for t in "${!GPU_OF[@]}"; do
      if [[ -n ${PID[$t]+x} && ${GPU_OF[$t]} == "$g" ]]; then c=$((c+1)); break; fi
    done
    if [[ -n ${HOLD_PID[$g]+x} ]] && kill -0 "${HOLD_PID[$g]}" 2>/dev/null; then
      # already counted if also in GPU_OF? holders only in HOLD_PID
      local also=0
      for t in "${!GPU_OF[@]}"; do
        [[ -n ${PID[$t]+x} && ${GPU_OF[$t]} == "$g" ]] && also=1
      done
      [[ $also -eq 0 ]] && c=$((c+1))
    fi
  done
  echo "$c"
}

start_hold() {
  local gpu=$1
  if [[ -n ${HOLD_PID[$gpu]+x} ]] && kill -0 "${HOLD_PID[$gpu]}" 2>/dev/null; then
    return
  fi
  # Don't hold if a real job is already on this GPU
  for t in "${!GPU_OF[@]}"; do
    if [[ -n ${PID[$t]+x} && ${GPU_OF[$t]} == "$gpu" ]]; then return; fi
  done
  echo "HOLD_START gpu=$gpu $(date -Is)" | tee -a "$LOG/queue.log"
  CUDA_VISIBLE_DEVICES=$gpu BELA_GPU_HOLD_GB=14 \
    "$PY" -u "$HOLD_PY" > "$LOG/hold_gpu${gpu}.log" 2>&1 &
  HOLD_PID[$gpu]=$!
}

stop_hold() {
  local gpu=$1
  if [[ -n ${HOLD_PID[$gpu]+x} ]] && kill -0 "${HOLD_PID[$gpu]}" 2>/dev/null; then
    echo "HOLD_STOP gpu=$gpu pid=${HOLD_PID[$gpu]} $(date -Is)" | tee -a "$LOG/queue.log"
    kill "${HOLD_PID[$gpu]}" 2>/dev/null || true
    wait "${HOLD_PID[$gpu]}" 2>/dev/null || true
  fi
  unset "HOLD_PID[$gpu]" || true
}

# Adopt in-flight workers.
while read -r line; do
  [[ -z "$line" ]] && continue
  pid=$(echo "$line" | awk '{print $1}')
  cmd=$line
  arm=""; task=""
  if [[ "$cmd" =~ --tasks[[:space:]]+([^ ]+) ]]; then task="${BASH_REMATCH[1]}"; fi
  if [[ "$cmd" =~ acc_a0_a4_s2qk_20261102/(NEXT7_T160_[A-Z0-9_]+)/ ]]; then
    arm="${BASH_REMATCH[1]}"
  fi
  [[ -z "$arm" || -z "$task" ]] && continue
  cv=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)
  gpu=${cv:-}
  [[ -z "$gpu" ]] && continue
  tag="${arm}__${task}"
  PID[$tag]=$pid
  GPU_OF[$tag]=$gpu
  echo "ADOPT $tag pid=$pid gpu=$gpu $(date -Is)" | tee -a "$LOG/queue.log"
done < <(pgrep -af 'lmms_bela_next_entry.py' || true)

# Adopt existing holders.
while read -r line; do
  [[ -z "$line" ]] && continue
  pid=$(echo "$line" | awk '{print $1}')
  cv=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)
  [[ -z "$cv" ]] && continue
  HOLD_PID[$cv]=$pid
  echo "ADOPT_HOLD gpu=$cv pid=$pid $(date -Is)" | tee -a "$LOG/queue.log"
done < <(pgrep -af 'gpu_hold.py' || true)

kill_arm_running() {
  local arm=$1 tag
  for tag in "${!PID[@]}"; do
    if [[ "${tag%__*}" == "$arm" ]]; then
      echo "KILL_ARM_JOB $tag pid=${PID[$tag]} $(date -Is)" | tee -a "$LOG/queue.log"
      kill "${PID[$tag]}" 2>/dev/null || true
      wait "${PID[$tag]}" 2>/dev/null || true
      unset "PID[$tag]"
      unset "GPU_OF[$tag]"
    fi
  done
}

launch() {
  local gpu=$1 arm=$2 task=$3
  local dest=$OUT_ROOT/$arm/$task
  local tag="${arm}__${task}"
  if [[ -n ${ARM_GATED[$arm]+x} && ${ARM_GATED[$arm]} != GO ]]; then
    echo "SKIP_LAUNCH $tag arm_already_skipped $(date -Is)" | tee -a "$LOG/queue.log"
    return 1
  fi
  if [[ -n ${PID[$tag]+x} ]]; then
    echo "RUNNING $tag" | tee -a "$LOG/queue.log"
    return 1
  fi
  if find "$dest" -name '*_results.json' 2>/dev/null | grep -q .; then
    echo "HAVE $tag" | tee -a "$LOG/queue.log"
    return 1
  fi
  stop_hold "$gpu"
  mkdir -p "$dest"
  echo "START gpu=$gpu $tag $(date -Is)" | tee -a "$LOG/queue.log"
  CUDA_VISIBLE_DEVICES=$gpu \
  BELA_NEXT_V2_ENABLE=1 \
  BELA_NEXT_ARM=$arm \
  BELA_NEXT_MAX_NEW=256 \
  "$PY" -u "$ENTRY" \
    --model llava_hf \
    --model_args pretrained=llava-hf/llava-v1.6-vicuna-7b-hf,dtype=float16,attn_implementation=sdpa,device_map=cuda:0 \
    --tasks "$task" --batch_size 1 --log_samples \
    --output_path "$dest" \
    > "$LOG/${tag}.log" 2>&1 &
  PID[$tag]=$!
  GPU_OF[$tag]=$gpu
  return 0
}

maybe_gate_arm() {
  # Call after every finished R4 task. Early SKIP if any task rel < 88%.
  local arm=$1
  [[ -n ${ARM_GATED[$arm]+x} ]] && return
  local verdict
  verdict=$("$PY" -u "$GATE_PY" --arm "$arm" --out-root "$OUT_ROOT" --single-floor 88 2>>"$LOG/gate.log") || true
  echo "GATE $arm $verdict $(date -Is)" | tee -a "$LOG/queue.log"
  if [[ "$verdict" == PENDING* ]]; then
    return
  fi
  if [[ "$verdict" == GO* ]]; then
    ARM_GATED[$arm]=GO
    for t in "${EXTRA_TASKS[@]}"; do
      JOBS+=("${arm}|${t}")
    done
    n=${#JOBS[@]}
    echo "ENQUEUE_6ACC $arm $(date -Is)" | tee -a "$LOG/queue.log"
    return
  fi
  # SKIP_6ACC (single_below_88 or r4_below_92): kill leftover arm jobs, drop pending.
  ARM_GATED[$arm]=SKIP
  kill_arm_running "$arm"
  local new_jobs=() j a t
  for j in "${JOBS[@]:$idx}"; do
    IFS='|' read -r a t <<< "$j"
    [[ "$a" == "$arm" ]] && continue
    new_jobs+=("$j")
  done
  JOBS=("${JOBS[@]:0:$idx}" "${new_jobs[@]}")
  n=${#JOBS[@]}
  echo "SKIP_6ACC $arm $verdict $(date -Is)" | tee -a "$LOG/queue.log"
}

pick_free_foreign() {
  # Card free of foreign+ours jobs (for claiming new slots up to MAX_HOLD).
  local g used busy
  for g in "${GPUS[@]}"; do
    busy=0
    for t in "${!GPU_OF[@]}"; do
      [[ -n ${PID[$t]+x} && ${GPU_OF[$t]} == "$g" ]] && busy=1
    done
    [[ -n ${HOLD_PID[$g]+x} ]] && kill -0 "${HOLD_PID[$g]}" 2>/dev/null && busy=1
    [[ $busy -eq 1 ]] && continue
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    if [[ ${used:-99999} -lt 800 ]]; then echo "$g"; return; fi
  done
}

echo "QUEUE_GATED_V4_RESUME_A1A4_BEGIN n=$n gpus=${GPUS[*]} pack4_never_release code=25a0f57 $(date -Is)" | tee -a "$LOG/queue.log"

for arm in "${ARMS[@]}"; do
  maybe_gate_arm "$arm"
done

declare -A RETRIES
# Main loop never exits while we hold GPUs (user: never release).
while true; do
  for tag in "${!PID[@]}"; do
    if ! kill -0 "${PID[$tag]}" 2>/dev/null; then
      set +e
      wait "${PID[$tag]}" 2>/dev/null
      rc=$?
      set -u
      gpu_done=${GPU_OF[$tag]}
      arm=${tag%__*}
      task=${tag#*__}
      dest=$OUT_ROOT/$arm/$task
      has_res=0
      if find "$dest" -name '*_results.json' 2>/dev/null | grep -q .; then has_res=1; fi
      echo "DONE $tag rc=$rc has_results=$has_res gpu=$gpu_done $(date -Is)" | tee -a "$LOG/queue.log"
      unset "PID[$tag]"
      unset "GPU_OF[$tag]"
      if [[ $has_res -eq 0 ]]; then
        r=${RETRIES[$tag]:-0}
        if [[ $r -lt 2 ]]; then
          RETRIES[$tag]=$((r + 1))
          JOBS+=("${arm}|${task}")
          n=${#JOBS[@]}
          echo "REQUEUE $tag try=${RETRIES[$tag]} $(date -Is)" | tee -a "$LOG/queue.log"
        else
          echo "GIVEUP $tag $(date -Is)" | tee -a "$LOG/queue.log"
        fi
      else
        maybe_gate_arm "$arm"
      fi
      # Immediately reuse same GPU for next pending job, else hold forever.
      launched=0
      while [[ $idx -lt $n ]]; do
        IFS='|' read -r a2 t2 <<< "${JOBS[$idx]}"
        idx=$((idx+1))
        if launch "$gpu_done" "$a2" "$t2"; then
          launched=1
          break
        fi
      done
      if [[ $launched -eq 0 ]]; then
        start_hold "$gpu_done"
      fi
    fi
  done

  # Fill up to MAX_HOLD with work or holders on newly free foreign cards.
  while [[ $(ours_gpu_count) -lt $MAX_HOLD ]]; do
    free=$(pick_free_foreign || true)
    [[ -z "${free:-}" ]] && break
    launched=0
    while [[ $idx -lt $n ]]; do
      IFS='|' read -r arm task <<< "${JOBS[$idx]}"
      idx=$((idx+1))
      if launch "$free" "$arm" "$task"; then
        launched=1
        break
      fi
    done
    if [[ $launched -eq 0 ]]; then
      start_hold "$free"
    fi
    sleep 2
  done

  if [[ $idx -ge $n && ${#PID[@]} -eq 0 ]]; then
    # All ACC work done: keep holders, summarize once, then idle-hold.
    if [[ ! -f "$LOG/summary_done.flag" ]]; then
      echo "QUEUE_ACC_DONE_HOLDING $(date -Is)" | tee -a "$LOG/queue.log"
      if [[ -f "$PKG/scripts/summarize_next7_v4_acc.py" ]]; then
        "$PY" -u "$PKG/scripts/summarize_next7_v4_acc.py" | tee -a "$LOG/summary.log" || true
      fi
      date -Is > "$LOG/summary_done.flag"
    fi
    # Ensure we still hold MAX_HOLD cards
    while [[ $(ours_gpu_count) -lt $MAX_HOLD ]]; do
      free=$(pick_free_foreign || true)
      [[ -z "${free:-}" ]] && break
      start_hold "$free"
      sleep 2
    done
  fi
  sleep 15
done
