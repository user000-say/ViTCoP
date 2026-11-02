#!/bin/bash
# NeXT-7B A0–A4 @94.4% native kit — 6 ACC tasks (R4 + SciQA + TextVQA-OCR).
# GPUs 4/5/6/7. Skips tasks that already have *_results.json (e.g. in-flight A0 R4).
set -u
ROOT=/home/xiangyue/ViTCoP
PKG=$ROOT/experiments/bela_next_7b13b_v100_20261101
PY=/home/xiangyue/envs/vitcop-v100-4model/bin/python
ENTRY=$PKG/scripts/lmms_bela_next_entry.py
OUT_ROOT=$PKG/logs/acc_a0_a4_94pct_v4
LOG=$PKG/logs/queue_v4
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
TASKS=(gqa pope mme mmbench_en_dev scienceqa_img textvqa_val_ocr)
GPUS=(4 5 6 7)

JOBS=()
for arm in "${ARMS[@]}"; do
  for task in "${TASKS[@]}"; do
    JOBS+=("${arm}|${task}")
  done
done

declare -A PID GPU_OF
idx=0
n=${#JOBS[@]}

# Adopt already-running lmms workers on 4/5/6/7 so we do not double-launch.
while read -r line; do
  [[ -z "$line" ]] && continue
  pid=$(echo "$line" | awk '{print $1}')
  cmd=$line
  arm=""; task=""; gpu=""
  if [[ "$cmd" =~ --tasks[[:space:]]+([^ ]+) ]]; then task="${BASH_REMATCH[1]}"; fi
  if [[ "$cmd" =~ acc_a0_a4_94pct_v4/(NEXT7_T160_[A-Z0-9_]+)/ ]]; then
    arm="${BASH_REMATCH[1]}"
  fi
  cv=$(tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)
  gpu=${cv:-}
  if [[ -n "$arm" && -n "$task" && -n "$gpu" ]]; then
    tag="${arm}__${task}"
    PID[$tag]=$pid
    GPU_OF[$tag]=$gpu
    echo "ADOPT $tag pid=$pid gpu=$gpu $(date -Is)" | tee -a "$LOG/queue.log"
  fi
done < <(pgrep -af 'lmms_bela_next_entry.py' || true)

launch() {
  local gpu=$1 arm=$2 task=$3
  local dest=$OUT_ROOT/$arm/$task
  local tag="${arm}__${task}"
  if [[ -n ${PID[$tag]+x} ]]; then
    echo "RUNNING $tag" | tee -a "$LOG/queue.log"
    return
  fi
  if find "$dest" -name '*_results.json' 2>/dev/null | grep -q .; then
    echo "HAVE $tag" | tee -a "$LOG/queue.log"
    return
  fi
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
}

echo "QUEUE5x6_BEGIN n=$n gpus=${GPUS[*]} $(date -Is)" | tee -a "$LOG/queue.log"

while [[ $idx -lt $n || ${#PID[@]} -gt 0 ]]; do
  for tag in "${!PID[@]}"; do
    if ! kill -0 "${PID[$tag]}" 2>/dev/null; then
      wait "${PID[$tag]}" 2>/dev/null || true
      rc=$?
      echo "DONE $tag rc=$rc gpu=${GPU_OF[$tag]} $(date -Is)" | tee -a "$LOG/queue.log"
      unset "PID[$tag]"
      unset "GPU_OF[$tag]"
    fi
  done
  while [[ $idx -lt $n ]]; do
    free=""
    for g in "${GPUS[@]}"; do
      busy=0
      for t in "${!GPU_OF[@]}"; do
        [[ -n ${PID[$t]+x} && ${GPU_OF[$t]} == "$g" ]] && busy=1
      done
      # also treat foreign high VRAM as busy
      if [[ $busy -eq 0 ]]; then
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
        if [[ ${used:-99999} -lt 800 ]]; then free=$g; break; fi
      fi
    done
    [[ -z $free ]] && break
    IFS='|' read -r arm task <<< "${JOBS[$idx]}"
    launch "$free" "$arm" "$task"
    idx=$((idx+1))
    sleep 2
  done
  sleep 20
done

echo "QUEUE5x6_DRAINED $(date -Is)" | tee -a "$LOG/queue.log"
"$PY" -u "$PKG/scripts/summarize_next7_v4_acc.py" | tee -a "$LOG/summary.log"
