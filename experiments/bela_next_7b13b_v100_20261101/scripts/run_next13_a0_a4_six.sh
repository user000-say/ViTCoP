#!/bin/bash
# NeXT-13B A0–A4 @94.4% — six tasks on GPUs 4/5/6/7. Cuts L15/L30.
# Requires smoke_next13_a0_a4 ALL_PASS first.
set -u
ROOT=/home/xiangyue/ViTCoP
PKG=$ROOT/experiments/bela_next_7b13b_v100_20261101
PY=/home/xiangyue/envs/vitcop-v100-4model/bin/python
ENTRY=$PKG/scripts/lmms_bela_next_entry.py
OUT_ROOT=$PKG/logs/acc_next13_a0_a4_94pct_v3
LOG=$PKG/logs/queue_next13_v3
mkdir -p "$OUT_ROOT" "$LOG"

SMOKE_VERDICT=$PKG/logs/smoke_next13_a0_a4/SMOKE_VERDICT.md
if ! grep -q 'ALL_PASS=True' "$SMOKE_VERDICT" 2>/dev/null; then
  echo "BLOCKED: NeXT-13B smoke not PASS ($SMOKE_VERDICT)" | tee -a "$LOG/queue.log"
  exit 2
fi

export PYTHONPATH=$PKG:$PKG/BELA_NEXT_7B13B_V100_CURSOR_20261101/src:$ROOT/lmms-eval:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export BELA_NEXT_V2_ENABLE=1

ARMS=(
  NEXT13_T160_A0_PAPER
  NEXT13_T160_A1_PAPER_VAR
  NEXT13_T160_A2_LATE
  NEXT13_T160_A3_LESS_S1
  NEXT13_T160_A4_STRONG_LATE
)
TASKS=(gqa pope mme mmbench_en_dev scienceqa_img textvqa_val_ocr)
GPUS=(4 5 6 7)
MODEL=llava-hf/llava-v1.6-vicuna-13b-hf

JOBS=()
for arm in "${ARMS[@]}"; do
  for task in "${TASKS[@]}"; do
    JOBS+=("${arm}|${task}")
  done
done

declare -A PID GPU_OF
idx=0
n=${#JOBS[@]}

launch() {
  local gpu=$1 arm=$2 task=$3
  local dest=$OUT_ROOT/$arm/$task
  local tag="${arm}__${task}"
  if [[ -f $dest/metrics.json ]]; then
    echo "HAVE $tag" | tee -a "$LOG/queue.log"
    return
  fi
  if find "$dest" -name '*_results.json' 2>/dev/null | grep -q .; then
    echo "HAVE_RESULTS $tag" | tee -a "$LOG/queue.log"
    return
  fi
  mkdir -p "$dest"
  echo "START gpu=$gpu $tag $(date -Is)" | tee -a "$LOG/queue.log"
  CUDA_VISIBLE_DEVICES=$gpu \
  BELA_NEXT_V2_ENABLE=1 \
  BELA_NEXT_ARM=$arm \
  "$PY" -u "$ENTRY" \
    --model llava_hf \
    --model_args pretrained=$MODEL,dtype=float16,attn_implementation=sdpa,device_map=cuda:0 \
    --tasks "$task" --batch_size 1 --log_samples \
    --output_path "$dest" \
    > "$LOG/${tag}.log" 2>&1 &
  PID[$tag]=$!
  GPU_OF[$tag]=$gpu
}

echo "QUEUE13_BEGIN n=$n gpus=${GPUS[*]} $(date -Is)" | tee -a "$LOG/queue.log"

while [[ $idx -lt $n || ${#PID[@]} -gt 0 ]]; do
  for tag in "${!PID[@]}"; do
    if ! kill -0 "${PID[$tag]}" 2>/dev/null; then
      wait "${PID[$tag]}"
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
      if [[ $busy -eq 0 ]]; then
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
        if [[ ${used:-99999} -lt 800 ]]; then free=$g; break; fi
      fi
    done
    [[ -z $free ]] && break
    IFS='|' read -r arm task <<< "${JOBS[$idx]}"
    launch "$free" "$arm" "$task"
    idx=$((idx+1))
    sleep 3
  done
  sleep 20
done

echo "QUEUE13_DRAINED $(date -Is)" | tee -a "$LOG/queue.log"
