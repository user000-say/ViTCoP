#!/bin/bash
# Unpruned Dense for the three models that are not LLaVA-NeXT-7B.
# NeXT-7B six-task scores are already on disk. One model at a time. GPUs 0, 2, 4, 7 only.
# Long tasks stay inside tmux so a dropped SSH session does not stop them.
set -u
shopt -s globstar
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ROOT=/home/xiangyue/ViTCoP
PY=/home/xiangyue/envs/vitcop-v100-4model/bin/python
LOGDIR=$ROOT/experiments/bela_v100_4model_qacover_20261028/logs
BASE=$ROOT/logs/e014_aee/full/bela_v100_4model_20261028
MASTER=$LOGDIR/remaining_dense.log
mkdir -p "$LOGDIR"
export PYTHONPATH=$ROOT/lmms-eval
export TOKENIZERS_PARALLELISM=false
GPUS=(0 2 4 7)

have_results() {
  compgen -G "$1/**/results.json" > /dev/null || compgen -G "$1/*results*.json" > /dev/null
}

run_one_model() {
  local name=$1 model=$2 pretrained=$3 args=$4
  shift 4
  local tasks=("$@")
  local out=$BASE/$name/DENSE
  mkdir -p "$out"
  echo "MODEL $name $(date -Is)" | tee -a "$MASTER"
  declare -A PID GPU_OF
  local idx=0 blocked=0
  while [[ $idx -lt ${#tasks[@]} || ${#PID[@]} -gt 0 ]]; do
    for task in "${!PID[@]}"; do
      if ! kill -0 "${PID[$task]}" 2>/dev/null; then
        wait "${PID[$task]}"
        local rc=$?
        echo "DONE $name $task rc=$rc $(date -Is)" | tee -a "$MASTER"
        if [[ $rc -ne 0 ]] && grep -q "CUDA out of memory" "$LOGDIR/${name}_${task}.log" 2>/dev/null; then
          echo "OOM $name stop launching further tasks" | tee -a "$MASTER"
          blocked=1
        fi
        unset "PID[$task]"
      fi
    done
    while [[ $blocked -eq 0 && $idx -lt ${#tasks[@]} ]]; do
      local task=${tasks[$idx]}
      local dest=$out/$task
      if have_results "$dest"; then
        echo "HAVE $name $task" | tee -a "$MASTER"
        idx=$((idx+1))
        continue
      fi
      local free=""
      for g in "${GPUS[@]}"; do
        local busy=0
        for t in "${!GPU_OF[@]}"; do
          [[ -n ${PID[$t]+x} && ${GPU_OF[$t]} == "$g" ]] && busy=1
        done
        if [[ $busy -eq 0 ]]; then
          local used
          used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
          if [[ ${used:-99999} -lt 800 ]]; then free=$g; break; fi
        fi
      done
      [[ -z $free ]] && break
      echo "START gpu=$free $name $task $(date -Is)" | tee -a "$MASTER"
      mkdir -p "$dest"
      CUDA_VISIBLE_DEVICES=$free "$PY" -m lmms_eval \
        --model "$model" \
        --model_args "$args" \
        --tasks "$task" --batch_size 1 --log_samples \
        --output_path "$dest" \
        > "$LOGDIR/${name}_${task}.log" 2>&1 &
      PID[$task]=$!
      GPU_OF[$task]=$free
      idx=$((idx+1))
    done
    if [[ $blocked -eq 1 && ${#PID[@]} -eq 0 ]]; then
      break
    fi
    sleep 20
  done
  echo "MODEL_DONE $name $(date -Is)" | tee -a "$MASTER"
}

NEXT13=/home/xiangyue/.cache/huggingface/hub/models--llava-hf--llava-v1.6-vicuna-13b-hf/snapshots/745cfbdb14dfeff9bcc6eb731937102e88ea3024
QWEN3VL=/home/xiangyue/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
QWEN35=/home/xiangyue/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
NEXT_TASKS=(gqa pope mme mmbench_en_dev scienceqa_img textvqa_val_ocr)
QWEN_TASKS=(mme mmbench_en_dev pope textvqa_val_ocr gqa scienceqa_img ai2d chartqa ocrbench mmstar)

echo "REMAINING_DENSE_START $(date -Is)" | tee -a "$MASTER"
run_one_model llava-next-13b llava_hf "$NEXT13" "pretrained=$NEXT13,dtype=float16,attn_implementation=sdpa,device_map=cuda:0" "${NEXT_TASKS[@]}"
run_one_model qwen3-vl-8b qwen3_vl "$QWEN3VL" "pretrained=$QWEN3VL,device_map=cuda:0" "${QWEN_TASKS[@]}"
run_one_model qwen3.5-9b qwen3_5 "$QWEN35" "pretrained=$QWEN35,device_map=cuda:0" "${QWEN_TASKS[@]}"
echo "ALL_REMAINING_DENSE_DONE $(date -Is)" | tee -a "$MASTER"
