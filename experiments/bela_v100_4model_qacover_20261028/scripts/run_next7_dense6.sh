#!/bin/bash
# Native LLaVA-NeXT-7B, no pruning. Isolated env only. One model, tasks queued on free project GPUs.
set -u
ROOT=/home/xiangyue/ViTCoP
PY=/home/xiangyue/envs/vitcop-v100-4model/bin/python
OUT=$ROOT/logs/e014_aee/full/bela_v100_4model_20261028/llava-next-7b/DENSE
LOG=$ROOT/experiments/bela_v100_4model_qacover_20261028/logs
mkdir -p "$OUT" "$LOG"
export PYTHONPATH=$ROOT/lmms-eval
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
TASKS=(gqa pope mme mmbench_en_dev scienceqa_img textvqa_val_ocr)
GPUS=(0 2 4 7)
declare -A PID GPU_OF
running=0
idx=0
launch() {
  local gpu=$1 task=$2
  local dest=$OUT/$task
  if [[ -f $dest/metrics.json ]]; then
    echo "HAVE $task" | tee -a "$LOG/next7_dense6.log"
    return
  fi
  mkdir -p "$dest"
  echo "START gpu=$gpu $task $(date -Is)" | tee -a "$LOG/next7_dense6.log"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -m lmms_eval \
    --model llava_hf \
    --model_args pretrained=llava-hf/llava-v1.6-vicuna-7b-hf,dtype=float16,attn_implementation=sdpa,device_map=cuda:0 \
    --tasks "$task" --batch_size 1 --log_samples \
    --output_path "$dest" \
    > "$LOG/next7_dense_${task}.log" 2>&1 &
  PID[$task]=$!
  GPU_OF[$task]=$gpu
}
while [[ $idx -lt ${#TASKS[@]} || ${#PID[@]} -gt 0 ]]; do
  for task in "${!PID[@]}"; do
    if ! kill -0 "${PID[$task]}" 2>/dev/null; then
      wait "${PID[$task]}"
      echo "DONE $task rc=$? $(date -Is)" | tee -a "$LOG/next7_dense6.log"
      unset "PID[$task]"
    fi
  done
  while [[ $idx -lt ${#TASKS[@]} ]]; do
    free=""
    for g in "${GPUS[@]}"; do
      busy=0
      for t in "${!GPU_OF[@]}"; do
        [[ -n ${PID[$t]+x} && ${GPU_OF[$t]} == "$g" ]] && busy=1
      done
      if [[ $busy -eq 0 ]]; then
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
        if [[ ${used:-99999} -lt 800 ]]; then free=$g; break; fi
      fi
    done
    [[ -z $free ]] && break
    launch "$free" "${TASKS[$idx]}"
    idx=$((idx+1))
  done
  sleep 20
done
echo "NEXT7_DENSE6_DRAINED $(date -Is)" | tee -a "$LOG/next7_dense6.log"
