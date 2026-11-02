#!/bin/bash
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
ARM=NEXT7_T160_A0_PAPER
TASKS=(gqa pope mme mmbench_en_dev)
GPUS=(4 5 6 7)
echo "A0_R4_BEGIN $(date -Is)" | tee -a "$LOG/queue.log"
idx=0
declare -A PID GPU_OF
while [[ $idx -lt ${#TASKS[@]} || ${#PID[@]} -gt 0 ]]; do
  for tag in "${!PID[@]}"; do
    if ! kill -0 "${PID[$tag]}" 2>/dev/null; then
      wait "${PID[$tag]}"; echo "DONE $tag rc=$? $(date -Is)" | tee -a "$LOG/queue.log"
      unset "PID[$tag]"; unset "GPU_OF[$tag]"
    fi
  done
  while [[ $idx -lt ${#TASKS[@]} ]]; do
    free=""
    for g in "${GPUS[@]}"; do
      busy=0
      for t in "${!GPU_OF[@]}"; do [[ -n ${PID[$t]+x} && ${GPU_OF[$t]} == "$g" ]] && busy=1; done
      if [[ $busy -eq 0 ]]; then
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
        [[ ${used:-99999} -lt 800 ]] && free=$g && break
      fi
    done
    [[ -z $free ]] && break
    task=${TASKS[$idx]}
    dest=$OUT_ROOT/$ARM/$task
    tag="${ARM}__${task}"
    if find "$dest" -name '*_results.json' 2>/dev/null | grep -q .; then
      echo "HAVE $tag" | tee -a "$LOG/queue.log"; idx=$((idx+1)); continue
    fi
    mkdir -p "$dest"
    echo "START gpu=$free $tag $(date -Is)" | tee -a "$LOG/queue.log"
    CUDA_VISIBLE_DEVICES=$free BELA_NEXT_ARM=$ARM BELA_NEXT_V2_ENABLE=1 BELA_NEXT_MAX_NEW=256 \
      "$PY" -u "$ENTRY" --model llava_hf \
      --model_args pretrained=llava-hf/llava-v1.6-vicuna-7b-hf,dtype=float16,attn_implementation=sdpa,device_map=cuda:0 \
      --tasks "$task" --batch_size 1 --log_samples --output_path "$dest" \
      > "$LOG/${tag}.log" 2>&1 &
    PID[$tag]=$!; GPU_OF[$tag]=$free; idx=$((idx+1)); sleep 2
  done
  sleep 15
done
echo "A0_R4_DRAINED $(date -Is)" | tee -a "$LOG/queue.log"
