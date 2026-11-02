#!/bin/bash
# Re-run tasks that finished without *_results.json (OOM crashes marked DONE).
set -u
PKG=/home/xiangyue/ViTCoP/experiments/bela_next_7b13b_v100_20261101
PY=/home/xiangyue/envs/vitcop-v100-4model/bin/python
ENTRY=$PKG/scripts/lmms_bela_next_entry.py
OUT_ROOT=$PKG/logs/acc_a0_a4_94pct_v3
LOG=$PKG/logs/queue_v3
export PYTHONPATH=$PKG:$PKG/BELA_NEXT_7B13B_V100_CURSOR_20261101/src:/home/xiangyue/ViTCoP/lmms-eval:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export BELA_NEXT_V2_ENABLE=1
export BELA_NEXT_MAX_NEW=64
MODEL=llava-hf/llava-v1.6-vicuna-7b-hf

# Wait until main queue drained or GPUs free enough for backlog
while pgrep -f 'run_next7_a0_a4_six.sh' >/dev/null; do sleep 60; done

ARMS=(NEXT7_T160_A0_PAPER NEXT7_T160_A1_PAPER_VAR NEXT7_T160_A2_LATE NEXT7_T160_A3_LESS_S1 NEXT7_T160_A4_STRONG_LATE)
TASKS=(mmbench_en_dev textvqa_val_ocr)
GPUS=(4 5 6 7)

JOBS=()
for arm in "${ARMS[@]}"; do
  for task in "${TASKS[@]}"; do
    dest=$OUT_ROOT/$arm/$task
    if find "$dest" -name '*_results.json' 2>/dev/null | grep -q .; then
      continue
    fi
    JOBS+=("$arm|$task")
  done
done
echo "REQUEUE n=${#JOBS[@]} $(date -Is)" | tee -a "$LOG/requeue.log"
idx=0
declare -A PID GPU_OF
while [[ $idx -lt ${#JOBS[@]} || ${#PID[@]} -gt 0 ]]; do
  for tag in "${!PID[@]}"; do
    if ! kill -0 "${PID[$tag]}" 2>/dev/null; then
      wait "${PID[$tag]}"; echo "DONE $tag rc=$? $(date -Is)" | tee -a "$LOG/requeue.log"
      unset "PID[$tag]"; unset "GPU_OF[$tag]"
    fi
  done
  while [[ $idx -lt ${#JOBS[@]} ]]; do
    free=""
    for g in "${GPUS[@]}"; do
      busy=0
      for t in "${!GPU_OF[@]}"; do [[ ${GPU_OF[$t]} == "$g" && -n ${PID[$t]+x} ]] && busy=1; done
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
      if [[ $busy -eq 0 && ${used:-99999} -lt 800 ]]; then free=$g; break; fi
    done
    [[ -z $free ]] && break
    IFS='|' read -r arm task <<< "${JOBS[$idx]}"
    tag="${arm}__${task}"
    dest=$OUT_ROOT/$arm/$task
    mkdir -p "$dest"
    echo "START gpu=$free $tag $(date -Is)" | tee -a "$LOG/requeue.log"
    CUDA_VISIBLE_DEVICES=$free BELA_NEXT_ARM=$arm BELA_NEXT_V2_ENABLE=1 BELA_NEXT_MAX_NEW=64 \
      "$PY" -u "$ENTRY" --model llava_hf \
      --model_args pretrained=$MODEL,dtype=float16,attn_implementation=sdpa,device_map=cuda:0 \
      --tasks "$task" --batch_size 1 --log_samples --output_path "$dest" \
      > "$LOG/requeue_${tag}.log" 2>&1 &
    PID[$tag]=$!; GPU_OF[$tag]=$free; idx=$((idx+1)); sleep 2
  done
  sleep 20
done
echo "REQUEUE_DRAINED $(date -Is)" | tee -a "$LOG/requeue.log"
