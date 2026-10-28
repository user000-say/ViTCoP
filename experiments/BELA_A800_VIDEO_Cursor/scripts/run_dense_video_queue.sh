#!/usr/bin/env bash
# Official Dense LLaVA-Video eval. Lives on the A800 node. Only physical GPUs 0 and 1.
# Pruned arms stay out of this queue until off==dense is proven.
set -u
ROOT=/data1/xiangyue
PY=$ROOT/envs/vitcop-a800-video/bin/python
OUT=$ROOT/logs/video_dense
LOG=$ROOT/logs/video_dense_queue.log
mkdir -p "$OUT" "$ROOT/hf" "$ROOT/tmp" "$ROOT/cache"
export PYTHONPATH=$ROOT/src/LLaVA-NeXT:$ROOT/src/lmms-eval
export HF_HOME=$ROOT/hf
export HUGGINGFACE_HUB_CACHE=$ROOT/hf/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR=$ROOT/tmp
MODEL=$ROOT/models/LLaVA-Video-7B-Qwen2
ARGS="pretrained=${MODEL},conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average,mm_spatial_pool_stride=2,attn_implementation=sdpa,force_sample=True,add_time_instruction=True,torch_dtype=bfloat16,video_decode_backend=decord"
TASKS=(videomme longvideobench_val_v mlvu_dev mvbench)
READY_FOR=(videomme longvideobench mlvu mvbench)
declare -A PID GPU_OF
idx=0
echo "QUEUE_START $(date -Is)" | tee -a "$LOG"
while [[ $idx -lt ${#TASKS[@]} || ${#PID[@]} -gt 0 ]]; do
  for task in "${!PID[@]}"; do
    if ! kill -0 "${PID[$task]}" 2>/dev/null; then
      wait "${PID[$task]}"
      echo "DONE $task rc=$? $(date -Is)" | tee -a "$LOG"
      unset "PID[$task]"
    fi
  done
  while [[ $idx -lt ${#TASKS[@]} ]]; do
    task=${TASKS[$idx]}
    flag=$ROOT/datasets/READY_${READY_FOR[$idx]}
    if [[ ! -f $flag ]]; then
      echo "WAIT_DATA $task $(date -Is)" | tee -a "$LOG"
      break
    fi
    if compgen -G "$OUT/$task/**/results.json" > /dev/null || compgen -G "$OUT/$task/*results*.json" > /dev/null; then
      echo "HAVE $task" | tee -a "$LOG"
      idx=$((idx+1))
      continue
    fi
    free=""
    for g in 0 1; do
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
    echo "START gpu=$free $task $(date -Is)" | tee -a "$LOG"
    mkdir -p "$OUT/$task"
    CUDA_VISIBLE_DEVICES=$free "$PY" -m lmms_eval \
      --model llava_vid \
      --model_args "$ARGS" \
      --tasks "$task" --batch_size 1 --log_samples \
      --output_path "$OUT/$task" \
      > "$OUT/${task}.log" 2>&1 &
    PID[$task]=$!
    GPU_OF[$task]=$free
    idx=$((idx+1))
  done
  sleep 30
done
echo "DENSE_VIDEO_QUEUE_DRAINED $(date -Is)" | tee -a "$LOG"
