#!/usr/bin/env bash
# Slim ACC queue: Dense + C (QA+Q_ATTN) + D (QA+RDRIFT) + STARPRO_T32 proxy.
# Uses free cards among physical GPUs 2/3 (0/1 often occupied by others).
set -u
ROOT=/data1/xiangyue
PY=$ROOT/envs/vitcop-a800-video/bin/python
OUT=$ROOT/logs/video_slim_acc
LOG=$ROOT/logs/video_slim_acc_queue.log
mkdir -p "$OUT" "$ROOT/tmp" "$ROOT/cache" "$ROOT/hf"
export PYTHONPATH=$ROOT/src/LLaVA-NeXT:$ROOT/src/lmms-eval:$ROOT/ViTCoP/experiments/BELA_A800_VIDEO_Cursor
export HF_HOME=$ROOT/hf HUGGINGFACE_HUB_CACHE=$ROOT/hf/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false TMPDIR=$ROOT/tmp
MODEL=$ROOT/models/LLaVA-Video-7B-Qwen2
ARGS="pretrained=${MODEL},conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average,mm_spatial_pool_stride=2,attn_implementation=sdpa,force_sample=True,add_time_instruction=True,torch_dtype=bfloat16,video_decode_backend=decord"

# Apply stage2 patch once
$PY $ROOT/ViTCoP/experiments/BELA_A800_VIDEO_Cursor/scripts/patch_llava_video_stage2.py >>"$LOG" 2>&1 || true

# arm|mode|stage2|tier|task
JOBS=(
  "DENSE|off|OFF|T32|videomme"
  "C_QA_QATTN|qacover|Q_ATTN_ALL|T32|videomme"
  "D_QA_RDRIFT|qacover|RDRIFT_FIRST|T32|videomme"
  "STARPRO_T32_PROXY|native|Q_ATTN_ALL|T32|videomme"
)

pick_gpu() {
  for g in 2 3; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
    if [[ "${used:-999999}" -lt 500 ]]; then
      echo "$g"
      return 0
    fi
  done
  return 1
}

echo "SLIM_ACC_START $(date -Is)" | tee -a "$LOG"
declare -A PID
idx=0
while [[ $idx -lt ${#JOBS[@]} ]]; do
  # reap
  for arm in "${!PID[@]}"; do
    if ! kill -0 "${PID[$arm]}" 2>/dev/null; then
      wait "${PID[$arm]}"
      rc=$?
      echo "DONE $arm rc=$rc $(date -Is)" | tee -a "$LOG"
      unset "PID[$arm]"
    fi
  done
  # launch if free gpu
  if g=$(pick_gpu); then
    IFS='|' read -r arm mode s2 tier task <<<"${JOBS[$idx]}"
    dest=$OUT/$arm/$task
    if compgen -G "$dest/**/*results*.json" > /dev/null; then
      echo "HAVE $arm" | tee -a "$LOG"
      idx=$((idx+1))
      continue
    fi
    mkdir -p "$dest"
    (
      export CUDA_VISIBLE_DEVICES=$g
      export BELA_VIDEO_MODE=$mode
      export BELA_VIDEO_TIER=$tier
      export BELA_VIDEO_STAGE2=$s2
      export BELA_LOCAL_DATASET=$ROOT/hf/ann/videomme
      echo "START $arm gpu=$g mode=$mode s2=$s2 $(date -Is)"
      $PY -m lmms_eval \
        --model llava \
        --model_args "$ARGS" \
        --tasks "$task" \
        --batch_size 1 \
        --output_path "$dest" \
        --log_samples
      echo "FINISH $arm rc=$? $(date -Is)"
    ) >>"$LOG" 2>&1 &
    PID[$arm]=$!
    echo "LAUNCHED $arm pid=${PID[$arm]} gpu=$g" | tee -a "$LOG"
    idx=$((idx+1))
    sleep 5
  else
    echo "WAIT_GPU $(date -Is)" | tee -a "$LOG"
    sleep 60
  fi
done
for arm in "${!PID[@]}"; do
  wait "${PID[$arm]}"
  echo "DONE $arm rc=$? $(date -Is)" | tee -a "$LOG"
done
echo "SLIM_ACC_END $(date -Is)" | tee -a "$LOG"
