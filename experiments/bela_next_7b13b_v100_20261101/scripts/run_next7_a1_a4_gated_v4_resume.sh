#!/bin/bash
# Resume pre-fidelity (v4) NeXT-7B gated ACC for unfinished arms A1–A4.
# A0 already finished R4 on v4 (R4=90.25 SKIP_6ACC) — not requeued.
# Policy: finish R4 for each arm; 6ACC only if R4>=92 and every task rel>=85.
set -u
ROOT=/home/xiangyue/ViTCoP
PKG=$ROOT/experiments/bela_next_7b13b_v100_20261101
PY=/home/xiangyue/envs/vitcop-v100-4model/bin/python
ENTRY=$PKG/scripts/lmms_bela_next_entry.py
GATE_PY=$PKG/scripts/gate_next7_arm_r4.py
OUT_ROOT=$PKG/logs/acc_a0_a4_94pct_v4
LOG=$PKG/logs/queue_v4_resume
mkdir -p "$OUT_ROOT" "$LOG"

export PYTHONPATH=$PKG:$PKG/BELA_NEXT_7B13B_V100_CURSOR_20261101/src:$ROOT/lmms-eval:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export BELA_NEXT_V2_ENABLE=1
export BELA_NEXT_MAX_NEW=256

ARMS=(
  NEXT7_T160_A1_PAPER_VAR
  NEXT7_T160_A2_LATE
  NEXT7_T160_A3_LESS_S1
  NEXT7_T160_A4_STRONG_LATE
)
R4_TASKS=(gqa pope mme mmbench_en_dev)
EXTRA_TASKS=(scienceqa_img textvqa_val_ocr)
# Project GPU set only
GPUS=(0 2 4 7)

JOBS=()
for arm in "${ARMS[@]}"; do
  for task in "${R4_TASKS[@]}"; do
    JOBS+=("${arm}|${task}")
  done
done

declare -A PID GPU_OF ARM_GATED
idx=0
n=${#JOBS[@]}

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

maybe_gate_arm() {
  local arm=$1
  [[ -n ${ARM_GATED[$arm]+x} ]] && return
  for t in "${R4_TASKS[@]}"; do
    if ! find "$OUT_ROOT/$arm/$t" -name '*_results.json' 2>/dev/null | grep -q .; then
      return
    fi
  done
  local verdict
  verdict=$("$PY" -u "$GATE_PY" --arm "$arm" --out-root "$OUT_ROOT" 2>>"$LOG/gate.log") || true
  ARM_GATED[$arm]=1
  echo "GATE $arm $verdict $(date -Is)" | tee -a "$LOG/queue.log"
  if [[ "$verdict" == GO* ]]; then
    for t in "${EXTRA_TASKS[@]}"; do
      JOBS+=("${arm}|${t}")
    done
    n=${#JOBS[@]}
    echo "ENQUEUE_6ACC $arm $(date -Is)" | tee -a "$LOG/queue.log"
  else
    echo "SKIP_6ACC $arm $verdict $(date -Is)" | tee -a "$LOG/queue.log"
  fi
}

echo "QUEUE_GATED_V4_RESUME_A1A4_BEGIN n=$n gpus=${GPUS[*]} r4_min=92 single_floor=85 code=25a0f57_pre_fidelity $(date -Is)" | tee -a "$LOG/queue.log"

for arm in "${ARMS[@]}"; do
  maybe_gate_arm "$arm"
done

declare -A RETRIES
while [[ $idx -lt $n || ${#PID[@]} -gt 0 ]]; do
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
    sleep 2
  done
  sleep 20
done

echo "QUEUE_GATED_V4_RESUME_DRAINED $(date -Is)" | tee -a "$LOG/queue.log"
if [[ -f "$PKG/scripts/summarize_next7_v4_acc.py" ]]; then
  "$PY" -u "$PKG/scripts/summarize_next7_v4_acc.py" | tee -a "$LOG/summary.log"
elif [[ -f "$PKG/scripts/summarize_next7_v5_acc.py" ]]; then
  # fall back: point summarize at v4 root via env if supported; else manual note
  echo "NOTE: use summarize against logs/acc_a0_a4_94pct_v4" | tee -a "$LOG/summary.log"
fi
