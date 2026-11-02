#!/bin/bash
# NeXT-7B A0–A4 @94.4% — six tasks on physical GPUs 4/5/6/7 only.
# Requires smoke PASS (logs/smoke_a0_a4/SMOKE_VERDICT.md ALL_PASS=True).
set -u
ROOT=/home/xiangyue/ViTCoP
PKG=$ROOT/experiments/bela_next_7b13b_v100_20261101
PY=/home/xiangyue/envs/vitcop-v100-4model/bin/python
ENTRY=$PKG/scripts/lmms_bela_next_entry.py
OUT_ROOT=$PKG/logs/acc_a0_a4_94pct_v3
LOG=$PKG/logs/queue_v3
mkdir -p "$OUT_ROOT" "$LOG"

export PYTHONPATH=$PKG:$PKG/BELA_NEXT_7B13B_V100_CURSOR_20261101/src:$ROOT/lmms-eval:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export BELA_NEXT_V2_ENABLE=1
export BELA_NEXT_MAX_NEW=64

ARMS=(
  NEXT7_T160_A0_PAPER
  NEXT7_T160_A1_PAPER_VAR
  NEXT7_T160_A2_LATE
  NEXT7_T160_A3_LESS_S1
  NEXT7_T160_A4_STRONG_LATE
)
TASKS=(gqa pope mme mmbench_en_dev scienceqa_img textvqa_val_ocr)
GPUS=(4 5 6 7)

# Build job list arm|task
JOBS=()
for arm in "${ARMS[@]}"; do
  for task in "${TASKS[@]}"; do
    JOBS+=("${arm}|${task}")
  done
done

declare -A PID GPU_OF JOB_OF
idx=0
n=${#JOBS[@]}

launch() {
  local gpu=$1
  local arm=$2
  local task=$3
  local dest=$OUT_ROOT/$arm/$task
  local tag="${arm}__${task}"
  if find "$dest" -name '*_results.json' 2>/dev/null | grep -q .; then
    echo "HAVE $tag" | tee -a "$LOG/queue.log"
    return
  fi
  mkdir -p "$dest"
  echo "START gpu=$gpu $tag $(date -Is)" | tee -a "$LOG/queue.log"
  CUDA_VISIBLE_DEVICES=$gpu \
  BELA_NEXT_V2_ENABLE=1 \
  BELA_NEXT_ARM=$arm \
  BELA_NEXT_MAX_NEW=64 \
  "$PY" -u "$ENTRY" \
    --model llava_hf \
    --model_args pretrained=llava-hf/llava-v1.6-vicuna-7b-hf,dtype=float16,attn_implementation=sdpa,device_map=cuda:0 \
    --tasks "$task" --batch_size 1 --log_samples \
    --output_path "$dest" \
    > "$LOG/${tag}.log" 2>&1 &
  PID[$tag]=$!
  GPU_OF[$tag]=$gpu
  JOB_OF[$tag]="$arm|$task"
}

echo "QUEUE_BEGIN n=$n gpus=${GPUS[*]} $(date -Is)" | tee -a "$LOG/queue.log"

while [[ $idx -lt $n || ${#PID[@]} -gt 0 ]]; do
  for tag in "${!PID[@]}"; do
    if ! kill -0 "${PID[$tag]}" 2>/dev/null; then
      wait "${PID[$tag]}"
      rc=$?
      echo "DONE $tag rc=$rc gpu=${GPU_OF[$tag]} $(date -Is)" | tee -a "$LOG/queue.log"
      unset "PID[$tag]"
      unset "GPU_OF[$tag]"
      unset "JOB_OF[$tag]"
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
  sleep 15
done

echo "QUEUE_DRAINED $(date -Is)" | tee -a "$LOG/queue.log"
# summary
"$PY" - <<'PY'
import json
from pathlib import Path
root=Path("/home/xiangyue/ViTCoP/experiments/bela_next_7b13b_v100_20261101/logs/acc_a0_a4_94pct_v3")
arms=["NEXT7_T160_A0_PAPER","NEXT7_T160_A1_PAPER_VAR","NEXT7_T160_A2_LATE","NEXT7_T160_A3_LESS_S1","NEXT7_T160_A4_STRONG_LATE"]
tasks=["gqa","pope","mme","mmbench_en_dev","scienceqa_img","textvqa_val_ocr"]
rows=[]
for a in arms:
  for t in tasks:
    hits=list((root/a/t).glob("**/*_results.json"))
    rows.append({"arm":a,"task":t,"have":bool(hits),"path":str(hits[0]) if hits else str(root/a/t)})
miss=[r for r in rows if not r["have"]]
print(json.dumps({"total":len(rows),"have":len(rows)-len(miss),"missing":miss},indent=2))
(root/"QUEUE_SUMMARY.json").write_text(json.dumps(rows,indent=2))
PY
