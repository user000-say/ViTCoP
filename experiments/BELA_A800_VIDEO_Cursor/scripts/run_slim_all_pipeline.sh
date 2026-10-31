#!/usr/bin/env bash
# Full slim video delivery on A800 — ACC then EFF. Idempotent + SSH-safe when launched via detach_slim_all.sh
# GPUs: prefer 2/3 (do not touch foreign jobs on 0/1).
set -u
ROOT=/data1/xiangyue
PY=$ROOT/envs/vitcop-a800-video/bin/python
SCR=$ROOT/ViTCoP/experiments/BELA_A800_VIDEO_Cursor/scripts
OUT_ACC=$ROOT/logs/video_slim_acc
OUT_EFF=$ROOT/logs/video_slim_eff
LOG=$ROOT/logs/video_slim_all_pipeline.log
STATUS=$ROOT/logs/VIDEO_SLIM_STATUS.json
mkdir -p "$OUT_ACC" "$OUT_EFF" "$ROOT/tmp" "$ROOT/cache" "$ROOT/hf" "$ROOT/logs"
export PYTHONPATH=$ROOT/src/LLaVA-NeXT:$ROOT/src/lmms-eval:$ROOT/ViTCoP/experiments/BELA_A800_VIDEO_Cursor
export HF_HOME=$ROOT/hf HUGGINGFACE_HUB_CACHE=$ROOT/hf/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false TMPDIR=$ROOT/tmp
MODEL=$ROOT/models/LLaVA-Video-7B-Qwen2
ARGS="pretrained=${MODEL},conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average,mm_spatial_pool_stride=2,attn_implementation=sdpa,force_sample=True,add_time_instruction=True,torch_dtype=bfloat16,video_decode_backend=decord"

write_status() {
  local phase=$1 msg=$2
  "$PY" - <<PY
import json,time,glob,os
from pathlib import Path
arms=["DENSE","C_QA_QATTN","D_QA_RDRIFT","STARPRO_T32_PROXY"]
acc={}
for a in arms:
  hits=glob.glob(f"$OUT_ACC/{a}/videomme/**/*results*.json", recursive=True)
  acc[a]=bool(hits)
eff={}
for a in arms:
  p=Path(f"$OUT_EFF/{a}/latency_summary.json")
  eff[a]=p.exists()
Path("$STATUS").write_text(json.dumps({
  "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
  "phase": "$phase",
  "msg": """$msg""",
  "acc_done": acc,
  "eff_done": eff,
  "pid": os.getpid(),
}, indent=2))
PY
}

have_acc() {
  local arm=$1
  compgen -G "$OUT_ACC/$arm/videomme/**/*results*.json" > /dev/null
}

arm_live() {
  local arm=$1
  pgrep -af "lmms_eval.*video_slim_acc/${arm}/" >/dev/null 2>&1
}

pick_gpu() {
  for g in 2 3; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$g" | tr -d ' ')
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null | grep -v '^$' | wc -l)
    if [[ "${used:-999999}" -lt 800 && "${util:-100}" -lt 10 && "$apps" -eq 0 ]]; then
      echo "$g"; return 0
    fi
  done
  return 1
}

echo "ALL_PIPELINE_START $(date -Is)" | tee -a "$LOG"
write_status "init" "pipeline start"
$PY "$SCR/patch_llava_video_stage2.py" >>"$LOG" 2>&1 || true
$PY "$SCR/patch_generate_mask_fix.py" >>"$LOG" 2>&1 || true

# arm|mode|stage2|tier
ARMS=(
  "DENSE|off|OFF|T32"
  "C_QA_QATTN|qacover|Q_ATTN_ALL|T32"
  "D_QA_RDRIFT|qacover|RDRIFT_FIRST|T32"
  "STARPRO_T32_PROXY|native|Q_ATTN_ALL|T32"
)

########################
# Phase A: Video-MME ACC
########################
write_status "acc" "Video-MME ACC queue"
declare -A PID
idx=0
while [[ $idx -lt ${#ARMS[@]} || ${#PID[@]} -gt 0 ]]; do
  for arm in "${!PID[@]}"; do
    if ! kill -0 "${PID[$arm]}" 2>/dev/null; then
      wait "${PID[$arm]}" || true
      echo "DONE_ACC $arm rc=$? $(date -Is)" | tee -a "$LOG"
      unset "PID[$arm]"
      write_status "acc" "finished $arm"
    fi
  done
  if [[ $idx -lt ${#ARMS[@]} ]]; then
    IFS='|' read -r arm mode s2 tier <<<"${ARMS[$idx]}"
    if have_acc "$arm"; then
      echo "HAVE_ACC $arm" | tee -a "$LOG"
      idx=$((idx+1)); continue
    fi
    if arm_live "$arm"; then
      echo "LIVE_EXTERNAL $arm — wait" | tee -a "$LOG"
      sleep 60; continue
    fi
    if g=$(pick_gpu); then
      dest=$OUT_ACC/$arm/videomme
      mkdir -p "$dest"
      (
        export CUDA_VISIBLE_DEVICES=$g
        export BELA_VIDEO_MODE=$mode BELA_VIDEO_TIER=$tier BELA_VIDEO_STAGE2=$s2
        export BELA_LOCAL_DATASET=$ROOT/hf/ann/videomme
        echo "START_ACC $arm gpu=$g $(date -Is)"
        $PY -m lmms_eval --model llava_vid --model_args "$ARGS" --tasks videomme \
          --batch_size 1 --output_path "$dest" --log_samples
        echo "FINISH_ACC $arm rc=$? $(date -Is)"
      ) >>"$LOG" 2>&1 &
      PID[$arm]=$!
      echo "LAUNCHED_ACC $arm pid=${PID[$arm]} gpu=$g" | tee -a "$LOG"
      idx=$((idx+1))
      sleep 8
    else
      echo "WAIT_GPU_ACC $(date -Is) busy=${#PID[@]} pending=$(( ${#ARMS[@]} - idx ))" | tee -a "$LOG"
      sleep 45
    fi
  else
    sleep 30
  fi
done
echo "ACC_PHASE_COMPLETE $(date -Is)" | tee -a "$LOG"
write_status "acc_done" "all ACC arms present or finished"

########################
# Phase B: EFF serial on one quiet GPU
########################
write_status "eff" "TTFT/Fixed32/FLOPs"
for spec in "${ARMS[@]}"; do
  IFS='|' read -r arm mode s2 tier <<<"$spec"
  if [[ -f $OUT_EFF/$arm/latency_summary.json ]]; then
    echo "HAVE_EFF $arm" | tee -a "$LOG"
    continue
  fi
  while ! g=$(pick_gpu); do
    echo "WAIT_GPU_EFF $arm $(date -Is)" | tee -a "$LOG"
    sleep 60
  done
  echo "START_EFF $arm gpu=$g $(date -Is)" | tee -a "$LOG"
  mkdir -p "$OUT_EFF/$arm"
  (
    export CUDA_VISIBLE_DEVICES=$g
    export BELA_EFF_GPU=0
    export BELA_VIDEO_MODE=$mode BELA_VIDEO_TIER=$tier BELA_VIDEO_STAGE2=$s2
    export BELA_EFF_ARM=$arm
    export BELA_EFF_OUT=$OUT_EFF/$arm/latency_summary.json
    export BELA_EFF_FRAMES=64 BELA_EFF_WARMUP=10 BELA_EFF_MEAS=50
    $PY "$SCR/bench_video_eff.py"
  ) >>"$LOG" 2>&1
  echo "FINISH_EFF $arm rc=$? $(date -Is)" | tee -a "$LOG"
  write_status "eff" "finished $arm"
done

########################
# Phase C: compact table
########################
$PY - <<'PY' >>"$LOG" 2>&1 || true
import json,glob
from pathlib import Path
root=Path("/data1/xiangyue/logs")
arms=["DENSE","C_QA_QATTN","D_QA_RDRIFT","STARPRO_T32_PROXY"]
rows=[]
for a in arms:
  hits=sorted(glob.glob(str(root/f"video_slim_acc/{a}/videomme/**/*results*.json"), recursive=True))
  acc=None
  if hits:
    d=json.loads(Path(hits[-1]).read_text())
    r=(d.get("results") or {}).get("videomme") or {}
    acc=r.get("videomme_perception_score,none")
  eff=None
  ep=root/f"video_slim_eff/{a}/latency_summary.json"
  if ep.exists():
    e=json.loads(ep.read_text())
    eff={"ttft_mean": e["ttft_wall_ms"]["mean"], "fixed32_mean": e["fixed32_wall_ms"]["mean"],
         "peak_gib": e.get("peak_allocated_gib"), "flops": e.get("flops")}
  rows.append({"arm":a,"videomme_acc":acc,"eff":eff})
out=root/"VIDEO_FINAL_PAPER_TABLE.md"
lines=["# VIDEO_FINAL_PAPER_TABLE","", "| Arm | Video-MME | TTFT ms | Fixed32 ms | peak GiB |", "|---|---:|---:|---:|---:|"]
for r in rows:
  e=r["eff"] or {}
  lines.append(f"| {r['arm']} | {r['videomme_acc']} | {e.get('ttft_mean','')} | {e.get('fixed32_mean','')} | {e.get('peak_gib','')} |")
out.write_text("\n".join(lines)+"\n")
(root/"VIDEO_REPRO_LOCK.json").write_text(json.dumps({"arms":rows,"note":"slim pipeline"}, indent=2))
print("WROTE", out)
PY

write_status "complete" "ACC+EFF+table done"
echo "ALL_PIPELINE_COMPLETE $(date -Is)" | tee -a "$LOG"
