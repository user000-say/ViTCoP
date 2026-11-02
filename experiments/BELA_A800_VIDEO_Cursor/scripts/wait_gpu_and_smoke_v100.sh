#!/usr/bin/env bash
# Wait for a project GPU (0/2/4/7) with >=28GiB free, then run fidelity smoke + optional limit ACC.
set -euo pipefail
ROOT=/home/xiangyue/ViTCoP-a800-video/experiments/BELA_A800_VIDEO_Cursor
PY=/home/xiangyue/envs/starpro/bin/python
LOGDIR=$ROOT/logs
mkdir -p "$LOGDIR"
NEED_MIB=${NEED_MIB:-28672}
LIMIT=${LIMIT:-0}  # 0 = smoke only; >0 = also videomme --limit

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits -i 0,2,4,7 \
    | awk -F', ' -v need="$NEED_MIB" '$2+0 >= need {print $1; exit}'
}

echo "[wait] need_free_mib=$NEED_MIB project_gpus=0,2,4,7"
while true; do
  g=$(pick_gpu || true)
  if [[ -n "${g:-}" ]]; then
    echo "[wait] got gpu=$g at $(date -Is)"
    break
  fi
  sleep 60
done

export BELA_SMOKE_GPU=$g BELA_SMOKE_FRAMES=${BELA_SMOKE_FRAMES:-8}
echo "[smoke] starting on gpu=$g"
"$PY" -u "$ROOT/scripts/v100_fidelity_smoke.py" | tee "$LOGDIR/v100_fidelity_smoke.log"
ok=$("$PY" -c "import json;print(json.load(open('$LOGDIR/v100_fidelity_smoke.json'))['ok'])")
if [[ "$ok" != "True" ]]; then
  echo "[smoke] FAIL"; exit 2
fi
echo "[smoke] PASS"

if [[ "$LIMIT" -gt 0 ]]; then
  echo "[acc] videomme limit=$LIMIT on gpu=$g (fp16 V100 bring-up; not A800 claim)"
  export CUDA_VISIBLE_DEVICES=$g
  export HF_HOME=/home/xiangyue/src/hf_video
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  export BELA_LOCAL_DATASET=${BELA_LOCAL_DATASET:-/home/xiangyue/src/hf_video/ann/videomme}
  export BELA_VIDEO_MODEL=/home/xiangyue/src/models/LLaVA-Video-7B-Qwen2
  mkdir -p "$BELA_LOCAL_DATASET" 2>/dev/null || true
  # Prefer parquet-backed cache under HF_HOME/videomme
  OUT=$LOGDIR/videomme_limit${LIMIT}_dual_qattn
  mkdir -p "$OUT"
  PYTHONPATH="/home/xiangyue/src/LLaVA-NeXT:$ROOT:/home/xiangyue/ViTCoP-a800-video/lmms-eval:${PYTHONPATH:-}" \
  BELA_VIDEO_MODE=dualcover BELA_VIDEO_TIER=T32 BELA_VIDEO_STAGE2=Q_ATTN_ALL \
  BELA_VIDEO_DUAL=1 BELA_VIDEO_STAGE2_SCORE=qk \
  "$PY" "$ROOT/scripts/lmms_eval_video_arm_v100.py" \
    --arm DUAL_QATTN_FIDELITY --mode dualcover --stage2 Q_ATTN_ALL --tier T32 \
    --limit "$LIMIT" --output "$OUT" \
    --model_args "pretrained=$BELA_VIDEO_MODEL,conv_template=qwen_1_5,max_frames_num=32,device_map=cuda:0,torch_dtype=float16"
fi
