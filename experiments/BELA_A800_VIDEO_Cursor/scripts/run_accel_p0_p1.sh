#!/usr/bin/env bash
# Detach-friendly P0 causal timing + P1 equivalence + EFF rebench (GPU 2/3).
set -euo pipefail
ROOT=/data1/xiangyue
SCR=$ROOT/ViTCoP/experiments/BELA_A800_VIDEO_Cursor/scripts
PY=$ROOT/envs/vitcop-a800-video/bin/python
export PYTHONPATH=$ROOT/src/LLaVA-NeXT:$ROOT/src/lmms-eval:$ROOT/ViTCoP/experiments/BELA_A800_VIDEO_Cursor
export BELA_VIDEO_QA_BATCH=1
LOGDIR=$ROOT/logs
mkdir -p "$LOGDIR" "$LOGDIR/video_accel_eff"
LOG=$LOGDIR/video_accel_p0p1.log
exec > >(tee -a "$LOG") 2>&1
echo "ACCEL_P0P1_START $(date -Is)"

$PY "$SCR/patch_llava_video_stage2.py" || true
$PY "$SCR/patch_generate_mask_fix.py" || true

echo "=== P1 equivalence (CUDA) ==="
CUDA_VISIBLE_DEVICES=2 BELA_EQ_OUT=$LOGDIR/VIDEO_QA_BATCH_EQUIVALENCE.json \
  $PY "$SCR/test_qa_batch_equivalence.py"

echo "=== P0 causal timing (batch=1) ==="
BELA_EFF_GPU=2 BELA_EFF_WARMUP=5 BELA_EFF_MEAS=20 \
  BELA_P0_OUT=$LOGDIR/VIDEO_CAUSAL_TIMING.json \
  $PY "$SCR/bench_causal_timing_p0.py"

echo "=== EFF rebench Dense / Native / QA_BATCH (GPU2 serial) ==="
for spec in "DENSE|off|OFF" "NATIVE_QATTN|native|Q_ATTN_ALL" "C_QA_QATTN_BATCH|qacover|Q_ATTN_ALL"; do
  IFS='|' read -r arm mode s2 <<<"$spec"
  echo "EFF $arm"
  BELA_EFF_GPU=2 BELA_VIDEO_MODE=$mode BELA_VIDEO_TIER=T32 BELA_VIDEO_STAGE2=$s2 \
    BELA_EFF_ARM=$arm BELA_EFF_OUT=$LOGDIR/video_accel_eff/$arm/latency_summary.json \
    BELA_EFF_WARMUP=10 BELA_EFF_MEAS=50 \
    $PY "$SCR/bench_video_eff.py"
done

# Also compare old serial QA path once (batch off) for tax delta
echo "EFF C_QA_SERIAL (batch off)"
BELA_EFF_GPU=2 BELA_VIDEO_MODE=qacover BELA_VIDEO_TIER=T32 BELA_VIDEO_STAGE2=Q_ATTN_ALL \
  BELA_VIDEO_QA_BATCH=0 \
  BELA_EFF_ARM=C_QA_QATTN_SERIAL BELA_EFF_OUT=$LOGDIR/video_accel_eff/C_QA_QATTN_SERIAL/latency_summary.json \
  BELA_EFF_WARMUP=10 BELA_EFF_MEAS=50 \
  $PY "$SCR/bench_video_eff.py"

$PY - <<'PY'
import json
from pathlib import Path
root=Path("/data1/xiangyue/logs/video_accel_eff")
rows=[]
for p in sorted(root.glob("*/latency_summary.json")):
    d=json.loads(p.read_text())
    rows.append((d["arm"], d["ttft_wall_ms"]["mean"], d["fixed32_wall_ms"]["mean"], d.get("peak_allocated_gib")))
md=["# VIDEO_ACCEL_EFF_REBENCH","","| Arm | TTFT ms | Fixed32 ms | peak GiB |","|---|---:|---:|---:|"]
for a,t,f,m in rows:
    md.append(f"| {a} | {t:.2f} | {f:.2f} | {m:.2f} |")
Path("/data1/xiangyue/logs/VIDEO_ACCEL_EFF_REBENCH.md").write_text("\n".join(md)+"\n")
print("\n".join(md))
PY

echo "ACCEL_P0P1_DONE $(date -Is)"
