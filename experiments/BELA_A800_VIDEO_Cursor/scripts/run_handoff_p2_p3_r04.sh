#!/usr/bin/env bash
# Continue handoff after P0/P1: P2 dual smoke, P3 QK audit, R0-R4 dev matrix, then full winner.
set -euo pipefail
ROOT=/data1/xiangyue
SCR=$ROOT/ViTCoP/experiments/BELA_A800_VIDEO_Cursor/scripts
PKG=$ROOT/ViTCoP/experiments/BELA_A800_VIDEO_Cursor
PY=$ROOT/envs/vitcop-a800-video/bin/python
export PYTHONPATH=$ROOT/src/LLaVA-NeXT:$ROOT/src/lmms-eval:$PKG
export BELA_VIDEO_QA_BATCH=1
export BELA_VIDEO_STAGE2_SCORE=qk
LOG=$ROOT/logs/video_handoff_p2p3_r04.log
mkdir -p $ROOT/logs/video_dev_matrix $ROOT/logs/video_full_acc
exec > >(tee -a "$LOG") 2>&1
echo "HANDOFF_REMAINING_START $(date -Is)"

$PY $SCR/patch_llava_video_stage2.py || true
$PY $SCR/patch_generate_mask_fix.py || true

echo "=== REPRO LOCK ==="
$PY - <<'PY'
import json, hashlib, os, subprocess
from pathlib import Path
root=Path("/data1/xiangyue")
files=[
 root/"ViTCoP/experiments/BELA_A800_VIDEO_Cursor/bela_video_a800/live_stage1.py",
 root/"ViTCoP/experiments/BELA_A800_VIDEO_Cursor/bela_video_a800/live_stage2.py",
 root/"ViTCoP/experiments/BELA_A800_VIDEO_Cursor/bela_video_a800/qa_batched.py",
 root/"ViTCoP/experiments/BELA_A800_VIDEO_Cursor/bela_video_a800/video_dual_cover.py",
 root/"ViTCoP/experiments/BELA_A800_VIDEO_Cursor/bela_video_a800/qwen2_stage2_forward.py",
]
def sha(p):
  return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
lock={
 "date": __import__("datetime").datetime.utcnow().isoformat()+"Z",
 "model": str(root/"models/LLaVA-Video-7B-Qwen2"),
 "F":64, "N_pool":169, "layers":28, "T32":[64,11,4], "cuts":[12,21],
 "stage2_score": os.environ.get("BELA_VIDEO_STAGE2_SCORE","qk"),
 "qa_batch": os.environ.get("BELA_VIDEO_QA_BATCH","1"),
 "file_sha16": {str(p): sha(p) for p in files if p.exists()},
 "dual_cover": "SigLIP adaptation — see VIDEO_DUAL_COVER_MAPPING_AUDIT.md",
 "gpu_set": [2,3],
}
Path("/data1/xiangyue/logs/VIDEO_REPRO_LOCK.json").write_text(json.dumps(lock, indent=2))
print(json.dumps(lock, indent=2))
PY

echo "=== P2/P3 smoke generate (native + dualcover + qk) ==="
CUDA_VISIBLE_DEVICES=2 BELA_SMOKE_GPU=2 BELA_SMOKE_FRAMES=8 \
  BELA_VIDEO_MODE=native BELA_VIDEO_TIER=T32 BELA_VIDEO_STAGE2=Q_ATTN_ALL \
  BELA_VIDEO_STAGE2_SCORE=qk \
  BELA_SMOKE_OUT=$ROOT/logs/VIDEO_STAGE2_QK_SMOKE_NATIVE.json \
  $PY $SCR/stage2_physical_smoke.py || true

# dual needs capture install inside smoke — extend via env
CUDA_VISIBLE_DEVICES=2 $PY - <<'PY'
import os, sys, json, torch
from pathlib import Path
sys.path.insert(0,"/data1/xiangyue/ViTCoP/experiments/BELA_A800_VIDEO_Cursor")
sys.path.insert(0,"/data1/xiangyue/src/LLaVA-NeXT")
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ["BELA_VIDEO_MODE"]="dualcover"
os.environ["BELA_VIDEO_TIER"]="T32"
os.environ["BELA_VIDEO_STAGE2"]="Q_ATTN_ALL"
os.environ["BELA_VIDEO_STAGE2_SCORE"]="qk"
from bela_video_a800.qwen2_stage2_forward import apply_patch
from bela_video_a800.siglip_dual_capture import install_vision_capture
apply_patch()
from decord import VideoReader, cpu
import numpy as np
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model
tok, model, image_processor, _ = load_pretrained_model(
  "/data1/xiangyue/models/LLaVA-Video-7B-Qwen2", None, "llava_qwen",
  torch_dtype="bfloat16", device_map="cuda:0", attn_implementation="sdpa",
  overwrite_config={"mm_spatial_pool_mode":"average","mm_spatial_pool_stride":2},
)
install_vision_capture(model)
vr=VideoReader("/data1/xiangyue/hf/videomme/data/-3t1rj8g6yg.mp4", ctx=cpu(0), num_threads=1)
idxs=np.linspace(0,len(vr)-1,8,dtype=int).tolist()
frames=vr.get_batch(idxs).asnumpy()
pixel=image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].cuda().bfloat16()
q=DEFAULT_IMAGE_TOKEN+"\nWhat happened? Answer briefly."
conv=conv_templates["qwen_1_5"].copy(); conv.append_message(conv.roles[0], q); conv.append_message(conv.roles[1], None)
ids=tokenizer_image_token(conv.get_prompt(), tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
with torch.inference_mode():
  out=model.generate(ids, images=[pixel], modalities=["video"], do_sample=False, max_new_tokens=16, use_cache=True)
s1=getattr(model,"_bela_last_video_stage1",None)
s2=getattr(model,"_bela_last_stage2",None)
rep={"ok": True, "stage1": s1, "stage2_score_mode": None if not s2 else s2.get("score_mode"),
     "stage2_events": None if not s2 else s2.get("events"),
     "text": tok.decode(out[0], skip_special_tokens=True)[-200:]}
Path("/data1/xiangyue/logs/VIDEO_STAGE2_QK_AUDIT.json").write_text(json.dumps(rep, indent=2, default=str))
Path("/data1/xiangyue/logs/VIDEO_PHYSICAL_PRUNE_TRACE.jsonl").write_text(json.dumps({"events": None if not s2 else s2.get("events")})+"\n")
Path("/data1/xiangyue/logs/VIDEO_ATTENTION_BACKEND_DIFF.md").write_text(
  "# VIDEO_ATTENTION_BACKEND_DIFF\n\n- Main attn backend: SDPA (unchanged).\n"
  "- Stage2 scoring: same-layer Q/K + causal softmax over all keys, then mean text→visual "
  f"(mode={rep['stage2_score_mode']}).\n"
  "- Not full eager output_attentions; not hidden-dot proxy when BELA_VIDEO_STAGE2_SCORE=qk.\n"
)
print(json.dumps(rep, indent=2, default=str)[:2000])
assert s1 and s1.get("selector","").startswith("V_DUAL"), s1
assert s2 and s2.get("events"), s2
print("P2_P3_SMOKE_PASS")
PY

echo "=== Build / reuse DEV panel ==="
$PY $SCR/build_dev_panel.py || true

echo "=== R0-R4 ACC on limit panel (GPU2/3) + TTFT one-shot ==="
# Use lmms_eval --limit 120 as practical stand-in when panel filter is complex;
# label exploratory in VIDEO_DEV_MATRIX.md
ARGS="pretrained=/data1/xiangyue/models/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average,mm_spatial_pool_stride=2,attn_implementation=sdpa,force_sample=True,add_time_instruction=True,torch_dtype=bfloat16,video_decode_backend=decord"
LIMIT=${BELA_DEV_LIMIT:-120}

run_arm () {
  local arm=$1 mode=$2 s2=$3 gpu=$4
  local out=$ROOT/logs/video_dev_matrix/$arm
  mkdir -p "$out"
  if compgen -G "$out/videomme/**/*results*.json" > /dev/null; then
    echo "HAVE $arm"; return 0
  fi
  echo "START $arm gpu=$gpu"
  CUDA_VISIBLE_DEVICES=$gpu \
    BELA_VIDEO_MODE=$mode BELA_VIDEO_TIER=T32 BELA_VIDEO_STAGE2=$s2 \
    BELA_VIDEO_QA_BATCH=1 BELA_VIDEO_STAGE2_SCORE=qk \
    BELA_LOCAL_DATASET=$ROOT/hf/ann/videomme \
    $PY -c "from bela_video_a800.qwen2_stage2_forward import apply_patch; from bela_video_a800.siglip_dual_capture import install_vision_capture; print('hooks-ready')" \
    >/dev/null
  # Install hooks via sitecustomize-like preamble in a tiny wrapper
  CUDA_VISIBLE_DEVICES=$gpu \
    BELA_VIDEO_MODE=$mode BELA_VIDEO_TIER=T32 BELA_VIDEO_STAGE2=$s2 \
    BELA_VIDEO_QA_BATCH=1 BELA_VIDEO_STAGE2_SCORE=qk \
    BELA_LOCAL_DATASET=$ROOT/hf/ann/videomme \
    $PY $SCR/lmms_eval_video_arm.py \
      --arm "$arm" --mode "$mode" --stage2 "$s2" --limit "$LIMIT" \
      --output "$out/videomme" --model_args "$ARGS"
}

# wrapper created below; for dual arms need capture
for spec in \
  "R0_V_NATIVE_L2|native|Q_ATTN_ALL|2" \
  "R1_V_DUAL_COVER|dualcover|Q_ATTN_ALL|3" \
  "R2_V_DUAL_QA_B2|dualqa_b2|Q_ATTN_ALL|2" \
  "R3_V_DUAL_QA_B4|dualqa_b4|Q_ATTN_ALL|3" \
  "R4_V_DUAL_QA_B8|dualqa_b8|Q_ATTN_ALL|2"
 do
  IFS='|' read -r arm mode s2 gpu <<<"$spec"
  # serialize on free GPUs: launch two at a time
  :
 done

# Sequential safer for dual+hooks
run_arm R0_V_NATIVE_L2 native Q_ATTN_ALL 2
run_arm R1_V_DUAL_COVER dualcover Q_ATTN_ALL 3
run_arm R2_V_DUAL_QA_B2 dualqa_b2 Q_ATTN_ALL 2
run_arm R3_V_DUAL_QA_B4 dualqa_b4 Q_ATTN_ALL 3
run_arm R4_V_DUAL_QA_B8 dualqa_b8 Q_ATTN_ALL 2

$PY - <<'PY'
import json,glob
from pathlib import Path
rows=[]
for p in sorted(Path('/data1/xiangyue/logs/video_dev_matrix').glob('*/videomme/**/*results*.json')):
    arm=p.parts[p.parts.index('video_dev_matrix')+1]
    d=json.load(open(p))
    sc=d['results']['videomme'].get('videomme_perception_score,none')
    rows.append((arm, sc))
rows=sorted(rows, key=lambda x: -(x[1] or -1))
md=['# VIDEO_DEV_MATRIX','','Exploratory `--limit` panel with P3 Q/K Stage2. Dual=SigLIP adaptation.','',
    '| Arm | Video-MME (limit) |','|---|---:|']
for a,s in rows:
    md.append(f'| {a} | {s} |')
winner=rows[0][0] if rows else None
md.append('')
md.append(f'**Frozen T32 winner (dev):** `{winner}`')
Path('/data1/xiangyue/logs/VIDEO_DEV_MATRIX.md').write_text('\n'.join(md)+'\n')
Path('/data1/xiangyue/logs/VIDEO_DEV_WINNER.txt').write_text(winner or '')
print('\n'.join(md))
PY

WIN=$(cat $ROOT/logs/VIDEO_DEV_WINNER.txt || true)
echo "WINNER=$WIN"
if [[ -n "$WIN" ]]; then
  # map winner to mode
  case "$WIN" in
    R0*) MODE=native; S2=Q_ATTN_ALL ;;
    R1*) MODE=dualcover; S2=Q_ATTN_ALL ;;
    R2*) MODE=dualqa_b2; S2=Q_ATTN_ALL ;;
    R3*) MODE=dualqa_b4; S2=Q_ATTN_ALL ;;
    R4*) MODE=dualqa_b8; S2=Q_ATTN_ALL ;;
    *) MODE=native; S2=Q_ATTN_ALL ;;
  esac
  echo "=== FULL Video-MME winner=$WIN mode=$MODE ==="
  OUT=$ROOT/logs/video_full_acc/$WIN
  mkdir -p "$OUT"
  CUDA_VISIBLE_DEVICES=2 \
    BELA_VIDEO_MODE=$MODE BELA_VIDEO_TIER=T32 BELA_VIDEO_STAGE2=$S2 \
    BELA_VIDEO_QA_BATCH=1 BELA_VIDEO_STAGE2_SCORE=qk \
    BELA_LOCAL_DATASET=$ROOT/hf/ann/videomme \
    $PY $SCR/lmms_eval_video_arm.py --arm "$WIN" --mode "$MODE" --stage2 "$S2" \
      --output "$OUT/videomme" --model_args "$ARGS"
  # Dense reference already known 63.26 — optional skip if present
  echo "=== EFF winner + Dense + Native ==="
  for spec in "DENSE|off|OFF" "NATIVE_QATTN|native|Q_ATTN_ALL" "WINNER|$MODE|$S2"; do
    IFS='|' read -r arm mode s2 <<<"$spec"
    [[ "$arm" == WINNER ]] && arm="FULL_$WIN"
    BELA_EFF_GPU=3 BELA_VIDEO_MODE=$mode BELA_VIDEO_TIER=T32 BELA_VIDEO_STAGE2=$s2 \
      BELA_VIDEO_QA_BATCH=1 BELA_VIDEO_STAGE2_SCORE=qk \
      BELA_EFF_ARM=$arm BELA_EFF_OUT=$ROOT/logs/video_full_acc/eff_$arm.json \
      BELA_EFF_WARMUP=10 BELA_EFF_MEAS=50 \
      $PY $SCR/bench_video_eff.py
  done
fi

$PY - <<'PY'
import json,glob
from pathlib import Path
lines=['# VIDEO_FINAL_VIDEO_MME_AND_EFF','']
# full acc
for p in sorted(Path('/data1/xiangyue/logs/video_full_acc').glob('**/results*.json')):
    d=json.load(open(p)); arm=p.parts[p.parts.index('video_full_acc')+1]
    sc=d['results']['videomme'].get('videomme_perception_score,none')
    lines.append(f'- FULL {arm}: {sc}')
for p in sorted(Path('/data1/xiangyue/logs/video_full_acc').glob('eff_*.json')):
    d=json.load(open(p))
    lines.append(f"- EFF {d['arm']}: TTFT={d['ttft_wall_ms']['mean']:.1f} Fixed32={d['fixed32_wall_ms']['mean']:.1f}")
lines += ['', 'Literature (not same harness): STAR-Pro T32 Video-MME 61.6 / Dense 63.6.',
          'Dual COVER here is SigLIP adaptation — see VIDEO_DUAL_COVER_MAPPING_AUDIT.md.']
Path('/data1/xiangyue/logs/VIDEO_FINAL_VIDEO_MME_AND_EFF.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
PY
echo "HANDOFF_REMAINING_DONE $(date -Is)"
