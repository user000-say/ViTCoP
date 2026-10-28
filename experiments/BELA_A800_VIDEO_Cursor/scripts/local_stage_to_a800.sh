#!/usr/bin/env bash
# Download on this machine (the A800 node has no DNS) and copy to /data1/xiangyue.
set -euo pipefail
LOG=/home/xiangyue/src/a800_stage.log
exec > >(tee -a "$LOG") 2>&1
echo "STAGE_START $(date -Is)"
export SSH_ASKPASS=/tmp/bela_askpass.sh
export SSH_ASKPASS_REQUIRE=force
export DISPLAY=none
RSH="ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o StrictHostKeyChecking=accept-new"
REMOTE=xiangyue@114.212.10.250
PY=/home/xiangyue/envs/vitcop-v100-4model/bin/python
WHEEL=/home/xiangyue/src/a800_wheels
MODEL=/home/xiangyue/src/models/LLaVA-Video-7B-Qwen2
PACK=/home/xiangyue/ViTCoP-a800-video/experiments/BELA_A800_VIDEO_Cursor
mkdir -p "$WHEEL" "$MODEL" /home/xiangyue/src/videos

copy() {
  setsid rsync -a --info=stats2 -e "$RSH" "$@"
}
setsid ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  "$REMOTE" 'mkdir -p /data1/xiangyue/ViTCoP/experiments /data1/xiangyue/src /data1/xiangyue/wheels /data1/xiangyue/models /data1/xiangyue/videos /data1/xiangyue/logs'

copy --exclude .git --exclude playground --exclude '__pycache__' --exclude '*.pyc' \
  /home/xiangyue/src/LLaVA-NeXT/ "$REMOTE:/data1/xiangyue/src/LLaVA-NeXT/"
copy "$PACK/" "$REMOTE:/data1/xiangyue/ViTCoP/experiments/BELA_A800_VIDEO_Cursor/"
echo "CODE_COPIED $(date -Is)"

if [[ ! -f $WHEEL/torch-2.1.2+cu121-cp310-cp310-linux_x86_64.whl ]]; then
  "$PY" -m pip download -d "$WHEEL" torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
fi
"$PY" -m pip download --no-deps -d "$WHEEL" \
  transformers==4.40.2 tokenizers==0.19.1 accelerate==0.29.3 \
  sentencepiece==0.1.99 'numpy==1.26.4' einops==0.6.1 einops-exts==0.0.4 \
  decord av pillow shortuuid safetensors==0.4.3 'huggingface_hub==0.23.5' \
  'protobuf==4.25.3' pyyaml tqdm requests ftfy 'opencv-python-headless==4.10.0.84' \
  setuptools wheel pip packaging regex filelock fsspec certifi charset-normalizer idna urllib3 typing-extensions \
  psutil wcwidth
echo "WHEELS_READY $(date -Is)"
copy "$WHEEL/" "$REMOTE:/data1/xiangyue/wheels/"
setsid ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  "$REMOTE" 'bash /data1/xiangyue/ViTCoP/experiments/BELA_A800_VIDEO_Cursor/scripts/remote_env_install.sh' \
  > /home/xiangyue/src/a800_env_install.log 2>&1
echo "ENV_INSTALL_EXIT $? $(date -Is)"

if [[ ! -f $MODEL/config.json ]]; then
  HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1 "$PY" - << PY
from huggingface_hub import snapshot_download
snapshot_download("lmms-lab/LLaVA-Video-7B-Qwen2", local_dir="$MODEL")
PY
fi
echo "MODEL_LOCAL $(date -Is)"
copy "$MODEL/" "$REMOTE:/data1/xiangyue/models/LLaVA-Video-7B-Qwen2/"
echo "MODEL_COPIED $(date -Is)"

setsid ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no "$REMOTE" 'bash -s' << 'REMOTE'
set -euo pipefail
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d " ")
if [[ ${used:-99999} -ge 800 ]]; then
  echo "GPU0 busy ${used} MiB"
  exit 3
fi
export PYTHONPATH=/data1/xiangyue/src/LLaVA-NeXT
/data1/xiangyue/envs/vitcop-a800-video/bin/python - << 'PY'
import av, numpy as np
from pathlib import Path
path = Path("/data1/xiangyue/videos/smoke_color.mp4")
path.parent.mkdir(parents=True, exist_ok=True)
container = av.open(str(path), mode="w")
stream = container.add_stream("mpeg4", rate=8)
stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
for i in range(64):
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:] = (220, 30, 30) if i < 32 else (30, 180, 40)
    frame = av.VideoFrame.from_ndarray(img, format="rgb24")
    for packet in stream.encode(frame):
        container.mux(packet)
for packet in stream.encode():
    container.mux(packet)
container.close()
print("VIDEO_WRITTEN", path, path.stat().st_size)
PY
export BELA_VIDEO_MODEL=/data1/xiangyue/models/LLaVA-Video-7B-Qwen2
export BELA_SMOKE_VIDEO=/data1/xiangyue/videos/smoke_color.mp4
export BELA_SMOKE_OUT=/data1/xiangyue/logs/VIDEO_DENSE_SMOKE.json
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
nohup /data1/xiangyue/envs/vitcop-a800-video/bin/python \
  /data1/xiangyue/ViTCoP/experiments/BELA_A800_VIDEO_Cursor/scripts/dense_official_smoke.py \
  > /data1/xiangyue/logs/dense_official_smoke.log 2>&1 &
echo SMOKE_PID $!
REMOTE
echo "STAGE_LAUNCHED $(date -Is)"
