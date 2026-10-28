#!/usr/bin/env bash
# Isolated A800 env. Installs only from a local wheelhouse. Does not touch other users.
set -euo pipefail
ROOT=/data1/xiangyue
ENV=$ROOT/envs/vitcop-a800-video
WHEEL=$ROOT/wheels
LLAVA=$ROOT/src/LLaVA-NeXT
mkdir -p "$ROOT/logs" "$ROOT/tmp" "$ROOT/cache/pip" "$ROOT/hf/hub" "$ROOT/hf/transformers" "$ROOT/torch"
# Weights, wheels, env, and caches stay on /data1. Do not use the system disk or GPUs 2 and 3.
export TMPDIR=$ROOT/tmp
export PIP_CACHE_DIR=$ROOT/cache/pip
export HF_HOME=$ROOT/hf
export HUGGINGFACE_HUB_CACHE=$ROOT/hf/hub
export TRANSFORMERS_CACHE=$ROOT/hf/transformers
export XDG_CACHE_HOME=$ROOT/cache
export TORCH_HOME=$ROOT/torch
if [[ ! -x $ENV/bin/python ]]; then
  python3 -m venv "$ENV"
fi
"$ENV/bin/python" -m pip install --no-index --find-links "$WHEEL" --upgrade pip setuptools wheel
"$ENV/bin/python" -m pip install --no-index --find-links "$WHEEL" \
  torch==2.1.2+cu121 torchvision==0.16.2+cu121
"$ENV/bin/python" -m pip install --no-index --find-links "$WHEEL" \
  transformers==4.40.2 tokenizers==0.15.2 accelerate==0.29.3 \
  sentencepiece==0.1.99 numpy==1.26.4 einops==0.6.1 einops-exts==0.0.4 \
  decord av pillow shortuuid safetensors huggingface_hub \
  protobuf pyyaml tqdm requests ftfy opencv-python-headless
"$ENV/bin/python" -m pip install --no-index --find-links "$WHEEL" --no-deps -e "$LLAVA"
export CUDA_VISIBLE_DEVICES=0,1
"$ENV/bin/python" - << 'PY'
import json, torch, transformers, platform
from pathlib import Path
out = {
    "host": platform.node(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "bf16": bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False,
    "gpus": [],
}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        out["gpus"].append({"index": i, "name": p.name, "total_mib": p.total_memory // 2**20})
Path("/data1/xiangyue/logs/VIDEO_ENV.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
PY
echo ENV_READY
