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
if ! "$ENV/bin/python" -m pip --version >/dev/null 2>&1; then
  rm -rf "$ENV"
  python3 -m venv --without-pip "$ENV"
  boot=$(mktemp -d "$ROOT/tmp/pipboot.XXXX")
  python3 - "$WHEEL"/pip-*.whl "$boot" << 'PY'
import sys, zipfile
zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])
PY
  PYTHONPATH="$boot" "$ENV/bin/python" -m pip install --ignore-installed --no-index --find-links "$WHEEL" pip setuptools wheel
  rm -rf "$boot"
fi
"$ENV/bin/python" -m pip install --no-index --find-links "$WHEEL" \
  torch==2.1.2+cu121 torchvision==0.16.2+cu121
"$ENV/bin/python" -m pip install --no-index --find-links "$WHEEL" \
  transformers==4.40.2 tokenizers==0.19.1 accelerate==0.29.3 \
  sentencepiece==0.1.99 numpy==1.26.4 einops==0.6.1 einops-exts==0.0.4 \
  decord av pillow shortuuid safetensors==0.4.3 huggingface_hub==0.23.5 \
  protobuf==4.25.3 pyyaml tqdm requests ftfy opencv-python-headless==4.10.0.84 \
  packaging regex filelock fsspec certifi charset-normalizer idna urllib3 typing-extensions \
  psutil wcwidth
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
