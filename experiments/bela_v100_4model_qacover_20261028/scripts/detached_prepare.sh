#!/bin/bash
# Isolated from vitcopreback and starpro. Survives the launching shell.
set -u
ROOT=/home/xiangyue/ViTCoP
ENV=/home/xiangyue/envs/vitcop-v100-4model
PY="$ENV/bin/python"
LOG=$ROOT/experiments/bela_v100_4model_qacover_20261028/logs/detached_prepare.log
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "START $(date -Is) env=$ENV"
echo "PROD_GUARD $(/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python -c 'import transformers; print(transformers.__version__)')"

install_env() {
  if "$PY" -c 'import torch,transformers; assert torch.__version__.startswith("2.6"); assert transformers.__version__.startswith("5.20")'; then
    echo "ENV_READY $($PY -c 'import torch,transformers; print(torch.__version__, transformers.__version__)')"
    return 0
  fi
  if pgrep -f "$ENV/bin/python -m pip install" >/dev/null; then
    echo "PIP_ALREADY_RUNNING"
    return 1
  fi
  echo "PIP_BEGIN $(date -Is)"
  "$PY" -m pip install --no-cache-dir torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
  "$PY" -m pip install --no-cache-dir 'transformers==5.20.2' accelerate pillow safetensors sentencepiece
  "$PY" -c 'import torch,transformers; print("ENV_READY", torch.__version__, transformers.__version__, torch.cuda.is_available())'
}

until install_env; do
  sleep 60
done

SNAP=/home/xiangyue/.cache/huggingface/hub/models--llava-hf--llava-v1.6-vicuna-7b-hf/snapshots/c916e6cdcd760b4cecd1dd4907f84ac649f93b23
until [[ -f "$SNAP/model-00001-of-00003.safetensors" && -f "$SNAP/model-00002-of-00003.safetensors" && -f "$SNAP/model-00003-of-00003.safetensors" ]]; do
  echo "WAIT_NEXT7 $(date -Is)"
  sleep 30
done

if [[ -f $ROOT/experiments/bela_v100_4model_qacover_20261028/audit/FP16_BACKEND_SMOKE.json ]]; then
  echo "G1_ALREADY_PRESENT"
  exit 0
fi

echo "G1_BEGIN $(date -Is)"
CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/experiments/bela_v100_4model_qacover_20261028/scripts/g1_native_smoke.py"
echo "G1_EXIT $? $(date -Is)"
