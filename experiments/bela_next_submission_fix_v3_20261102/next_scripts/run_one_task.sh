#!/usr/bin/env bash
# Isolated, one task on one explicitly selected GPU. Never touches GPU holders/jobs.
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo 'Usage: CUDA_VISIBLE_DEVICES=4 bash next_scripts/run_one_task.sh ARM TASK OUTPUT_DIR' >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo 'Explicit CUDA_VISIBLE_DEVICES required (do not hijack a live job)' >&2
  exit 2
fi
ARM="$1"; TASK="$2"; OUT="$3"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${BELA_PYTHON:-python}"
ENTRY="$ROOT/next_scripts/lmms_bela_next_entry.py"
if find "$OUT" -name '*_results.json' -print -quit 2>/dev/null | grep -q .; then
  echo 'Output already contains results: use a NEW directory, do not mix retries' >&2
  exit 2
fi
mkdir -p "$OUT"
export PYTHONPATH="$ROOT${LMMS_EVAL_ROOT:+:$LMMS_EVAL_ROOT}:${PYTHONPATH:-}"
export BELA_NEXT_V2_ENABLE=1 BELA_NEXT_ARM="$ARM"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PY" "$ROOT/tools/preflight.py" --runtime --arm "$ARM" --out "$OUT/preflight.json"
LIMIT_ARGS=()
if [[ -n "${BELA_LMMS_LIMIT:-}" ]]; then
  if ! [[ "${BELA_LMMS_LIMIT}" =~ ^[0-9]+$ ]]; then
    echo 'BELA_LMMS_LIMIT must be an unsigned integer' >&2; exit 2
  fi
  LIMIT_ARGS=(--limit "${BELA_LMMS_LIMIT}")
fi
"$PY" -u "$ENTRY" \
  --model llava_hf \
  --model_args 'pretrained=llava-hf/llava-v1.6-vicuna-7b-hf,dtype=float16,attn_implementation=sdpa,device_map=cuda:0' \
  --tasks "$TASK" --batch_size 1 --log_samples --output_path "$OUT" "${LIMIT_ARGS[@]}"
