#!/usr/bin/env bash
# Read-only production checkout audit. Do NOT blindly apply historic E014 files.
set -euo pipefail
echo "[BELA NEXT] current checkout:"
pwd
git rev-parse HEAD || true
git status --short || true
for file in \
  llava/model/multimodal_encoder/clip_encoder.py \
  llava/model/language_model/vitcop_modeling_llama.py; do
  if [[ -f "$file" ]]; then
    sha256sum "$file"
    grep -n -E "e014_choose|RDRIFT|E012|last_dynamic_info|e006_last" "$file" | head -25 || true
  else
    echo "MISSING: $file"
  fi
done
echo "Find the actual installed import paths:"
python - <<'PY'
import importlib.util
for path in ('llava.model.multimodal_encoder.clip_encoder',
             'llava.model.language_model.vitcop_modeling_llama'):
    try:
        spec=importlib.util.find_spec(path)
        print(path, spec.origin if spec else 'UNRESOLVED')
    except Exception as e:
        print(path, 'ERROR', str(e))
PY
echo "READ-ONLY preflight finished. Nothing has been overwritten."
