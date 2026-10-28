#!/usr/bin/env bash
set -euo pipefail
# Read-only. Do not paste login details, passwords or access tokens.
echo "[BELA] Read-only A800 hardware and software inspection"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is missing"; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_mode --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv || true
python3 - <<'PY'
import sys, importlib.util
print("Python:",sys.version.split()[0])
for name in ("torch","transformers","accelerate","decord","av","lmms_eval","flash_attn"):
    print(name, "installed:",bool(importlib.util.find_spec(name)))
if importlib.util.find_spec("torch"):
    import torch
    print("PyTorch:",torch.__version__,"CUDA:",torch.version.cuda)
    for i in range(torch.cuda.device_count()):
        p=torch.cuda.get_device_properties(i)
        print("GPU",i,p.name,round(p.total_memory/1024**3,2),"GiB","compute",p.major,p.minor)
    print("BF16 is supported:",torch.cuda.is_bf16_supported())
PY
echo "ffmpeg: $(command -v ffmpeg || echo missing)"
echo "ffprobe: $(command -v ffprobe || echo missing)"
df -h "${BELA_WORKDIR:-$HOME}" | tail -n 1
echo "[BELA] Complete. No model loaded."
