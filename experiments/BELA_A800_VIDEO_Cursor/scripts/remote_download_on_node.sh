#!/usr/bin/env bash
# Download one public video set on the A800 data disk. Pass mlvu or mvbench.
# The proxy address lives only in /data1/xiangyue/proxy.env on that node.
set -euo pipefail
WHICH=${1:?pass mlvu or mvbench}
ROOT=/data1/xiangyue
# shellcheck disable=SC1091
source "$ROOT/proxy.env"
export HF_HOME=$ROOT/hf
export HUGGINGFACE_HUB_CACHE=$ROOT/hf/hub
export HF_HUB_ENABLE_HF_TRANSFER=0
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
mkdir -p "$ROOT/hf" "$ROOT/logs" "$ROOT/tmp"
export TMPDIR=$ROOT/tmp
exec >> "$ROOT/logs/remote_dataset_${WHICH}.log" 2>&1
echo "NODE_DOWNLOAD_START $WHICH $(date -Is)"
export WHICH
"$ROOT/envs/vitcop-a800-video/bin/python" - << 'PY'
import os, glob, zipfile
from huggingface_hub import snapshot_download

which = os.environ["WHICH"]
hf = "/data1/xiangyue/hf"
if which == "mlvu":
    print("DOWNLOAD sy1998/MLVU_dev", flush=True)
    mlvu_snap = snapshot_download("sy1998/MLVU_dev", repo_type="dataset")
    dst = os.path.join(hf, "mlvu")
    os.makedirs(dst, exist_ok=True)
    for z in sorted(glob.glob(os.path.join(mlvu_snap, "*.zip"))):
        print("UNZIP", os.path.basename(z), flush=True)
        with zipfile.ZipFile(z) as zz:
            zz.extractall(dst)
    print("NODE_DOWNLOAD_DONE mlvu", mlvu_snap, flush=True)
elif which == "mvbench":
    print("DOWNLOAD OpenGVLab/MVBench revision=video", flush=True)
    mv = snapshot_download("OpenGVLab/MVBench", repo_type="dataset", revision="video")
    link = os.path.join(hf, "mvbench_video")
    if not os.path.lexists(link):
        os.symlink(mv, link)
    print("NODE_DOWNLOAD_DONE mvbench", mv, flush=True)
else:
    raise SystemExit(f"unknown set {which}")
PY
echo "NODE_DOWNLOAD_EXIT $WHICH $(date -Is)"
