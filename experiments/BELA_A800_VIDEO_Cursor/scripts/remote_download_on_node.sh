#!/usr/bin/env bash
# Download the remaining public video sets on the A800 data disk.
# The proxy address lives only in /data1/xiangyue/proxy.env on that node.
set -euo pipefail
ROOT=/data1/xiangyue
# shellcheck disable=SC1091
source "$ROOT/proxy.env"
export HF_HOME=$ROOT/hf
export HUGGINGFACE_HUB_CACHE=$ROOT/hf/hub
export HF_HUB_ENABLE_HF_TRANSFER=0
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
mkdir -p "$ROOT/hf" "$ROOT/logs" "$ROOT/tmp"
export TMPDIR=$ROOT/tmp
exec >> "$ROOT/logs/remote_dataset_fetch.log" 2>&1
echo "NODE_DOWNLOAD_START $(date -Is)"
"$ROOT/envs/vitcop-a800-video/bin/python" - << 'PY'
import os, glob, zipfile
from huggingface_hub import snapshot_download

hf = "/data1/xiangyue/hf"
print("DOWNLOAD sy1998/MLVU_dev", flush=True)
mlvu_snap = snapshot_download("sy1998/MLVU_dev", repo_type="dataset")
dst = os.path.join(hf, "mlvu")
os.makedirs(dst, exist_ok=True)
for z in sorted(glob.glob(os.path.join(mlvu_snap, "*.zip"))):
    print("UNZIP", os.path.basename(z), flush=True)
    with zipfile.ZipFile(z) as zz:
        zz.extractall(dst)
print("DOWNLOAD OpenGVLab/MVBench revision=video", flush=True)
mv = snapshot_download("OpenGVLab/MVBench", repo_type="dataset", revision="video")
link = os.path.join(hf, "mvbench_video")
if os.path.lexists(link) and not os.path.samefile(link, mv):
    if os.path.islink(link) or (os.path.isdir(link) and not os.listdir(link)):
        if os.path.islink(link):
            os.remove(link)
        else:
            os.rmdir(link)
if not os.path.lexists(link):
    os.symlink(mv, link)
print("NODE_DOWNLOAD_DONE", mlvu_snap, mv, flush=True)
PY
echo "NODE_DOWNLOAD_EXIT $(date -Is)"
