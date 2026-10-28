#!/usr/bin/env bash
# Download video benchmarks here (the A800 node has no DNS) and copy them to /data1.
set -u
PY=/home/xiangyue/envs/vitcop-v100-4model/bin/python
HF=/home/xiangyue/src/hf_video
LOG=/home/xiangyue/src/video_dataset_fetch.log
export HF_HOME=$HF
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF"
exec >> "$LOG" 2>&1
echo "FETCH_START $(date -Is)"
export SSH_ASKPASS=/tmp/bela_askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=none
RSH="ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no"
REMOTE=xiangyue@114.212.10.250

mark() {
  local name=$1
  setsid ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no "$REMOTE" "mkdir -p /data1/xiangyue/datasets; touch /data1/xiangyue/datasets/READY_${name}"
  echo "READY $name $(date -Is)"
}

unpack_and_push() {
  local repo=$1 cache=$2 ready=$3
  echo "DOWNLOAD $repo $(date -Is)"
  HF_HOME=$HF "$PY" - << PY
from huggingface_hub import snapshot_download
snapshot_download(${repo@Q}, repo_type="dataset", local_dir="${HF}/repos/${cache}")
PY
  HF_HOME=$HF "$PY" - << PY
import os, glob, zipfile, tarfile
src = "${HF}/repos/${cache}"
dst = os.path.join("${HF}", "${cache}")
os.makedirs(dst, exist_ok=True)
for z in glob.glob(src + "/**/*.zip", recursive=True):
    print("UNZIP", z, flush=True)
    with zipfile.ZipFile(z) as zz:
        zz.extractall(dst)
for t in glob.glob(src + "/**/*.tar*", recursive=True):
    print("UNTAR", t, flush=True)
    with tarfile.open(t) as tt:
        tt.extractall(dst)
print("UNPACK_DONE", dst, flush=True)
PY
  echo "RSYNC $cache $(date -Is)"
  setsid rsync -a --info=stats2 -e "$RSH" "$HF/" "$REMOTE:/data1/xiangyue/hf/"
  mark "$ready"
}

unpack_and_push "lmms-lab/Video-MME" videomme videomme
unpack_and_push "longvideobench/LongVideoBench" longvideobench longvideobench
unpack_and_push "sy1998/MLVU_dev" mlvu mlvu
"$PY" - << 'PY'
from huggingface_hub import snapshot_download
import os
os.environ["HF_HOME"] = "/home/xiangyue/src/hf_video"
snapshot_download("OpenGVLab/MVBench", repo_type="dataset", revision="video", local_dir="/home/xiangyue/src/hf_video/repos/mvbench_video")
PY
HF_HOME=$HF "$PY" - << 'PY'
import os, glob, zipfile, tarfile
src = "/home/xiangyue/src/hf_video/repos/mvbench_video"
dst = "/home/xiangyue/src/hf_video/mvbench_video"
os.makedirs(dst, exist_ok=True)
for z in glob.glob(src + "/**/*.zip", recursive=True):
    print("UNZIP", z, flush=True)
    with zipfile.ZipFile(z) as zz:
        zz.extractall(dst)
for t in glob.glob(src + "/**/*.tar*", recursive=True):
    print("UNTAR", t, flush=True)
    with tarfile.open(t) as tt:
        tt.extractall(dst)
print("UNPACK_DONE", dst, flush=True)
PY
setsid rsync -a --info=stats2 -e "$RSH" "$HF/" "$REMOTE:/data1/xiangyue/hf/"
mark mvbench
echo "FETCH_DONE $(date -Is)"
