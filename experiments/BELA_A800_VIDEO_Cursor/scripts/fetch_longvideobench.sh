#!/usr/bin/env bash
# LongVideoBench is gated. Uses the local Hugging Face token file; never prints it.
set -u
PY=/home/xiangyue/envs/vitcop-v100-4model/bin/python
HF=/home/xiangyue/src/hf_video
LOG=/home/xiangyue/src/longvideobench_fetch.log
export HF_HOME=$HF
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF"
exec >> "$LOG" 2>&1
echo "LVB_START $(date -Is)"
export SSH_ASKPASS=/tmp/bela_askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=none
RSH="ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no"
REMOTE=xiangyue@114.212.10.250
"$PY" - << 'PY'
import os, glob, tarfile
from huggingface_hub import snapshot_download
hf = "/home/xiangyue/src/hf_video"
src = os.path.join(hf, "repos", "longvideobench")
dst = os.path.join(hf, "longvideobench")
print("DOWNLOAD longvideobench", flush=True)
snapshot_download("longvideobench/LongVideoBench", repo_type="dataset", local_dir=src)
os.makedirs(dst, exist_ok=True)
parts = sorted(glob.glob(os.path.join(src, "videos.tar.part.*")))
if not parts:
    raise SystemExit("no video tar parts")
out = os.path.join(src, "videos.tar")
if not os.path.exists(out):
    print("CONCAT", len(parts), flush=True)
    with open(out, "wb") as w:
        for part in parts:
            print("PART", os.path.basename(part), flush=True)
            with open(part, "rb") as r:
                while True:
                    buf = r.read(8 * 1024 * 1024)
                    if not buf:
                        break
                    w.write(buf)
print("UNTAR", out, flush=True)
with tarfile.open(out) as t:
    t.extractall(dst)
sub = os.path.join(src, "subtitles.tar")
if os.path.exists(sub):
    with tarfile.open(sub) as t:
        t.extractall(dst)
n = 0
for root, dirs, files in os.walk(dst):
    n += sum(name.endswith((".mp4", ".mkv", ".webm")) for name in files)
print("UNPACK_DONE", dst, "media", n, flush=True)
if n < 100:
    raise SystemExit(f"too few videos: {n}")
PY
echo "RSYNC longvideobench $(date -Is)"
setsid ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no "$REMOTE" "mkdir -p /data1/xiangyue/hf/ann"
rsync -a --exclude '*.zip' --info=stats2 -e "$RSH" "$HF/longvideobench/" "$REMOTE:/data1/xiangyue/hf/longvideobench/"
rsync -a --exclude '*.tar*' --exclude 'videos.tar.part.*' -e "$RSH" "$HF/repos/longvideobench/" "$REMOTE:/data1/xiangyue/hf/ann/longvideobench/"
echo "LVB_DONE $(date -Is)"
