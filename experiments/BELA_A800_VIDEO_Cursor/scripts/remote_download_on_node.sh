#!/usr/bin/env bash
# Download one video set on the A800 data disk. Pass mlvu, mvbench, or longvideobench.
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
import glob, os, tarfile, time, zipfile
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
elif which == "longvideobench":
    token = open("/data1/xiangyue/secrets/hf_token").read().strip()
    src = os.path.join(hf, "repos", "longvideobench")
    os.makedirs(src, exist_ok=True)
    print("DOWNLOAD longvideobench/LongVideoBench", flush=True)
    err = None
    for attempt in range(30):
        try:
            snapshot_download("longvideobench/LongVideoBench", repo_type="dataset", local_dir=src, token=token)
            err = None
            break
        except Exception as exc:
            err = type(exc).__name__
            print("RETRY", attempt, err, flush=True)
            time.sleep(20)
    if err:
        raise SystemExit("download failed: " + err)
    parts = sorted(glob.glob(os.path.join(src, "videos.tar.part.*")))
    print("PARTS", len(parts), flush=True)
    if len(parts) < 10:
        raise SystemExit("too few tar parts")
    out = os.path.join(src, "videos.tar")
    need = sum(os.path.getsize(part) for part in parts)
    if (not os.path.exists(out)) or os.path.getsize(out) != need:
        print("CONCAT", len(parts), need, flush=True)
        with open(out, "wb") as w:
            for part in parts:
                print("PART", os.path.basename(part), flush=True)
                with open(part, "rb") as r:
                    while True:
                        buf = r.read(8 * 1024 * 1024)
                        if not buf:
                            break
                        w.write(buf)
    dst = os.path.join(hf, "longvideobench")
    os.makedirs(dst, exist_ok=True)
    print("UNTAR", out, flush=True)
    with tarfile.open(out) as packed:
        packed.extractall(dst)
    sub = os.path.join(src, "subtitles.tar")
    if os.path.exists(sub):
        with tarfile.open(sub) as packed:
            packed.extractall(dst)
    media = 0
    for root, dirs, files in os.walk(dst):
        media += sum(name.endswith((".mp4", ".mkv", ".webm")) for name in files)
    print("NODE_DOWNLOAD_DONE longvideobench", media, flush=True)
    if media < 100:
        raise SystemExit("too few videos: %s" % media)
else:
    raise SystemExit(f"unknown set {which}")
PY
echo "NODE_DOWNLOAD_EXIT $WHICH $(date -Is)"
