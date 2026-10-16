#!/usr/bin/env bash
# Collect E020 pairdata on one free GPU, then fit four same-capacity heads.
# Requires project_e020_backend.py (real teacher-forced adapter). Exits 2 if missing.
set -euo pipefail
REPO=/home/xiangyue/ViTCoP
PKG=$REPO/experiments/e020_continuation
PY=/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python
mkdir -p "$PKG/logs" "$PKG/models" "$PKG/data"

if [[ ! -f $PKG/project_e020_backend.py ]]; then
  echo "[$(date -Is)] BLOCKER: project_e020_backend.py not installed; cannot collect real pairs"
  echo "learned_blocked_no_backend $(date -Is)" > "$PKG/state/learned_blocked.txt"
  exit 2
fi

# pick first free GPU among 0,2,4,6 with <2GiB used
GPU=$("$PY" - <<'PY'
import subprocess
want={0,2,4,6}
text=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],text=True)
for line in text.strip().splitlines():
    i,m=line.split(","); i=int(i); m=int(m)
    if i in want and m<2000:
        print(i); break
else:
    print(-1)
PY
)
if [[ "$GPU" -lt 0 ]]; then
  echo "[$(date -Is)] no free GPU for collect; will retry later"
  exit 3
fi

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$PKG:$REPO:$REPO/e020_lib
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

OUT=$PKG/data/e020_pairs.npz
CFG=$PKG/data/pair_config.json
cat > "$CFG" <<EOF
{
  "backend": "project_e020_backend:make",
  "source": $("$PY" -c "import json; print(json.dumps(json.load(open('$PKG/runtime.json'))['source']))"),
  "schedules": ["FIRST_SHIFT", "EARLY"],
  "arms": ["COVER", "BASE"],
  "max_examples": 192,
  "backend_config": {
    "qa_jsonl": "$REPO/logs/sdmc_13b_r1/20260927_1345/manifest/calibration_train_qa.jsonl",
    "model": "liuhaotian/llava-v1.5-7b",
    "device": "cuda:0"
  }
}
EOF

echo "[$(date -Is)] collect GPU=$GPU out=$OUT"
cd "$PKG"
"$PY" pairdata.py --adapter replay_adapter:make --config "$CFG" --out "$OUT" --states 384 --actions 6
echo "[$(date -Is)] fit models → $PKG/models"
"$PY" fit_models.py "$OUT" --out "$PKG/models" --epochs 160
echo "[$(date -Is)] collect_and_fit OK"
ls -la "$PKG/models"
