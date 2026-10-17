#!/usr/bin/env bash
# SSH-safe: E019 B/C finish → E020.1 R2 smoke/collect/fit → core learned ACC (0/2/4/6).
# Never kill other experiments. Survive hangup via setsid + disown of children.
set -euo pipefail
trap '' HUP
REPO=/home/xiangyue/ViTCoP
PKG=$REPO/experiments/e020_1_r2
E019=$REPO/experiments/e019_progressive_evidence
PY=/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python
LOG=$PKG/logs/orchestrate_e019_r2.log
mkdir -p "$PKG/logs" "$PKG/state" "$PKG/data" "$PKG/models"
echo $$ > "$PKG/state/orchestrate_e019_r2.pid"

# Log only to file+stderr so command substitutions never capture log lines.
log(){ echo "[$(date -Is)] $*" | tee -a "$LOG" >&2; }

wait_e019() {
  log "wait E019 remaining ACC until 144/144 markers"
  while true; do
    # prefer live queue pid
    if [[ -f $E019/state/acc_remaining.pid ]]; then
      pid=$(cat "$E019/state/acc_remaining.pid" || true)
      if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
        :
      fi
    fi
    done=$("$PY" - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0,"/home/xiangyue/ViTCoP")
from e014_queue import PHASE_ARMS
root=Path("/home/xiangyue/ViTCoP/logs/e014_aee/full/full_e019_remaining")
arms=PHASE_ARMS["full_e019_remaining"]
tasks=["gqa","pope","mme","mmbench_en_dev"]
print(sum(1 for a in arms for t in tasks if (root/a/t/"metrics.json").exists()))
print(len(arms)*4)
PY
)
    n=$(echo "$done" | head -1)
    tot=$(echo "$done" | tail -1)
    log "E019 progress $n/$tot"
    if [[ "$n" -ge "$tot" ]]; then
      log "E019 remaining COMPLETE"
      return 0
    fi
    # if queue died early with incomplete jobs, relaunch once
    if [[ -f $E019/state/acc_remaining.pid ]]; then
      pid=$(cat "$E019/state/acc_remaining.pid")
      if ! kill -0 "$pid" 2>/dev/null; then
        log "E019 queue dead at $n/$tot — relaunch campaign_acc_remaining.sh"
        bash "$E019/scripts/campaign_acc_remaining.sh" >>"$E019/logs/campaign.log" 2>&1 || true
      fi
    else
      log "no E019 pid — launch remaining"
      bash "$E019/scripts/campaign_acc_remaining.sh" >>"$E019/logs/campaign.log" 2>&1 || true
    fi
    sleep 120
  done
}

pick_gpu() {
  "$PY" - <<'PY'
import subprocess
want={0,2,4,6}
text=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],text=True)
free=[]
for line in text.strip().splitlines():
    i,m=line.split(","); i=int(i); m=int(m)
    if i in want and m<2000: free.append(i)
print(free[0] if free else -1)
PY
}

wait_gpus_free() {
  log "wait GPUs 0/2/4/6 mostly free (<2GiB)"
  while true; do
    g=$(pick_gpu)
    if [[ "$g" -ge 0 ]]; then
      n=$("$PY" - <<'PY'
import subprocess
want={0,2,4,6}
text=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],text=True)
print(sum(1 for line in text.strip().splitlines() if int(line.split(",")[0]) in want and int(line.split(",")[1])<2000))
PY
)
      log "free_gpus_count=$n pick=$g"
      if [[ "$n" -ge 1 && "$g" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "$g"
        return 0
      fi
    fi
    sleep 60
  done
}

run_collect_fit() {
  local GPU=$1
  export CUDA_VISIBLE_DEVICES=$GPU
  export PYTHONPATH=$PKG:$REPO:$REPO/e020_lib:$REPO/LLaVA
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  log "collect/fit on physical GPU=$GPU"

  CFG=$PKG/data/replay_config.json
  "$PY" - <<PY
import json
from pathlib import Path
rt=json.loads(Path("$PKG/runtime.json").read_text())
cfg={
  "backend":"project_e020_backend:make",
  "source":rt["source"],
  "schedules":["FIRST_SHIFT","FIRST_NEXT_WMATCH"],
  "arms":["COVER","BASE"],
  "max_examples":128,
  "backend_config":{
    "qa_jsonl":"$REPO/logs/sdmc_13b_r1/20260927_1345/manifest/calibration_train_qa.jsonl",
    "model":"liuhaotian/llava-v1.5-7b",
    "device":"cuda:0",
  },
}
Path("$CFG").write_text(json.dumps(cfg,indent=2))
PY

  cd "$PKG"
  if [[ ! -f $PKG/state/smoke_ok.txt ]]; then
    log "SMOKE states=8"
    set +e
    "$PY" pairdata.py --adapter replay_adapter:make --config "$CFG" \
      --out "$PKG/data/e0201r2_smoke.npz" --states 8 --actions 16 \
      --candidate-limit 16 --shortlist 16 --group-sizes 1 --noop-every 8 \
      >>"$PKG/logs/collect_smoke.log" 2>&1
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      log "SMOKE FAILED rc=$rc"
      echo "smoke_failed $(date -Is)" > "$PKG/state/smoke_failed.txt"
      exit $rc
    fi
    echo "ok $(date -Is)" > "$PKG/state/smoke_ok.txt"
    log "SMOKE OK"
  else
    log "skip smoke (already ok)"
  fi

  OUT=$PKG/data/e0201r2_pairs.npz
  if [[ ! -f $OUT ]]; then
    log "FULL collect states=256"
    "$PY" pairdata.py --adapter replay_adapter:make --config "$CFG" \
      --out "$OUT" --states 256 --actions 16 \
      --candidate-limit 16 --shortlist 16 --group-sizes 1 --noop-every 8 \
      >>"$PKG/logs/collect_full.log" 2>&1
  else
    log "reuse existing $OUT"
  fi

  if [[ ! -f $PKG/models/current.npz ]]; then
    log "FIT models → $PKG/models"
    "$PY" fit_models.py "$OUT" --out "$PKG/models" --modes current,no_context,shuffled,full \
      >>"$PKG/logs/fit.log" 2>&1
  else
    log "models already present"
  fi

  "$PY" diagnostics.py "$OUT" --models "$PKG/models" --group-size 1 --candidate-limit 16 \
    --out "$PKG/data/e0201r2_diagnostics.json" >>"$PKG/logs/diagnostics.log" 2>&1 || true

  # expose models to e014_arms E020_MODELS_DIR (continuation/models)
  mkdir -p "$REPO/experiments/e020_continuation/models"
  for m in current no_context shuffled full; do
    if [[ -f $PKG/models/$m.npz ]]; then
      ln -sfn "$PKG/models/$m.npz" "$REPO/experiments/e020_continuation/models/$m.npz"
    fi
  done
  echo "collect_fit_ok $(date -Is)" > "$PKG/state/collect_fit_ok.txt"
  log "collect+fit complete"
}

launch_learned_acc() {
  log "LAUNCH full_e020_learned ACC on gpus 0,2,4,6"
  # sync runtime code
  "$PY" - <<'PY'
from pathlib import Path
import shutil, sys
sys.path.insert(0,"/home/xiangyue/ViTCoP")
from e010_common import sha256
REPO=Path("/home/xiangyue/ViTCoP"); RUNTIME=REPO/"logs/e014_aee/code_runtime"
for name in [
 "LLaVA/llava/model/e018_stage2_support.py",
 "LLaVA/llava/model/e019_evidence_torch.py",
 "LLaVA/llava/model/e019_selector_hook.py",
]:
  src,dst=REPO/name, RUNTIME/name
  if src.exists() and (not dst.exists() or sha256(dst)!=sha256(src)):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src,dst); print("synced", name)
lib_src=REPO/"e020_lib"; lib_dst=RUNTIME/"e020_lib"
lib_dst.mkdir(parents=True, exist_ok=True)
for p in lib_src.glob("*.py"):
  d=lib_dst/p.name
  if not d.exists() or sha256(d)!=sha256(p):
    shutil.copy2(p,d); print("synced lib", p.name)
PY
  for g in 0 2 4 6; do
    echo $$ > "$REPO/experiments/e014_latency_fair_256/state/snatch/worker_gpu${g}.pid" 2>/dev/null || true
  done
  setsid "$PY" "$REPO/e014_queue.py" \
    --phase full_e020_learned --gpus 0,2,4,6 --mem-free-mib 2000 \
    </dev/null >>"$PKG/logs/learned_acc_queue.log" 2>&1 &
  echo $! > "$PKG/state/learned_acc.pid"
  log "learned_acc pid=$(cat $PKG/state/learned_acc.pid)"
  # wait until complete or exhausted
  while kill -0 "$(cat $PKG/state/learned_acc.pid)" 2>/dev/null; do
    sleep 120
    n=$("$PY" - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0,"/home/xiangyue/ViTCoP")
import e014_arms as m
root=Path("/home/xiangyue/ViTCoP/logs/e014_aee/full/full_e020_learned")
tasks=["gqa","pope","mme","mmbench_en_dev"]
print(sum(1 for a in m.FULL_E020_LEARNED for t in tasks if (root/a/t/"metrics.json").exists()))
print(len(m.FULL_E020_LEARNED)*4)
PY
)
    log "learned ACC $(echo "$n" | head -1)/$(echo "$n" | tail -1)"
  done
  log "learned ACC queue exited"
}

summarize() {
  "$PY" "$REPO/experiments/e020_continuation/scripts/analyze_hypothesis.py" \
    >"$PKG/HYPOTHESIS_AFTER_R2.txt" 2>&1 || true
  "$PY" - <<'PY' >>"$PKG/HYPOTHESIS_AFTER_R2.txt" 2>&1 || true
from pathlib import Path
import json, e014_arms as m
DENOM={'gqa':61.9,'pope':85.8,'mme':1852.0,'mmbench_en_dev':64.1}
TASKS=['gqa','pope','mme','mmbench_en_dev']
def pull(root,arm):
  vals={}
  for t in TASKS:
    p=Path(root)/arm/t/'metrics.json'
    if not p.exists(): return None
    d=json.loads(p.read_text())
    if t=='gqa': vals['GQA']=100*float(d['exact_match'])
    elif t=='pope': vals['POPE']=100*float(d['pope_accuracy'])
    else:
      r=json.loads(Path(d['results_json']).read_text())['results'][t]
      vals['MME' if t=='mme' else 'MMB']=(r['mme_perception_score,none']+r['mme_cognition_score,none']) if t=='mme' else r['gpt_eval_score,none']
  vals['R4']=25*(vals['GQA']/61.9+vals['POPE']/85.8+vals['MME']/1852+vals['MMB']/64.1)
  return vals
root=Path('/home/xiangyue/ViTCoP/logs/e014_aee/full/full_e020_learned')
rows=[]
for a in m.FULL_E020_LEARNED:
  s=pull(root,a)
  if s: rows.append((a,s['R4']))
rows.sort(key=lambda x:-x[1])
print('learned complete', len(rows), '/', len(m.FULL_E020_LEARNED))
for a,r in rows[:15]:
  print(f'{r:.2f} {a}')
PY
  log "summary → $PKG/HYPOTHESIS_AFTER_R2.txt"
  echo "orchestrate_complete $(date -Is)" > "$PKG/state/orchestrate_complete.txt"
}

log "ORCHESTRATE E019→R2 start pid=$$"
wait_e019
GPU=$(wait_gpus_free)
run_collect_fit "$GPU"
launch_learned_acc
summarize
log "ALL DONE"
