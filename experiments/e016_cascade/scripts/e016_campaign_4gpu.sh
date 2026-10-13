#!/usr/bin/env bash
# E016 cascade: full ACC on 0/2/6 + 1024 TTFT on GPU4. Never release cards.
set -uo pipefail
REPO=/home/xiangyue/ViTCoP
PKG=$REPO/experiments/e016_cascade
FAIR=$REPO/experiments/e014_latency_fair_256
PY_ACC=/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python
ACC_GPUS=(0 2 6)
LAT_GPU=4
ALL_GPUS=(0 2 4 6)
LOG=$PKG/logs/campaign.log
mkdir -p "$PKG/logs" "$PKG/state"
echo $$ >"$PKG/state/campaign.pid"
exec >>"$LOG" 2>&1
trap '' HUP TERM INT
echo "E016_CAMPAIGN_START $(date -Is)"

log(){ echo "$(date '+%F %T') $*"; }

# Reload snatch recognition
[[ -f $FAIR/state/snatch/snatch.pid ]] && kill -9 "$(tr -d ' \n' <$FAIR/state/snatch/snatch.pid)" 2>/dev/null || true
pgrep -f gpu_snatch_on_idle.py | while read -r p; do kill -9 "$p" 2>/dev/null || true; done
sleep 1
bash "$FAIR/scripts/ensure_never_release.sh" || true
sleep 2

kill_gpu() {
  local g="$1"
  if [[ -f $FAIR/state/snatch/worker_gpu${g}.pid ]]; then
    wp=$(tr -d ' \n' <$FAIR/state/snatch/worker_gpu${g}.pid)
    kill -9 -"$wp" 2>/dev/null || kill -9 "$wp" 2>/dev/null || true
    rm -f $FAIR/state/snatch/worker_gpu${g}.pid
  fi
  for p in $(pgrep -f gpu_hold_vram.py || true); do
    envg=$(tr '\0' '\n' </proc/$p/environ 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)
    [[ "$envg" == "$g" ]] && kill -9 "$p" 2>/dev/null || true
  done
}
for g in "${ALL_GPUS[@]}"; do kill_gpu "$g"; done
sleep 1
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$LOG"
for g in "${ALL_GPUS[@]}"; do echo $$ >$FAIR/state/snatch/worker_gpu${g}.pid; done

# Force-sync decoder into ACC runtime
cp -a $REPO/LLaVA/llava/model/language_model/vitcop_modeling_llama.py \
  $REPO/logs/e014_aee/code_runtime/LLaVA/llava/model/language_model/vitcop_modeling_llama.py

log "LAUNCH_ACC full_e016_cascade"
setsid "$PY_ACC" "$REPO/e014_queue.py" \
  --phase full_e016_cascade \
  --gpus "$(IFS=,; echo "${ACC_GPUS[*]}")" \
  --mem-free-mib 2000 \
  </dev/null >>"$PKG/logs/acc_queue.log" 2>&1 &
echo $! >"$PKG/state/acc_queue.pid"

log "LAUNCH_LATENCY gpu=$LAT_GPU"
setsid bash "$PKG/scripts/run_latency.sh" "$LAT_GPU" </dev/null &>/dev/null &
echo $! >"$PKG/state/latency.pid"

ARM=CASCADE_L8_96_L10_64_L16_32_L20_16
acc_done() {
  local n=0
  for t in gqa pope mme mmbench_en_dev; do
    [[ -f $REPO/logs/e014_aee/full/full_e016_cascade/$ARM/$t/metrics.json ]] && n=$((n+1))
  done
  [[ $n -eq 4 ]]
}
lat_done() {
  [[ -f $PKG/latency/runs/gpu${LAT_GPU}/$ARM/latency_summary.json ]]
}

while true; do
  a=0; l=0
  acc_done && a=1
  lat_done && l=1
  log "WATCH acc=$a lat=$l"
  if [[ $a -eq 1 && $l -eq 1 ]]; then
    log "ALL_DONE — PERMA_HOLD"
    break
  fi
  for g in "${ALL_GPUS[@]}"; do echo $$ >$FAIR/state/snatch/worker_gpu${g}.pid; done
  sleep 60
done

for g in "${ALL_GPUS[@]}"; do rm -f $FAIR/state/snatch/worker_gpu${g}.pid; done
sleep 1
for g in "${ALL_GPUS[@]}"; do
  setsid bash $FAIR/scripts/snatch_worker.sh "$g" </dev/null >>$FAIR/logs/snatch_worker_gpu${g}.log 2>&1 &
  echo $! >$FAIR/state/snatch/worker_gpu${g}.pid
  log "HOLD gpu=$g pid=$!"
done
log "E016_CAMPAIGN_COMPLETE $(date -Is)"
while true; do sleep 3600; done
