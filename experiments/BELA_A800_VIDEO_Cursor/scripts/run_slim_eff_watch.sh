#!/usr/bin/env bash
# Wait until all slim ACC arms have results.json, then run EFF only.
# Safe to run alongside the existing ACC queue (does not re-launch ACC).
set -u
ROOT=/data1/xiangyue
SCR=$ROOT/ViTCoP/experiments/BELA_A800_VIDEO_Cursor/scripts
LOG=$ROOT/logs/video_slim_eff_watch.log
OUT_ACC=$ROOT/logs/video_slim_acc
ARMS=(DENSE C_QA_QATTN D_QA_RDRIFT STARPRO_T32_PROXY)
mkdir -p "$ROOT/logs"
echo "EFF_WATCH_START $(date -Is)" | tee -a "$LOG"

all_acc() {
  local a
  for a in "${ARMS[@]}"; do
    if ! compgen -G "$OUT_ACC/$a/videomme/**/*results*.json" > /dev/null; then
      return 1
    fi
  done
  return 0
}

while ! all_acc; do
  echo "WAIT_ACC $(date -Is)" | tee -a "$LOG"
  for a in "${ARMS[@]}"; do
    if compgen -G "$OUT_ACC/$a/videomme/**/*results*.json" > /dev/null; then
      echo "  HAVE $a"
    elif pgrep -af "lmms_eval.*video_slim_acc/${a}/" >/dev/null 2>&1; then
      echo "  LIVE $a"
    else
      echo "  MISSING $a"
    fi
  done | tee -a "$LOG"
  # If ACC queue died and something missing, kick full pipeline once
  if ! pgrep -f 'run_slim_metrics_queue.sh|run_slim_all_pipeline.sh' >/dev/null 2>&1; then
    missing=0
    for a in "${ARMS[@]}"; do
      if ! compgen -G "$OUT_ACC/$a/videomme/**/*results*.json" > /dev/null \
         && ! pgrep -af "lmms_eval.*video_slim_acc/${a}/" >/dev/null 2>&1; then
        missing=1
      fi
    done
    if [[ $missing -eq 1 ]]; then
      echo "ACC_QUEUE_DEAD — launching all_pipeline $(date -Is)" | tee -a "$LOG"
      setsid nohup bash "$SCR/run_slim_all_pipeline.sh" >>"$ROOT/logs/video_slim_all_pipeline.nohup" 2>&1 &
      echo "PIPELINE_PID $!" | tee -a "$LOG"
      # pipeline includes EFF; exit this watcher to avoid double EFF
      echo "EFF_WATCH_HANDOFF_TO_PIPELINE $(date -Is)" | tee -a "$LOG"
      exit 0
    fi
  fi
  sleep 120
done

echo "ALL_ACC_READY — start EFF via all_pipeline EFF section $(date -Is)" | tee -a "$LOG"
# Reuse pipeline but ACC will HAVE-skip quickly then run EFF
setsid nohup bash "$SCR/run_slim_all_pipeline.sh" >>"$ROOT/logs/video_slim_all_pipeline.nohup" 2>&1 &
echo "PIPELINE_PID $!" | tee -a "$LOG"
wait $! || true
echo "EFF_WATCH_DONE $(date -Is)" | tee -a "$LOG"
