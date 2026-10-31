#!/usr/bin/env bash
# Detach full slim ACC+EFF pipeline so SSH logout cannot stop it.
set -euo pipefail
ROOT=/data1/xiangyue
SCR=$ROOT/ViTCoP/experiments/BELA_A800_VIDEO_Cursor/scripts
mkdir -p "$ROOT/logs"
chmod +x "$SCR"/run_slim_all_pipeline.sh "$SCR"/run_slim_eff_watch.sh "$SCR"/bench_video_eff.py "$SCR"/run_slim_metrics_queue.sh || true

# Prefer not to double-start ACC if metrics queue already alive.
if pgrep -f 'run_slim_metrics_queue.sh' >/dev/null 2>&1 \
   || pgrep -f 'run_slim_all_pipeline.sh' >/dev/null 2>&1; then
  echo "ACC_ALREADY_RUNNING — attach EFF watch only"
  if ! pgrep -f 'run_slim_eff_watch.sh' >/dev/null 2>&1; then
    setsid nohup bash "$SCR/run_slim_eff_watch.sh" \
      >"$ROOT/logs/video_slim_eff_watch.nohup" 2>&1 < /dev/null &
    echo "EFF_WATCH_PID $!"
  else
    echo "EFF_WATCH_ALREADY"
  fi
else
  setsid nohup bash "$SCR/run_slim_all_pipeline.sh" \
    >"$ROOT/logs/video_slim_all_pipeline.nohup" 2>&1 < /dev/null &
  echo "PIPELINE_PID $!"
fi

# Heartbeat file
nohup bash -c '
  while true; do
    date -Is > /data1/xiangyue/logs/VIDEO_SLIM_HEARTBEAT.txt
    pgrep -af "lmms_eval.*video_slim|run_slim_|bench_video_eff" >> /data1/xiangyue/logs/VIDEO_SLIM_HEARTBEAT.txt 2>/dev/null || true
    sleep 300
  done
' >/data1/xiangyue/logs/video_slim_heartbeat.nohup 2>&1 < /dev/null &
echo "HEARTBEAT_PID $!"
echo "DETACH_OK $(date -Is)"
