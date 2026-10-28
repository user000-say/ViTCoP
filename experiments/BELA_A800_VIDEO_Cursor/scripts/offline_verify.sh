#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m unittest discover -s tests -v
python3 -m bela_video_a800.schedule --tier T32 --frames 64
python3 -m bela_video_a800.schedule --tier T16 --frames 64
