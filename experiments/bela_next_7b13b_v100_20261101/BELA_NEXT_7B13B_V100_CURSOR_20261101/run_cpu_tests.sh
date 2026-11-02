#!/usr/bin/env bash
set -euo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PKG/src:$PKG:${PYTHONPATH:-}"
python -m unittest discover -s "$PKG/tests" -v
python "$PKG/tools/print_arm_env.py" --arm NEXT7_T160_A0_PAPER
python "$PKG/tools/print_arm_env.py" --arm NEXT13_T320_A2_LATE
python "$PKG/tools/plan_priority_queue.py"
echo "LOCAL CONFIG CHECKS PASSED. Actual V100/model integration NOT TESTED."
