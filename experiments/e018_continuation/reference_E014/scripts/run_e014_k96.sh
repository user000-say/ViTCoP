#!/usr/bin/env bash
set -euo pipefail
if (( $# == 0 )); then
  echo "Usage: $(basename "$0") <EXISTING-EVAL-COMMAND> [args ...]" >&2
  echo "Example: E014_VARIANT=AEE E014_AUDIT=True $(basename "$0") python YOUR_SMOKE_RUNNER.py" >&2
  exit 64
fi
source "$(dirname "${BASH_SOURCE[0]}")/e014_env.sh" 96
exec "$@"
