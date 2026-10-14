#!/usr/bin/env bash
# source scripts/e018_env.sh 32|64|96 [STATIC|COLLECT|ONLINE|PROXY]
set -euo pipefail
E018_K="${1:-32}"
E018_MODE_ARG="${2:-STATIC}"
case "${E018_K}" in
  32) E018_MID=102 ;;
  64) E018_MID=118 ;;
  96) E018_MID=134 ;;
  *) echo 'K must be 32,64,96' >&2; return 2 ;;
esac
case "${E018_MODE_ARG}" in STATIC|COLLECT|ONLINE|PROXY) ;; *) echo 'Mode must be STATIC/COLLECT/ONLINE/PROXY' >&2; return 2 ;; esac
# E014 legacy anchor is only for existing guards/Stage1. E018 bypasses its decoder.
export E014_LAYER=8 E014_VARIANT="${E014_VARIANT:-FIXED_COVER}" E014_B="${E014_B:-32}"
source "$(dirname "${BASH_SOURCE[0]}")/e014_env.sh" "${E018_K}"
export E018_ENABLE=True E018_MODE="${E018_MODE_ARG}"
export E018_EVENTS="${E018_EVENTS:-4:${E018_MID},12:${E018_K}}"
export E018_SELECTOR="${E018_SELECTOR:-Q_ATTN}"
export E018_CACHE_POLICY="${E018_CACHE_POLICY:-layer_local}"
export E018_LAMBDA="${E018_LAMBDA:-0}"
export E018_MAX_WORK="${E018_MAX_WORK:-0}"
export E018_TRACE="${E018_TRACE:-}"
export E007_TOKEN_AUDIT=False E014_AUDIT=False E012_DIAG=False E012_KV_AUDIT=False E006_PROFILE=False
export COMPUTE_EFFICIENCY=False
if [[ ( "${E018_MODE_ARG}" == COLLECT || "${E018_MODE_ARG}" == ONLINE ) && -z "${E018_PROTOCOL_ID:-}" ]]; then
  echo 'Set E018_PROTOCOL_ID to frozen actual weights/source/evaluation identity' >&2; return 2
fi
if [[ "${E018_MODE_ARG}" == ONLINE && ! -f "${E018_CONTROLLER:-}" ]]; then
  echo 'ONLINE requires your trained E018_CONTROLLER JSON' >&2; return 2
fi
