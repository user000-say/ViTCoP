#!/usr/bin/env bash
# Source from run_e014_k{96,64,32}.sh; args: final K.
# E014-only controls: frozen E007 Core172, E010/E012 S=D, QED unchanged.
set -euo pipefail
E014_FINAL_K="${1:?final K required}"
case "${E014_FINAL_K}" in 96|64|32) ;; *) echo "Unsupported E014 final K" >&2; return 2 ;; esac

export USE_VITCOP=True
export E006_ENABLE=True E006_SELECTOR=QED E006_LAYER=12
export E006_TARGET_K="${E014_FINAL_K}"
export E007_ENABLE=True E007_STAGE1_MODE=ORIGINAL_172
export E007_SKIP_UNUSED_VIC=True
export E007_TOKEN_AUDIT="${E007_TOKEN_AUDIT:-False}"
export E010_ENABLE=True
export E014_LAYER="${E014_LAYER:-8}"
case "${E014_LAYER}" in 8|12) ;; *) echo "E014_LAYER must be 8 or 12" >&2; return 2 ;; esac
export E010_QED_LAYER="${E014_LAYER}" E010_QED_SIDE_ONLY=False
export E010_CONTEXT_EXIT_LAYER=-1 E010_KEEP_CONTEXT_AFTER_QED=False
export E010_ANALYSIS_FULL576=False E010_COLLECT_LOCAL_INFLUENCE=False
export E012_ENABLE=True E012_DELETE_LAYER="${E014_LAYER}"
export E012_DIAG="${E012_DIAG:-False}"
export E012_KV_AUDIT="${E012_KV_AUDIT:-False}"
export E013_ENABLE=False E008_ENABLE=False E009_ENABLE=False
export E014_ENABLE=True
export E014_VARIANT="${E014_VARIANT:-AEE}"
export E014_B="${E014_B:-16}"
export E014_LAMBDA="${E014_LAMBDA:-1.0}"
export E014_PENALTY="${E014_PENALTY:-0.0}"
export E014_SEED="${E014_SEED:-20261009}"
export E014_AUDIT="${E014_AUDIT:-False}"
export STAGE3_POLICY=skip
export USE_STAGE2_SRE=False USE_STAGE2_LITE_ROUTER=False
export USE_STAGE2_WEIGHTED_MERGE=False USE_T3GUIDE=False
export USE_DYNAMIC_RATIO=False USE_ADAPTIVE_SELECT=False
export USE_DEBIAS=False USE_ZONE_ROUTING=False USE_COORD_TOKENS=False
export USE_SPATIAL_AWARE=False
