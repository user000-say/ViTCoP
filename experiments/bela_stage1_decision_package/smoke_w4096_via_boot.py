#!/usr/bin/env python3
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
REPO = Path("/home/xiangyue/ViTCoP")
sys.path.insert(0, str(REPO / "experiments/bela_query_anchor"))
from launch_qacover_pilot import PY, env_for
DEC = REPO / "experiments/bela_stage1_decision_package"
R3 = REPO / "experiments/elva_r3"
QA = REPO / "experiments/bela_query_anchor"
CFG = DEC / "configs/S1_QANCHOR_W4096.json"
OUT = REPO / "logs/e014_aee/smoke/bela_w4096_k256_20261031"
LOG = DEC / "logs" / "smoke_w4096_k256.log"
GPU = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0")
OUT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
child = os.environ.copy()
child.update(env_for(CFG, OUT))
child["BELA_STAGE1_INIT"] = "cover"
child["CUDA_VISIBLE_DEVICES"] = str(GPU)
child["PYTHONPATH"] = f"{REPO}:{REPO/'experiments'}:{R3}:{QA}:{DEC}:{REPO/'LLaVA'}:{REPO/'e020_lib'}"
child["PYTHONUNBUFFERED"] = "1"
child["HF_HUB_OFFLINE"] = child["HF_DATASETS_OFFLINE"] = child["TRANSFORMERS_OFFLINE"] = "1"
assert child["ELVA_R3_ENTRY_K"] == "256", child["ELVA_R3_ENTRY_K"]
assert child["E018_EVENTS"] == "11:74,19:36", child["E018_EVENTS"]
assert child["E020_ENABLE"] == "True"
print(f"SMOKE_W4096 gpu={GPU} entry={child['ELVA_R3_ENTRY_K']} events={child['E018_EVENTS']}", flush=True)
cmd = [PY, str(REPO/"e014_boot.py"), "--profile", "ANCHOR_A", "--task", "gqa",
       "--output", str(OUT), "--gpu-index", str(GPU), "--layer", "8", "--limit", "2"]
rc = subprocess.call(cmd, env=child, stdout=LOG.open("w"), stderr=subprocess.STDOUT)
print(f"rc={rc} log={LOG}", flush=True)
# verify traces
trace = OUT / "entry_trace.jsonl"
if trace.exists():
    rows = [json.loads(l) for l in trace.read_text().splitlines() if l.strip()]
    ks = [len(r.get("final_ids") or []) for r in rows]
    print("trace_ks", ks, "swaps", [r.get("swaps") for r in rows], flush=True)
    if not ks or any(k != 256 for k in ks):
        raise SystemExit(f"FAIL entry K not 256: {ks}")
else:
    # fallback: grep log for BLOCKED / entry
    text = LOG.read_text()[-4000:]
    print("NO_TRACE tail:\n", text, flush=True)
    if rc != 0:
        raise SystemExit("SMOKE_FAIL")
if rc != 0:
    raise SystemExit("SMOKE_FAIL")
print("SMOKE_PASS", flush=True)
