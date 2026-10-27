"""GPU smoke limit=2 for L2_QCP and L2_QCP_RANDOM."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
PKG = REPO / "experiments/bela_qcp"
R3 = REPO / "experiments/elva_r3"
PY = "/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python"
SMOKE = REPO / "logs/e014_aee/smoke/bela_qcp_20261027"
BASE_ARM = "E018_COVER_TWO32"
ARMS = (("L2_QCP", 0), ("L2_QCP_RANDOM", 2))


def env_for(cfg_path: Path, audit_dir: Path) -> dict:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(R3))
    from e014_arms import apply_arm_env
    from apply_llava import overlay

    os.environ["ELVA_R3_CONFIG"] = str(cfg_path)
    os.environ["E014_ARM"] = BASE_ARM
    meta = overlay(apply_arm_env(BASE_ARM))
    cfg = json.loads(cfg_path.read_text())
    qcp = cfg["method"].get("qcp") or {}
    keep = {}
    for key, value in os.environ.items():
        if key.startswith(("E0", "ELVA_", "USE_", "STAGE", "COMPUTE", "BELA_")) or key == "E014_ARM":
            keep[key] = value
    keep.update({
        "ELVA_R3_CONFIG": str(cfg_path),
        "ELVA_R3_AUDIT_DIR": str(audit_dir),
        "E014_ARM": meta["arm"],
        "BELA_QCP_MODE": str(qcp.get("mode") or "off"),
        "BELA_QCP_MAX_SWAPS": str(int(qcp.get("max_swaps") or 2)),
        "BELA_QCP_DONOR_POOL": str(int(qcp.get("donor_pool") or 16)),
        "BELA_QCP_DROP_POOL": str(int(qcp.get("drop_pool") or 12)),
        "BELA_QCP_LAMBDA": str(float(qcp.get("lambda") or 0.5)),
        "BELA_QCP_TAU": str(float(qcp.get("tau") or 1e-4)),
        "BELA_QCP_SEED": str(int(qcp.get("seed") or 20261027)),
        "BELA_QCP_TRACE": "True",
        "BELA_ICER_MODE": "off",
    })
    return keep


def main() -> None:
    SMOKE.mkdir(parents=True, exist_ok=True)
    (PKG / "logs").mkdir(parents=True, exist_ok=True)
    procs = []
    for arm_id, gpu in ARMS:
        cfg = PKG / "configs" / f"{arm_id}.json"
        out = SMOKE / f"{arm_id}_gqa"
        out.mkdir(parents=True, exist_ok=True)
        child = os.environ.copy()
        child.update(env_for(cfg, out))
        child["CUDA_VISIBLE_DEVICES"] = str(gpu)
        child["PYTHONPATH"] = (
            f"{REPO}:{REPO / 'experiments'}:{R3}:{PKG}:{REPO / 'LLaVA'}:{REPO / 'e020_lib'}"
        )
        child["PYTHONUNBUFFERED"] = "1"
        child["HF_HUB_OFFLINE"] = child["HF_DATASETS_OFFLINE"] = child["TRANSFORMERS_OFFLINE"] = "1"
        log = PKG / "logs" / f"smoke_{arm_id}.log"
        cmd = [PY, str(REPO / "e014_boot.py"), "--profile", "ANCHOR_A", "--task", "gqa",
               "--output", str(out), "--gpu-index", str(gpu), "--layer", "8", "--limit", "2"]
        print(f"SMOKE {arm_id} gpu={gpu} QCP={child['BELA_QCP_MODE']} EV={child.get('E018_EVENTS')}", flush=True)
        procs.append((arm_id, subprocess.Popen(cmd, env=child, stdout=log.open("w"), stderr=subprocess.STDOUT), out, log))
    failed = []
    for arm_id, proc, out, log in procs:
        rc = proc.wait()
        ok = rc == 0 and (out / "metrics.json").exists()
        print(f"SMOKE DONE {arm_id} rc={rc} metrics={ok}", flush=True)
        if (out / "audit.json").exists():
            a = json.loads((out / "audit.json").read_text())
            print(f"  audit stage1={a.get('stage1_tokens')} final={a.get('final_visual_tokens')} events={a.get('e018_events')}", flush=True)
        if (out / "qcp_trace.jsonl").exists():
            lines = (out / "qcp_trace.jsonl").read_text().splitlines()
            print(f"  qcp_trace rows={len(lines)} first={lines[0][:200] if lines else ''}", flush=True)
        if not ok:
            failed.append(arm_id)
            print(log.read_text()[-1500:], flush=True)
    if failed:
        raise SystemExit(f"smoke failed: {failed}")
    print("ALL_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
