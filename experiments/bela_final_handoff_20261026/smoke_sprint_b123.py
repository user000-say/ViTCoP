"""Minimal GPU smoke for sprint B1/B2/B3 (limit=2 GQA)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
PKG = REPO / "experiments/bela_final_handoff_20261026"
CFG_DIR = PKG / "configs"
R3 = REPO / "experiments/elva_r3"
PY = "/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python"
SMOKE = REPO / "logs/e014_aee/smoke/sprint_b123_20261027"
LOG_DIR = PKG / "logs"
BASE_ARM = "E018_COVER_TWO32"
ARMS = (
    ("BELA_COVER32_FLAT", 0),
    ("BELA64_ENTRYORDER_L2", 2),
    ("BELA64_S16_NATIVE", 4),
)


def env_for(cfg_path: Path, audit_dir: Path) -> dict:
    sys.path.insert(0, str(R3))
    from e014_arms import apply_arm_env
    from apply_llava import overlay

    os.environ["ELVA_R3_CONFIG"] = str(cfg_path)
    os.environ["E014_ARM"] = BASE_ARM
    meta = overlay(apply_arm_env(BASE_ARM))
    keep = {}
    for key, value in os.environ.items():
        if key.startswith(("E0", "ELVA_", "USE_", "STAGE", "COMPUTE")) or key == "E014_ARM":
            keep[key] = value
    keep["ELVA_R3_CONFIG"] = str(cfg_path)
    keep["ELVA_R3_AUDIT_DIR"] = str(audit_dir)
    keep["E014_ARM"] = meta["arm"]
    return keep


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE.mkdir(parents=True, exist_ok=True)
    procs = []
    for arm_id, gpu in ARMS:
        cfg = CFG_DIR / f"{arm_id}.json"
        out = SMOKE / f"{arm_id}_gqa"
        out.mkdir(parents=True, exist_ok=True)
        child = os.environ.copy()
        child.update(env_for(cfg, out))
        child["CUDA_VISIBLE_DEVICES"] = str(gpu)
        child["PYTHONPATH"] = (
            f"{REPO}:{REPO / 'experiments'}:{R3}:{REPO / 'LLaVA'}:{REPO / 'e020_lib'}"
        )
        child["PYTHONUNBUFFERED"] = "1"
        child["HF_HUB_OFFLINE"] = "1"
        child["HF_DATASETS_OFFLINE"] = "1"
        child["TRANSFORMERS_OFFLINE"] = "1"
        log = LOG_DIR / f"smoke_{arm_id}.log"
        cmd = [
            PY,
            str(REPO / "e014_boot.py"),
            "--profile",
            "ANCHOR_A",
            "--task",
            "gqa",
            "--output",
            str(out),
            "--gpu-index",
            str(gpu),
            "--layer",
            "8",
            "--limit",
            "2",
        ]
        print(
            f"SMOKE {arm_id} gpu={gpu} ENTRY={child.get('ELVA_R3_ENTRY_K')} "
            f"SEL={child.get('E018_SELECTOR')} EV={child.get('E018_EVENTS')}",
            flush=True,
        )
        handle = log.open("w")
        procs.append(
            (
                arm_id,
                subprocess.Popen(
                    cmd, env=child, stdout=handle, stderr=subprocess.STDOUT
                ),
                out,
                log,
            )
        )
    failed = []
    for arm_id, proc, out, log in procs:
        rc = proc.wait()
        ok = (out / "metrics.json").exists() and rc == 0
        print(f"SMOKE DONE {arm_id} rc={rc} metrics={ok}", flush=True)
        if not ok:
            failed.append(arm_id)
            print(log.read_text()[-2000:], flush=True)
        else:
            # B2 must have written stage1 ids length 64
            s1 = out / "stage1.jsonl"
            if s1.exists():
                import json

                row = json.loads(s1.read_text().splitlines()[0])
                print(f"  stage1 n_ids={len(row['ids'])} audit_slots={row['audit'].get('replacement_slots')}", flush=True)
    if failed:
        raise SystemExit(f"smoke failed: {failed}")
    print("ALL_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
