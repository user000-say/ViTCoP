"""QCP pilot C1/C2 on W1024 L2 schedule. C0 = reuse L2_LATE_64_17_10 metrics.

GPUs {0,2,4,7}. Tasks: gqa, pope, mmbench_en_dev, textvqa_val_ocr.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
from gpu_exclusive import PROJECT_GPUS, release, try_acquire  # noqa: E402

PKG = REPO / "experiments/bela_qcp"
CFG_DIR = PKG / "configs"
R3 = REPO / "experiments/elva_r3"
PY = "/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python"
OUT_ROOT = REPO / "logs/e014_aee/full/bela_qcp_20261027"
LOG_DIR = PKG / "logs"
BASE_ARM = "E018_COVER_TWO32"
TASKS = ("gqa", "pope", "mmbench_en_dev", "textvqa_val_ocr")
ARMS = ("L2_QCP", "L2_QCP_RANDOM")


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
    mode = str(qcp.get("mode") or "off")
    keep = {}
    for key, value in os.environ.items():
        if key.startswith(("E0", "ELVA_", "USE_", "STAGE", "COMPUTE", "BELA_")) or key == "E014_ARM":
            keep[key] = value
    keep["ELVA_R3_CONFIG"] = str(cfg_path)
    keep["ELVA_R3_AUDIT_DIR"] = str(audit_dir)
    keep["E014_ARM"] = meta["arm"]
    keep["BELA_QCP_MODE"] = mode
    keep["BELA_QCP_MAX_SWAPS"] = str(int(qcp.get("max_swaps") or 2))
    keep["BELA_QCP_DONOR_POOL"] = str(int(qcp.get("donor_pool") or 16))
    keep["BELA_QCP_DROP_POOL"] = str(int(qcp.get("drop_pool") or 12))
    keep["BELA_QCP_LAMBDA"] = str(float(qcp.get("lambda") or 0.5))
    keep["BELA_QCP_TAU"] = str(float(qcp.get("tau") or 1e-4))
    keep["BELA_QCP_SEED"] = str(int(qcp.get("seed") or 20261027))
    keep["BELA_QCP_TRACE"] = "True"
    keep["BELA_ICER_MODE"] = "off"  # mutually exclusive
    return keep


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for arm_id in ARMS:
        cfg = CFG_DIR / f"{arm_id}.json"
        if not cfg.exists():
            raise FileNotFoundError(cfg)
        root = OUT_ROOT / arm_id
        root.mkdir(parents=True, exist_ok=True)
        for task in TASKS:
            if (root / task / "metrics.json").exists():
                print(f"HAVE {arm_id}/{task}", flush=True)
                continue
            jobs.append((cfg, arm_id, task, root / task))
    print(f"QCP pilot pending {len(jobs)}", flush=True)
    print(json.dumps([f"{a}:{t}" for _, a, t, _ in jobs], indent=2), flush=True)
    running: list[tuple[subprocess.Popen, object, str]] = []
    while jobs or running:
        still = []
        for proc, lease, name in running:
            if proc.poll() is None:
                still.append((proc, lease, name))
            else:
                release(lease)  # type: ignore[arg-type]
                print(f"DONE {name} rc={proc.returncode}", flush=True)
        running = still
        while jobs:
            leased = None
            for gpu in PROJECT_GPUS:
                if any(g.gpu == gpu for _, g, _ in running):  # type: ignore[attr-defined]
                    continue
                leased = try_acquire(
                    gpu, f"qcp:{jobs[0][1]}:{jobs[0][2]}", allowed=PROJECT_GPUS
                )
                if leased is not None:
                    break
            if leased is None:
                break
            cfg_path, arm_id, task, out = jobs.pop(0)
            out.mkdir(parents=True, exist_ok=True)
            log = LOG_DIR / f"acc_{arm_id}_{task}.log"
            child_env = os.environ.copy()
            child_env.update(env_for(cfg_path, out))
            child_env["CUDA_VISIBLE_DEVICES"] = str(leased.gpu)
            child_env["PYTHONPATH"] = (
                f"{REPO}:{REPO / 'experiments'}:{R3}:{PKG}:{REPO / 'LLaVA'}:{REPO / 'e020_lib'}"
            )
            child_env["PYTHONUNBUFFERED"] = "1"
            child_env["HF_HUB_OFFLINE"] = "1"
            child_env["HF_DATASETS_OFFLINE"] = "1"
            child_env["TRANSFORMERS_OFFLINE"] = "1"
            cmd = [
                PY,
                str(REPO / "e014_boot.py"),
                "--profile",
                "ANCHOR_A",
                "--task",
                task,
                "--output",
                str(out),
                "--gpu-index",
                str(leased.gpu),
                "--layer",
                "8",
            ]
            handle = log.open("w")
            proc = subprocess.Popen(
                cmd,
                env=child_env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            running.append((proc, leased, f"{arm_id}:{task}"))
            print(f"START gpu={leased.gpu} {arm_id} {task} pid={proc.pid}", flush=True)
        time.sleep(20)
    print("QCP pilot drained", flush=True)
    missing = []
    for arm_id in ARMS:
        for task in TASKS:
            if not (OUT_ROOT / arm_id / task / "metrics.json").exists():
                missing.append(f"{arm_id}:{task}")
    if missing:
        print("MISSING", missing, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
