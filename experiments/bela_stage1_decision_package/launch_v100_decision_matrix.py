#!/usr/bin/env python3
"""V100 Stage1 decision matrix from BELA_CVPR2027_STAGE1_DECISION_EXPERIMENT_KIT.

Reuses production elva_r3 + bela_query_anchor host (vitcopreback).
GPUs: only free cards from {0,2,4,7}. Skips busy / foreign jobs.
Does not regenerate completed S1_QANCHOR_L2 six-task metrics.
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
sys.path.insert(0, str(REPO / "experiments/bela_query_anchor"))
from gpu_exclusive import PROJECT_GPUS, release, try_acquire  # noqa: E402
from launch_qacover_pilot import PY, R3, PKG as QA_PKG, env_for, write_run_meta  # noqa: E402

DEC = REPO / "experiments/bela_stage1_decision_package"
OUT_ROOT = REPO / "logs/e014_aee/full/bela_stage1_decision_20261029"
LOG_DIR = DEC / "logs"
CFG_DIR = DEC / "configs"

# P1 B-scan on GQA+POPE (B8 already measured as S1_QANCHOR_L2 — skip).
B_SCAN = [0, 4, 12, 16, 24]
# Structure arms that need new runs (S0/S1/S6 partially on disk).
STRUCT = [
    "S2_SALIENT_QA_B8",
    "S3_JOINT",
    "S4_JOINT_CAP8",
    "S5_QUERY_TOPK",
    "S8_NO_NOVELTY",
    "S8_NO_COVERAGE",
    "S8_QUERY_ONLY_GAIN",
]
# Finish incomplete historical controls under the decision out root (copy-or-run).
FINISH = [
    ("L2_NATIVE", QA_PKG / "configs/L2_NATIVE.json", ("mme", "mmbench_en_dev", "scienceqa_img", "textvqa_val_ocr")),
    ("S1_MATCHED_RANDOM", QA_PKG / "configs/S1_MATCHED_RANDOM.json", ("mme", "mmbench_en_dev", "scienceqa_img", "textvqa_val_ocr")),
    ("S1_QUERY_SHUFFLE", QA_PKG / "configs/S1_QUERY_SHUFFLE.json", ("pope", "mme", "mmbench_en_dev", "scienceqa_img", "textvqa_val_ocr")),
]
CORE = ("gqa", "pope")


def _have(root: Path, task: str) -> bool:
    return (root / task / "metrics.json").exists()


def _jobs():
    jobs = []
    for b in B_SCAN:
        arm = f"S1_QANCHOR_L2_B{b}"
        cfg = CFG_DIR / f"{arm}.json"
        root = OUT_ROOT / arm
        for task in CORE:
            if _have(root, task):
                print(f"HAVE {arm}/{task}", flush=True)
                continue
            jobs.append((cfg, arm, task, root / task))
    for arm in STRUCT:
        cfg = CFG_DIR / f"{arm}.json"
        root = OUT_ROOT / arm
        for task in CORE:
            if _have(root, task):
                print(f"HAVE {arm}/{task}", flush=True)
                continue
            jobs.append((cfg, arm, task, root / task))
    for arm, cfg, tasks in FINISH:
        # Prefer historical dir if already complete; else write under decision root.
        hist = REPO / "logs/e014_aee/full/bela_stage1_queryanchor_20261027" / arm
        root = OUT_ROOT / arm
        for task in tasks:
            if _have(hist, task) or _have(root, task):
                print(f"HAVE {arm}/{task}", flush=True)
                continue
            jobs.append((cfg, arm, task, root / task))
    return jobs


def _child_env(cfg_path: Path, out: Path) -> dict:
    child = os.environ.copy()
    child.update(env_for(cfg_path, out))
    cfg = json.loads(cfg_path.read_text())
    init = cfg.get("method", {}).get("stage1_init", "cover")
    child["BELA_STAGE1_INIT"] = init
    # Prefer project GPUs that are free; caller sets CUDA_VISIBLE_DEVICES.
    child["PYTHONPATH"] = (
        f"{REPO}:{REPO / 'experiments'}:{R3}:{QA_PKG}:{DEC}:{REPO / 'LLaVA'}:{REPO / 'e020_lib'}"
    )
    child["PYTHONUNBUFFERED"] = "1"
    child["HF_HUB_OFFLINE"] = "1"
    child["HF_DATASETS_OFFLINE"] = "1"
    child["TRANSFORMERS_OFFLINE"] = "1"
    return child


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = _jobs()
    print(f"DECISION_MATRIX pending {len(jobs)} on {PROJECT_GPUS}", flush=True)
    # Avoid GPUs already held by our Dense queue / other users (>800 MiB).
    def free_project_gpus():
        out = []
        for g in PROJECT_GPUS:
            try:
                used = int(
                    subprocess.check_output(
                        [
                            "nvidia-smi",
                            f"--id={g}",
                            "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                    ).strip()
                )
            except Exception:
                used = 99999
            if used < 800:
                out.append(g)
        return out

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
        free = free_project_gpus()
        while jobs:
            leased = None
            for gpu in free:
                if any(g.gpu == gpu for _, g, _ in running):  # type: ignore[attr-defined]
                    continue
                leased = try_acquire(gpu, f"decision:{jobs[0][1]}:{jobs[0][2]}", allowed=PROJECT_GPUS)
                if leased is not None:
                    break
            if leased is None:
                break
            cfg_path, arm_id, task, out = jobs.pop(0)
            out.mkdir(parents=True, exist_ok=True)
            log = LOG_DIR / f"acc_{arm_id}_{task}.log"
            child_env = _child_env(cfg_path, out)
            child_env["CUDA_VISIBLE_DEVICES"] = str(leased.gpu)
            write_run_meta(out, arm_id, cfg_path, child_env)
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
            proc = subprocess.Popen(
                cmd,
                env=child_env,
                stdout=log.open("w"),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            running.append((proc, leased, f"{arm_id}:{task}"))
            print(f"START gpu={leased.gpu} {arm_id} {task} pid={proc.pid}", flush=True)
            free = free_project_gpus()
        time.sleep(30)
    print("DECISION_MATRIX_DRAINED", flush=True)


if __name__ == "__main__":
    main()
