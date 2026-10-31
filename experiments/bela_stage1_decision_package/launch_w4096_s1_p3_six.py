#!/usr/bin/env python3
"""LLaVA-1.5-7B W4096 / layer-avg 77.8% (256→74→36): S1 vs P3 on six ACC tasks.

Schedule A from BELA_V100_CROSSMODEL_TOKEN_PLAN_KIT_20261031.
Arms: S1_QANCHOR_W4096 (RDRIFT→Q_ATTN) and P3_QA_QATTN_W4096 (Q_ATTN→Q_ATTN).
Tasks: gqa, pope, mme, mmbench_en_dev, scienceqa_img, textvqa_val_ocr.
GPUs: VITCOP_DECISION_GPUS or project 0/2/4/7; skips cards with >800 MiB used.
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
from gpu_exclusive import PROJECT_GPUS as _DEFAULT_GPUS, release, try_acquire  # noqa: E402
from launch_qacover_pilot import PY, R3, PKG as QA_PKG, env_for, write_run_meta  # noqa: E402

DEC = REPO / "experiments/bela_stage1_decision_package"
OUT_ROOT = REPO / "logs/e014_aee/full/bela_w4096_77p8_20261031"
LOG_DIR = DEC / "logs"
CFG_DIR = DEC / "configs"

_env_gpus = os.environ.get("VITCOP_DECISION_GPUS", "").strip()
PROJECT_GPUS = (
    tuple(int(x) for x in _env_gpus.split(",") if x.strip() != "")
    if _env_gpus
    else _DEFAULT_GPUS
)

SIX_TASKS = ("gqa", "pope", "mme", "mmbench_en_dev", "scienceqa_img", "textvqa_val_ocr")
ARMS = (
    ("S1_QANCHOR_W4096", CFG_DIR / "S1_QANCHOR_W4096.json"),
    ("P3_QA_QATTN_W4096", CFG_DIR / "P3_QA_QATTN_W4096.json"),
)


def _have(root: Path, task: str) -> bool:
    return (root / task / "metrics.json").exists()


def _live_outputs() -> set[str]:
    live: set[str] = set()
    try:
        out = subprocess.check_output(["ps", "-eo", "args="], text=True)
    except Exception:
        return live
    needle = str(OUT_ROOT) + "/"
    for line in out.splitlines():
        if "e014_boot.py" not in line or "--output" not in line:
            continue
        parts = line.split()
        for i, p in enumerate(parts):
            if p == "--output" and i + 1 < len(parts):
                path = parts[i + 1]
                if path.startswith(needle):
                    live.add(path.rstrip("/"))
                break
    return live


def _jobs(running_names: set[str] | None = None):
    running_names = running_names or set()
    live = _live_outputs()
    jobs = []
    for arm_id, cfg in ARMS:
        root = OUT_ROOT / arm_id
        for task in SIX_TASKS:
            key = f"{arm_id}:{task}"
            if key in running_names:
                continue
            if _have(root, task):
                print(f"HAVE {arm_id}/{task}", flush=True)
                continue
            out = root / task
            if str(out) in live:
                print(f"LIVE {arm_id}/{task}", flush=True)
                continue
            jobs.append((cfg, arm_id, task, out))
    return jobs


def _child_env(cfg_path: Path, out: Path) -> dict:
    child = os.environ.copy()
    child.update(env_for(cfg_path, out))
    cfg = json.loads(cfg_path.read_text())
    init = cfg.get("method", {}).get("stage1_init", "cover")
    child["BELA_STAGE1_INIT"] = init
    child["PYTHONPATH"] = (
        f"{REPO}:{REPO / 'experiments'}:{R3}:{QA_PKG}:{DEC}:{REPO / 'LLaVA'}:{REPO / 'e020_lib'}"
    )
    child["PYTHONUNBUFFERED"] = "1"
    child["HF_HUB_OFFLINE"] = "1"
    child["HF_DATASETS_OFFLINE"] = "1"
    child["TRANSFORMERS_OFFLINE"] = "1"
    # Contract asserts for W4096 schedule A.
    assert child.get("ELVA_R3_ENTRY_K") == "256", child.get("ELVA_R3_ENTRY_K")
    assert child.get("ELVA_R3_FINAL_K") == "36", child.get("ELVA_R3_FINAL_K")
    assert child.get("E018_EVENTS") == "11:74,19:36", child.get("E018_EVENTS")
    return child


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


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(
        f"W4096_S1_P3_SIX tasks={SIX_TASKS} arms={[a for a, _ in ARMS]} gpus={PROJECT_GPUS}",
        flush=True,
    )
    running: list[tuple[subprocess.Popen, object, str]] = []
    while True:
        still = []
        for proc, lease, name in running:
            if proc.poll() is None:
                still.append((proc, lease, name))
            else:
                release(lease)  # type: ignore[arg-type]
                print(f"DONE {name} rc={proc.returncode}", flush=True)
        running = still
        running_names = {name for _, _, name in running}
        jobs = _jobs(running_names)
        # Prefer gqa/pope first for both arms.
        gate = {"gqa", "pope"}
        jobs.sort(key=lambda j: (0 if j[2] in gate else 1, j[1], j[2]))
        print(f"PENDING {len(jobs)} running={sorted(running_names)}", flush=True)
        if not jobs and not running:
            break
        free = free_project_gpus()
        while jobs:
            leased = None
            for gpu in free:
                if any(g.gpu == gpu for _, g, _ in running):  # type: ignore[attr-defined]
                    continue
                leased = try_acquire(gpu, f"w4096:{jobs[0][1]}:{jobs[0][2]}", allowed=PROJECT_GPUS)
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
    print("W4096_S1_P3_SIX_DRAINED", flush=True)


if __name__ == "__main__":
    main()
