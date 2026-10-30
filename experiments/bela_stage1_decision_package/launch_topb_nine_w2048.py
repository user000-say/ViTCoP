#!/usr/bin/env python3
"""Expand top B arms (B6/B8/B10) to official 9-task ACC, plus native E3 W2048.

W1024 L2: reuse existing S1_QANCHOR_L2_B{6,8,10} six-task metrics; fill
  mmmu_val / ai2d / vizwiz_vqa_val.

W2048 E3: native schedule 172/120/24/16, budget 2048, same QA hyperparams
  with max_swaps ∈ {6,8,10}. Hist S1_QANCHOR_E3 (B8) six-task is reused;
  B6/B10 run full nine; B8 only fills the three extras.

GPUs: VITCOP_DECISION_GPUS (default project 0,2,4,7). Skips busy cards.
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
from elva_acc_tasks import ACC_TASKS  # noqa: E402
from gpu_exclusive import PROJECT_GPUS as _DEFAULT_GPUS, release, try_acquire  # noqa: E402
from launch_qacover_pilot import PY, R3, PKG as QA_PKG, env_for, write_run_meta  # noqa: E402

DEC = REPO / "experiments/bela_stage1_decision_package"
OUT_ROOT = REPO / "logs/e014_aee/full/bela_stage1_decision_20261029"
HIST = REPO / "logs/e014_aee/full/bela_stage1_queryanchor_20261027"
LOG_DIR = DEC / "logs"
CFG_DIR = DEC / "configs"

_env_gpus = os.environ.get("VITCOP_DECISION_GPUS", "").strip()
PROJECT_GPUS = (
    tuple(int(x) for x in _env_gpus.split(",") if x.strip() != "")
    if _env_gpus
    else _DEFAULT_GPUS
)

TOP_B = [6, 8, 10]
# Already measured on L2 for top B / hist E3 B8.
HAVE_SIX = ("gqa", "pope", "mme", "mmbench_en_dev", "scienceqa_img", "textvqa_val_ocr")
FILL_THREE = ("mmmu_val", "ai2d", "vizwiz_vqa_val")
assert tuple(ACC_TASKS) == HAVE_SIX[:4] + ("scienceqa_img", "mmmu_val", "ai2d", "textvqa_val_ocr", "vizwiz_vqa_val")


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

    def add(cfg: Path, arm: str, task: str, out: Path, hist: Path | None = None) -> None:
        key = f"{arm}:{task}"
        if key in running_names:
            return
        if _have(out.parent, task) or (hist is not None and _have(hist, task)):
            print(f"HAVE {arm}/{task}", flush=True)
            return
        if str(out) in live:
            print(f"LIVE {arm}/{task}", flush=True)
            return
        jobs.append((cfg, arm, task, out))

    # --- W1024 L2 top-B: fill three extras ---
    for b in TOP_B:
        arm = f"S1_QANCHOR_L2_B{b}"
        cfg = CFG_DIR / f"{arm}.json"
        root = OUT_ROOT / arm
        hist = HIST / "S1_QANCHOR_L2" if b == 8 else None
        for task in FILL_THREE:
            add(cfg, arm, task, root / task, hist=None)  # hist L2 has no fill-three

    # --- W2048 E3 top-B ---
    for b in TOP_B:
        arm = f"S1_QANCHOR_E3_B{b}"
        cfg = CFG_DIR / f"{arm}.json"
        root = OUT_ROOT / arm
        # Official hist E3 is B8 six-task.
        hist = HIST / "S1_QANCHOR_E3" if b == 8 else None
        tasks = FILL_THREE if b == 8 else ACC_TASKS
        for task in tasks:
            add(cfg, arm, task, root / task, hist=hist)
        # For B8 also skip re-queue of six if somehow listed — tasks already FILL only.

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
    return child


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(
        f"TOPB_NINE_W2048 top_B={TOP_B} fill={FILL_THREE} gpus={PROJECT_GPUS} "
        f"acc9={ACC_TASKS}",
        flush=True,
    )

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
        # Prefer W1024 fill-three, then E3 gate (gqa/pope), then rest.
        def sort_key(j):
            arm, task = j[1], j[2]
            w = 0 if "_L2_" in arm else 1
            gate = 0 if task in ("gqa", "pope") else 1 if task in FILL_THREE else 2
            return (w, gate, arm, task)

        jobs.sort(key=sort_key)
        print(f"PENDING {len(jobs)} running={sorted(running_names)}", flush=True)
        if not jobs and not running:
            break
        free = free_project_gpus()
        while jobs:
            leased = None
            for gpu in free:
                if any(g.gpu == gpu for _, g, _ in running):  # type: ignore[attr-defined]
                    continue
                leased = try_acquire(gpu, f"topb:{jobs[0][1]}:{jobs[0][2]}", allowed=PROJECT_GPUS)
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
            # Sanity: W2048 must expose entry 172
            if "_E3_" in arm_id:
                assert child_env.get("ELVA_R3_ENTRY_K") == "172", child_env.get("ELVA_R3_ENTRY_K")
                assert child_env.get("BELA_QA_MAX_SWAPS") in {"6", "8", "10"}
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
    print("TOPB_NINE_W2048_DRAINED", flush=True)


if __name__ == "__main__":
    main()
