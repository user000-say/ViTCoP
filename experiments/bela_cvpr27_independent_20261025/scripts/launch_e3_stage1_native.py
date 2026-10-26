"""Independent P2: E3 schedule, Stage1 guide ablations (pen / l9), dual=E3 reuse.

Same Stage2 COVER+RDRIFT host. Full 6-task ACC on GPUs {0,2,4,7}.
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

PKG = REPO / "experiments/bela_cvpr27_independent_20261025"
CFG_DIR = PKG / "configs" / "e3_stage1"
R3 = REPO / "experiments/elva_r3"
PY = "/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python"
OUT_ROOT = REPO / "logs/e014_aee/full/bela_independent_e3_stage1_20261026"
LOG_DIR = PKG / "logs"
BASE_ARM = "E018_COVER_TWO32"

# E3 schedule: 172×4 + 120×8 + 24×10 + 16×10 = 2048
E3_COUNTS = (172, 120, 24, 16)
E3_LENGTHS = (4, 8, 10, 10)

TASKS6 = (
    "gqa",
    "pope",
    "mme",
    "mmbench_en_dev",
    "scienceqa_img",
    "textvqa_val_ocr",
)

# (arm_id, guide, beta) — dual already measured as E3_RET24_TO16
ARMS = (
    ("E3_STAGE1_PENULTIMATE_ONLY", "pen", 0.0),
    ("E3_STAGE1_LAYER9_ONLY", "l9", 0.0),
)


def write_cfg(arm_id: str, guide: str, beta: float) -> Path:
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    path = CFG_DIR / f"{arm_id}.json"
    cfg = {
        "id": arm_id,
        "phase": "cvpr2027_independent_e3_stage1",
        "method": {
            "original_tokens": 576,
            "depth": 32,
            "budget": 2048,
            "schedule": {"counts": list(E3_COUNTS), "lengths": list(E3_LENGTHS)},
            "stage1": {
                "guide": guide,
                "cover_family": "legacy_native",
                "cover": True,
                "beta": beta,
                "score_guide": "same",
                "payload": "pen",
                "saliency": "BIND_ORIGINAL_SAME_RULE",
            },
            "stage2": {"first_selector": "RDRIFT", "later_selector": "Q_ATTN"},
            "cache_mode": "native",
            "merge": False,
            "learned": False,
        },
    }
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    return path


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
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for arm_id, guide, beta in ARMS:
        cfg = write_cfg(arm_id, guide, beta)
        env_for(cfg, CFG_DIR / f"_overlay_{arm_id}")
        print(
            f"overlay {arm_id} guide={guide} ENTRY={os.environ.get('ELVA_R3_ENTRY_K')} "
            f"FINAL={os.environ.get('ELVA_R3_FINAL_K')} EV={os.environ.get('E018_EVENTS')}",
            flush=True,
        )
        root = OUT_ROOT / arm_id
        root.mkdir(parents=True, exist_ok=True)
        for task in TASKS6:
            if (root / task / "metrics.json").exists():
                print(f"HAVE {arm_id}/{task}", flush=True)
                continue
            jobs.append((cfg, arm_id, task, root / task))
    print(f"E3 Stage1 ACC pending {len(jobs)}", flush=True)

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
                    gpu, f"acc:{jobs[0][1]}:{jobs[0][2]}", allowed=PROJECT_GPUS
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
                f"{REPO}:{REPO / 'experiments'}:{R3}:{REPO / 'LLaVA'}:{REPO / 'e020_lib'}"
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
            print(
                f"START gpu={leased.gpu} {arm_id} {task} pid={proc.pid}",
                flush=True,
            )
        time.sleep(20)
    print("E3 Stage1 ACC drained", flush=True)


if __name__ == "__main__":
    main()
