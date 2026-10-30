#!/usr/bin/env python3
"""P2 structure necessity matrix (handoff §E) — mandatory, same RDRIFT / W1024.

Arms (configs under decision package or bela_query_anchor):
  S0 L2_NATIVE              Native Dual COVER, QA off
  S1 S1_QANCHOR_L2 (hist)   COVER→QA B8 — six-task already on disk, skipped
  S2 S2_SALIENT_QA_B8       salient Top-K → same QA B8
  S3 S3_JOINT               one-shot joint cover K64
  S4 S4_JOINT_CAP8          joint with ≤8 new vs native
  S5 S5_QUERY_TOPK          query-only Top-K64
  S6 S1_MATCHED_RANDOM      matched-random swaps
  S7 S1_QUERY_SHUFFLE       mismatched query
  S8_*                      novelty / coverage / query-gain ablations

Default: official six-task ACC (GQA/POPE/MME/MMBench/SciQA/TextVQA-OCR).
Does not invent algorithms; uses existing stage1_init + QA env bindings.
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

SIX = ("gqa", "pope", "mme", "mmbench_en_dev", "scienceqa_img", "textvqa_val_ocr")
GATE = ("gqa", "pope")

# (arm_id, cfg_path, hist_alias_or_None)
# hist_alias: if metrics exist under HIST/alias/task, treat as HAVE.
ARMS: list[tuple[str, Path, str | None]] = [
    ("L2_NATIVE", QA_PKG / "configs/L2_NATIVE.json", "L2_NATIVE"),  # S0
    # S1 already complete as HIST/S1_QANCHOR_L2 — listed for HAVE logging only
    ("S1_QANCHOR_L2", QA_PKG / "configs/S1_QANCHOR_L2.json", "S1_QANCHOR_L2"),
    ("S2_SALIENT_QA_B8", CFG_DIR / "S2_SALIENT_QA_B8.json", None),
    ("S3_JOINT", CFG_DIR / "S3_JOINT.json", None),
    ("S4_JOINT_CAP8", CFG_DIR / "S4_JOINT_CAP8.json", None),
    ("S5_QUERY_TOPK", CFG_DIR / "S5_QUERY_TOPK.json", None),
    ("S1_MATCHED_RANDOM", QA_PKG / "configs/S1_MATCHED_RANDOM.json", "S1_MATCHED_RANDOM"),  # S6
    ("S1_QUERY_SHUFFLE", QA_PKG / "configs/S1_QUERY_SHUFFLE.json", "S1_QUERY_SHUFFLE"),  # S7
    ("S8_NO_NOVELTY", CFG_DIR / "S8_NO_NOVELTY.json", None),
    ("S8_NO_COVERAGE", CFG_DIR / "S8_NO_COVERAGE.json", None),
    ("S8_QUERY_ONLY_GAIN", CFG_DIR / "S8_QUERY_ONLY_GAIN.json", None),
]


def _have(root: Path, task: str) -> bool:
    return (root / task / "metrics.json").exists()


def _metric(write: Path, hist_alias: str | None, task: str) -> Path | None:
    if _have(write, task):
        return write / task / "metrics.json"
    if hist_alias and _have(HIST / hist_alias, task):
        return HIST / hist_alias / task / "metrics.json"
    return None


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
    for arm, cfg, hist_alias in ARMS:
        if not cfg.exists():
            print(f"MISSING_CFG {cfg}", flush=True)
            continue
        write = OUT_ROOT / arm
        for task in SIX:
            key = f"{arm}:{task}"
            if key in running_names:
                continue
            if _metric(write, hist_alias, task) is not None:
                print(f"HAVE {arm}/{task}", flush=True)
                continue
            out = write / task
            if str(out) in live:
                print(f"LIVE {arm}/{task}", flush=True)
                continue
            jobs.append((cfg, arm, task, out))
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
    print(f"P2_STRUCT six={SIX} arms={[a[0] for a in ARMS]} gpus={PROJECT_GPUS}", flush=True)

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
        # Prefer GQA/POPE gate for every arm before extras.
        jobs.sort(key=lambda j: (0 if j[2] in GATE else 1, j[1], j[2]))
        print(f"PENDING {len(jobs)} running={sorted(running_names)}", flush=True)
        if not jobs and not running:
            break
        free = free_project_gpus()
        while jobs:
            leased = None
            for gpu in free:
                if any(g.gpu == gpu for _, g, _ in running):  # type: ignore[attr-defined]
                    continue
                leased = try_acquire(gpu, f"p2:{jobs[0][1]}:{jobs[0][2]}", allowed=PROJECT_GPUS)
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
            print(
                f"ENV {arm_id} init={child_env.get('BELA_STAGE1_INIT')} "
                f"qa={child_env.get('BELA_QA_ENTRY_MODE')} swaps={child_env.get('BELA_QA_MAX_SWAPS')}",
                flush=True,
            )
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
    print("P2_STRUCT_DRAINED", flush=True)


if __name__ == "__main__":
    main()
