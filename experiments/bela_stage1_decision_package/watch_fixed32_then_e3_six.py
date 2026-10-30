#!/usr/bin/env python3
"""After Fixed32 S1 vs P3 finishes, run W2048 E3 six-task ACC for both Stage2 arms.

Arms:
  E3_QA_RDRIFT  = Dual+QA B8 + first RDRIFT (hist S1_QANCHOR_E3 reused if present)
  E3_QA_QATTN   = Dual+QA B8 + all-cut Q_ATTN (new; same Stage1 as E3)

GPUs: VITCOP_DECISION_GPUS (default 0,2,4,6,7). Does not kill foreign jobs.
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

from gpu_exclusive import PROJECT_GPUS as _DEFAULT, release, try_acquire  # noqa: E402
from launch_qacover_pilot import PY, R3, PKG as QA_PKG, env_for, write_run_meta  # noqa: E402

DEC = REPO / "experiments/bela_stage1_decision_package"
OUT_ROOT = REPO / "logs/e014_aee/full/bela_stage1_decision_20261029"
HIST = REPO / "logs/e014_aee/full/bela_stage1_queryanchor_20261027"
LAT = DEC / "latency"
LOG_DIR = DEC / "logs"
SIX = ("gqa", "pope", "mme", "mmbench_en_dev", "scienceqa_img", "textvqa_val_ocr")

ARMS = [
    # (arm_id, cfg, hist_alias_or_None)
    ("E3_QA_RDRIFT", QA_PKG / "configs/S1_QANCHOR_E3.json", "S1_QANCHOR_E3"),
    ("E3_QA_QATTN", DEC / "configs/E3_QA_QATTN.json", None),
]

_env_gpus = os.environ.get("VITCOP_DECISION_GPUS", "").strip()
PROJECT_GPUS = (
    tuple(int(x) for x in _env_gpus.split(",") if x.strip() != "")
    if _env_gpus
    else _DEFAULT
)


def _have(root: Path, task: str) -> bool:
    return (root / task / "metrics.json").exists()


def _fixed32_done() -> bool:
    a = LAT / "S1_QANCHOR_L2_RDRIFT" / "latency_summary.json"
    b = LAT / "P3_QA_QATTN" / "latency_summary.json"
    return a.exists() and b.exists()


def _pending() -> list[tuple[Path, str, str, Path]]:
    jobs = []
    for arm, cfg, hist in ARMS:
        write = OUT_ROOT / arm
        for task in SIX:
            if _have(write, task):
                continue
            if hist and _have(HIST / hist, task):
                print(f"REUSE_HIST {arm}/{task} <- {hist}", flush=True)
                continue
            jobs.append((cfg, arm, task, write / task))
    return jobs


def _child_env(cfg_path: Path, out: Path) -> dict:
    child = os.environ.copy()
    child.update(env_for(cfg_path, out))
    cfg = json.loads(cfg_path.read_text())
    child["BELA_STAGE1_INIT"] = cfg.get("method", {}).get("stage1_init", "cover")
    child.pop("BELA_QA_PASS_CAPS", None)
    child["PYTHONPATH"] = (
        f"{REPO}:{REPO / 'experiments'}:{R3}:{QA_PKG}:{DEC}:{REPO / 'LLaVA'}:{REPO / 'e020_lib'}"
    )
    child["PYTHONUNBUFFERED"] = "1"
    child["HF_HUB_OFFLINE"] = "1"
    child["HF_DATASETS_OFFLINE"] = "1"
    child["TRANSFORMERS_OFFLINE"] = "1"
    # E3 entry K must be 172
    if child.get("ELVA_R3_ENTRY_K") != "172":
        # overlay should set from schedule; assert for safety
        counts = cfg["method"]["schedule"]["counts"]
        if int(counts[0]) != 172:
            raise RuntimeError(f"E3 expected entry 172, got {counts[0]}")
    return child


def _free_gpus(held: set[int]) -> list[int]:
    out = []
    for g in PROJECT_GPUS:
        if g in held:
            continue
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


def wait_fixed32() -> None:
    print("WAIT_FIXED32 for S1_QANCHOR_L2_RDRIFT + P3_QA_QATTN latency_summary.json", flush=True)
    while not _fixed32_done():
        # still alive?
        time.sleep(30)
        print(f"WAIT_FIXED32 tick have_s1={ (LAT/'S1_QANCHOR_L2_RDRIFT'/'latency_summary.json').exists() } "
              f"have_p3={ (LAT/'P3_QA_QATTN'/'latency_summary.json').exists() }", flush=True)
    print("FIXED32_BOTH_DONE — starting E3 six ACC", flush=True)
    for arm in ("S1_QANCHOR_L2_RDRIFT", "P3_QA_QATTN"):
        s = json.loads((LAT / arm / "latency_summary.json").read_text())
        print(
            f"  {arm}: TTFT_mean={s['ttft_wall_ms']['mean']:.2f} "
            f"Fixed32_mean={s['fixed32_wall_ms']['mean']:.2f}",
            flush=True,
        )


def run_acc() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = _pending()
    print(f"E3_SIX pending={len(jobs)} gpus={PROJECT_GPUS}", flush=True)
    if not jobs:
        print("E3_SIX_NOTHING_TO_RUN (hist reuse covers RDRIFT; QATTN already done?)", flush=True)
        return

    # Prefer QATTN tasks first (RDRIFT may be empty after hist reuse).
    jobs.sort(key=lambda j: (0 if "QATTN" in j[1] else 1, j[2], j[1]))
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
        held = {lease.gpu for _, lease, _ in running}  # type: ignore[attr-defined]
        while jobs:
            free = _free_gpus(held)
            if not free:
                break
            leased = None
            for gpu in free:
                leased = try_acquire(gpu, f"e3six:{jobs[0][1]}:{jobs[0][2]}", allowed=PROJECT_GPUS)
                if leased is not None:
                    break
            if leased is None:
                break
            cfg, arm, task, out = jobs.pop(0)
            if out.exists() and not (out / "metrics.json").exists():
                import shutil

                shutil.rmtree(out, ignore_errors=True)
            out.mkdir(parents=True, exist_ok=True)
            child = _child_env(cfg, out)
            child["CUDA_VISIBLE_DEVICES"] = str(leased.gpu)
            write_run_meta(out, arm, cfg, child)
            assert child.get("ELVA_R3_ENTRY_K") == "172", child.get("ELVA_R3_ENTRY_K")
            log = LOG_DIR / f"acc_{arm}_{task}.log"
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
                cmd, env=child, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
            )
            running.append((proc, leased, f"{arm}:{task}"))
            held.add(leased.gpu)
            print(f"START gpu={leased.gpu} {arm}/{task} pid={proc.pid}", flush=True)
        time.sleep(8)
    print("E3_SIX_COMPLETE", flush=True)


def _stop_master_queue() -> None:
    """Free project GPUs for E3 six; ACC master can be restarted later."""
    subprocess.call(["pkill", "-f", "experiments/bela_stage1_decision_package/launch_master_queue.py"])
    time.sleep(2)
    print("MASTER_QUEUE_STOPPED for E3 six priority", flush=True)


def main() -> None:
    wait_fixed32()
    _stop_master_queue()
    # brief settle so Fixed32 process has released GPU4 VRAM
    time.sleep(15)
    run_acc()


if __name__ == "__main__":
    main()
