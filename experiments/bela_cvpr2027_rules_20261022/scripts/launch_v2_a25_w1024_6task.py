"""Handoff V2: A25 (172/125/18) + W1024_probe (96/50/12) — 6-task ACC then TTFT.

6 tasks: gqa, pope, mme, mmbench_en_dev, scienceqa_img, textvqa_val_ocr.
Uses gpu_exclusive flock before every spawn. ACC on {0,2,4,7}; TTFT sequential after ACC.
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
from gpu_exclusive import PROJECT_GPUS, acquire_one, release, try_acquire  # noqa: E402

PKG = REPO / "experiments/bela_cvpr2027_rules_20261022"
R3 = REPO / "experiments/elva_r3"
PY = "/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python"
PYBENCH = str(REPO / "experiments/e014_latency_fair_256/bin/python")
BENCH = REPO / "experiments/e014_latency_fair_256/scripts/benchmark_llava_e2e_anygpu.py"
PANEL = REPO / "experiments/e014_latency_fair_256/panel/speed_256.jsonl"
MODEL = REPO / "experiments/e014_latency_fair_256/models_llava"
OUT_ROOT = REPO / "logs/e014_aee/full/bela_cvpr27_handoff_v2_20261024"
LAT_ROOT = PKG / "latency"
BASE_ARM = "E018_COVER_TWO32"

TASKS6 = (
    "gqa",
    "pope",
    "mme",
    "mmbench_en_dev",
    "scienceqa_img",
    "textvqa_val_ocr",
)

ARMS = (
    (PKG / "configs/A25_COVER_RDRIFT_W2048.json", "A25_COVER_RDRIFT"),
    (PKG / "configs/W1024_PROBE_COVER_RDRIFT.json", "W1024_PROBE_COVER_RDRIFT"),
)

TTFT_ARMS = (
    "A25_COVER_RDRIFT_W2048",
    "W1024_PROBE_COVER_RDRIFT",
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


def run_acc() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for cfg_path, out_id in ARMS:
        root = OUT_ROOT / out_id
        root.mkdir(parents=True, exist_ok=True)
        for task in TASKS6:
            if (root / task / "metrics.json").exists():
                print(f"HAVE {out_id}/{task}", flush=True)
                continue
            jobs.append((cfg_path, out_id, task, root / task))
    print(f"ACC pending {len(jobs)} (A25 + W1024_probe × 6)", flush=True)
    print(json.dumps([f"{o}:{t}" for _, o, t, _ in jobs], indent=2), flush=True)

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
                owner = f"acc:{jobs[0][1]}:{jobs[0][2]}"
                leased = try_acquire(gpu, owner, allowed=PROJECT_GPUS)
                if leased is not None:
                    break
            if leased is None:
                break
            cfg_path, out_id, task, out = jobs.pop(0)
            out.mkdir(parents=True, exist_ok=True)
            log = PKG / "logs" / f"acc_{out_id}_{task}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
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
            handle = log.open("w")
            proc = subprocess.Popen(
                cmd,
                env=child_env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            running.append((proc, leased, f"{out_id}:{task}"))
            print(
                f"START gpu={leased.gpu} {out_id} {task} pid={proc.pid} (locked)",
                flush=True,
            )
        time.sleep(20)
    print("ACC queue drained", flush=True)


def run_ttft() -> None:
    LAT_ROOT.mkdir(parents=True, exist_ok=True)
    for arm in TTFT_ARMS:
        cfg_path = PKG / "configs" / f"{arm}.json"
        out = LAT_ROOT / arm
        if (out / "latency_summary.json").exists():
            print(f"SKIP TTFT {arm}", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        log = PKG / "logs" / f"ttft_{arm}.log"
        lease = acquire_one(f"ttft:{arm}", allowed=PROJECT_GPUS)
        try:
            child_env = os.environ.copy()
            child_env.update(env_for(cfg_path, out))
            child_env["CUDA_VISIBLE_DEVICES"] = str(lease.gpu)
            child_env["PYTHONPATH"] = (
                f"{REPO}:{REPO / 'experiments'}:{R3}:{REPO / 'LLaVA'}:{REPO / 'e020_lib'}"
            )
            child_env["PYTHONUNBUFFERED"] = "1"
            child_env["HF_HUB_OFFLINE"] = "1"
            child_env["TRANSFORMERS_OFFLINE"] = "1"
            child_env["HF_DATASETS_OFFLINE"] = "1"
            cmd = [
                PYBENCH,
                str(BENCH),
                "--repo-root",
                str(REPO),
                "--panel",
                str(PANEL),
                "--model-path",
                str(MODEL),
                "--conv-mode",
                "llava_v1",
                "--out-dir",
                str(out),
                "--warmups",
                "8",
                "--limit",
                "0",
            ]
            print(f"TTFT START gpu={lease.gpu} {arm} (locked)", flush=True)
            with log.open("w") as handle:
                rc = subprocess.call(
                    cmd, env=child_env, stdout=handle, stderr=subprocess.STDOUT
                )
            if rc != 0 or not (out / "latency_summary.json").exists():
                raise SystemExit(f"TTFT FAIL {arm} rc={rc}")
            s = json.loads((out / "latency_summary.json").read_text())
            print(
                f"TTFT DONE {arm} ttft={s['ttft_wall_ms']['mean']:.1f} "
                f"fixed32={s['fixed32_wall_ms']['mean']:.1f}",
                flush=True,
            )
        finally:
            release(lease)
    print("TTFT queue drained", flush=True)


def main() -> None:
    run_acc()
    run_ttft()
    print("V2 A25+W1024_probe ACC+TTFT ALL DONE", flush=True)


if __name__ == "__main__":
    main()
