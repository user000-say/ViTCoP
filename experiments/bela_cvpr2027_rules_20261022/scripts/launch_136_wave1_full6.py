"""136-arm handoff Wave-1: progressive full-6 ACC (no subset sampling).

W2048 curated schedule arms + W1024 k0=96 grid (skip already-measured).
COVER+RDRIFT host. Exclusive GPU leases on {0,2,4,7}. TTFT after ACC for
arms that finish all 6 metrics.

Judgment tasks: gqa, pope, mme, mmbench_en_dev, scienceqa_img, textvqa_val_ocr.
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
CFG_DIR = PKG / "configs" / "arms_136_wave1"
R3 = REPO / "experiments/elva_r3"
PY = "/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python"
PYBENCH = str(REPO / "experiments/e014_latency_fair_256/bin/python")
BENCH = REPO / "experiments/e014_latency_fair_256/scripts/benchmark_llava_e2e_anygpu.py"
PANEL = REPO / "experiments/e014_latency_fair_256/panel/speed_256.jsonl"
MODEL = REPO / "experiments/e014_latency_fair_256/models_llava"
OUT_ROOT = REPO / "logs/e014_aee/full/bela_136_wave1_full6"
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

# Progressive Wave-1 (paper+B1/B2 informed). Already-measured A2 172/120/20 skipped.
# Format: (arm_id, k0, kmid, kdeep, W)
WAVE1 = (
    # W2048 around A2 / deep recovery / STAR-like entry
    ("S_2048_172_105_26", 172, 105, 26, 2048),
    ("S_2048_172_125_18", 172, 125, 18, 2048),  # was A25; final-K allowlist fixed
    ("S_2048_172_135_14", 172, 135, 14, 2048),
    ("S_2048_172_95_30", 172, 95, 30, 2048),
    ("S_2048_160_126_20", 160, 126, 20, 2048),
    ("S_2048_160_96_32", 160, 96, 32, 2048),
    ("S_2048_128_107_34", 128, 107, 34, 2048),
    ("S_2048_128_117_30", 128, 117, 30, 2048),
    # W1024 COVER@96 grid (skip 96/50/12 already measured as W1024_PROBE)
    ("S_1024_96_25_22", 96, 25, 22, 1024),
    ("S_1024_96_30_20", 96, 30, 20, 1024),
    ("S_1024_96_35_18", 96, 35, 18, 1024),
    ("S_1024_96_40_16", 96, 40, 16, 1024),
    ("S_1024_96_45_14", 96, 45, 14, 1024),
    ("S_1024_96_55_10", 96, 55, 10, 1024),
    ("S_1024_96_60_8", 96, 60, 8, 1024),
)


def write_cfg(arm_id: str, k0: int, kmid: int, kdeep: int, budget: int) -> Path:
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    path = CFG_DIR / f"{arm_id}.json"
    assert k0 * 4 + kmid * 8 + kdeep * 20 == budget, (arm_id, budget)
    cfg = {
        "id": arm_id,
        "phase": "cvpr2027_136_wave1",
        "method": {
            "original_tokens": 576,
            "depth": 32,
            "budget": budget,
            "schedule": {"counts": [k0, kmid, kdeep], "lengths": [4, 8, 20]},
            "stage1": {
                "guide": "dual",
                "cover_family": "legacy_native",
                "cover": True,
                "beta": 0.5,
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


def run_acc() -> list[str]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for arm_id, k0, kmid, kdeep, budget in WAVE1:
        cfg = write_cfg(arm_id, k0, kmid, kdeep, budget)
        root = OUT_ROOT / arm_id
        root.mkdir(parents=True, exist_ok=True)
        for task in TASKS6:
            if (root / task / "metrics.json").exists():
                print(f"HAVE {arm_id}/{task}", flush=True)
                continue
            jobs.append((cfg, arm_id, task, root / task))
    print(f"Wave1 ACC pending {len(jobs)}", flush=True)
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
                owner = f"acc:{jobs[0][1]}:{jobs[0][2]}"
                leased = try_acquire(gpu, owner, allowed=PROJECT_GPUS)
                if leased is not None:
                    break
            if leased is None:
                break
            cfg_path, arm_id, task, out = jobs.pop(0)
            out.mkdir(parents=True, exist_ok=True)
            log = PKG / "logs" / f"acc_{arm_id}_{task}.log"
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
            running.append((proc, leased, f"{arm_id}:{task}"))
            print(
                f"START gpu={leased.gpu} {arm_id} {task} pid={proc.pid} (locked)",
                flush=True,
            )
        time.sleep(20)
    print("Wave1 ACC drained", flush=True)
    complete = []
    for arm_id, *_ in WAVE1:
        if all((OUT_ROOT / arm_id / t / "metrics.json").exists() for t in TASKS6):
            complete.append(arm_id)
    return complete


def run_ttft(arm_ids: list[str]) -> None:
    LAT_ROOT.mkdir(parents=True, exist_ok=True)
    for arm_id in arm_ids:
        cfg_path = CFG_DIR / f"{arm_id}.json"
        out = LAT_ROOT / arm_id
        if (out / "latency_summary.json").exists():
            print(f"SKIP TTFT {arm_id}", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        log = PKG / "logs" / f"ttft_{arm_id}.log"
        lease = acquire_one(f"ttft:{arm_id}", allowed=PROJECT_GPUS)
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
            print(f"TTFT START gpu={lease.gpu} {arm_id} (locked)", flush=True)
            with log.open("w") as handle:
                rc = subprocess.call(
                    cmd, env=child_env, stdout=handle, stderr=subprocess.STDOUT
                )
            if rc != 0 or not (out / "latency_summary.json").exists():
                print(f"TTFT FAIL {arm_id} rc={rc}", flush=True)
                continue
            s = json.loads((out / "latency_summary.json").read_text())
            print(
                f"TTFT DONE {arm_id} ttft={s['ttft_wall_ms']['mean']:.1f} "
                f"fixed32={s['fixed32_wall_ms']['mean']:.1f}",
                flush=True,
            )
        finally:
            release(lease)


def main() -> None:
    # smoke one overlay
    cfg = write_cfg("S_2048_172_125_18", 172, 125, 18, 2048)
    env_for(cfg, CFG_DIR / "_smoke")
    print(
        "smoke FINAL_K",
        os.environ.get("ELVA_R3_FINAL_K"),
        "EVENTS",
        os.environ.get("E018_EVENTS"),
        flush=True,
    )
    done = run_acc()
    print(f"complete arms for TTFT: {done}", flush=True)
    run_ttft(done)
    print("Wave1 ALL DONE", flush=True)


if __name__ == "__main__":
    main()
