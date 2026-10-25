"""FINAL handoff P1: retirement / late-first-cut + W1024 entry ACC boost.

Pre-specified schedules only (no test-set retuning). COVER+RDRIFT host.
GPUs {0,2,4,7} via gpu_exclusive. Full 6-task ACC; TTFT after complete arms.
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

PKG = REPO / "experiments/bela_cvpr27_final_handoff_20261025"
CFG_DIR = PKG / "configs" / "p1_arms"
R3 = REPO / "experiments/elva_r3"
PY = "/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python"
PYBENCH = str(REPO / "experiments/e014_latency_fair_256/bin/python")
BENCH = REPO / "experiments/e014_latency_fair_256/scripts/benchmark_llava_e2e_anygpu.py"
PANEL = REPO / "experiments/e014_latency_fair_256/panel/speed_256.jsonl"
MODEL = REPO / "experiments/e014_latency_fair_256/models_llava"
OUT_ROOT = REPO / "logs/e014_aee/full/bela_final_p1_20261025"
LAT_ROOT = PKG / "latency"
LOG_DIR = PKG / "logs"
BASE_ARM = "E018_COVER_TWO32"

TASKS6 = (
    "gqa",
    "pope",
    "mme",
    "mmbench_en_dev",
    "scienceqa_img",
    "textvqa_val_ocr",
)

# (arm_id, counts, lengths, W, note)
# counts/lengths must sum lengths==32 and W==sum(c*l)
P1_ARMS = (
    # W2048 retirement (handoff P1)
    ("E1_RET32_TO12", (172, 120, 32, 12), (4, 8, 8, 12), 2048, "keep K32 to L20 then 12"),
    ("E2_RET32_TO16", (172, 120, 32, 16), (4, 8, 5, 15), 2048, "shorter K32 survival"),
    ("E3_RET24_TO16", (172, 120, 24, 16), (4, 8, 10, 10), 2048, "moderate late survival"),
    ("E5_DELAY_128_12_20", (128, 37, 18), (12, 8, 12), 2048, "STAR-Pro T64 shape, BELA selector"),
    # W1024 late / mid first cut (handoff P1)
    ("L2_LATE_64_17_10", (64, 17, 10), (12, 8, 12), 1024, "STAR-Pro T32 shape"),
    ("L3_MID_80_24_12", (80, 24, 12), (8, 8, 16), 1024, "mid-depth first cut"),
    # ACC-boost: higher Stage1 entry @W1024 (Wave1 only swept K96)
    ("S_1024_128_24_16", (128, 24, 16), (4, 8, 20), 1024, "higher entry early-cut"),
    ("S_1024_160_28_8", (160, 28, 8), (4, 8, 20), 1024, "aggressive entry early-cut"),
)


def write_cfg(arm_id: str, counts: tuple, lengths: tuple, budget: int) -> Path:
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    path = CFG_DIR / f"{arm_id}.json"
    assert sum(lengths) == 32, (arm_id, lengths)
    assert sum(c * l for c, l in zip(counts, lengths)) == budget, (arm_id, budget)
    assert len(counts) == len(lengths)
    assert len(counts) - 1 <= 3, "E018 max 3 events"
    cfg = {
        "id": arm_id,
        "phase": "cvpr2027_final_p1",
        "method": {
            "original_tokens": 576,
            "depth": 32,
            "budget": budget,
            "schedule": {"counts": list(counts), "lengths": list(lengths)},
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
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for arm_id, counts, lengths, budget, _note in P1_ARMS:
        cfg = write_cfg(arm_id, counts, lengths, budget)
        root = OUT_ROOT / arm_id
        root.mkdir(parents=True, exist_ok=True)
        for task in TASKS6:
            if (root / task / "metrics.json").exists():
                print(f"HAVE {arm_id}/{task}", flush=True)
                continue
            jobs.append((cfg, arm_id, task, root / task))
    print(f"P1 ACC pending {len(jobs)}", flush=True)
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
            log = LOG_DIR / f"acc_{arm_id}_{task}.log"
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
    print("P1 ACC drained", flush=True)
    complete = []
    for arm_id, *_ in P1_ARMS:
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
        log = LOG_DIR / f"ttft_{arm_id}.log"
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


def smoke_overlay() -> None:
    for arm_id, counts, lengths, budget, _ in P1_ARMS[:2]:
        cfg = write_cfg(arm_id, counts, lengths, budget)
        env_for(cfg, CFG_DIR / f"_smoke_{arm_id}")
        print(
            f"overlay {arm_id} ENTRY={os.environ.get('ELVA_R3_ENTRY_K')} "
            f"FINAL={os.environ.get('ELVA_R3_FINAL_K')} "
            f"EVENTS={os.environ.get('E018_EVENTS')}",
            flush=True,
        )


def main() -> None:
    smoke_overlay()
    done = run_acc()
    print(f"complete arms for TTFT: {done}", flush=True)
    run_ttft(done)
    print("P1 FINAL ALL DONE", flush=True)


if __name__ == "__main__":
    main()
