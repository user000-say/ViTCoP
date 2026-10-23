"""T1 A0/A2/A3 Fixed32+TTFT on fair-256 panel (same bench as H1).

Waits for an exclusive free card from {0,2,4,7}; runs arms sequentially
(one GPU at a time) to avoid contention contamination.
A0 included as same-session anchor (172/90/32); A2=20 final, A3=16 final.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
PKG = REPO / "experiments/bela_cvpr2027_rules_20261022"
R3 = REPO / "experiments/elva_r3"
PYBENCH = str(REPO / "experiments/e014_latency_fair_256/bin/python")
BENCH = REPO / "experiments/e014_latency_fair_256/scripts/benchmark_llava_e2e_anygpu.py"
PANEL = REPO / "experiments/e014_latency_fair_256/panel/speed_256.jsonl"
MODEL = REPO / "experiments/e014_latency_fair_256/models_llava"
OUT_ROOT = PKG / "latency"
GPUS = (0, 2, 4, 7)
BASE_ARM = "E018_COVER_TWO32"

ARMS = (
    "A0_COVER_RDRIFT_W2048",
    "A2_COVER_RDRIFT_W2048",
    "A3_COVER_RDRIFT_W2048",
)


def env_for(cfg_path: Path, audit_dir: Path) -> dict:
    sys.path.insert(0, str(REPO))
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


def free_gpus(max_used_mib: int = 4000) -> list[int]:
    """Return GPUs from the project set with little memory in use."""
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    free = []
    for line in out.strip().splitlines():
        idx_s, used_s = [x.strip() for x in line.split(",")]
        idx, used = int(idx_s), int(float(used_s))
        if idx in GPUS and used < max_used_mib:
            free.append(idx)
    return free


def run_one(arm: str, gpu: int) -> None:
    cfg_path = PKG / "configs" / f"{arm}.json"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    out = OUT_ROOT / arm
    if (out / "latency_summary.json").exists():
        print(f"SKIP {arm}", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    log = PKG / "logs" / f"ttft_{arm}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env.update(env_for(cfg_path, out))
    child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    child_env["PYTHONPATH"] = (
        f"{REPO}:{REPO / 'experiments'}:{R3}:{REPO / 'LLaVA'}:{REPO / 'e020_lib'}"
    )
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["HF_HUB_OFFLINE"] = "1"
    child_env["TRANSFORMERS_OFFLINE"] = "1"
    child_env["HF_DATASETS_OFFLINE"] = "1"
    # do not let child rewrite CUDA_VISIBLE_DEVICES
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
    print(f"TTFT START gpu={gpu} {arm}", flush=True)
    with log.open("w") as handle:
        rc = subprocess.call(cmd, env=child_env, stdout=handle, stderr=subprocess.STDOUT)
    if rc != 0 or not (out / "latency_summary.json").exists():
        raise SystemExit(f"TTFT FAIL {arm} rc={rc}")
    summary = json.loads((out / "latency_summary.json").read_text())
    print(
        f"TTFT DONE {arm} "
        f"ttft={summary['ttft_wall_ms']['mean']:.1f} "
        f"fixed32={summary['fixed32_wall_ms']['mean']:.1f}",
        flush=True,
    )


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pending = [
        a
        for a in ARMS
        if not (OUT_ROOT / a / "latency_summary.json").exists()
    ]
    print(f"T1 TTFT pending {pending}", flush=True)
    for arm in pending:
        while True:
            free = free_gpus()
            if free:
                gpu = free[0]
                print(f"exclusive gpu={gpu} for {arm} (free={free})", flush=True)
                run_one(arm, gpu)
                break
            print("waiting for free GPU in {0,2,4,7} ...", flush=True)
            time.sleep(60)
    print("T1 A0/A2/A3 TTFT queue drained", flush=True)
    # brief table
    for arm in ARMS:
        p = OUT_ROOT / arm / "latency_summary.json"
        if not p.exists():
            print(f"{arm}: MISSING", flush=True)
            continue
        s = json.loads(p.read_text())
        print(
            f"{arm}: TTFT={s['ttft_wall_ms']['mean']:.2f} "
            f"Fixed32={s['fixed32_wall_ms']['mean']:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
