#!/usr/bin/env python3
"""V100 Stage1 decision matrix from BELA_CVPR2027_STAGE1_DECISION_EXPERIMENT_KIT.

Reuses production elva_r3 + bela_query_anchor host (vitcopreback).

B-scan (priority): B4..B24 step 2, six-task ACC.
Early drop: after GQA+POPE, discard a B if GQA < 55 or POPE F1 < 83.8
(no MME/MMBench/SciQA/TextVQA for that B).

Hist B8 six-task under bela_stage1_queryanchor_20261027/S1_QANCHOR_L2 is reused.
STRUCT/FINISH only if VITCOP_DECISION_INCLUDE_STRUCT=1.
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
HIST_ROOT = REPO / "logs/e014_aee/full/bela_stage1_queryanchor_20261027"
LOG_DIR = DEC / "logs"
CFG_DIR = DEC / "configs"

_env_gpus = os.environ.get("VITCOP_DECISION_GPUS", "").strip()
if _env_gpus:
    PROJECT_GPUS = tuple(int(x) for x in _env_gpus.split(",") if x.strip() != "")
else:
    PROJECT_GPUS = _DEFAULT_GPUS

INCLUDE_STRUCT = os.environ.get("VITCOP_DECISION_INCLUDE_STRUCT", "0") == "1"

# B4..B24 every 2 (user denser scan). B8 hist reused.
B_SCAN = list(range(4, 26, 2))
GATE_TASKS = ("gqa", "pope")
EXTRA_TASKS = ("mme", "mmbench_en_dev", "scienceqa_img", "textvqa_val_ocr")
SIX_TASKS = GATE_TASKS + EXTRA_TASKS

# Absolute floors (GQA exact_match %; POPE F1 %). Below → drop remaining extras.
GQA_MIN = float(os.environ.get("VITCOP_BSCAN_GQA_MIN", "55.0"))
POPE_F1_MIN = float(os.environ.get("VITCOP_BSCAN_POPE_F1_MIN", "83.8"))

STRUCT = [
    "S2_SALIENT_QA_B8",
    "S3_JOINT",
    "S4_JOINT_CAP8",
    "S5_QUERY_TOPK",
    "S8_NO_NOVELTY",
    "S8_NO_COVERAGE",
    "S8_QUERY_ONLY_GAIN",
]
FINISH = [
    ("L2_NATIVE", QA_PKG / "configs/L2_NATIVE.json", ("mme", "mmbench_en_dev", "scienceqa_img", "textvqa_val_ocr")),
    ("S1_MATCHED_RANDOM", QA_PKG / "configs/S1_MATCHED_RANDOM.json", ("mme", "mmbench_en_dev", "scienceqa_img", "textvqa_val_ocr")),
    ("S1_QUERY_SHUFFLE", QA_PKG / "configs/S1_QUERY_SHUFFLE.json", ("pope", "mme", "mmbench_en_dev", "scienceqa_img", "textvqa_val_ocr")),
]


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


def _arm_roots(b: int) -> tuple[str, Path, Path | None]:
    """Return (arm_id, write_root, hist_root_or_None). B8 reads hist S1_QANCHOR_L2."""
    arm = f"S1_QANCHOR_L2_B{b}"
    write = OUT_ROOT / arm
    if b == 8:
        return arm, write, HIST_ROOT / "S1_QANCHOR_L2"
    return arm, write, None


def _metric_path(write: Path, hist: Path | None, task: str) -> Path | None:
    for root in (write, hist) if hist is not None else (write,):
        if root is None:
            continue
        p = root / task / "metrics.json"
        if p.exists():
            return p
    return None


def _gqa_pct(metrics_path: Path) -> float | None:
    m = json.loads(metrics_path.read_text())
    v = m.get("exact_match")
    rj = m.get("results_json")
    if rj and Path(rj).exists():
        d = json.loads(Path(rj).read_text())
        g = (d.get("results") or {}).get("gqa") or {}
        v = g.get("exact_match,none", g.get("exact_match", v))
    if v is None:
        return None
    v = float(v)
    return v * 100.0 if v <= 1.5 else v


def _pope_f1(metrics_path: Path) -> float | None:
    m = json.loads(metrics_path.read_text())
    rj = m.get("results_json")
    f1 = None
    if rj and Path(rj).exists():
        d = json.loads(Path(rj).read_text())
        p = (d.get("results") or {}).get("pope") or {}
        for k, val in p.items():
            if "f1" in k.lower() and isinstance(val, (int, float)):
                f1 = float(val)
                break
    if f1 is None:
        return None
    return f1 * 100.0 if f1 <= 1.5 else f1


def _gate_status(write: Path, hist: Path | None) -> tuple[str, float | None, float | None]:
    """Return (pending|pass|drop, gqa_pct, pope_f1)."""
    gp = _metric_path(write, hist, "gqa")
    pp = _metric_path(write, hist, "pope")
    if gp is None or pp is None:
        return "pending", None, None
    gqa = _gqa_pct(gp)
    f1 = _pope_f1(pp)
    if gqa is None or f1 is None:
        return "pending", gqa, f1
    if gqa < GQA_MIN or f1 < POPE_F1_MIN:
        return "drop", gqa, f1
    return "pass", gqa, f1


def _jobs(running_names: set[str] | None = None) -> list[tuple[Path, str, str, Path]]:
    running_names = running_names or set()
    jobs: list[tuple[Path, str, str, Path]] = []
    live = _live_outputs()

    def maybe_add(cfg: Path, arm: str, task: str, out: Path) -> None:
        key = f"{arm}:{task}"
        if key in running_names:
            return
        write_root = out.parent
        arm_b = None
        if arm.startswith("S1_QANCHOR_L2_B") and arm[len("S1_QANCHOR_L2_B") :].isdigit():
            arm_b = int(arm[len("S1_QANCHOR_L2_B") :])
        hist = HIST_ROOT / "S1_QANCHOR_L2" if arm_b == 8 else None
        if _metric_path(write_root, hist, task) is not None:
            print(f"HAVE {arm}/{task}", flush=True)
            return
        if str(out) in live:
            print(f"LIVE {arm}/{task}", flush=True)
            return
        jobs.append((cfg, arm, task, out))

    for b in B_SCAN:
        arm, write, hist = _arm_roots(b)
        cfg = CFG_DIR / f"{arm}.json"
        if not cfg.exists():
            print(f"MISSING_CFG {cfg}", flush=True)
            continue
        # Always finish gate tasks first (order gqa then pope).
        for task in GATE_TASKS:
            maybe_add(cfg, arm, task, write / task)
        status, gqa, f1 = _gate_status(write, hist)
        if status == "drop":
            print(
                f"DROP {arm} gqa={gqa:.2f}<{GQA_MIN} or pope_f1={f1:.2f}<{POPE_F1_MIN} — skip extras",
                flush=True,
            )
            continue
        if status == "pending":
            print(f"GATE_WAIT {arm} (need GQA+POPE before extras)", flush=True)
            continue
        print(f"GATE_PASS {arm} gqa={gqa:.2f} pope_f1={f1:.2f} — queue extras", flush=True)
        for task in EXTRA_TASKS:
            maybe_add(cfg, arm, task, write / task)

    if INCLUDE_STRUCT:
        for arm in STRUCT:
            cfg = CFG_DIR / f"{arm}.json"
            root = OUT_ROOT / arm
            for task in GATE_TASKS:
                maybe_add(cfg, arm, task, root / task)
        for arm, cfg, tasks in FINISH:
            hist = HIST_ROOT / arm
            root = OUT_ROOT / arm
            for task in tasks:
                if _have(hist, task) or _have(root, task):
                    print(f"HAVE {arm}/{task}", flush=True)
                    continue
                out = root / task
                key = f"{arm}:{task}"
                if key in running_names or str(out) in live:
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
    print(
        f"DECISION_MATRIX B_SCAN={B_SCAN} gate GQA>={GQA_MIN} POPE_F1>={POPE_F1_MIN} "
        f"gpus={PROJECT_GPUS} struct={INCLUDE_STRUCT}",
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
        # Prefer unfinished gate tasks (gqa/pope) before extras / later B extras.
        jobs.sort(key=lambda j: (0 if j[2] in GATE_TASKS else 1, B_SCAN.index(int(j[1].split("_B")[-1])) if "_B" in j[1] and j[1].split("_B")[-1].isdigit() else 99, j[2]))
        print(f"PENDING {len(jobs)} running={sorted(running_names)}", flush=True)
        if not jobs and not running:
            break
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
