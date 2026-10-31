#!/usr/bin/env python3
"""W4096 equal-budget schedule selection: A/B/C with QA B8 + first RDRIFT.

Handoff: compare 256→74→36 / 256→80→32 / 256→68→40 on full GQA+POPE+MME.
Schedule A reuses finished S1_QANCHOR_W4096 metrics when present.
GPUs: VITCOP_DECISION_GPUS default 0,4,6,7.
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
from gpu_exclusive import release, try_acquire  # noqa: E402
from launch_qacover_pilot import PY, R3, PKG as QA_PKG, env_for, write_run_meta  # noqa: E402

DEC = REPO / "experiments/bela_stage1_decision_package"
OUT_ROOT = REPO / "logs/e014_aee/full/bela_w4096_sched_abc_20261031"
HIST_A = REPO / "logs/e014_aee/full/bela_w4096_77p8_20261031/S1_QANCHOR_W4096"
LOG_DIR = DEC / "logs"
CFG_DIR = DEC / "configs"

_env_gpus = os.environ.get("VITCOP_DECISION_GPUS", "").strip()
PROJECT_GPUS = (
    tuple(int(x) for x in _env_gpus.split(",") if x.strip() != "")
    if _env_gpus
    else (0, 4, 6, 7)
)

GATE = ("gqa", "pope", "mme")
# arm_id, cfg, expected events, hist root or None
ARMS = (
    ("S1_W4096_A", CFG_DIR / "S1_QANCHOR_W4096.json", "11:74,19:36", HIST_A),
    ("S1_W4096_B", CFG_DIR / "S1_QANCHOR_W4096_B.json", "11:80,19:32", None),
    ("S1_W4096_C", CFG_DIR / "S1_QANCHOR_W4096_C.json", "11:68,19:40", None),
)


def _have(write: Path, hist: Path | None, task: str) -> bool:
    if (write / task / "metrics.json").exists():
        return True
    if hist is not None and (hist / task / "metrics.json").exists():
        return True
    return False


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
    for arm_id, cfg, _ev, hist in ARMS:
        root = OUT_ROOT / arm_id
        for task in GATE:
            key = f"{arm_id}:{task}"
            if key in running_names:
                continue
            if _have(root, hist, task):
                src = "hist" if hist and (hist / task / "metrics.json").exists() and not (root / task / "metrics.json").exists() else "write"
                print(f"HAVE {arm_id}/{task} ({src})", flush=True)
                continue
            out = root / task
            if str(out) in live:
                print(f"LIVE {arm_id}/{task}", flush=True)
                continue
            jobs.append((cfg, arm_id, task, out))
    return jobs


def _child_env(cfg_path: Path, out: Path, expect_events: str) -> dict:
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
    assert child.get("ELVA_R3_ENTRY_K") == "256", child.get("ELVA_R3_ENTRY_K")
    assert child.get("E018_EVENTS") == expect_events, (child.get("E018_EVENTS"), expect_events)
    assert child.get("E020_ENABLE") == "True"
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


def _write_summary() -> None:
    dense = {"gqa": 61.9, "pope": 85.8, "mme": 1852.0}
    rows = []
    for arm_id, cfg, _ev, hist in ARMS:
        write = OUT_ROOT / arm_id
        vals = {}
        ok = True
        for task in GATE:
            p = write / task / "metrics.json"
            if not p.exists() and hist is not None:
                p = hist / task / "metrics.json"
            if not p.exists():
                ok = False
                break
            m = json.loads(p.read_text())
            if task == "gqa":
                v = m.get("exact_match")
                if v is not None and v <= 1.5:
                    v *= 100
            elif task == "pope":
                v = m.get("pope_f1_score")
                if v is not None and v <= 1.5:
                    v *= 100
            else:
                v = float(m.get("mme_perception_score") or 0) + float(m.get("mme_cognition_score") or 0)
            vals[task] = float(v)
        if not ok:
            continue
        rel = {t: 100 * vals[t] / dense[t] for t in GATE}
        mean3 = sum(rel.values()) / 3
        counts = json.loads(cfg.read_text())["method"]["schedule"]["counts"]
        rows.append((arm_id, counts, vals, rel, mean3))
    lines = [
        "# W4096 schedule A/B/C (S1 RDRIFT, QA B8) — GQA+POPE+MME",
        "",
        "| Arm | schedule | GQA | POPE F1 | MME | mean3 rel% |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm_id, counts, vals, rel, mean3 in sorted(rows, key=lambda r: -r[4]):
        lines.append(
            f"| {arm_id} | {counts[0]}→{counts[1]}→{counts[2]} | "
            f"{vals['gqa']:.2f} ({rel['gqa']:.1f}%) | "
            f"{vals['pope']:.2f} ({rel['pope']:.1f}%) | "
            f"{vals['mme']:.1f} ({rel['mme']:.1f}%) | {mean3:.2f} |"
        )
    winner = rows[0][0] if rows else ""
    if rows:
        rows_sorted = sorted(rows, key=lambda r: -r[4])
        winner = rows_sorted[0][0]
        lines += ["", f"**Winner (mean3 rel):** `{winner}`"]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "SCHEDULE_ABC_SUMMARY.md").write_text("\n".join(lines) + "\n")
    (OUT_ROOT / "SCHEDULE_ABC_WINNER.txt").write_text(winner + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(
        f"W4096_SCHED_ABC gate={GATE} arms={[a for a, *_ in ARMS]} gpus={PROJECT_GPUS}",
        flush=True,
    )
    expect = {a: ev for a, _c, ev, _h in ARMS}
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
        jobs.sort(key=lambda j: (GATE.index(j[2]), j[1]))
        print(f"PENDING {len(jobs)} running={sorted(running_names)}", flush=True)
        if not jobs and not running:
            break
        free = free_project_gpus()
        while jobs:
            leased = None
            for gpu in free:
                if any(g.gpu == gpu for _, g, _ in running):  # type: ignore[attr-defined]
                    continue
                leased = try_acquire(gpu, f"w4096abc:{jobs[0][1]}:{jobs[0][2]}", allowed=PROJECT_GPUS)
                if leased is not None:
                    break
            if leased is None:
                break
            cfg_path, arm_id, task, out = jobs.pop(0)
            out.mkdir(parents=True, exist_ok=True)
            log = LOG_DIR / f"acc_{arm_id}_{task}.log"
            child_env = _child_env(cfg_path, out, expect[arm_id])
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
    _write_summary()
    print("W4096_SCHED_ABC_DRAINED", flush=True)


if __name__ == "__main__":
    main()
