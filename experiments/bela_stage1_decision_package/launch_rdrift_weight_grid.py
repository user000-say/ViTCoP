#!/usr/bin/env python3
"""RDRIFT (w,1-w) × {77.8,88.9,94.4}% ACC after ABC winner.

Queue: R4 → R6 remainder → R10 remainder. GPUs: VITCOP_DECISION_GPUS default 0,4,6,7.
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
CFG_DIR = DEC / "configs" / "rdrift_weight_grid"
OUT_ROOT = REPO / "logs/e014_aee/full/bela_rdrift_weight_grid_20261031"
ABC_ROOT = REPO / "logs/e014_aee/full/bela_w4096_sched_abc_20261031"
HIST_A = REPO / "logs/e014_aee/full/bela_w4096_77p8_20261031/S1_QANCHOR_W4096"
HIST_L2 = REPO / "logs/e014_aee/full/bela_stage1_queryanchor_20261027/S1_QANCHOR_L2"
LOG_DIR = DEC / "logs"

_env_gpus = os.environ.get("VITCOP_DECISION_GPUS", "").strip()
PROJECT_GPUS = (
    tuple(int(x) for x in _env_gpus.split(",") if x.strip() != "")
    if _env_gpus
    else (0, 4, 6, 7)
)

R4 = ("gqa", "pope", "mme", "mmbench_en_dev")
R6_EXTRA = ("scienceqa_img", "textvqa_val_ocr")
R10_EXTRA = ("mmmu_val", "ai2d", "vizwiz_vqa_val", "mmvet")
# Priority bands for queue sort.
BAND = {t: 0 for t in R4}
BAND.update({t: 1 for t in R6_EXTRA})
BAND.update({t: 2 for t in R10_EXTRA})
ALL_TASKS = R4 + R6_EXTRA + R10_EXTRA


def _letter() -> str:
    p = ABC_ROOT / "SCHEDULE_ABC_WINNER.txt"
    if not p.exists():
        raise SystemExit("missing SCHEDULE_ABC_WINNER.txt — run schedule ABC first")
    raw = p.read_text().strip()
    # Winner ids are S1_W4096_A / B / C
    if raw.endswith("_A") or raw == "A":
        return "A"
    if raw.endswith("_B") or raw == "B":
        return "B"
    if raw.endswith("_C") or raw == "C":
        return "C"
    raise SystemExit(f"unrecognized ABC winner: {raw!r}")


def _hist_for(arm_id: str, task: str) -> Path | None:
    """Reuse prior metrics when schedule+rate+weight match a finished run."""
    # W60@77.8 production default
    if arm_id.endswith("_R778_W60"):
        if arm_id.startswith("RD_SA_"):
            p = HIST_A / task / "metrics.json"
            if p.exists():
                return p
            p2 = ABC_ROOT / "S1_W4096_A" / task / "metrics.json"
            if p2.exists():
                return p2
        if arm_id.startswith("RD_SB_") and task in ("gqa", "pope", "mme"):
            p = ABC_ROOT / "S1_W4096_B" / task / "metrics.json"
            if p.exists():
                return p
        if arm_id.startswith("RD_SC_") and task in ("gqa", "pope", "mme"):
            p = ABC_ROOT / "S1_W4096_C" / task / "metrics.json"
            if p.exists():
                return p
    # W60@94.4 = L2 64→17→10 default RDRIFT (same counts as schedule A/C @944)
    if arm_id.endswith("_R944_W60") and (
        arm_id.startswith("RD_SA_") or arm_id.startswith("RD_SC_")
    ):
        p = HIST_L2 / task / "metrics.json"
        if p.exists():
            return p
    return None


def _have(write: Path, arm_id: str, task: str) -> bool:
    if (write / task / "metrics.json").exists():
        return True
    return _hist_for(arm_id, task) is not None


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


def _arms(letter: str) -> list[tuple[str, Path]]:
    man = json.loads((CFG_DIR / "MANIFEST.json").read_text())
    want = set(man.get("arms_to_run") or [])
    arms = []
    for path in sorted(CFG_DIR.glob(f"RD_S{letter}_*.json")):
        name = path.name
        if want and name not in want:
            continue
        # never queue W60 or deferred 88.9%
        if path.stem.endswith("_W60") or "_R889_" in path.stem:
            continue
        arms.append((path.stem, path))
    # 3 @778 + 3 @944 = 6
    if len(arms) != 6:
        raise SystemExit(
            f"expected 6 run arms for schedule {letter} "
            f"(W20/W40/W80 × 77.8/94.4), got {len(arms)} in {CFG_DIR}"
        )
    return arms


def _jobs(letter: str, running_names: set[str]):
    live = _live_outputs()
    jobs = []
    for arm_id, cfg in _arms(letter):
        root = OUT_ROOT / arm_id
        for task in ALL_TASKS:
            key = f"{arm_id}:{task}"
            if key in running_names:
                continue
            if _have(root, arm_id, task):
                src = "hist" if _hist_for(arm_id, task) and not (root / task / "metrics.json").exists() else "write"
                print(f"HAVE {arm_id}/{task} ({src})", flush=True)
                continue
            out = root / task
            if str(out) in live:
                print(f"LIVE {arm_id}/{task}", flush=True)
                continue
            jobs.append((cfg, arm_id, task, out, BAND[task]))
    jobs.sort(key=lambda j: (j[4], j[1], ALL_TASKS.index(j[2])))
    return [(c, a, t, o) for c, a, t, o, _ in jobs]


def _child_env(cfg_path: Path, out: Path) -> dict:
    child = os.environ.copy()
    child.update(env_for(cfg_path, out))
    cfg = json.loads(cfg_path.read_text())
    child["BELA_STAGE1_INIT"] = cfg.get("method", {}).get("stage1_init", "cover")
    child["PYTHONPATH"] = (
        f"{REPO}:{REPO / 'experiments'}:{R3}:{QA_PKG}:{DEC}:{REPO / 'LLaVA'}:{REPO / 'e020_lib'}"
    )
    child["PYTHONUNBUFFERED"] = "1"
    child["HF_HUB_OFFLINE"] = "1"
    child["HF_DATASETS_OFFLINE"] = "1"
    child["TRANSFORMERS_OFFLINE"] = "1"
    w = float(cfg["method"]["stage2"]["rdrift_current_weight"])
    assert child.get("E020_ENABLE") == "True"
    assert child.get("E020_KIND") == "rank_drift"
    assert abs(float(child.get("E020_CURRENT_WEIGHT", "-1")) - w) < 1e-9, (
        child.get("E020_CURRENT_WEIGHT"),
        w,
    )
    assert child.get("ELVA_R3_ENTRY_K") == str(cfg["method"]["schedule"]["counts"][0])
    if out.name == "mmvet" and not child.get("OPENAI_API_KEY"):
        raise SystemExit("mmvet requires OPENAI_API_KEY")
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


def _write_summary(letter: str) -> None:
    dense = {
        "gqa": 61.9,
        "pope": 85.8,
        "mme": 1852.0,
        "mmbench_en_dev": 64.1,
        "scienceqa_img": 69.51,
        "textvqa_val_ocr": 58.20,
        "mmmu_val": 35.60,
        "ai2d": 55.18,
        "vizwiz_vqa_val": 50.68,
    }
    lines = [
        f"# RDRIFT weight grid (schedule {letter})",
        "",
        "score = w*rank_now + (1-w)*(rank_now-rank_old)",
        "",
    ]
    for rate, rate_lab in (("778", "77.8%"), ("944", "94.4%")):
        lines += [f"## {rate_lab}", "", "| arm | w | R4 mean rel% | R6 mean rel% | notes |", "|---|---:|---:|---:|---|"]
        for w in (0.2, 0.4, 0.6, 0.8):
            arm_id = f"RD_S{letter}_R{rate}_W{int(round(w * 100)):02d}"
            root = OUT_ROOT / arm_id
            rels = {}
            for task in ALL_TASKS:
                p = root / task / "metrics.json"
                if not p.exists():
                    hp = _hist_for(arm_id, task)
                    p = hp if hp else p
                if not p.exists():
                    continue
                m = json.loads(p.read_text())
                if task == "gqa":
                    v = m.get("exact_match")
                    if v is not None and v <= 1.5:
                        v = float(v) * 100
                elif task == "pope":
                    v = m.get("pope_f1_score")
                    if v is not None and v <= 1.5:
                        v = float(v) * 100
                elif task == "mme":
                    v = float(m.get("mme_perception_score") or 0) + float(m.get("mme_cognition_score") or 0)
                elif task == "mmbench_en_dev":
                    v = m.get("gpt_eval_score", m.get("exact_match"))
                    if v is not None and float(v) <= 1.5:
                        v = float(v) * 100
                elif task == "mmvet":
                    v = m.get("gpt_eval_score")
                    rels[task] = float(v) if v is not None else None
                    continue
                else:
                    key = {
                        "scienceqa_img": "exact_match",
                        "textvqa_val_ocr": "exact_match",
                        "mmmu_val": "mmmu_acc",
                        "ai2d": "exact_match",
                        "vizwiz_vqa_val": "exact_match",
                    }[task]
                    v = m.get(key)
                    if v is not None and float(v) <= 1.5:
                        v = float(v) * 100
                if v is not None and task in dense:
                    rels[task] = 100.0 * float(v) / dense[task]
            r4 = [rels[t] for t in R4 if t in rels]
            r6 = [rels[t] for t in (R4 + R6_EXTRA) if t in rels]
            r4m = sum(r4) / len(r4) if len(r4) == 4 else float("nan")
            r6m = sum(r6) / len(r6) if len(r6) == 6 else float("nan")
            miss = [t for t in ALL_TASKS if t not in rels]
            note = "complete" if not miss else f"missing {len(miss)}"
            lines.append(
                f"| {arm_id} | {w:g} | {r4m:.2f} | {r6m:.2f} | {note} |"
            )
        lines.append("")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "RDRIFT_WEIGHT_GRID_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    letter = os.environ.get("RDRIFT_GRID_SCHEDULE", "").strip().upper() or _letter()
    if not (CFG_DIR / "MANIFEST.json").exists():
        raise SystemExit(f"configs missing; run gen_rdrift_weight_configs.py --schedule {letter}")
    man = json.loads((CFG_DIR / "MANIFEST.json").read_text())
    if man.get("schedule_letter") != letter:
        raise SystemExit(
            f"MANIFEST schedule={man.get('schedule_letter')} != winner {letter}; regenerate configs"
        )
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(
        f"RDRIFT_WEIGHT_GRID schedule={letter} tasks={ALL_TASKS} gpus={PROJECT_GPUS}",
        flush=True,
    )
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
        jobs = _jobs(letter, running_names)
        print(f"PENDING {len(jobs)} running={sorted(running_names)}", flush=True)
        if not jobs and not running:
            break
        free = free_project_gpus()
        while jobs:
            leased = None
            for gpu in free:
                if any(getattr(g, "gpu", None) == gpu for _, g, _ in running):
                    continue
                leased = try_acquire(gpu, f"rdriftw:{jobs[0][1]}:{jobs[0][2]}", allowed=PROJECT_GPUS)
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
            # annotate weight in run_meta
            meta_p = out / "run_meta.json"
            meta = json.loads(meta_p.read_text())
            meta["rdrift_current_weight"] = child_env.get("E020_CURRENT_WEIGHT")
            meta_p.write_text(json.dumps(meta, indent=2) + "\n")
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
    _write_summary(letter)
    print("RDRIFT_WEIGHT_GRID_DRAINED", flush=True)


if __name__ == "__main__":
    main()
