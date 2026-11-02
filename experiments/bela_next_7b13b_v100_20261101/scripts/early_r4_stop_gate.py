#!/usr/bin/env python3
"""Wait for A0 R4 four tasks; if mean relative ACC < 92% vs NeXT-7B Dense → STOP all.

Relative = 100 * abs / Dense_abs (next7_dense6_absolute.json).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

PKG = Path("/home/xiangyue/ViTCoP/experiments/bela_next_7b13b_v100_20261101")
OUT = PKG / "logs" / "acc_a0_a4_94pct_v4"
ARM = "NEXT7_T160_A0_PAPER"
TASKS = ("gqa", "pope", "mme", "mmbench_en_dev")
THRESH = float(os.environ.get("BELA_EARLY_R4_MIN", "92.0"))
DENSE = json.loads(
    Path(
        "/home/xiangyue/ViTCoP/experiments/bela_v100_4model_qacover_20261028/results/next7_dense6_absolute.json"
    ).read_text()
)["absolute"]
DENSE_MAP = {
    "gqa": float(DENSE["gqa"]),
    "pope": float(DENSE["pope_f1"]),
    "mme": float(DENSE["mme"]),
    "mmbench_en_dev": float(DENSE["mmbench_en_dev"]),
}
GATE = OUT / "EARLY_R4_STOP_GATE.json"
LOG = PKG / "logs" / "queue_v3" / "early_r4_gate.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def extract_abs(task: str, results: dict) -> float | None:
    root = results.get("results", {})
    block = root.get(task, root)
    if not isinstance(block, dict):
        return None
    if task == "gqa":
        v = block.get("exact_match,none")
        return None if v is None else 100.0 * float(v)
    if task == "pope":
        for k in ("pope_f1_score,none", "f1_score,none", "f1,none"):
            if k in block:
                return 100.0 * float(block[k])
        return None
    if task == "mme":
        perc = block.get("mme_perception_score,none")
        cog = block.get("mme_cognition_score,none")
        if perc is not None and cog is not None:
            return float(perc) + float(cog)
        if "mme_totale_score,none" in block:
            return float(block["mme_totale_score,none"])
        return None
    if task == "mmbench_en_dev":
        for k in (
            "gpt_eval_score,none",
            "gpt_eval_acc,none",
            "mmbench_acc,none",
            "accuracy,none",
        ):
            if k in block:
                return float(block[k])
        return None
    return None


def find_results(task: str) -> Path | None:
    root = OUT / ARM / task
    if not root.exists():
        return None
    hits = sorted(root.glob("**/*_results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def stop_all() -> None:
    cmds = [
        ["pkill", "-f", "scripts/run_next7_a0_a4_six.sh"],
        ["pkill", "-f", "scripts/run_next13_a0_a4_six.sh"],
        ["pkill", "-f", "scripts/chain_next13_after_next7"],
        ["pkill", "-f", "scripts/requeue_failed"],
        ["pkill", "-f", "lmms_bela_next_entry.py"],
    ]
    for c in cmds:
        subprocess.run(c, check=False, capture_output=True)


def resume_full_queue() -> None:
    """Restart full 7B queue (skips tasks that already have results) + 13B chain."""
    subprocess.Popen(
        ["bash", str(PKG / "scripts" / "run_next7_a0_a4_six.sh")],
        stdout=open(PKG / "logs" / "queue_v3" / "nohup.out", "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    subprocess.Popen(
        ["bash", str(PKG / "scripts" / "chain_next13_after_next7_v3.sh")],
        stdout=open(PKG / "logs" / "queue_next13_v3" / "nohup_chain.out", "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def main() -> int:
    log(f"EARLY_R4_GATE_BEGIN arm={ARM} thresh={THRESH} dense={DENSE_MAP}")
    abs_scores: dict[str, float] = {}
    rel_scores: dict[str, float] = {}
    while len(abs_scores) < 4:
        for t in TASKS:
            if t in abs_scores:
                continue
            p = find_results(t)
            if p is None:
                continue
            data = json.loads(p.read_text())
            # mmbench may finish without usable score if OOM — require extractable abs
            a = extract_abs(t, data)
            if a is None:
                log(f"WAIT {t} results present but metric missing path={p}")
                continue
            abs_scores[t] = a
            rel_scores[t] = 100.0 * a / DENSE_MAP[t]
            log(f"GOT {t} abs={a:.4f} rel={rel_scores[t]:.2f} path={p}")
        missing = [t for t in TASKS if t not in abs_scores]
        if missing:
            # still running?
            alive = subprocess.run(
                ["pgrep", "-af", "lmms_bela_next_entry"], capture_output=True, text=True
            ).stdout
            log(f"waiting missing={missing} alive_procs={bool(alive.strip())}")
            if not alive.strip() and missing:
                # workers died without results
                payload = {
                    "verdict": "STOP",
                    "reason": "workers_exited_before_r4_complete",
                    "missing": missing,
                    "have": abs_scores,
                    "thresh": THRESH,
                }
                GATE.write_text(json.dumps(payload, indent=2))
                log(f"STOP incomplete {payload}")
                stop_all()
                return 2
            time.sleep(60)
            continue
        break

    r4 = sum(rel_scores[t] for t in TASKS) / 4.0
    payload = {
        "verdict": "GO" if r4 >= THRESH else "STOP",
        "arm": ARM,
        "thresh": THRESH,
        "r4_mean_rel": r4,
        "abs": abs_scores,
        "rel": rel_scores,
        "dense_ref": DENSE_MAP,
        "note": "Relative vs NeXT-7B Dense (next7_dense6_absolute.json), not LLaVA-1.5.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    GATE.write_text(json.dumps(payload, indent=2))
    log(f"R4_MEAN_REL={r4:.2f} verdict={payload['verdict']}")

    if r4 < THRESH:
        log("STOP: below 92% — killing all NeXT ACC workers")
        stop_all()
        return 1

    log("GO: >=92% — A0 R4 passed; await user to expand arms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
