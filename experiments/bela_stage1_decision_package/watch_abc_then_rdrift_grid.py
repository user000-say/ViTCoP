#!/usr/bin/env python3
"""Wait for W4096 schedule ABC drain, then generate configs + launch RDRIFT weight grid."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
DEC = REPO / "experiments/bela_stage1_decision_package"
ABC_ROOT = REPO / "logs/e014_aee/full/bela_w4096_sched_abc_20261031"
ABC_LOG = DEC / "logs" / "w4096_schedule_abc_queue.log"
PY = "/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python"
LOG = DEC / "logs" / "rdrift_weight_grid_queue.log"


def _winner_letter() -> str | None:
    p = ABC_ROOT / "SCHEDULE_ABC_WINNER.txt"
    if not p.exists():
        return None
    raw = p.read_text().strip()
    for suf, let in (("_A", "A"), ("_B", "B"), ("_C", "C")):
        if raw.endswith(suf) or raw == let:
            return let
    return None


def _abc_alive() -> bool:
    try:
        out = subprocess.check_output(["pgrep", "-af", "launch_w4096_schedule_abc"], text=True)
    except subprocess.CalledProcessError:
        return False
    return "launch_w4096_schedule_abc.py" in out


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    print("WATCH_ABC_THEN_RDRIFT_GRID", flush=True)
    while True:
        letter = _winner_letter()
        drained = False
        if ABC_LOG.exists() and "W4096_SCHED_ABC_DRAINED" in ABC_LOG.read_text():
            drained = True
        if letter and drained and not _abc_alive():
            break
        if letter and not _abc_alive():
            # winner file written at end of ABC; safe to proceed
            break
        print(
            f"WAIT_ABC letter={letter} alive={_abc_alive()} drained={drained}",
            flush=True,
        )
        time.sleep(60)
    letter = _winner_letter()
    if not letter:
        raise SystemExit("ABC finished but no winner file")
    print(f"ABC_WINNER schedule={letter}", flush=True)
    subprocess.check_call(
        [PY, str(DEC / "gen_rdrift_weight_configs.py"), "--schedule", letter],
        cwd=str(REPO),
    )
    env = os.environ.copy()
    env["VITCOP_DECISION_GPUS"] = env.get("VITCOP_DECISION_GPUS", "0,4,6,7")
    env["RDRIFT_GRID_SCHEDULE"] = letter
    with LOG.open("w") as fh:
        proc = subprocess.Popen(
            [PY, "-u", str(DEC / "launch_rdrift_weight_grid.py")],
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"STARTED_RDRIFT_GRID pid={proc.pid} log={LOG}", flush=True)
    rc = proc.wait()
    print(f"RDRIFT_GRID_WATCH_DONE rc={rc}", flush=True)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
