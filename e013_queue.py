#!/usr/bin/env python3
"""Fill 8 GPUs with E013 panel/full jobs. One VLM per GPU. Survives SSH via setsid children."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
ROOT = REPO / "logs/e013_p1"
PY = "/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python"
from e013_arms import ARMS, BRIDGE12, SCREEN1, SCREEN2

TASKS = ("gqa", "pope", "mme", "mmbench_en_dev")
DEFAULT_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
# Full-split ACC only (user rule): never treat panel/limit runs as accuracy evidence.
FULL_COMPARE_ARMS = [
    "BASE_S8",
    "B0_S8",
    "REPLACE_RANDOM16",
    "REPLACE_ATTN16",
    "REPLACE_COVER16",
    "EXTRA_RANDOM16",
    "EXTRA_ATTN16",
    "EXTRA_COVER16",
    "EXTRA_NULL16",
    "EXTRA_DUPLICATE16",
]


def memory() -> dict[int, int]:
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    out = {}
    for line in text.splitlines():
        index, used = [part.strip() for part in line.split(",")]
        out[int(index)] = int(used)
    return out


def alive(pid: int) -> bool:
    return pid > 0 and Path(f"/proc/{pid}").exists()


def claim_pid(name: str) -> int:
    path = ROOT / "claims" / f"{name}.json"
    if not path.exists():
        return 0
    try:
        return int(json.loads(path.read_text()).get("pid") or 0)
    except (OSError, ValueError):
        return 0


def attempts(name: str) -> int:
    path = ROOT / "claims" / f"{name}.attempts"
    if not path.exists():
        return 0
    try:
        return int(path.read_text().strip() or "0")
    except ValueError:
        return 0


def tasks(phase: str, arms: list[str], limit: int) -> list[dict]:
    items = []
    bucket = "panel" if phase.startswith("screen") or phase == "bridge12" else "full"
    for arm in arms:
        layer = int(ARMS[arm][4])  # S
        for task in TASKS:
            output = ROOT / bucket / phase / arm / task
            items.append({
                "name": f"{phase}_{arm}_{task}",
                "arm": arm,
                "marker": str(output / "metrics.json"),
                "cmd": (
                    f"export CUDA_VISIBLE_DEVICES=GPU PYTHONPATH={REPO} PYTHONUNBUFFERED=1 "
                    f"HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false "
                    f"E013_ARM={arm}; "
                    f"{PY} {REPO / 'e013_boot.py'} --profile ANCHOR_A --task {task} "
                    f"--output {output} --gpu-index GPU --layer {layer}"
                    + (f" --limit {limit}" if limit else "")
                ),
            })
    return items


def launch(item: dict, gpu: int) -> None:
    name = item["name"]
    claims = ROOT / "claims"
    fills = ROOT / "fills"
    logs = ROOT / "logs"
    claims.mkdir(parents=True, exist_ok=True)
    fills.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    claim = claims / f"{name}.json"
    wrapper = fills / f"{name}.sh"
    command = item["cmd"].replace("GPU", str(gpu))
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -uo pipefail\n"
        f"cd {REPO}\n"
        "if [[ -z \"${OPENAI_API_KEY:-}\" && -f \"$HOME/.bashrc\" ]]; then\n"
        "  source \"$HOME/.bashrc\"\n"
        "fi\n"
        f"export CUDA_VISIBLE_DEVICES={gpu}\n"
        f"export E013_ARM={item['arm']}\n"
        f"echo '{{\"pid\": '$$', \"gpu\": {gpu}, \"name\": \"{name}\"}}' > {claim}\n"
        f"{command}\n"
        f"ec=$?\n"
        f"echo exit=$ec >> {logs}/{name}.outer.log\n"
        f"exit $ec\n"
    )
    wrapper.chmod(0o755)
    subprocess.Popen(
        ["setsid", "bash", str(wrapper)],
        stdout=(logs / f"{name}.outer.log").open("a"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        cwd=str(REPO),
    )
    (claims / f"{name}.attempts").write_text(str(attempts(name) + 1))


def log(text: str) -> None:
    line = f"[{time.strftime('%F %T')}] {text}"
    print(line, flush=True)
    with (ROOT / "queue.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("screen1", "screen2", "bridge12", "full"))
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=512)
    ap.add_argument(
        "--gpus",
        default=",".join(str(g) for g in DEFAULT_GPUS),
        help="comma-separated physical GPU indices",
    )
    ap.add_argument("--once", action="store_true", help="schedule available free GPUs once then exit")
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    if args.phase == "screen1":
        arms = args.arms or SCREEN1
        limit = args.limit
    elif args.phase == "screen2":
        arms = args.arms or SCREEN2
        limit = args.limit
    elif args.phase == "bridge12":
        arms = args.arms or BRIDGE12
        limit = args.limit
    else:
        # User rule: ACC decisions require full 4-dataset splits only.
        arms = args.arms or FULL_COMPARE_ARMS
        limit = 0
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    items = tasks(args.phase, arms, limit)
    while True:
        pending = []
        for item in items:
            if Path(item["marker"]).exists():
                continue
            pid = claim_pid(item["name"])
            if alive(pid):
                continue
            # Skip thrashing: if attempts>=2 and marker missing after dead pid, reset once for resume.
            if attempts(item["name"]) >= 2 and not alive(pid):
                # allow one more after operator/reset; still try
                pass
            pending.append(item)
        if not pending:
            log(f"{args.phase} complete all markers")
            return
        mem = memory()
        free = [g for g in gpus if mem.get(g, 99999) < 500]
        # Also treat GPU as busy if a live claim owns it.
        busy = set()
        for path in (ROOT / "claims").glob("*.json"):
            try:
                claim = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if alive(int(claim.get("pid") or 0)):
                busy.add(int(claim.get("gpu", -1)))
        free = [g for g in free if g not in busy]
        if not free:
            log(f"{args.phase} wait free gpu pending={len(pending)}")
            if args.once:
                return
            time.sleep(30)
            continue
        for gpu, item in zip(free, pending):
            log(f"launch gpu={gpu} {item['name']}")
            launch(item, gpu)
        if args.once:
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
