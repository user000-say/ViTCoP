#!/usr/bin/env python3
"""Fill 8 GPUs with E014 full/panel jobs. One VLM per GPU. SSH-safe via setsid."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
ROOT = REPO / "logs/e014_aee"
PY = "/home/xiangyue/envs/vitcopreback/vitcopreback/bin/python"
from e014_arms import (
    ARMS,
    FULL_E015_DYN_COVER_K96,
    FULL_E015_K64,
    FULL_E016_CASCADE,
    FULL_K32,
    FULL_K32_L10,
    FULL_K32_L12,
    FULL_K64,
    FULL_K96,
    SMOKE_K32,
    SMOKE_K32_L10,
    SMOKE_K32_L12,
    SMOKE_K64,
    SMOKE_K96,
)

TASKS = ("gqa", "pope", "mme", "mmbench_en_dev")
DEFAULT_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]

PHASE_ARMS = {
    "smoke_k96": SMOKE_K96,
    "smoke_k64": SMOKE_K64,
    "smoke_k32": SMOKE_K32,
    "smoke_k32_l10": SMOKE_K32_L10,
    "smoke_k32_l12": SMOKE_K32_L12,
    "full_k96": FULL_K96,
    "full_k64": FULL_K64,
    "full_k32": FULL_K32,
    "full_k32_l10": FULL_K32_L10,
    "full_k32_l12": FULL_K32_L12,
    "full_e015_k64": FULL_E015_K64,
    "full_e015_dyn_cover_k96": FULL_E015_DYN_COVER_K96,
    "full_e016_cascade": FULL_E016_CASCADE,
}


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
    except (OSError, ValueError, json.JSONDecodeError):
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
    bucket = "smoke" if phase.startswith("smoke") else "full"
    for arm in arms:
        layer = int(ARMS[arm][2])  # S (=D)
        for task in TASKS:
            output = ROOT / bucket / phase / arm / task
            items.append({
                "name": f"{phase}_{arm}_{task}",
                "arm": arm,
                "marker": str(output / "metrics.json"),
                "cmd": (
                    f"export CUDA_VISIBLE_DEVICES=GPU PYTHONPATH={REPO} PYTHONUNBUFFERED=1 "
                    f"HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false "
                    f"E014_ARM={arm}; "
                    f"{PY} {REPO / 'e014_boot.py'} --profile ANCHOR_A --task {task} "
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
        f"export E014_ARM={item['arm']}\n"
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
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "queue.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=tuple(PHASE_ARMS))
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = full split; >0 for smoke/panel only")
    ap.add_argument("--gpus", default=",".join(str(g) for g in DEFAULT_GPUS))
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--mem-free-mib", type=int, default=2000)
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    arms = args.arms or PHASE_ARMS[args.phase]
    items = tasks(args.phase, arms, args.limit)
    ROOT.mkdir(parents=True, exist_ok=True)
    log(f"phase={args.phase} arms={len(arms)} jobs={len(items)} limit={args.limit}")

    while True:
        mem = memory()
        busy = set()
        for item in items:
            name = item["name"]
            if Path(item["marker"]).exists():
                continue
            pid = claim_pid(name)
            if alive(pid):
                try:
                    busy.add(int(json.loads((ROOT / "claims" / f"{name}.json").read_text())["gpu"]))
                except Exception:
                    pass
        free = [g for g in gpus if mem.get(g, 99999) < args.mem_free_mib and g not in busy]
        pending = [i for i in items if not Path(i["marker"]).exists() and not alive(claim_pid(i["name"]))]
        for gpu, item in zip(free, pending):
            if attempts(item["name"]) >= 3:
                log(f"SKIP max attempts {item['name']}")
                continue
            log(f"LAUNCH gpu={gpu} {item['name']}")
            launch(item, gpu)
            time.sleep(2)
        done = sum(1 for i in items if Path(i["marker"]).exists())
        running = sum(1 for i in items if alive(claim_pid(i["name"])) and not Path(i["marker"]).exists())
        log(f"progress done={done}/{len(items)} running={running} free_gpus={free}")
        if done == len(items):
            log("COMPLETE")
            return
        if args.once:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
