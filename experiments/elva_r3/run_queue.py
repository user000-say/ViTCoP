"""Explicit phase queue, current GPU checks, source-bound per-sample results."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import math
import tempfile
from make_catalog import digest, TASKS
from elva.workflow import (blocked_reason, result_key, validate_binding, validate_result,
                           validate_decision, collect_complete)
from gpu_resources import inventory, choose, respect_visible, Lease


def run_process(argv, *, env, log, timeout):
    process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
    except BaseException:
        # This process group was created above for this exact queue job.
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except ProcessLookupError:
            pass
        raise
    if code:
        raise subprocess.CalledProcessError(code, argv)


def run(catalog, binding, command, output, *, phase="stage1_core", requested=("auto",),
        workers=3, reserve=1, hours=6., tasks=TASKS, budgets=(2048, 1024),
        decisions=None, gpu_probe=inventory, lease_directory=None, allow_fixture=False, ids=None):
    validate_binding(binding, allow_fixture=allow_fixture)
    if not math.isfinite(hours) or hours <= 0 or phase not in ("stage1_core", "stage1_optional", "stage2"):
        raise ValueError("finite positive deadline and explicit supported phase required")
    if not budgets or set(budgets)-{2048, 1024} or set(tasks)-set(TASKS):
        raise ValueError("only the two declared budgets and bound four tasks are accepted")
    markers = ("{config}", "{task}", "{out}", "{run_key}", "{binding}")
    if (not isinstance(command, list) or not command or any(not isinstance(v, str) for v in command)
            or any(not any(p in token for token in command) for p in markers)):
        raise ValueError("real runner argv must contain all five routing placeholders")
    configs = catalog["configs"]
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    binding_path = out/("binding-"+digest(binding)+".json")
    binding_path.write_text(json.dumps(binding, indent=2)+"\n")
    jobs, states = deque(), []
    for budget in budgets:
        if phase == "stage1_optional":
            collect_complete(configs, binding, out, budget, allow_fixture=allow_fixture)
        if phase == "stage2":
            if decisions is None:
                raise ValueError("Stage2 cannot start before the completed Stage1 result freeze")
            validate_decision(decisions, binding, configs, out, budget, allow_fixture=allow_fixture)
    selected_configs = [c for c in configs if c["phase"] == phase and c["method"]["budget"] in budgets]
    if ids is not None:
        if not ids or set(ids)-{c["id"] for c in selected_configs}:
            raise ValueError("--ids must exactly name configurations in the requested phase/budgets")
        selected_configs = [c for c in selected_configs if c["id"] in ids]
    if not selected_configs:
        raise ValueError("no configs for this phase; generate Stage2 from the actual freeze first")
    for config in selected_configs:
        reason = blocked_reason(config, binding)
        if reason:
            states.append(dict(config=config["id"], status="BLOCKED", reason=reason))
            continue
        for task in tasks:
            jobs.append((config, task, result_key(config, binding, task)))
    selected, unused = choose(respect_visible(gpu_probe(), os.environ.get("CUDA_VISIBLE_DEVICES")), requested, workers, reserve)
    lease_directory = lease_directory or os.environ.get("ELVA_GPU_LOCK_DIR") or str(Path(tempfile.gettempdir())/f"elva-gpu-leases-{os.getuid()}")
    state_lock = threading.Lock()
    deadline = time.monotonic()+hours*3600
    def save_state(status):
        with state_lock:
            states.append(status)
            tmp = out/"status.tmp"
            tmp.write_text(json.dumps(dict(phase=phase, jobs=states,
                           reserved_or_unused=[g.uuid for g in unused]), indent=2)+"\n")
            tmp.replace(out/"status.json")
    def worker(device):
        try:
            with Lease(lease_directory, device.uuid):
                while time.monotonic() < deadline:
                    current = next((g for g in gpu_probe() if g.uuid == device.uuid), None)
                    if current is None or not current.available():
                        save_state(dict(gpu=device.uuid, status="GPU_BECAME_BUSY"))
                        return
                    with state_lock:
                        if not jobs:
                            return
                        config, task, key = jobs.popleft()
                    folder = out/key
                    folder.mkdir(exist_ok=True)
                    config_path, result_path = folder/"config.json", folder/"result.json"
                    config_path.write_text(json.dumps(config, indent=2)+"\n")
                    status = dict(config=config["id"], task=task, run_key=key,
                                  physical_gpu=device.index, gpu_uuid=device.uuid)
                    try:
                        reused = False
                        if result_path.exists():
                            try:
                                value = json.loads(result_path.read_text())
                                validate_result(value, config, binding, task, key, folder, allow_fixture=allow_fixture)
                                reused = True
                            except Exception:
                                result_path.replace(folder/f"rejected-{time.time_ns()}.json")
                        if not reused:
                            replacements = dict(zip(markers, (str(config_path), task, str(result_path), key, str(binding_path))))
                            argv = command.copy()
                            for marker, value in replacements.items():
                                argv = [token.replace(marker, value) for token in argv]
                            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device.uuid,
                                       ELVA_GPU_UUID=device.uuid, ELVA_RUN_KEY=key)
                            with (folder/"runner.log").open("w") as log:
                                run_process(argv, env=env, log=log, timeout=max(.01, deadline-time.monotonic()))
                            value = json.loads(result_path.read_text())
                            validate_result(value, config, binding, task, key, folder, allow_fixture=allow_fixture)
                        status["status"] = "REUSED_VERIFIED" if reused else "DONE"
                    except Exception as error:
                        status.update(status="FAILED", reason=f"{type(error).__name__}: {error}")
                    save_state(status)
        except BlockingIOError:
            save_state(dict(gpu=device.uuid, status="LEASE_BUSY"))
        except Exception as error:
            save_state(dict(gpu=device.uuid, status="GPU_WORKER_FAILED", reason=f"{type(error).__name__}: {error}"))
    if selected and jobs:
        with ThreadPoolExecutor(max_workers=len(selected)) as executor:
            list(executor.map(worker, selected))
    for config, task, key in jobs:
        states.append(dict(config=config["id"], task=task, run_key=key, status="PENDING_RESOURCE_OR_DEADLINE"))
    save_state(dict(status="PHASE_QUEUE_FINISHED", selected_gpu_uuids=[g.uuid for g in selected]))
    return states


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="configs/catalog.json")
    parser.add_argument("--binding", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--out", default="results")
    parser.add_argument("--phase", choices=("stage1_core", "stage1_optional", "stage2"), default="stage1_core")
    parser.add_argument("--gpus", nargs="+", default=["auto"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--reserve", type=int, default=1)
    parser.add_argument("--budgets", nargs="+", type=int, default=[2048, 1024])
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--hours", type=float, default=6.)
    parser.add_argument("--decisions")
    parser.add_argument("--lease-dir")
    parser.add_argument("--ids", nargs="+", help="exact config IDs; use to run a bounded optional ablation")
    args = parser.parse_args()
    load = lambda path: json.loads(Path(path).read_text())
    run(load(args.catalog), load(args.binding), load(args.runner), args.out, phase=args.phase,
        requested=args.gpus, workers=args.workers, reserve=args.reserve, hours=args.hours,
        tasks=args.tasks, budgets=args.budgets, decisions=load(args.decisions) if args.decisions else None,
        lease_directory=args.lease_dir, ids=args.ids)
