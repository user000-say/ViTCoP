"""Resumable three-worker ACC queue; caller supplies the REAL native runner.

No shell interpolation. GPU IDs are configurable; reserve a separate card
for TTFT. Official scorer output and complete trace are mandatory for DONE.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import threading
import time


REQUIRED_BINDING = ("source_commit", "model_sha", "processor_sha", "scorer_sha",
                    "prompt_sha", "tokenizer_sha", "dataset_sha", "backend", "cache_mode",
                    "native_selectors", "original_tokens", "depth")


def run_process(argv, *, env, log, timeout):
    process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
    except BaseException:
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


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def conditional_ready(config, binding):
    if not config["method"]["conditional"]:
        return True
    selector = config["method"]["stage2"]["first_selector"]
    if not binding.get("transient_adapter_ready", False):
        return False
    if selector == "FROZEN_FINAL_ANCHOR":
        return True
    gate = binding.get("transient_gate", {})
    return (gate.get("schema") == "transient_survival_swap_v1"
            and gate.get("status") == "READY" and gate.get("cache_mode") == "native"
            and bool(gate.get("weights_sha")) and bool(gate.get("catalog_hash"))
            and gate.get("real_labels") is True and gate.get("heldout_gain_over_noise") is True
            and gate.get("generation_checked") is True)


def validate_result(value, config, binding, task, run_key):
    method = config["method"]
    if value.get("run_key") != run_key or value.get("config_hash") != config["config_hash"]:
        raise ValueError("stale or missing provenance fingerprint")
    if value.get("task") != task or value.get("backend") != binding["backend"] or value.get("cache_mode") != method["cache_mode"]:
        raise ValueError("task/backend/cache mismatch")
    count = value.get("samples")
    if type(count) is not int or count <= 0 or value.get("scorer_sha") != binding["scorer_sha"]:
        raise ValueError("real scored samples and bound scorer required")
    metrics = value.get("metrics")
    if not isinstance(metrics, dict) or not metrics or not all(type(v) in (int, float) and math.isfinite(v) for v in metrics.values()):
        raise ValueError("missing/nonfinite official metrics")
    expected = [k for k, d in zip(method["schedule"]["counts"], method["schedule"]["lengths"]) for _ in range(d)]
    audit = value.get("trace_audit", {})
    if (audit.get("expected") != expected or audit.get("observed_samples") != count
            or audit.get("mismatch_count") != 0):
        raise ValueError("missing/incorrect all-sample trace audit")
    if value.get("binding_hash") != digest(binding):
        raise ValueError("binding changed; do not reuse old results")


def run(catalog, binding, command, output, gpus, tasks, hours, allow_conditional=False):
    if any(key not in binding or not binding[key] for key in REQUIRED_BINDING):
        raise ValueError("complete server binding manifest required")
    if (binding["original_tokens"], binding["depth"]) != (576, 32):
        raise ValueError("this catalog is for 576x32; regenerate budgets for other architectures")
    if not gpus or len(set(gpus)) != len(gpus) or hours <= 0 or not math.isfinite(hours):
        raise ValueError("distinct ACC GPU IDs and finite positive time budget required")
    if not isinstance(command, list) or not command or any(not isinstance(v, str) for v in command):
        raise ValueError("runner command must be an argv array")
    required = ("{config}", "{task}", "{out}", "{run_key}", "{binding}")
    if any(not any(p in token for token in command) for p in required):
        raise ValueError("runner argv missing mandatory routing placeholders")
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out/"binding.json").write_text(json.dumps(binding, indent=2))
    deadline = time.monotonic()+hours*3600
    jobs, states, lock = [], [], threading.Lock()
    for config in catalog:
        if config["method"]["conditional"] and (not allow_conditional or not conditional_ready(config, binding)):
            states.append(dict(config=config["id"], status="DEFERRED_CONDITIONAL"))
            continue
        for task in tasks:
            key = digest(dict(config=config, binding=binding, task=task))
            jobs.append((config, task, key))

    def worker(gpu):
        while time.monotonic() < deadline:
            with lock:
                if not jobs:
                    return
                config, task, key = jobs.pop(0)
            path = out/key
            path.mkdir(exist_ok=True)
            result_path, config_path = path/"result.json", path/"config.json"
            config_path.write_text(json.dumps(config, indent=2))
            status = dict(config=config["id"], task=task, run_key=key, gpu=gpu)
            try:
                if result_path.exists():
                    value = json.loads(result_path.read_text())
                    validate_result(value, config, binding, task, key)
                    status["status"] = "REUSED_VERIFIED"
                else:
                    replacements = {"{config}": str(config_path), "{task}": task,
                                    "{out}": str(result_path), "{run_key}": key,
                                    "{binding}": str(out/"binding.json")}
                    argv = command.copy()
                    for marker, value in replacements.items():
                        argv = [token.replace(marker, value) for token in argv]
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                    remaining = max(.01, deadline-time.monotonic())
                    with (path/"runner.log").open("w") as log:
                        run_process(argv, env=env, log=log, timeout=remaining)
                    value = json.loads(result_path.read_text())
                    validate_result(value, config, binding, task, key)
                    status["status"] = "DONE"
            except Exception as error:
                status.update(status="FAILED", reason=f"{type(error).__name__}: {error}")
                if result_path.exists():
                    result_path.replace(path/f"rejected-{time.time_ns()}.json")
            with lock:
                states.append(status)
                (out/"status.json").write_text(json.dumps(states, indent=2, ensure_ascii=False))

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in gpus]
        for future in futures:
            future.result()
    states.extend(dict(config=c["id"], task=t, run_key=k, status="PENDING_DEADLINE") for c, t, k in jobs)
    (out/"status.json").write_text(json.dumps(states, indent=2, ensure_ascii=False))
    return states


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default="configs/catalog.json")
    p.add_argument("--binding", required=True)
    p.add_argument("--runner", required=True, help="JSON file containing argv array")
    p.add_argument("--out", default="results")
    p.add_argument("--gpus", nargs="+", default=["0", "2", "4"])
    p.add_argument("--tasks", nargs="+", default=["GQA", "POPE", "MME", "MMBench"])
    p.add_argument("--hours", type=float, default=6.)
    p.add_argument("--allow-conditional", action="store_true")
    args = p.parse_args()
    load = lambda path: json.loads(Path(path).read_text())
    run(load(args.catalog)["configs"], load(args.binding), load(args.runner), args.out,
        args.gpus, args.tasks, args.hours, args.allow_conditional)
