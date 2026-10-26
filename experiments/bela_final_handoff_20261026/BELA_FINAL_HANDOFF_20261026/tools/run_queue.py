"""Small argv-only, phase-based queue. Uses one serial worker per physical GPU.

Dry-run is the default. This host has no production adapter or CUDA validation.
Prefer the server's existing exclusive queue if already available; this runner
is a fallback, not a requirement to replace working infrastructure.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from check_plan import ROOT, validate


PHASES = ["pilot", "pilot_low", "pilot_high", "candidate", "controls", "confirm", "ablation", "diagnostic_extra"]


def gate_value(gates, key):
    if key is None:
        return True
    value = gates.get("gates", {}).get(key)
    return isinstance(value, dict) and value.get("value") is True and bool(value.get("evidence"))


def jobs_for_phase(config, gates, phase):
    if phase == "pilot":
        return jobs_for_phase(config, gates, "pilot_low") + jobs_for_phase(config, gates, "pilot_high")
    arms = {a["id"]: a for a in config["arms"]}
    four = ["gqa", "pope", "textvqa_val_ocr", "mmbench_en_dev"]
    chosen = []
    def add(name, tasks, seed=17, required=None, ids_scope="pilot"):
        arm = arms[name]
        for task in tasks:
            gpu = (0 if arm["budget"] == 1024 else 2) if task in ("gqa", "pope") else (7 if task in ("mme", "mmbench_en_dev") else 4)
            spec_hash = hashlib.sha256(json.dumps(arm, sort_keys=True).encode()).hexdigest()
            # Stable IDs enable an audited resume across phases, not silent reuse
            # of results from a different runtime tag or frozen sample manifest.
            job = {"id": f"{name}__{task}__{ids_scope}__s{seed}",
                   "arm": name, "task": task, "budget": arm["budget"],
                   "gpu": gpu, "seed": seed, "ids_scope": ids_scope,
                   "spec_hash": spec_hash,
                   "required_gate": required if required is not None else arm.get("gate")}
            chosen.append(job)
    if phase == "pilot_low":
        for a in ["L2_NATIVE", "CD32_FLAT", "COVER32_FLAT", "CD64_L2_NATIVE", "CD64_L2_STATIC"]:
            add(a, four)
    elif phase == "pilot_high":
        for a in ["E3_NATIVE", "E3_L9", "CD64_FLAT"]:
            add(a, four)
    elif phase == "candidate":
        for w in (1024, 2048):
            for prefix in ["A", "S", "AS", "RAND", "SHUFFLE"]:
                add(f"{prefix}_{w}", four)
    elif phase == "controls":
        for w in (1024, 2048):
            winner = gates.get("winners", {}).get(str(w))
            if winner == f"AS_{w}":
                controls = [f"RAND_AS_{w}", f"SHUFFLE_{w}", f"SOURCE_ONLY_AS_{w}"]
            elif winner == f"A_{w}":
                controls = [f"RAND_{w}"]
            elif winner == f"S_{w}":
                controls = [f"SHUFFLE_S_{w}", f"SOURCE_ONLY_S_{w}"]
            else:
                continue
            for a in controls:
                seeds = [config["seeds"]["main"]] if a.startswith("SOURCE_ONLY") else config["seeds"]["negative_controls"]
                for seed in seeds:
                    add(a, four, seed, required=f"confirm_{w}")
    elif phase == "confirm":
        for w, native, cd in [(1024, "L2_NATIVE", "CD32_FLAT"), (2048, "E3_NATIVE", "CD64_FLAT")]:
            winner = gates.get("winners", {}).get(str(w))
            if winner is not None:
                if winner not in arms or arms[winner]["budget"] != w:
                    raise ValueError("unknown or wrong-budget winner")
                for a in dict.fromkeys([native, cd, winner]):
                    add(a, config["tasks"], required=f"confirm_{w}", ids_scope="full")
    elif phase == "ablation":
        for w in (1024, 2048):
            if gates.get("winners", {}).get(str(w)) == f"AS_{w}":
                for a in [f"A_{w}", f"S_{w}"]:
                    add(a, config["tasks"], required=f"confirm_{w}", ids_scope="full")
    elif phase == "diagnostic_extra":
        for a in ["COVER64_FLAT", "CD172_E3_NATIVE"]:
            add(a, four)
    else:
        raise ValueError("unknown phase")
    return chosen


def command_for(job, bindings):
    output = Path(bindings["output_root"]) / bindings["run_tag"] / job["id"]
    ids_file = bindings["ids_files"][job["ids_scope"]][job["task"]]
    fields = dict(job, python=bindings["python"], adapter=bindings["adapter"],
                  output=str(output), run_tag=bindings["run_tag"], ids_file=ids_file)
    argv = [str(s).format_map(fields) for s in bindings["command"]]
    return argv, output, ids_file


def matching_result(path, job, run_tag, expected_ids_hash):
    try:
        m = json.loads((path / "run_meta.json").read_text())
        metrics = json.loads((path / "metrics.json").read_text())
        return (all(m.get(k) == v for k, v in {"arm": job["arm"], "task": job["task"],
                   "actual_W": job["budget"], "seed": job["seed"],
                   "run_tag": run_tag, "spec_hash": job["spec_hash"]}.items())
                and m.get("sample_count", 0) > 0
                and m.get("complete") is True
                and m.get("ids_sha256") == expected_ids_hash and bool(m.get("scorer_hash"))
                and bool(m.get("checkpoint_hash")) and bool(m.get("latency_fix_commit"))
                and (path / "predictions.jsonl").is_file()
                and m.get("predictions_sha256") == hashlib.sha256((path / "predictions.jsonl").read_bytes()).hexdigest()
                and metrics.get("status") == "complete")
    except (OSError, ValueError, TypeError):
        return False


def gpu_is_free(gpu):
    try:
        txt = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
            "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    except (OSError, subprocess.CalledProcessError):
        raise RuntimeError("cannot establish physical GPU availability")
    lines = [line.strip() for line in txt.splitlines() if line.strip()]
    if not lines or all("No running processes found" in line for line in lines):
        return True
    if any(not line.isdigit() for line in lines):
        raise RuntimeError("unrecognized nvidia-smi availability response")
    return False


def execute_worker(gpu, jobs, bindings, wall_hours):
    results = []
    start = time.monotonic()
    for job in jobs:
        if wall_hours is not None and time.monotonic() - start >= wall_hours * 3600:
            results.append({"job": job["id"], "status": "DEFERRED_WINDOW"})
            continue
        argv, output, ids_file = command_for(job, bindings)
        ids_hash = hashlib.sha256(Path(ids_file).read_bytes()).hexdigest()
        if matching_result(output, job, bindings["run_tag"], ids_hash):
            results.append({"job": job["id"], "status": "REUSED_VERIFIED"})
            continue
        if not gpu_is_free(gpu):
            results.append({"job": job["id"], "status": "DEFERRED_BUSY_GPU"})
            continue
        if output.exists() and any(output.iterdir()):
            results.append({"job": job["id"], "status": "INCOMPLETE_OUTPUT_REQUIRES_ADAPTER_RESUME"})
            continue
        output.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        env["BELA_EXPECTED_SPEC_HASH"] = job["spec_hash"]
        env["BELA_EXPECTED_IDS_SCOPE"] = job["ids_scope"]
        env["BELA_HANDOFF_CONFIG"] = bindings["handoff_config"]
        with (output / "process.log").open("w") as log:
            proc = subprocess.run(argv, cwd=bindings["repo"], env=env,
                                  stdout=log, stderr=subprocess.STDOUT, shell=False)
        passed = proc.returncode == 0 and matching_result(output, job, bindings["run_tag"], ids_hash)
        results.append({"job": job["id"], "status": "COMPLETE" if passed else "FAILED_OR_INVALID_ARTIFACT",
                        "returncode": proc.returncode})
        if not passed:
            # Do not spend a whole card on repeated versions of the same failure.
            break
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=PHASES, default="pilot")
    p.add_argument("--config", type=Path, default=ROOT / "config/experiments.json")
    p.add_argument("--bindings", type=Path, default=ROOT / "config/bindings.example.json")
    p.add_argument("--gates", type=Path, default=ROOT / "config/gates.example.json")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--wall-hours", type=float)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    validate(config)
    gates = json.loads(args.gates.read_text())
    bindings = json.loads(args.bindings.read_text())
    bindings["handoff_config"] = str(args.config.resolve())
    jobs = jobs_for_phase(config, gates, args.phase)
    ready, blocked = [], []
    for job in jobs:
        (ready if gate_value(gates, "runtime_bound") and gate_value(gates, job["required_gate"]) else blocked).append(job)
    out = {"phase": args.phase, "mode": "execute" if args.execute else "dry_run",
           "jobs_ready": len(ready), "jobs_blocked": len(blocked),
           "jobs": [dict(j, blocked=j in blocked, argv=command_for(j, bindings)[0]) for j in jobs]}
    if args.execute:
        if "UNBOUND" in json.dumps(bindings):
            p.error("bind real production paths, corrected commit, and frozen IDs first")
        for key in ("python", "adapter", "repo"):
            if not Path(bindings[key]).exists():
                p.error("missing bound path: " + key)
        if args.wall_hours is not None and args.wall_hours <= 0:
            p.error("wall-hours must be positive")
        for job in ready:
            ids_file = command_for(job, bindings)[2]
            if not Path(ids_file).is_file():
                p.error("missing frozen IDs file: " + ids_file)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(execute_worker, gpu,
                [j for j in ready if j["gpu"] == gpu], bindings, args.wall_hours)
                for gpu in config["runtime_policy"]["gpu_ids"]]
            out["results"] = [r for f in futures for r in f.result()]
    txt = json.dumps(out, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(txt + "\n")
    print(txt)
    if args.execute and any(r["status"] == "FAILED_OR_INVALID_ARTIFACT" for r in out.get("results", [])):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
