#!/usr/bin/env python3
"""E014 full-accuracy scheduler using the PROJECT'S EXISTING eval commands.

This utility does not invent any GQA/POPE/MME/MMBench evaluator arguments.
Provide a JSON map of command templates with {out_dir}, {gpu}, {k}, ...
For publication, require per-run eval_summary.json with exact sample N.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from queue import Queue
import shlex
import subprocess
import sys

BENCHMARKS = ("GQA", "POPE", "MME", "MMBench")

@dataclass(frozen=True)
class Job:
    benchmark: str
    k: int
    method: str
    b: int
    layer: int

    @property
    def tag(self):
        return f"L{self.layer}_K{self.k}_{self.method}_B{self.b}_{self.benchmark}"

def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", choices=("k96","k64","k32","all"), default="k96")
    p.add_argument("--fixed-b", type=int, choices=(0,8,16,24,32), default=16)
    p.add_argument("--layer", type=int, choices=(8,12), default=8)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--commands-json", help='{"GQA":"python ... {out_dir} ...",...}')
    p.add_argument("--expected-n-json", help='{"GQA":N,...} from YOUR full dataset splits')
    p.add_argument("--out", default="logs/e014_suite")
    p.add_argument("--cwd", default=".")
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--lambda-value", type=float, default=1.0)
    p.add_argument("--penalty", type=float, default=0.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-unverified", action="store_true",
                   help="smoke/prototyping ONLY; results NOT publication-grade")
    p.add_argument("--allow-subset", action="store_true",
                   help="smoke ONLY; full-accuracy comparisons require all four")
    return p.parse_args()

def create_jobs(plan, layer, fixed_b, benchmarks):
    ks = ([96,64,32] if plan == "all" else [int(plan[1:])])
    for k in ks:
        configs = (
            [("BASE",0)] +
            [("FIXED_COVER",b) for b in (0,8,16,24,32)] +
            [("FIXED_RISK",16),("ADAPTIVE_B_ONLY",16),("AEE",16)]
            if k == 96 else
            [("BASE",0),("FIXED_COVER",fixed_b),
             ("FIXED_RISK",fixed_b),("ADAPTIVE_B_ONLY",fixed_b),("AEE",fixed_b)]
        )
        for method,b in configs:
            for benchmark in benchmarks:
                yield Job(benchmark,k,method,b,layer)

def main():
    args = parse()
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise SystemExit("GPU list must be nonempty and unique")
    cmds = json.loads(Path(args.commands_json).read_text(encoding="utf-8")) if args.commands_json else {}
    benches = BENCHMARKS if not args.allow_subset else tuple(x for x in BENCHMARKS if x in cmds)
    if not args.dry_run and not cmds:
        raise SystemExit("--commands-json is required for execution")
    if not args.dry_run and not args.allow_subset and set(cmds) != set(BENCHMARKS):
        raise SystemExit("Require exactly four full benchmark commands; use --allow-subset for smoke")
    if not benches:
        raise SystemExit("No benchmarks selected")
    jobs = list(create_jobs(args.plan,args.layer,args.fixed_b,benches))

    if args.dry_run:
        print("Planned jobs:", len(jobs), "benchmarks:", ",".join(benches))
        for j in jobs:
            print(j.tag)
        return 0

    expected = (
        json.loads(Path(args.expected_n_json).read_text(encoding="utf-8"))
        if args.expected_n_json else {}
    )
    if not args.allow_unverified:
        if set(expected) != set(benches) or any(
            not isinstance(expected.get(b),int) or expected[b] <= 0 for b in benches
        ):
            raise SystemExit(
                "Full runs require --expected-n-json with exact positive N "
                "for ALL benches and per-job eval_summary.json; "
                "--allow-unverified is for smoke ONLY"
            )

    # Fail early if the exact original E013 COVER dependency is unavailable.
    cwd = Path(args.cwd).resolve()
    legacy_helper = cwd / "llava/model/multimodal_encoder/e013_stage1_support.py"
    if any(j.method == "FIXED_COVER" for j in jobs) and not legacy_helper.exists():
        raise SystemExit(
            f"Missing {legacy_helper}. E014 FIXED_COVER deliberately uses the "
            "original E013 helper. Restore it from your repository, or modify "
            "the matrix to use FIXED_COVER_PROXY with an explicit method rename."
        )

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    gpu_q = Queue()
    for gpu in gpus:
        gpu_q.put(gpu)
    scripts = Path(__file__).resolve().parent

    def execute(job):
        gpu = gpu_q.get()
        destination = out / job.tag
        destination.mkdir(exist_ok=True, parents=True)
        log = destination / "run.log"
        summary_file = destination / "eval_summary.json"
        values = {
            "out_dir": str(destination),
            "gpu": gpu, "k": job.k,
            "b": job.b, "variant": job.method, "method": job.method,
            "benchmark": job.benchmark, "bench": job.benchmark,
            "layer": job.layer, "seed": args.seed,
        }
        started = datetime.now(timezone.utc).isoformat()
        record = {**values,"tag":job.tag,"started_utc":started,"log":str(log)}
        try:
            rendered = cmds[job.benchmark].format_map(values)
            cmd = ["bash", str(scripts/f"run_e014_k{job.k}.sh")] + shlex.split(rendered)
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES":str(gpu),
                "E014_VARIANT":job.method,
                "E014_B":str(job.b),
                "E014_LAYER":str(job.layer),
                "E014_SEED":str(args.seed),
                "E014_LAMBDA":str(args.lambda_value),
                "E014_PENALTY":str(args.penalty),
                "E014_AUDIT":"False", "E012_KV_AUDIT":"False",
                "E012_DIAG":"False", "E007_TOKEN_AUDIT":"False",
            })
            record["argv"] = cmd
            with log.open("w",encoding="utf-8") as out_log:
                ret = subprocess.run(
                    cmd, env=env, cwd=cwd, stdout=out_log,
                    stderr=subprocess.STDOUT, check=False
                )
            record["returncode"] = int(ret.returncode)
            record["counts_verified"] = False
            if ret.returncode == 0 and summary_file.exists():
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
                n = summary.get("n")
                record["n"] = n
                record["summary"] = summary
                if job.benchmark in expected:
                    record["counts_verified"] = n == expected[job.benchmark]
            if ret.returncode != 0:
                record["status"] = "FAILED_RUN"
            elif not args.allow_unverified and not record["counts_verified"]:
                record["status"] = "FAILED_FULL_N"
            elif args.allow_unverified:
                record["status"] = "UNVERIFIED_SMOKE"
            else:
                record["status"] = "VERIFIED_FULL_N"
        except Exception as e:
            record["status"] = "FAILED_DRIVER"
            record["error"] = repr(e)
        finally:
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            gpu_q.put(gpu)
        return record

    records = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(execute,j) for j in jobs]
        for future in as_completed(futures):
            rec = future.result()
            records.append(rec)
            print(rec["status"], rec["tag"], flush=True)

    manifest = out / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(rec,ensure_ascii=False)+"\n" for rec in records),
        encoding="utf-8"
    )
    failed = [r for r in records if r["status"].startswith("FAILED")]
    print("Tasks:",len(records),"failed:",len(failed),"manifest:",manifest)
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
