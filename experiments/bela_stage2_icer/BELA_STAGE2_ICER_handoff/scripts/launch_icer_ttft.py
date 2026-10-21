"""BELA ICER TTFT/FLOPs on GPU 7 for the five new arms."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
PKG = REPO / "experiments/bela_stage2_icer/BELA_STAGE2_ICER_handoff"
R3 = REPO / "experiments/elva_r3"
PYBENCH = str(REPO / "experiments/e014_latency_fair_256/bin/python")
BENCH = REPO / "experiments/e014_latency_fair_256/scripts/benchmark_llava_e2e_anygpu.py"
PANEL = REPO / "experiments/e014_latency_fair_256/panel/speed_256.jsonl"
MODEL = REPO / "experiments/e014_latency_fair_256/models_llava"
TTFT_GPU = 7
BASE_ARM = "E018_COVER_TWO32"
OUT_ROOT = PKG / "server_results" / "timing"
CFG = PKG / "configs"

ARMS = (
    ("S2_W2048_QATTN_ICER", "icer"),
    ("S3_W2048_QATTN_ICER_DRIFT", "icer_drift"),
    ("S4_W2048_RDRIFT_ICER", "icer"),
    ("S5_W2048_SINGLETON_PROXY", "singleton_proxy"),
    ("S7_W2048_RANDOM_SWAP", "random_swap"),
)


def env_for(cfg_path: Path, audit_dir: Path, mode: str) -> tuple[dict, dict]:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(R3))
    from e014_arms import apply_arm_env
    from apply_llava import overlay
    os.environ["ELVA_R3_CONFIG"] = str(cfg_path)
    os.environ["E014_ARM"] = BASE_ARM
    meta = overlay(apply_arm_env(BASE_ARM))
    keep = {}
    for key, value in os.environ.items():
        if key.startswith(("E0", "ELVA_", "USE_", "STAGE", "COMPUTE", "BELA_")) or key == "E014_ARM":
            keep[key] = value
    keep["ELVA_R3_CONFIG"] = str(cfg_path)
    keep["ELVA_R3_AUDIT_DIR"] = str(audit_dir)
    keep["E014_ARM"] = meta["arm"]
    keep["BELA_ICER_MODE"] = mode
    keep["BELA_ICER_MAX_SWAPS"] = "3"
    keep["BELA_ICER_DONOR_POOL"] = "24"
    keep["BELA_ICER_DROP_POOL"] = "16"
    keep["BELA_ICER_MIN_QMASS"] = "0.97"
    keep["BELA_ICER_DRIFT_WEIGHT"] = "0.20"
    keep["BELA_ICER_DIAG"] = "False"
    return keep, meta


def flops_report(cfg: dict, meta: dict, latency_summary: dict, mode: str) -> dict:
    counts = [int(x) for x in cfg["method"]["schedule"]["counts"]]
    lengths = [int(x) for x in cfg["method"]["schedule"]["lengths"]]
    w = sum(c * l for c, l in zip(counts, lengths))
    full = 576 * 32
    d = 4096
    return {
        "status": "ESTIMATE_NOT_FUSED_KERNEL_PROFILE",
        "schedule_counts": counts,
        "schedule_lengths": lengths,
        "actual_W_token_layers": w,
        "full_visual_token_layers_576x32": full,
        "visual_compression_proxy_1_minus_W_over_full": 1.0 - w / full,
        "e018_events": meta.get("e018_events"),
        "entry_k": meta.get("visual"),
        "final_k": meta.get("target_k"),
        "visual_attn_proxy_flops": 2.0 * d * w * d,
        "bela_icer_mode": mode,
        "note": (
            "W is visual token-layer budget, not whole-model FLOPs. "
            "TTFT includes selector+ICER+gather inside generate()."
        ),
        "ttft_wall_median_ms": latency_summary.get("ttft_wall_ms", {}).get("median"),
        "ttft_cuda_median_ms": latency_summary.get("ttft_cuda_ms", {}).get("median"),
        "fixed32_wall_median_ms": latency_summary.get("fixed32_wall_ms", {}).get("median"),
        "peak_gpu_allocated_mb_median": latency_summary.get("peak_gpu_allocated_mb", {}).get("median"),
        "gpu_physical_id": str(TTFT_GPU),
        "protocol": "warmup+panel; CUDA Event + wall; Fixed32; GPU7 exclusive",
    }


def run_one(out_id: str, mode: str):
    cfg_path = CFG / f"{out_id}.json"
    out = OUT_ROOT / out_id
    if (out / "latency_summary.json").exists():
        print("SKIP", out_id, flush=True)
        return
    if out.exists():
        for child in out.iterdir():
            if child.is_file():
                child.unlink()
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(cfg_path.read_text())
    child_env = os.environ.copy()
    keep, meta = env_for(cfg_path, out, mode)
    child_env.update(keep)
    child_env["CUDA_VISIBLE_DEVICES"] = str(TTFT_GPU)
    child_env["PYTHONPATH"] = (
        f"{REPO}:{REPO / 'experiments'}:{R3}:{REPO / 'LLaVA'}:"
        f"{REPO / 'e020_lib'}:{PKG}"
    )
    child_env["HF_HUB_OFFLINE"] = "1"
    child_env["HF_DATASETS_OFFLINE"] = "1"
    child_env["TRANSFORMERS_OFFLINE"] = "1"
    child_env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        PYBENCH, str(BENCH),
        "--repo-root", str(REPO), "--panel", str(PANEL),
        "--model-path", str(MODEL), "--conv-mode", "llava_v1",
        "--out-dir", str(out), "--warmups", "12", "--limit", "0",
    ]
    log = PKG / "logs" / f"ttft_{out_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"TTFT gpu={TTFT_GPU} {out_id} mode={mode}", flush=True)
    with log.open("w") as handle:
        subprocess.check_call(cmd, env=child_env, stdout=handle, stderr=subprocess.STDOUT)
    summary = json.loads((out / "latency_summary.json").read_text())
    flops = flops_report(cfg, meta, summary, mode)
    (out / "flops_report.json").write_text(json.dumps(flops, indent=2) + "\n")
    flops_dir = PKG / "server_results" / "flops"
    flops_dir.mkdir(parents=True, exist_ok=True)
    (flops_dir / f"{out_id}.json").write_text(json.dumps(flops, indent=2) + "\n")
    print("TTFT done", out_id, "TTFT_med", summary["ttft_wall_ms"]["median"], flush=True)


def main():
    while True:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        mem7 = None
        for line in out.strip().splitlines():
            idx, mem = [x.strip() for x in line.split(",")]
            if int(idx) == TTFT_GPU:
                mem7 = float(mem)
        if mem7 is not None and mem7 < 2000:
            break
        print(f"wait GPU {TTFT_GPU} free (mem={mem7})", flush=True)
        time.sleep(30)
    for out_id, mode in ARMS:
        run_one(out_id, mode)
    print("ICER TTFT queue drained", flush=True)


if __name__ == "__main__":
    main()
