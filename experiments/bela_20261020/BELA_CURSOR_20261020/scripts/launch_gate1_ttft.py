"""Gate1 SparseVLM-style latency on GPU 7 only.

Protocol (aligned with handoff / SparseVLM fair timing notes):
- batch=1, exclusive physical GPU 7, SDPA backend
- warmup real generate() then panel measure
- report wall + CUDA-event TTFT (1 token) and Fixed32 (32 tokens)
- peak GPU allocated MB; selector/CLIP/gather inside generate wall
- NOT HTTP/service E2E; NOT cross-GPU absolute claims

FLOPs: writes decoder visual-token-layer proxy + rough LLaMA-7B prefill
estimate from measured W; fused kernels may be undercounted (labeled).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
PKG = REPO / "experiments/bela_20261020/BELA_CURSOR_20261020"
R3 = REPO / "experiments/elva_r3"
CVPR = REPO / "experiments/elva_cvpr_final/ELVA_FINAL_HANDOFF_20261020"
PYBENCH = str(REPO / "experiments/e014_latency_fair_256/bin/python")
BENCH = REPO / "experiments/e014_latency_fair_256/scripts/benchmark_llava_e2e_anygpu.py"
PANEL = REPO / "experiments/e014_latency_fair_256/panel/speed_256.jsonl"
MODEL = REPO / "experiments/e014_latency_fair_256/models_llava"
TTFT_GPU = 7
BASE_ARM = "E018_COVER_TWO32"
OUT_ROOT = PKG / "server_results" / "timing"

ARMS = (
    (R3 / "configs/R3_W2048_V5_a5a7f59fc3.json", "W2048_V5_COVER_QATTN"),
    (CVPR / "configs/H2_W2048_V5_COVER_RDRIFT.json", "W2048_V5_COVER_RDRIFT"),
    (R3 / "configs/R3_W1024_V4_PCOV60_B12_08205a4243.json", "W1024_V4_PCOV_QATTN"),
)


def env_for(cfg_path: Path, audit_dir: Path) -> dict:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(R3))
    from e014_arms import apply_arm_env
    from apply_llava import overlay
    os.environ["ELVA_R3_CONFIG"] = str(cfg_path)
    os.environ["E014_ARM"] = BASE_ARM
    meta = overlay(apply_arm_env(BASE_ARM))
    keep = {}
    for key, value in os.environ.items():
        if key.startswith(("E0", "ELVA_", "USE_", "STAGE", "COMPUTE")) or key == "E014_ARM":
            keep[key] = value
    keep["ELVA_R3_CONFIG"] = str(cfg_path)
    keep["ELVA_R3_AUDIT_DIR"] = str(audit_dir)
    keep["E014_ARM"] = meta["arm"]
    return keep, meta


def flops_report(cfg: dict, meta: dict, latency_summary: dict) -> dict:
    """Documented estimates — not fused-kernel profiler counts."""
    counts = [int(x) for x in cfg["method"]["schedule"]["counts"]]
    lengths = [int(x) for x in cfg["method"]["schedule"]["lengths"]]
    w = sum(c * l for c, l in zip(counts, lengths))
    full = 576 * 32
    # LLaMA-7B rough: per-layer attn+MLP ~ 2*d*d*(seq) style proxy with d=4096, L=32
    # Prefill FLOPs �seq) style proxy with d=4096, L=32
    # Prefill FLOPs ∝ sum over layers of O(seq_len_layer). Use visual W as visual part proxy.
    d = 4096
    visual_proxy_ratio = w / full
    # Transparent O(d^2 * W) visual-attn proxy (not fused-kernel profile).
    visual_attn_proxy = 2.0 * d * w * d
    return {
        "status": "ESTIMATE_NOT_FUSED_KERNEL_PROFILE",
        "schedule_counts": counts,
        "schedule_lengths": lengths,
        "actual_W_token_layers": w,
        "full_visual_token_layers_576x32": full,
        "visual_compression_proxy_1_minus_W_over_full": 1.0 - visual_proxy_ratio,
        "e018_events": meta.get("e018_events"),
        "entry_k": meta.get("visual"),
        "final_k": meta.get("target_k"),
        "visual_attn_proxy_flops": visual_attn_proxy,
        "note": (
            "W is visual token-layer budget, not whole-model FLOPs. "
            "visual_attn_proxy_flops is a transparent O(d^2 * W) estimate; "
            "FlashAttention/fused kernels are NOT counted here. "
            "TTFT wall/cuda already include selector+gather inside generate()."
        ),
        "ttft_wall_median_ms": latency_summary.get("ttft_wall_ms", {}).get("median"),
        "ttft_cuda_median_ms": latency_summary.get("ttft_cuda_ms", {}).get("median"),
        "fixed32_wall_median_ms": latency_summary.get("fixed32_wall_ms", {}).get("median"),
        "peak_gpu_allocated_mb_median": latency_summary.get("peak_gpu_allocated_mb", {}).get("median"),
        "gpu_physical_id": latency_summary.get("gpu_physical_id"),
        "protocol": "warmup+panel; CUDA Event + wall; Fixed32 separate generate; SparseVLM-style model-level not service E2E",
    }


def run_one(cfg_path: Path, out_id: str):
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
    keep, meta = env_for(cfg_path, out)
    child_env.update(keep)
    child_env["CUDA_VISIBLE_DEVICES"] = str(TTFT_GPU)
    child_env["PYTHONPATH"] = (
        f"{REPO}:{REPO / 'experiments'}:{R3}:{REPO / 'LLaVA'}:{REPO / 'e020_lib'}:{PKG}"
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
    print(f"TTFT gpu={TTFT_GPU} {out_id}", flush=True)
    with log.open("w") as handle:
        subprocess.check_call(cmd, env=child_env, stdout=handle, stderr=subprocess.STDOUT)
    summary = json.loads((out / "latency_summary.json").read_text())
    flops = flops_report(cfg, meta, summary)
    (out / "flops_report.json").write_text(json.dumps(flops, indent=2) + "\n")
    (PKG / "server_results" / "flops" / f"{out_id}.json").parent.mkdir(parents=True, exist_ok=True)
    (PKG / "server_results" / "flops" / f"{out_id}.json").write_text(json.dumps(flops, indent=2) + "\n")
    print("TTFT done", out_id, "TTFT_med", summary["ttft_wall_ms"]["median"], flush=True)


def main():
    for cfg_path, out_id in ARMS:
        run_one(cfg_path, out_id)
    print("TTFT/FLOPs queue drained", flush=True)


if __name__ == "__main__":
    main()
