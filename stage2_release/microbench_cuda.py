"""CUDA microbench for Stage2 release (GPU 0 only via CUDA_VISIBLE_DEVICES)."""

from __future__ import annotations

import time

import torch

from stage2_release.release_effect import ReleaseEffect


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_once(
    h: int = 32,
    t: int = 8,
    d_h: int = 128,
    n: int = 640,
    m: int = 576,
    warmup: int = 10,
    iters: int = 50,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = h * d_h
    q = torch.randn(h, t, d_h, device=device)
    k = torch.randn(h, n, d_h, device=device)
    v = torch.randn(h, n, d_h, device=device)
    vis_idx = torch.arange(n - m, n, device=device)
    o_proj = torch.randn(d_model, d_model, device=device)
    hh = torch.randn(t, d_model, device=device)
    calc = ReleaseEffect(k_grid=(96, 64, 48, 32, 16, 0))

    for _ in range(warmup):
        calc.compute(q, k, v, vis_idx, o_proj, hh, tau=0.05)
    _sync()

    t0 = time.perf_counter()
    out = None
    for _ in range(iters):
        out = calc.compute(q, k, v, vis_idx, o_proj, hh, tau=0.05)
    _sync()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iters

    return {
        "device": str(device),
        "shape": f"H={h} T={t} Dh={d_h} N={n} M={m}",
        "ms_per_call": round(elapsed_ms, 3),
        "chosen_k": out.chosen_k if out is not None else None,
        "epsilon_per_k": {
            kk: round(vv, 6) for kk, vv in (out.epsilon_per_k.items() if out else [])
        },
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for microbench")
    print("cuda_device:", torch.cuda.get_device_name(0))
    print("visible_count:", torch.cuda.device_count())
    result = bench_once()
    for key, val in result.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()
