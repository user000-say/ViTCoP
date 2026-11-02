#!/usr/bin/env python3
"""Hold a GPU with allocated VRAM until killed. Used between queue jobs."""
from __future__ import annotations

import os
import signal
import time

import torch


def main() -> None:
    gb = float(os.environ.get("BELA_GPU_HOLD_GB", "14"))
    device = torch.device("cuda:0")
    n = int(gb * (1024**3) / 4)  # float32 bytes
    # Allocate in chunks to tolerate fragmentation.
    bufs = []
    chunk = n // 8
    left = n
    while left > 0:
        m = min(chunk, left)
        try:
            bufs.append(torch.empty(m, dtype=torch.float32, device=device))
            left -= m
        except RuntimeError:
            chunk = max(chunk // 2, 1_000_000)
            if chunk < 1_000_000:
                break
    torch.cuda.synchronize()
    used = torch.cuda.memory_allocated(device) / (1024**3)
    print(f"[gpu_hold] pid={os.getpid()} hold≈{used:.2f}GiB chunks={len(bufs)}", flush=True)

    def _stop(*_):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
