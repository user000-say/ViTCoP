"""Exclusive GPU leases for ViTCoP experiment launchers.

Race that this prevents
-----------------------
`nvidia-smi` memory can stay low for tens of seconds while a job is still
loading weights. A second launcher that only checks memory will claim the
same card → contaminated TTFT / ACC. Always acquire a flock *before* spawn
and hold it until the child exits.

Project GPUs: 0, 2, 4, 7 only.
"""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_GPUS = (0, 2, 4, 7)
LOCK_ROOT = Path(os.environ.get("VITCOP_GPU_LOCK_DIR", "/tmp/vitcop_gpu_locks"))
# Treat as occupied if used memory exceeds this (MiB). Idle cards are ~0–20.
DEFAULT_MAX_USED_MIB = 800


def _nvidia_used() -> dict[int, int]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    used: dict[int, int] = {}
    for line in out.strip().splitlines():
        idx_s, used_s = [x.strip() for x in line.split(",")]
        used[int(idx_s)] = int(float(used_s))
    return used


@dataclass
class GpuLease:
    gpu: int
    owner: str
    path: Path
    _fd: int

    def release(self) -> None:
        try:
            meta = self.path.with_suffix(".json")
            if meta.exists():
                meta.unlink(missing_ok=True)  # type: ignore[call-arg]
        except OSError:
            pass
        try:
            import fcntl

            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
        except OSError:
            pass


_HELD: list[GpuLease] = []


def _cleanup() -> None:
    while _HELD:
        _HELD.pop().release()


atexit.register(_cleanup)


def try_acquire(
    gpu: int,
    owner: str,
    *,
    max_used_mib: int = DEFAULT_MAX_USED_MIB,
    allowed: tuple[int, ...] = PROJECT_GPUS,
) -> GpuLease | None:
    """Non-blocking: lock + verify memory still free. None if unavailable."""
    import fcntl

    if gpu not in allowed:
        return None
    used = _nvidia_used().get(gpu, 10**9)
    if used >= max_used_mib:
        return None

    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    path = LOCK_ROOT / f"gpu_{gpu}.lock"
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None

    # Re-check memory after lock (another process may have started loading).
    used2 = _nvidia_used().get(gpu, 10**9)
    if used2 >= max_used_mib:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        return None

    meta = {
        "gpu": gpu,
        "owner": owner,
        "pid": os.getpid(),
        "ts": time.time(),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    lease = GpuLease(gpu=gpu, owner=owner, path=path, _fd=fd)
    _HELD.append(lease)
    return lease


def acquire_one(
    owner: str,
    *,
    preferred: tuple[int, ...] | None = None,
    max_used_mib: int = DEFAULT_MAX_USED_MIB,
    allowed: tuple[int, ...] = PROJECT_GPUS,
    poll_s: float = 15.0,
    timeout_s: float | None = None,
) -> GpuLease:
    """Block until one exclusive free GPU from `allowed` is leased."""
    order = preferred or allowed
    t0 = time.time()
    while True:
        for gpu in order:
            if gpu not in allowed:
                continue
            lease = try_acquire(gpu, owner, max_used_mib=max_used_mib, allowed=allowed)
            if lease is not None:
                print(
                    f"[gpu_exclusive] LEASE gpu={lease.gpu} owner={owner} "
                    f"used_mib<{max_used_mib}",
                    flush=True,
                )
                return lease
        if timeout_s is not None and (time.time() - t0) > timeout_s:
            raise TimeoutError(f"no free GPU in {allowed} for owner={owner}")
        print(
            f"[gpu_exclusive] waiting owner={owner} allowed={allowed} ...",
            flush=True,
        )
        time.sleep(poll_s)


def release(lease: GpuLease) -> None:
    if lease in _HELD:
        _HELD.remove(lease)
    lease.release()
    print(f"[gpu_exclusive] RELEASE gpu={lease.gpu} owner={lease.owner}", flush=True)


def free_candidates(
    *,
    busy: set[int] | None = None,
    max_used_mib: int = DEFAULT_MAX_USED_MIB,
    allowed: tuple[int, ...] = PROJECT_GPUS,
) -> list[int]:
    """Memory-only preview (may race). Prefer try_acquire for real claims."""
    busy = busy or set()
    used = _nvidia_used()
    return [
        g
        for g in allowed
        if g not in busy and used.get(g, 10**9) < max_used_mib
    ]
