"""Read-only GPU inventory and per-UUID leases. Never kill another job."""
import argparse
import csv
from dataclasses import dataclass
import fcntl
import io
import json
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class GPU:
    index: str
    uuid: str
    used_mib: int
    utilization: int
    pids: tuple

    def available(self, max_used_mib=1024, max_utilization=10):
        return not self.pids and self.used_mib <= max_used_mib and self.utilization <= max_utilization


def parse_inventory(gpus_csv, processes_csv):
    processes = {}
    for row in csv.reader(io.StringIO(processes_csv)):
        if not row:
            continue
        if len(row) != 2:
            raise ValueError("unexpected GPU process query format; cannot assume an empty card")
        uuid, pid = (s.strip() for s in row)
        processes.setdefault(uuid, []).append(int(pid))
    result = []
    for row in csv.reader(io.StringIO(gpus_csv)):
        if not row:
            continue
        if len(row) != 4:
            raise ValueError("unexpected GPU inventory format")
        index, uuid, used, utilization = (s.strip() for s in row)
        if not uuid.startswith("GPU-"):
            raise ValueError("unsupported UUID/MIG mapping; do not guess physical device ownership")
        result.append(GPU(index, uuid, int(used), int(utilization), tuple(processes.get(uuid, ()))))
    if not result or len({g.uuid for g in result}) != len(result):
        raise ValueError("missing or duplicate GPU UUIDs")
    if set(processes)-{g.uuid for g in result}:
        raise ValueError("processes refer to unrecognized UUIDs; inventory is not safe to schedule")
    return result


def inventory():
    def query(argument):
        return subprocess.run(["nvidia-smi", argument, "--format=csv,noheader,nounits"],
                              check=True, capture_output=True, text=True, timeout=15).stdout
    return parse_inventory(query("--query-gpu=index,uuid,memory.used,utilization.gpu"),
                           query("--query-compute-apps=gpu_uuid,pid"))


def respect_visible(gpus, mask):
    if mask is None:
        return list(gpus)
    values = [s.strip() for s in mask.split(",") if s.strip()]
    if not values or values == ["-1"]:
        return []
    known = {g.index for g in gpus}|{g.uuid for g in gpus}
    if set(values)-known:
        raise ValueError("inherited CUDA_VISIBLE_DEVICES cannot be mapped safely to physical UUIDs")
    return [g for g in gpus if g.index in values or g.uuid in values]


def choose(gpus, requested, workers, reserve):
    if workers < 1 or reserve < 0:
        raise ValueError("positive ACC worker count and nonnegative reserve required")
    requested = list(map(str, requested))
    if requested != ["auto"] and len(set(requested)) != len(requested):
        raise ValueError("duplicate requested GPU IDs")
    if requested != ["auto"]:
        known = {g.index for g in gpus}|{g.uuid for g in gpus}
        if set(requested)-known:
            raise ValueError("requested GPU is absent from the current inventory")
    free = [g for g in gpus if g.available() and
            (requested == ["auto"] or g.index in requested or g.uuid in requested)]
    usable = max(0, min(workers, len(free)-reserve))
    return free[:usable], free[usable:]


class Lease:
    def __init__(self, directory, uuid):
        if not uuid.startswith("GPU-") or "/" in uuid:
            raise ValueError("valid physical GPU UUID required")
        self.path = Path(directory)/(uuid+".lock")
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:
            self.stream.close()
            self.stream = None
            raise
        return self

    def __exit__(self, *args):
        if self.stream is not None:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", default=["auto"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--reserve", type=int, default=1)
    args = parser.parse_args()
    devices = inventory()
    selected, rest = choose(devices, args.gpus, args.workers, args.reserve)
    print(json.dumps(dict(inventory=[dict(index=g.index, uuid=g.uuid, used_mib=g.used_mib,
          utilization=g.utilization, pids=g.pids, eligible=g.available()) for g in devices],
          acc=[g.uuid for g in selected], reserved_or_unused=[g.uuid for g in rest]), indent=2))
