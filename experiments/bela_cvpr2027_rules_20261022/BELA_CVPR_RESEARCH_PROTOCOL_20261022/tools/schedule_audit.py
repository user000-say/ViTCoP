#!/usr/bin/env python3
"""Offline, dependency-free validation of BELA/ELVA monotone decoder-token schedules.

No GPU score or measured latency is inferred from this tool.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

TOTAL_LAYERS = 32
FULL_VISION_TOKENS = 576


@dataclass(frozen=True)
class Schedule:
    arm: str
    sections: tuple[tuple[int, int], ...]  # (tokens, number of decoder blocks)

    @property
    def layer_count(self) -> int:
        return sum(n for _, n in self.sections)

    @property
    def visual_token_layers(self) -> int:
        return sum(k * n for k, n in self.sections)

    @property
    def layer_average(self) -> float:
        return self.visual_token_layers / self.layer_count

    @property
    def proxy_reduction(self) -> float:
        return 1 - self.visual_token_layers / (FULL_VISION_TOKENS * TOTAL_LAYERS)

    def validate(self, target_w: int, full_tokens: int = FULL_VISION_TOKENS) -> None:
        if not self.sections:
            raise ValueError("missing schedule")
        if any(type(k) is not int or type(n) is not int for k, n in self.sections):
            raise ValueError("token and depth values must be integers")
        if any(n <= 0 for _, n in self.sections):
            raise ValueError("section depth must be positive")
        if any(k < 0 or k > full_tokens for k, _ in self.sections):
            raise ValueError("token count out of range")
        if self.layer_count != TOTAL_LAYERS:
            raise ValueError(f"expected {TOTAL_LAYERS} decoder blocks; got {self.layer_count}")
        if any(self.sections[i][0] < self.sections[i+1][0]
               for i in range(len(self.sections)-1)):
            raise ValueError("retained tokens must be non-increasing")
        if self.visual_token_layers != target_w:
            raise ValueError(
                f"bad W: {self.visual_token_layers}, expected {target_w}"
            )
        # Reverse/inverse check W == average * 32, with integer-safe arithmetic:
        if abs(self.layer_average * TOTAL_LAYERS - target_w) > 1e-9:
            raise ValueError("failed reverse arithmetic check")


def read_experiments(path: str | Path) -> list[tuple[Schedule, int, str]]:
    records = []
    with open(path, newline="", encoding="utf-8-sig") as fp:
        for row in csv.DictReader(fp):
            s = Schedule(
                row["arm"],
                ((int(row["k0"]), int(row["n0"])),
                 (int(row["k1"]), int(row["n1"])),
                 (int(row["k2"]), int(row["n2"]))),
            )
            gate = row["gate"]
            target = 1024 if gate == "T3" else 2048
            records.append((s, target, gate))
    return records


def enumerate_mid_for_tail(
    initial_tokens: int, n0: int, n1: int, n2: int,
    target_w: int, tails: range | list[int]
) -> list[Schedule]:
    """Enumerate exact-W feasible triples; never guess mid-token counts."""
    results = []
    if n1 <= 0:
        raise ValueError("n1 must be positive")
    for last in tails:
        numerator = target_w - initial_tokens*n0 - last*n2
        if numerator < 0 or numerator % n1:
            continue
        middle = numerator//n1
        s = Schedule(
            f"{initial_tokens}x{n0}+{middle}x{n1}+{last}x{n2}",
            ((initial_tokens,n0),(middle,n1),(last,n2)),
        )
        try:
            s.validate(target_w)
        except ValueError:
            continue
        results.append(s)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path, help="experiment_matrix.csv")
    args = parser.parse_args()
    records = read_experiments(args.matrix)
    count = 0
    for sched, target, gate in records:
        sched.validate(target)
        print(
            f"{sched.arm:3s} {gate} "
            f"W={sched.visual_token_layers:4d} "
            f"avg={sched.layer_average:6.2f} "
            f"proxy_reduction={100*sched.proxy_reduction:6.2f}% "
            f"sections={sched.sections}"
        )
        count += 1
    assert count > 0, "empty experiment matrix"
    print(f"PASS: {count} valid exact-W schedules; NO GPU runs claimed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
