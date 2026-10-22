"""Exact visual token-layer accounting for post-decoder-block pruning.

A boundary `cut1=4` means the first FOUR decoder blocks have run with k0
visual tokens. Blocks 5 through `cut2` use kmid; all remaining blocks use kfinal.
This is a prefill proxy, NEVER a claim of end-to-end FLOPs/latency equality.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class Schedule:
    layers: int
    k0: int
    cut1: int
    kmid: int
    cut2: int
    kfinal: int

    def __post_init__(self):
        ints = (self.layers, self.k0, self.cut1, self.kmid, self.cut2, self.kfinal)
        if any(type(v) is not int for v in ints):
            raise TypeError("Schedule fields must be Python integers")
        if not (0 < self.cut1 < self.cut2 < self.layers):
            raise ValueError("Require 0 < cut1 < cut2 < layers")
        if not (0 < self.kfinal <= self.kmid <= self.k0):
            raise ValueError("Require 0 < kfinal <= kmid <= k0")
        if self.k0 > 576:
            raise ValueError("k0 exceeds native 576 CLIP patches")

    def counts(self) -> Tuple[int, ...]:
        return (
            (self.k0,) * self.cut1 +
            (self.kmid,) * (self.cut2 - self.cut1) +
            (self.kfinal,) * (self.layers - self.cut2)
        )

    @property
    def W(self) -> int:
        return sum(self.counts())

    @property
    def average_tokens(self) -> float:
        return self.W / self.layers

    @property
    def visual_work_reduction(self) -> float:
        return 1.0 - self.W / (576 * self.layers)

    def as_dict(self):
        return {
            "layers": self.layers, "k0": self.k0, "cut1": self.cut1,
            "kmid": self.kmid, "cut2": self.cut2, "kfinal": self.kfinal,
            "W": self.W, "average_tokens": self.average_tokens,
            "visual_work_reduction": self.visual_work_reduction,
        }

@dataclass(frozen=True)
class PolicyProfile:
    name: str
    k0: int
    target_W: int
    cut1: int = 4
    cut2: int = 12
    kfinal: int = 32
    layers: int = 32

    def schedule(self) -> Schedule:
        return schedule_for_cut(self, self.cut2)

PROFILES = {
    # Deliberately falsifiable initial policy, NOT a tuned BELA result.
    "easy": PolicyProfile("easy", 96, 1536),
    "normal": PolicyProfile("normal", 160, 2048),
    "hard": PolicyProfile("hard", 172, 2560),
}

def schedule_for_cut(profile: PolicyProfile, cut2: int) -> Schedule:
    """Highest feasible integer kmid at this cut2, without exceeding target_W.

    Exact target_W may be impossible due to integer divisibility. Record W
    actually realized rather than silently claiming nominal budget equality.
    """
    if not (profile.cut1 < cut2 < profile.layers):
        raise ValueError("Invalid second pruning boundary")
    remaining = profile.target_W - profile.cut1 * profile.k0
    remaining -= (profile.layers - cut2) * profile.kfinal
    span = cut2 - profile.cut1
    kmid = min(profile.k0, remaining // span)
    if kmid < profile.kfinal:
        raise ValueError("Requested target_W infeasible for this schedule")
    schedule = Schedule(
        profile.layers, profile.k0, profile.cut1, int(kmid),
        cut2, profile.kfinal
    )
    assert schedule.W <= profile.target_W
    return schedule

def candidate_schedules(profile: PolicyProfile, cuts=(12, 16, 20)):
    out = []
    for cut in cuts:
        try:
            out.append(schedule_for_cut(profile, cut))
        except ValueError:
            continue
    if not out:
        raise ValueError("No feasible schedule candidates")
    return tuple(out)

def verify_monotone_survivors(previous_ids, next_ids, expected):
    """Validate patch-ID nesting at a pruning event, without any GPU operations."""
    old = [int(i) for i in previous_ids]
    new = [int(i) for i in next_ids]
    if len(new) != expected or len(new) != len(set(new)):
        raise ValueError("Wrong survivor count or duplicate patch IDs")
    if len(old) != len(set(old)):
        raise ValueError("Duplicate prior patch IDs")
    if not set(new).issubset(set(old)):
        raise ValueError("Deleted token resurrected or patch ID changed")
    positions = {patch_id: pos for pos, patch_id in enumerate(old)}
    survivor_positions = [positions[i] for i in new]
    if survivor_positions != sorted(survivor_positions):
        raise ValueError("Survivor order changed; decoder position semantics unsafe")
    return True
