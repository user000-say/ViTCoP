"""An OFF-by-default adaptive W policy, calibrated without accuracy labels.

This is a falsifiable RESEARCH PROPOSAL, not a validated improvement.
Calibrate thresholds on an unlabeled, image-disjoint DEV pool. Freeze JSON
before inference on validation/test. Mean W is NOT guaranteed on new data;
the realized W must be audited and reported.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import json
import math
import numpy as np

from .budget import PROFILES, Schedule, candidate_schedules

@dataclass(frozen=True)
class QuantilePolicy:
    lower: float
    upper: float
    source_split: str = "dev"
    model_revision: str = "unspecified"
    signal: str = "cosine_feature_coverage_residual"

    @classmethod
    def calibrate(cls, difficulties, source_split="dev",
                  model_revision="unspecified", lower_q=.25, upper_q=.75):
        if source_split.lower() not in ("dev","calibration","unlabeled_dev"):
            raise ValueError("No calibration on hidden test or scored benchmark")
        x = np.asarray(difficulties, dtype=float)
        if x.ndim != 1 or len(x) < 4 or not np.isfinite(x).all():
            raise ValueError("At least four finite DEV difficulty measurements required")
        if not 0 <= lower_q < upper_q <= 1:
            raise ValueError("Quantile parameters must satisfy 0<=lo<hi<=1")
        lo, hi = np.quantile(x, [lower_q,upper_q])
        return cls(float(lo),float(hi),source_split,model_revision)

    def __post_init__(self):
        if (not math.isfinite(self.lower) or not math.isfinite(self.upper)
                or self.lower > self.upper):
            raise ValueError("Invalid difficulty thresholds")

    def choose_profile(self, difficulty: float) -> str:
        if not math.isfinite(difficulty):
            raise ValueError("Non-finite difficulty")
        if self.lower == self.upper:  # no discriminative input signal
            return "normal"
        if difficulty < self.lower: return "easy"
        if difficulty > self.upper: return "hard"
        return "normal"

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(self.__dict__, fp, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path,encoding="utf-8") as fp:
            obj=json.load(fp)
        if set(obj) != set(cls.__dataclass_fields__):
            raise ValueError("Unexpected policy schema")
        return cls(**obj)

@dataclass(frozen=True)
class Plan:
    profile: str
    schedule: Schedule
    difficulty: float
    stage2_mode: str
    drift: Optional[float] = None
    reason: str = ""

    def as_dict(self):
        return {"profile":self.profile, "difficulty":self.difficulty,
                "stage2_mode":self.stage2_mode, "drift":self.drift,
                "reason":self.reason, "schedule":self.schedule.as_dict()}

def plan_for_image(policy: QuantilePolicy, difficulty:float,
                   stage2_mode="fixed", early_drift:Optional[float]=None) -> Plan:
    """Return schedule following an already decided Stage1 entry size.

    stage2_mode=adaptive is EXPLORATORY. It is not enabled for the main BELA
    until paired accuracy and latency tests beat fixed at matched actual W.
    All alternatives are under the same PER-IMAGE nominal W.
    """
    profile=policy.choose_profile(difficulty)
    cfg=PROFILES[profile]
    if stage2_mode == "fixed":
        sched=cfg.schedule()
        return Plan(profile,sched,difficulty,"fixed",None,"frozen_second_cut12")
    if stage2_mode != "adaptive":
        raise ValueError("stage2_mode must be fixed or adaptive")
    if early_drift is None or not math.isfinite(early_drift) or not 0<=early_drift<=1:
        raise ValueError("Need audited early-stage same-ID drift in [0,1]")
    # Pre-registered, deliberately simple heuristic: HIGH rank-turnover keeps a
    # wider candidate pool until cut12; LOW turnover keeps a narrower pool longer.
    # Its thresholds are hypotheses, not learned constants or published gains.
    # CUT=12 and CUT=20 are BOTH exactly target_W for each frozen profile.
    # CUT=16 usually falls a few token-layers below target due to divisibility;
    # therefore exclude it from the causally clean online comparison.
    chosen_cut = 12 if early_drift >= 0.40 else 20
    opts={s.cut2:s for s in candidate_schedules(cfg, cuts=(12,20))}
    sched=opts[chosen_cut]
    if sched.W != cfg.target_W:
        raise RuntimeError("Dynamic Stage2 comparison must match exact per-image W")
    return Plan(profile,sched,difficulty,"adaptive",float(early_drift),
                "unvalidated_early_rank_turnover")

def audit_plans(plans):
    plans=list(plans)
    if not plans:
        raise ValueError("Need >=1 realized plan")
    counts={p:sum(q.profile==p for q in plans) for p in PROFILES}
    realized=float(np.mean([p.schedule.W for p in plans]))
    return {"n":len(plans),"profile_counts":counts,
            "mean_realized_W":realized, "min_W":min(p.schedule.W for p in plans),
            "max_W":max(p.schedule.W for p in plans),
            "target_mean_W_not_guaranteed":2048}
