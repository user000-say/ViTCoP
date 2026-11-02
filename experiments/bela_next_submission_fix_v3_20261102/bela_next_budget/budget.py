"""Integer layer-average visual PATCH token budget verification.

Do NOT mistake a paper's *final 160 tokens* for a 160-token layer average.
Reference 2880 means five times 576 *patch* tokens and excludes special
newlines/padding/text. Actual AnyRes N must be observed after processing.
This module never modifies a model.
"""
from dataclasses import dataclass
import math
from typing import Optional

@dataclass(frozen=True)
class BudgetSchedule:
    name: str
    layers: int
    cuts: tuple[int, int]
    k0: int
    k1: int
    k2: int
    reference_n: int = 2880
    target_t: Optional[int] = None

    @property
    def w(self):
        a,b = self.cuts
        return a*self.k0+(b-a)*self.k1+(self.layers-b)*self.k2

    @property
    def avg(self):
        return self.w/self.layers

    def validate(self, exact=True):
        a,b=self.cuts
        if not (self.layers in (32,40) and 0<a<b<self.layers):
            raise ValueError("Architecture or cut boundary incompatible")
        if not (1<=self.k2<=self.k1<=self.k0<=self.reference_n):
            raise ValueError("Invalid monotonically nested budgets")
        if exact and self.target_t is not None and self.w != self.layers*self.target_t:
            raise ValueError(f"Unexpected W={self.w}, expected {self.layers*self.target_t}")
        return self

    def as_dict(self):
        return dict(name=self.name,layers=self.layers,cuts=list(self.cuts),
                    k0=self.k0,k1=self.k1,k2=self.k2,patch_W=self.w,
                    achieved_avg=self.avg,ref_n=self.reference_n,
                    nominal_target_t=self.target_t)

def source_schedule(arm: dict) -> BudgetSchedule:
    x=BudgetSchedule(arm["arm_id"],arm["layers"],tuple(arm["cuts_after_layers"]),
                     *arm["stage_k"],reference_n=arm["reference_patch_N"],
                     target_t=arm["budget_T"])
    return x.validate()

def _nearest_multiple(num: float, step: int) -> int:
    # avoid banker's rounding for a fixed positive budget
    return max(step, int(math.floor(num / step + 0.5))*step)

def dynamic_ratio_plan(src: BudgetSchedule, actual_n: int, *, tolerance=32):
    """An exploratory per-image matched-ratio schedule for native dynamic AnyRes.

    Scale by actual PATCH count N/2880 (not by full sequence incl. newlines).
    Exactly reproduce the paper fixed five-crop schedule if N=2880.
    Otherwise solve integer counts for W nearest gcd(cut-lengths) to target.
    Achieved (and requested) W are emitted for honest reporting. If no feasible
    schedule is found fail CLOSED, do not change compression silently.

    For strict paper-repro experiments use reference five-crop input instead.
    """
    src.validate()
    if not isinstance(actual_n,int) or actual_n<64:
        raise ValueError("actual_n must be integer >=64; inspect model crop groups")
    if actual_n==src.reference_n:
        return {**src.as_dict(), "requested_patch_W":src.w,
                "reference_equivalent":True, "observed_actual_n":actual_n,
                "ideal_avg":src.avg}
    a,b=src.cuts
    c=[a,b-a,src.layers-b]
    gcd=math.gcd(*c)
    ideal_t=src.target_t*actual_n/src.reference_n
    ideal_W=ideal_t*src.layers
    W_req=_nearest_multiple(ideal_W,gcd)
    scales=actual_n/src.reference_n
    desired=[src.k0*scales,src.k1*scales,src.k2*scales]
    best=None
    # Search near proportionally scaled original counts, not a different method.
    for delta_w in (0,-gcd,gcd,-2*gcd,2*gcd,-3*gcd,3*gcd):
        W=W_req+delta_w
        if W<=0: continue
        k0_ctr=round(desired[0])
        for k0 in range(max(1,k0_ctr-tolerance),min(actual_n,k0_ctr+tolerance)+1):
            k1_ctr=round(desired[1])
            for k1 in range(max(1,k1_ctr-tolerance),min(k0,k1_ctr+tolerance)+1):
                residual=W-c[0]*k0-c[1]*k1
                if residual%c[2]: continue
                k2=residual//c[2]
                if not (1<=k2<=k1): continue
                rel=((k0-desired[0])**2+(k1-desired[1])**2+
                     (k2-desired[2])**2)
                loss=(abs(W-ideal_W)/max(1,src.layers))*100+rel
                cand=(loss,k0,k1,k2,W)
                if best is None or cand<best: best=cand
        if best is not None and delta_w==0: break
    if best is None:
        raise RuntimeError("No valid integer matched-ratio schedule near source; fail CLOSED")
    _,k0,k1,k2,W=best
    candidate=BudgetSchedule(src.name+f"_N{actual_n}",src.layers,src.cuts,
                              k0,k1,k2,reference_n=actual_n)
    candidate.validate(exact=False)
    return {**candidate.as_dict(),"observed_actual_n":actual_n,
            "ideal_avg":ideal_t,"requested_patch_W":W_req,
            "ideal_patch_W":ideal_W,"reference_equivalent":False,
            "ratio_error_pp":100*(candidate.avg-ideal_t)/actual_n,
            "unrounded_reference_W":src.w}
