#!/usr/bin/env python3
"""BELA V100 budget reference verifier (CPU only).

Primary formula:
    layer_average_tokens = sum_i( segment_layers[i] * keep[i] ) / decoder_layers
    layer_average_reduction = 1 - layer_average_tokens / actual_original_visual_tokens

Run:
    python BELA_V100_BUDGET_ALLOCATOR_20261031.py --self-test
    python BELA_V100_BUDGET_ALLOCATOR_20261031.py --model qwen25_7b --n 1296
    python BELA_V100_BUDGET_ALLOCATOR_20261031.py --model qwen3_vl_8b --n 832

Important: this utility NEVER modifies the model or defines correctness of
position_ids, DeepStack, M-RoPE, crop grouping, or KV / linear attention states.
It supports per-image dynamic N in token arithmetic only.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass

@dataclass(frozen=True)
class Schedule:
    layers: int
    cuts: tuple[int,int] | None
    k: tuple[int,...]
    n: int
    def report(self):
        if not (isinstance(self.n,int) and self.n >= 1):
            raise ValueError("N must be a positive integer")
        if self.cuts is None:
            if len(self.k)!=1:
                raise ValueError("stage1-only requires one keep K")
            counts=(self.layers,)
        else:
            c1,c2=self.cuts
            if not 0<c1<c2<self.layers:
                raise ValueError("invalid pruning layers")
            counts=(c1,c2-c1,self.layers-c2)
            if len(self.k)!=3:
                raise ValueError("progressive requires 3 keep counts")
        if not all(isinstance(q,int) and not isinstance(q,bool) for q in self.k):
            raise ValueError("K values must be integers")
        if not (self.n >= self.k[0] >= self.k[-1] >= 1):
            raise ValueError("invalid K order/range")
        if any(a<b for a,b in zip(self.k,self.k[1:])):
            raise ValueError("keep must be monotone")
        work=sum(a*b for a,b in zip(counts,self.k))
        avg=work/self.layers
        return {"n":self.n, "layers":self.layers, "cuts":self.cuts,
                "keep":self.k, "segments":counts,
                "visual_token_layers":work,
                "average_visual_tokens":avg,
                "average_reduction_pct":100*(1-avg/self.n)}

TEMPLATES = {
    # baseline N, L, pruning layers, k0,k1,k2 (STAR-Pro where noted)
    "llava15_7b":(576,32,(12,20),(256,74,36)),
    "next_7b":(2880,32,(12,20),(1280,367,182)),
    "next_13b":(2880,40,(15,30),(1280,320,160)),
    "qwen25_7b":(1296,28,(10,18),(576,198,72)), # BELA design
    "qwen25_3b":(1296,36,(14,24),(576,144,72)), # BELA design
    "qwen3_vl_8b":(1296,36,(14,24),(576,144,72)), # BELA strict 77.8
    "qwen35_9b":(1024,32,None,(228,)), # Stage1 ONLY
}

def _round_nearest_ratio(n,denom=9,numer=2):
    if n < 1:
        raise ValueError("N must be >=1")
    return max(1,(n*numer*2+denom)//(2*denom))

def solve_scaled(model:str, n:int)->Schedule:
    """Enumerate all monotone K1 possibilities, derive nearest K2, and choose
    lowest layer-budget mismatch, then proximity to proportionally scaled
    reference. This makes ZERO claim of optimal downstream Accuracy.
    """
    if model not in TEMPLATES:
        raise ValueError(f"unknown model: {model}")
    n0,L,cuts,base=TEMPLATES[model]
    if n<1 or isinstance(n,bool):
        raise ValueError("N must be positive int")
    if cuts is None:
        kept=_round_nearest_ratio(n)
        return Schedule(L,None,(kept,),n)
    c1,c2=cuts
    a,b,c=c1,c2-c1,L-c2
    # Keep exact known reference profile when the input corresponds to the paper setup.
    if n==n0:
        return Schedule(L,cuts,base,n)
    T=_round_nearest_ratio(n)
    desired_work=L*T
    k0=min(n, max(1, 2*T))
    scale=T/(sum(x*y for x,y in zip((a,b,c),base))/L)
    t1,t2=base[1]*scale,base[2]*scale
    best=None
    best_score=None
    for k1 in range(1,k0+1):
        raw=(desired_work-a*k0-b*k1)/c
        opts={1,k1,int(raw),int(raw)+1,round(raw),int(t2)}
        for k2 in opts:
            if not isinstance(k2,int) or k2<1 or k2>k1:
                continue
            work=a*k0+b*k1+c*k2
            score=(abs(work-desired_work), abs(k1-t1)+abs(k2-t2),abs(k1-t1),k1)
            if best_score is None or score<best_score:
                best_score=score
                best=(k0,k1,k2)
    assert best is not None
    return Schedule(L,cuts,best,n)

def self_test():
    expectations={
        "llava15_7b":(256,74,36,4096),
        "next_7b":(1280,367,182,20480),
        "next_13b":(1280,320,160,25600),
        "qwen25_7b":(576,198,72,8064),
        "qwen25_3b":(576,144,72,10368),
        "qwen3_vl_8b":(576,144,72,10368),
        "qwen35_9b":(228,None,None,7296)
    }
    for name,(k0,k1,k2,work) in expectations.items():
        n=TEMPLATES[name][0]
        report=solve_scaled(name,n).report()
        assert report["visual_token_layers"]==work,(name,report)
        expected=(k0,) if k1 is None else (k0,k1,k2)
        assert tuple(report["keep"])==expected
        assert abs(report["average_reduction_pct"]-77.77777777777)<0.05
    checked=0
    for name in TEMPLATES:
        for n in (1,2,3,5,8,31,64,128,256,576,1024,1296,2880,4096):
            x=solve_scaled(name,n).report()
            assert x["n"]==n
            assert 1<=min(x["keep"])<=max(x["keep"])<=n
            assert x["visual_token_layers"] == sum(a*b for a,b in zip(x["segments"],x["keep"]))
            checked+=1
    for bad in (0,-1,False):
        try:
            solve_scaled("qwen25_7b",bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid N was accepted: {bad}")
    for invalid in (
        Schedule(32,(12,20),(80,90,30),576),
        Schedule(32,(20,12),(256,74,36),576),
        Schedule(32,(12,20),(256,74,36),200),
        Schedule(32,None,(256,64),576)
    ):
        try: invalid.report()
        except ValueError: pass
        else: raise AssertionError("invalid schedule was accepted")
    print(f"PASS: 7 reference budgets; {checked} dynamic-N checks; invalid-N and schedule boundaries")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--model",choices=sorted(TEMPLATES))
    p.add_argument("--n",type=int)
    args=p.parse_args()
    if args.self_test:
        self_test()
        return
    if args.model is None or args.n is None:
        p.error("--model and --n required unless --self-test")
    print(json.dumps(solve_scaled(args.model,args.n).report(),ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()
