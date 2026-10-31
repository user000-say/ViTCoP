#!/usr/bin/env python3
"""BELA reference: batched *exact* coverage-delta calculation for one donor/drop pair.

Assumption (must be verified against the real qa_core.py before use):
    C(S)=mean_i max_{j in S} sim(i,j).
The implementation vectorizes F frames and every pair of proposed donor/drop IDs.
It is a reference for one expensive sub-operation, NOT a drop-in implementation
of the entire BELA QA selector and NOT a tested A800 CUDA kernel.

Production constraints:
- preserve the exact production similarity fusion, thresholds, donor/drop pools,
  tie breaking, and per-frame selection order;
- do not perform debug tensor-to-Python assertions within CUDA hot loops;
- use fp32 for acceptance margins near thresholds if production does;
- never call .cpu() or .item() inside the selection step.
"""
from __future__ import annotations
import argparse
import torch

def batched_coverage_delta(
    sim: torch.Tensor,
    keep: torch.Tensor,
    donors: torch.Tensor,
    drops: torch.Tensor,
    *,
    validate: bool = False,
) -> torch.Tensor:
    """Returns ΔC as [frames, n_drop, n_donor].

    Args:
        sim: [F,N,N], i->j similarities (usually cosine); CUDA tensors accepted.
        keep: [F,K], unique original patch IDs already selected.
        donors: [F,D], unique original patch IDs not selected.
        drops: [F,R], unique IDs selected and eligible to drop.

    Key invariant: removing a single kept representative changes an item's best
    similarity only if that representative attained its maximum. Therefore the
    first two maxima are sufficient to compute every one-for-one exchange.
    """
    if sim.ndim != 3 or sim.shape[1] != sim.shape[2]:
        raise ValueError("sim must have shape [F,N,N]")
    if keep.ndim != 2 or donors.ndim != 2 or drops.ndim != 2:
        raise ValueError("keep, donors, drops must have shape [F,K/D/R]")
    F,N,_=sim.shape
    if any(t.shape[0]!=F for t in (keep,donors,drops)):
        raise ValueError("frame dimension mismatch")
    if min(N,keep.shape[1],donors.shape[1],drops.shape[1])<1:
        raise ValueError("nonempty N, K, donor and drop sets required")
    if sim.device != keep.device or sim.device != donors.device or sim.device != drops.device:
        raise ValueError("all tensors must share a device")
    if any(t.dtype != torch.long for t in (keep,donors,drops)):
        raise ValueError("index tensors must be torch.long")
    if not sim.is_floating_point():
        raise ValueError("sim must be floating point")

    if validate:
        # Validation is CPU/GPU synchronization and is FOR TESTS ONLY.
        for t in (keep,donors,drops):
            if bool(((t<0)|(t>=N)).any()):
                raise ValueError("index out of range")
            if bool((t.sort(dim=-1).values[:,1:] ==
                     t.sort(dim=-1).values[:,:-1]).any()):
                raise ValueError("duplicate indices within a set")
        if not bool((keep[:,:,None]==drops[:,None,:]).any(dim=1).all()):
            raise ValueError("drop must be in keep")
        if bool((keep[:,:,None]==donors[:,None,:]).any()):
            raise ValueError("donor cannot be in keep")

    K,D,R=keep.shape[1],donors.shape[1],drops.shape[1]
    selected_sim=sim.gather(2,keep[:,None,:].expand(F,N,K))
    largest=selected_sim.topk(k=min(K,2),dim=-1)
    best=largest.values[:,:,0]                    # [F,N]
    best_pos=largest.indices[:,:,0]               # [F,N], position in keep
    if K==1:
        second=best.new_full((F,N),-float("inf"))
    else:
        second=largest.values[:,:,1]

    # The validation guard guarantees drops appear exactly once in keep.
    drop_pos=(keep[:,:,None] == drops[:,None,:]).to(torch.int64).argmax(dim=1) # [F,R]
    remaining=torch.where(
        best_pos[:,:,None]==drop_pos[:,None,:],
        second[:,:,None],best[:,:,None]
    )                                              # [F,N,R]
    new_sim=sim.gather(2,donors[:,None,:].expand(F,N,D))
    new_best=torch.maximum(remaining[:,:,:,None],new_sim[:,:,None,:]) # [F,N,R,D]
    return new_best.mean(dim=1)-best.mean(dim=1)[:,None,None]          # [F,R,D]

def brute_force(sim, keep, donors, drops):
    F,N,_=sim.shape
    out=sim.new_empty((F,drops.shape[1],donors.shape[1]))
    for f in range(F):
        prev=sim[f][:,keep[f]].max(dim=1).values.mean()
        for r in range(drops.shape[1]):
            for d in range(donors.shape[1]):
                ids=keep[f].tolist()
                ids.remove(int(drops[f,r]))
                ids.append(int(donors[f,d]))
                out[f,r,d]=sim[f][:,ids].max(dim=1).values.mean()-prev
    return out

def case(F,N,K,D,R, seed, tie=False):
    g=torch.Generator().manual_seed(seed)
    x=torch.randn((F,N,N),generator=g)
    if tie:
        x=(x*2).round()/2
    keep=[]; donors=[]; drops=[]
    for f in range(F):
        ids=torch.randperm(N,generator=g)
        k=ids[:K]; d=ids[K:K+D]; r=k[torch.randperm(K,generator=g)[:R]]
        keep.append(k); donors.append(d); drops.append(r)
    keep=torch.stack(keep); donors=torch.stack(donors); drops=torch.stack(drops)
    fast=batched_coverage_delta(x,keep,donors,drops,validate=True)
    slow=brute_force(x,keep,donors,drops)
    if not torch.allclose(fast,slow,rtol=1e-4,atol=2e-6):
        raise AssertionError(f"Vectorized/reference mismatch: {abs(fast-slow).max().item()}")
    return F*R*D

def self_test():
    comparisons=0
    for args in (
        (1,8,1,2,1,1,False),      # single selected representative
        (1,8,3,2,2,2,True),       # tied similarities
        (3,16,6,4,3,3,False),
        (8,32,8,4,4,4,True),
        (64,169,64,2,2,5,False), # realistic 64-frame dimension
    ):
        comparisons += case(*args)
    x=torch.randn((1,6,6))
    k=torch.tensor([[0,1]],dtype=torch.long)
    d=torch.tensor([[4]],dtype=torch.long)
    r=torch.tensor([[1]],dtype=torch.long)
    invalid=[
        (x,k,d,torch.tensor([[2]],dtype=torch.long)), # invalid drop
        (x,k,torch.tensor([[1]],dtype=torch.long),r), # donor selected
        (x,torch.tensor([[0,0]],dtype=torch.long),d,r), # duplicate
        (x,k,torch.tensor([[6]],dtype=torch.long),r), # out-of-range
        (x,k,d,torch.empty((1,0),dtype=torch.long)), # empty
        (x,k,d.float(),r),                          # dtype
    ]
    for fields in invalid:
        try: batched_coverage_delta(*fields,validate=True)
        except ValueError: pass
        else: raise AssertionError("invalid input was accepted")
    print(f"PASS: {comparisons} brute-force pair comparisons; F=1/3/8/64, K=1, ties, 6 invalid inputs")
    print("Scope: exact per-swap coverage delta under facility-location assumption; no claims about production QA or A800 speed.")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:
        self_test()
    else:
        print(__doc__)
if __name__=="__main__":
    main()
