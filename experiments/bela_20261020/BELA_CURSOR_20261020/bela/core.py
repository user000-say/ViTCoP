"""BELA: budget-exact, model-agnostic visual evidence lifetime core.

Design: no merging, no hidden modification, no model-specific KV/position logic.
A boundary b is AFTER decoder block b (1-indexed). The first b blocks
process entry_count tokens; the next blocks process the reduced count.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence
import math
import numpy as np

@dataclass(frozen=True)
class Event:
    after_block: int
    keep: int

@dataclass(frozen=True)
class Plan:
    blocks: int
    entry: int
    events: tuple[Event, ...]
    def __post_init__(self):
        if not isinstance(self.blocks,int) or self.blocks <= 0:
            raise ValueError("blocks must be positive integer")
        if not isinstance(self.entry,int) or self.entry <= 0:
            raise ValueError("entry must be positive integer")
        prev_b, prev_k = 0, self.entry
        for event in self.events:
            if not isinstance(event.after_block,int) or not (prev_b < event.after_block < self.blocks):
                raise ValueError("events must be strictly increasing inside decoder")
            if not isinstance(event.keep,int) or not (0 < event.keep <= prev_k):
                raise ValueError("keep must be positive and nonincreasing")
            prev_b,prev_k = event.after_block,event.keep
    def layer_counts(self) -> tuple[int,...]:
        counts=[]
        k=self.entry
        lookup={e.after_block:e.keep for e in self.events}
        for block in range(1,self.blocks+1):
            counts.append(k)
            if block in lookup:
                k=lookup[block]
        return tuple(counts)
    @property
    def work(self) -> int:
        return sum(self.layer_counts())
    def to_dict(self):
        return {"blocks":self.blocks,"entry":self.entry,
                "events":[{"after_block":e.after_block,"keep":e.keep} for e in self.events],
                "work":self.work, "counts":list(self.layer_counts())}

def find_exact_plans(*,blocks:int,entry:int,target:int,boundaries:Sequence[int],
                     allowed_counts:Sequence[int],final_keep:int|None=None,
                     limit:int=30)->list[Plan]:
    """Enumerate exact-budget monotone plans. All listed boundaries must prune.

    Finite candidate search only; a no-solution result is explicit, not rounded.
    """
    bs=tuple(boundaries)
    ks=sorted(set(int(v) for v in allowed_counts),reverse=True)
    if sorted(bs)!=list(bs) or len(set(bs))!=len(bs) or any(b<=0 or b>=blocks for b in bs):
        raise ValueError("invalid boundaries")
    if target<=0 or limit<=0: raise ValueError("invalid target/limit")
    out=[]
    def rec(i,previous,events):
        if len(out)>=limit: return
        if i==len(bs):
            p=Plan(blocks,entry,tuple(events))
            if p.work==target and (final_keep is None or (events[-1].keep if events else entry)==final_keep):
                out.append(p)
            return
        for k in ks:
            if k>=previous: continue
            if i==len(bs)-1 and final_keep is not None and k!=final_keep: continue
            rec(i+1,k,events+[Event(bs[i],k)])
    rec(0,entry,[])
    return out

def stable_keep(scores:np.ndarray,patch_ids:np.ndarray,k:int,forced_ids:Sequence[int]=())->np.ndarray:
    """Select active slot indices; tie break by stable ORIGINAL patch ID."""
    s=np.asarray(scores,dtype=np.float64)
    ids=np.asarray(patch_ids)
    if s.ndim!=1 or ids.ndim!=1 or len(s)!=len(ids) or not np.all(np.isfinite(s)):
        raise ValueError("scores and ids must be finite aligned vectors")
    if len(np.unique(ids))!=len(ids): raise ValueError("duplicated original patch ids")
    if not isinstance(k,int) or not 0<k<=len(ids): raise ValueError("invalid k")
    forced=set(map(int,forced_ids))
    pos={int(x):i for i,x in enumerate(ids)}
    if not forced.issubset(pos) or len(forced)>k: raise ValueError("invalid forced IDs")
    first=sorted((pos[x] for x in forced),key=lambda i:int(ids[i]))
    candidates=(i for i in range(len(ids)) if int(ids[i]) not in forced)
    ranked=sorted(candidates,key=lambda i:(-s[i],int(ids[i])))
    return np.array(first+ranked[:k-len(first)],dtype=np.int64)

def lifetime_scores(*,current:Sequence[float],older:Sequence[float]|None=None,
                    prior:Sequence[float]|None=None,coverage:Sequence[float]|None=None,
                    weight_current:float=0.6, weight_drift:float=0.4,
                    weight_prior:float=0.,weight_coverage:float=0.)->np.ndarray:
    """Transparent ranking surrogate; NOT a learned future-utility oracle.

    The historical RDRIFT arm uses current_weight=0.6 and ranks,
    but the exact model implementation must be matched in Cursor.
    """
    x=np.asarray(current,dtype=np.float64)
    if x.ndim!=1 or not np.isfinite(x).all(): raise ValueError("invalid current signal")
    out=weight_current*x.copy()
    for arr,weight,term in ((older,weight_drift,"drift"),
                            (prior,weight_prior,"prior"),(coverage,weight_coverage,"coverage")):
        if arr is None:
            if weight and term!="drift": raise ValueError(f"missing {term}")
            if weight and term=="drift": raise ValueError("missing older for drift")
            continue
        a=np.asarray(arr,dtype=np.float64)
        if a.shape!=x.shape or not np.isfinite(a).all(): raise ValueError("invalid aligned signal")
        out += weight*(x-a if term=="drift" else a)
    if not np.isfinite(out).all(): raise ValueError("nonfinite score")
    return out

class LifetimeState:
    """Records visual ORIGINAL ids and protects budget accounting.

    Caller MUST apply pruning physically to model hidden/attention masks/KV, and
    MUST call record_block on each processed decoder block before event.
    """
    def __init__(self,plan:Plan,patch_ids:Sequence[int]):
        ids=np.asarray(patch_ids,dtype=np.int64)
        if ids.ndim!=1 or len(ids)!=plan.entry or len(set(map(int,ids)))!=len(ids):
            raise ValueError("entry IDs mismatch")
        self.plan=plan
        self.ids=ids.copy()
        self.block=0
        self.counts=[]
        self.done=set()
    def record_block(self,block:int,observed_visual_tokens:int):
        if block!=self.block+1 or block>self.plan.blocks:
            raise ValueError("block order mismatch")
        if observed_visual_tokens!=len(self.ids):
            raise RuntimeError("actual decoder visual token count mismatch")
        self.block=block
        self.counts.append(observed_visual_tokens)
    def prune(self,after_block:int,score:Sequence[float],forced_ids:Sequence[int]=()):
        events={e.after_block:e.keep for e in self.plan.events}
        if after_block!=self.block or after_block not in events or after_block in self.done:
            raise ValueError("pruning at wrong/duplicate boundary")
        slots=stable_keep(np.asarray(score),self.ids,events[after_block],forced_ids)
        self.ids=self.ids[slots]
        self.done.add(after_block)
        return slots.copy(),self.ids.copy()
    def finish(self)->dict:
        if self.block!=self.plan.blocks or len(self.done)!=len(self.plan.events):
            raise RuntimeError("incomplete trace")
        if tuple(self.counts)!=self.plan.layer_counts():
            raise RuntimeError("trace doesn't match plan")
        if sum(self.counts)!=self.plan.work:
            raise RuntimeError("work mismatch")
        return {"work":sum(self.counts),"counts":self.counts.copy(),
                "final_ids":self.ids.tolist(),"events_applied":len(self.done)}

def equal_budget_swap(ids:Sequence[int],base_score:Sequence[float],protected_final:Sequence[int],
                      group_a:Sequence[int],group_b:Sequence[int],interim_keep:int):
    """Generate two interim survival interventions, same final set and cardinality.

    Caller must run BOTH through the actual VLM and compare paired answers.
    """
    allids=list(map(int,ids))
    protected=set(map(int,protected_final))
    a=set(map(int,group_a)); b=set(map(int,group_b))
    if any(len(x)!=len(set(x)) for x in (allids,list(group_a),list(group_b))):
        raise ValueError("duplicates")
    if not a or len(a)!=len(b) or a&b or a&protected or b&protected:
        raise ValueError("groups must be equally sized, disjoint from protected")
    if not (protected|a|b).issubset(allids) or interim_keep<len(protected)+len(a):
        raise ValueError("invalid membership/capacity")
    eligible=set(allids)-(a|b)
    scores=np.asarray(base_score,dtype=float)
    if scores.shape!=(len(allids),): raise ValueError("invalid scores")
    def branch(group,excluded):
        allowed=np.array([i for i,x in enumerate(allids) if x not in excluded],dtype=int)
        local=stable_keep(scores[allowed],np.array(allids)[allowed],
                          interim_keep,tuple(sorted(protected|group)))
        return np.array(allids)[allowed][local].tolist()
    left=branch(a,b)
    right=branch(b,a)
    if len(left)!=len(right) or not protected.issubset(left) or not protected.issubset(right):
        raise RuntimeError("invalid survival comparison")
    return {"A_interim_ids":left,"B_interim_ids":right,
            "final_ids":sorted(protected),
            "interim_count":interim_keep}
