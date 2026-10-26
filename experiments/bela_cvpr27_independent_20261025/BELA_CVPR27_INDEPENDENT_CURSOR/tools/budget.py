"""Strict visual token-layer workload and decoder event validator. CPU-only."""
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class Schedule:
    name:str
    stages:Tuple[Tuple[int,int], ...]
    def validate(self):
        if not self.name or not self.stages:raise ValueError("missing schedule")
        if any(len(s)!=2 or any(type(v)!=int or v<=0 for v in s) for s in self.stages):
            raise ValueError("stages must be positive integer (layers,K)")
        if sum(n for n,k in self.stages)!=32:raise ValueError("LLaVA-1.5 7B needs 32 decoder blocks")
        if any(self.stages[i+1][1]>self.stages[i][1] for i in range(len(self.stages)-1)):
            raise ValueError("pruned visual IDs cannot be reintroduced")
        return self
    @property
    def W(self):
        self.validate()
        return sum(n*k for n,k in self.stages)
    @property
    def Q(self):
        self.validate()
        return sum(n*k*k for n,k in self.stages)
    @property
    def cuts_after(self):
        self.validate()
        out=[];c=0
        for n,k in self.stages[:-1]:
            c+=n;out.append(c)
        return out
    @property
    def active_counts(self):
        self.validate()
        return [k for n,k in self.stages for _ in range(n)]

def read_schedule_file(path):
    import json
    from pathlib import Path
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    res={}
    for item in data["runs"]:
        x=Schedule(item["arm"],tuple(tuple(s) for s in item["stages"])).validate()
        if x.name in res:raise ValueError("duplicate arm name")
        if x.W!=item["W"]:raise ValueError(f"{x.name}: declared W={item['W']} actual={x.W}")
        res[x.name]=(x,item)
    return res

def validate_trace(sc,trace):
    """Validate original Patch IDs in a **prefill-only**, batch-1 trace.
    Does not establish whether KV or mask semantics are correct.
    """
    sc.validate()
    if trace.get("active_visual_by_layer")!=sc.active_counts:
        raise ValueError("runtime K differs from declared schedule")
    cuts=trace.get("cuts")
    if not isinstance(cuts,list) or len(cuts)!=len(sc.cuts_after):
        raise ValueError("cut count differs")
    prior=None
    for i,(event,after) in enumerate(zip(cuts,sc.cuts_after)):
        before=event.get("original_before"); kept=event.get("original_after")
        if event.get("after_decoder_layer")!=after:raise ValueError("cut at wrong decoder layer")
        if not isinstance(before,list) or not isinstance(kept,list):raise ValueError("missing IDs")
        if any(type(x)!=int or not 0<=x<576 for x in before+kept):
            raise ValueError("bad image Patch IDs")
        if len(set(before))!=len(before) or len(set(kept))!=len(kept):
            raise ValueError("duplicate Patch IDs")
        if len(before)!=sc.active_counts[after-1] or len(kept)!=sc.active_counts[after]:
            raise ValueError("unexpected candidate counts")
        if not set(kept).issubset(before):
            raise ValueError("reintroduced a pruned visual Patch")
        if prior is not None and set(before)!=set(prior):
            raise ValueError("patch identities changed between successive cuts")
        if not event.get("selector_callsite"):raise ValueError("actual selector function missing")
        prior=kept
    return True

if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser();p.add_argument("file",nargs="?",default="configs/experiment_matrix.json")
 args=p.parse_args()
 for sc,item in read_schedule_file(args.file).values():
     print(f"{sc.name:30} W={sc.W:4} Q={sc.Q:7} cuts={sc.cuts_after} status={item['status']} gate={item['gate']}")
