"""Strict decoder visual-token budget auditor. No model/GPU imports."""
import json
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Schedule:
    name: str
    stages: tuple
    def check(self):
        if not isinstance(self.name,str) or not self.name:
            raise ValueError("empty schedule name")
        if not isinstance(self.stages,(list,tuple)) or not self.stages:
            raise ValueError("empty stages")
        if any(len(s)!=2 or any(type(x)!=int or x<=0 for x in s) for s in self.stages):
            raise ValueError("each stage must be [positive integer layer_count, token_count]")
        if sum(s[0] for s in self.stages)!=32:
            raise ValueError("LLaVA-1.5-7B decoder requires 32 layers")
        if any(self.stages[i+1][1]>self.stages[i][1] for i in range(len(self.stages)-1)):
            raise ValueError("visual token counts cannot increase after hard pruning")
        return self
    @property
    def counts(self):
        self.check()
        return [int(k) for n,k in self.stages for _ in range(n)]
    @property
    def W(self):
        return sum(self.counts)
    @property
    def Q_visual(self):
        return sum(v*v for v in self.counts)
    @property
    def cuts_after(self):
        self.check()
        cuts=[]
        pos=0
        for n,k in self.stages[:-1]:
            pos+=n
            cuts.append(pos)
        return cuts
    def audit(self):
        return dict(name=self.name,W=self.W,meanK=self.W/32,
                    Q_visual=self.Q_visual,
                    stages=[list(x) for x in self.stages],
                    cuts_after=self.cuts_after)

def load(path):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    output=[]
    for e in data["arms"]:
        sc=Schedule(e["id"],tuple(tuple(t) for t in e["stages"]))
        sc.check()
        expected=e.get("W_target")
        if expected is not None and sc.W!=expected:
            raise ValueError(f"{e['id']}: got W={sc.W}, expected {expected}")
        output.append((e,sc))
    if len(set(sc.name for _,sc in output))!=len(output):
        raise ValueError("duplicate schedule IDs")
    return output

def validate_trace(row, expected):
    """Read-only runtime trace; actual counts and original patch IDs."""
    expected.check()
    counts=row["active_visual_tokens_by_decoder_layer"]
    if counts!=expected.counts:
        raise ValueError("runtime counts differ from schedule (prefill only)")
    events=row["prune_events"]
    if len(events)!=len(expected.cuts_after):
        raise ValueError("wrong number of pruning events")
    prior=None
    for index,(evt,cut) in enumerate(zip(events,expected.cuts_after)):
        if evt.get("after_layer")!=cut:
            raise ValueError(f"bad cut layer for event {index}")
        before=evt.get("original_patch_ids_before")
        after=evt.get("original_patch_ids_after")
        if not isinstance(before,list) or not isinstance(after,list):
            raise ValueError("missing patch ID sets")
        if any(type(i)!=int or i<0 for i in before+after):
            raise ValueError("invalid patch IDs")
        if len(set(before))!=len(before) or len(set(after))!=len(after):
            raise ValueError("duplicate patch IDs")
        bcount=expected.counts[cut-1]
        acount=expected.counts[cut] if cut<32 else None
        if acount is None or len(before)!=bcount or len(after)!=acount:
            raise ValueError("observed cut sizes do not match actual schedule")
        if not set(after).issubset(before):
            raise ValueError("pruning reintroduced absent patch IDs")
        if prior is not None and set(before)!=set(prior):
            raise ValueError("IDs changed between pruning boundaries")
        if not evt.get("score_function_path") or not evt.get("callsite"):
            raise ValueError("missing real selector provenance")
        prior=after
    return True

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("file",nargs="?",default="config/experiments.json")
    args=p.parse_args()
    for desc,sc in load(args.file):
        print(f"{sc.name:24s} {sc.W:5d} avgK={sc.W/32:5.1f} "
              f"Q={sc.Q_visual:7d} cuts={sc.cuts_after} "
              f"phase={desc.get('phase')}")
