"""Validate token-ID cardinality and monotonicity in model-produced JSONL traces.

Does not prove correctness of position IDs / RoPE / KV / DeepStack implementation.
"""
import argparse,json
from pathlib import Path
from tools.budget_planner import compute_budget

def verify_one(trace:dict):
    model=trace["model"]
    tier=trace["budget"]
    n=int(trace["observed_n"])
    rec=compute_budget(model,tier,n)
    ids=trace["entry_ids"]
    stage_lists=[ids]+trace.get("post_cut_ids",[])
    if len(stage_lists)!=len(rec.counts):
        raise ValueError(f"expected {len(rec.counts)} token ID sets, got {len(stage_lists)}")
    previous=set(range(n))
    for i,(chosen,k) in enumerate(zip(stage_lists,rec.counts)):
        if not isinstance(chosen,list) or len(chosen)!=k:
            raise ValueError(f"stage {i}: expected {k}, got {len(chosen)}")
        if any(type(x)!=int or x<0 or x>=n for x in chosen):
            raise ValueError("Invalid original post-merge visual patch id")
        if len(set(chosen))!=len(chosen):
            raise ValueError("Duplicate token IDs")
        if not set(chosen).issubset(previous):
            raise ValueError("Token resurrection or mismatched original IDs")
        previous=set(chosen)
    return {"model":model,"budget":tier,"observed_n":n,"W":rec.W,
            "checked_stage_K":list(rec.counts),
            "warning":"Patch IDs validated; RoPE/DeepStack/cache MUST be separately verified"}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--trace",required=True,help="JSON Lines with original visual IDs per pruning stage")
    a=p.parse_args()
    successes=0
    with open(a.trace,encoding="utf-8") as f:
        for line in f:
            if not line.strip():continue
            verify_one(json.loads(line))
            successes+=1
    print(json.dumps({"validated":successes,"scope":"IDs/cardinality only"},indent=2))


if __name__=="__main__":
    main()
