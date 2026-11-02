"""Print pre-registered first-pass screen order, no model launches.

20 available schedules are *not* 20 full-benchmark jobs. Screen at most
3-4 per model/budget on a separate dev panel, then secondary QA and crop
quota isolated tests and ONE final official run on frozen winner.
"""
import argparse
import json
from pathlib import Path

PRIORITY={
 ("next7",160):["A0_PAPER","A2_LATE","A3_LESS_S1","A4_STRONG_LATE","A1_PAPER_VAR"],
 ("next7",320):["A0_PAPER","A2_LATE","A4_STRONG_LATE","A1_PAPER_VAR"],
 ("next13",160):["A0_PAPER","A2_LATE","A4_STRONG_LATE","A1_PAPER_VAR"],
 ("next13",320):["A0_PAPER","A2_LATE","A4_STRONG_LATE","A1_PAPER_VAR"],
}
def priority(cfg, *,take=4):
    out=[]
    for model,budget in PRIORITY:
        for suffix in PRIORITY[model,budget][:take]:
            found=[a for a in cfg["arm_matrix"]
                   if a["model"]==model and a["budget_T"]==budget and
                   a["arm_id"].endswith(suffix)]
            if len(found)!=1: raise ValueError("Cannot resolve experiment")
            out.append(found[0]["arm_id"])
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--max-per-model-budget",type=int,default=4)
    p.add_argument("--config",default=str(Path(__file__).resolve().parents[1]/"configs"/"arms.json"))
    a=p.parse_args()
    if not 1<=a.max_per_model_budget<=4:
        raise SystemExit("Bounded initial screen: choose 1..4 candidates per cell")
    cfg=json.loads(Path(a.config).read_text())
    ids=priority(cfg,take=a.max_per_model_budget)
    print(json.dumps({"status":"PLANNED_ONLY_NO_GPU_LAUNCHED",
                      "screen_order":ids,
                      "note":"Cursor must use actual repo entry, model SHA, valid 4 physical GPUs and preflight"},
                     ensure_ascii=False,indent=2))
if __name__=="__main__":main()
