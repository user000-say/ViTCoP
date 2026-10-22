#!/usr/bin/env python3
"""Validate one JSONL audit per prefill, including patch IDs and measured W.

Row contract:
{sample_id, schedule: {layers,k0,cut1,kmid,cut2,kfinal},
 entry_patch_ids:[...], events:[{post_block,selected_patch_ids:[...]}],
 observed_visual_counts_per_block:[32 values]}

This checks logical ID/count contracts, NOT correct KV cache or model logits.
"""
from pathlib import Path
import json, argparse, sys
from bela_next.budget import Schedule
from bela_next.stage2_adapter import PruningContract

def check(row):
    s=row["schedule"]
    sched=Schedule(*(int(s[k]) for k in ("layers","k0","cut1","kmid","cut2","kfinal")))
    tracker=PruningContract(sched,tuple(row["entry_patch_ids"]))
    ev=row["events"]
    if len(ev)!=2: raise ValueError("Expected exactly two prefill pruning events")
    for e in ev:
        tracker.step_after_block(int(e["post_block"]),e["selected_patch_ids"])
    answer=tracker.validate_end_prefill()
    observed=row.get("observed_visual_counts_per_block")
    if observed is None:
        raise ValueError("Actual prefill count trace missing; don't infer real W")
    if tuple(observed)!=sched.counts():
        raise ValueError("Actual per-block token count differs from schedule")
    answer["observed_W"]=sum(observed)
    answer["sample_id"]=row["sample_id"]
    return answer

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True)
    parser.add_argument("--out")
    args=parser.parse_args()
    rows=[]
    failures=[]
    for i,line in enumerate(Path(args.input).read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():continue
        try: rows.append(check(json.loads(line)))
        except Exception as exc: failures.append({"line":i,"error":str(exc)})
    response={"passed":len(rows),"failed":len(failures),"failures":failures,
              "mean_observed_W":(sum(r["observed_W"] for r in rows)/len(rows)
                                 if rows else None)}
    result=json.dumps(response,ensure_ascii=False,indent=2)
    if args.out:Path(args.out).write_text(result,encoding="utf-8")
    print(result)
    if failures:sys.exit(1)

if __name__=="__main__":main()
