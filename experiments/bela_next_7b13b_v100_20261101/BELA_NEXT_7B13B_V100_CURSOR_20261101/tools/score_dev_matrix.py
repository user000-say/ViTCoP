"""Score an INDEPENDENT developer panel; never choose on official test data.

Each row e.g.
{"arm_id":"NEXT7_T160_A0_PAPER","task":"GQA",
 "score":0.52, "dense_score":0.59,"n":100,"split":"DEV_NON_OVERLAP_V1"}
Group by model and T separately. Reject mixed dense scores within groups.
Use scorer-provided task metrics, NO MM-E raw-point averaging with accuracies.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

def rank_rows(rows):
    seen=set(); data=defaultdict(lambda:defaultdict(list))
    for x in rows:
        fields=["arm_id","model","budget_T","task","score","dense_score","n","split"]
        if any(f not in x for f in fields):raise ValueError("Missing required field")
        if "TEST" in x["split"].upper() or "FINAL" in x["split"].upper():
            raise ValueError("Refusing to tune on TEST / FINAL benchmark split")
        if x["dense_score"]<=0 or x["n"]<=0:
            raise ValueError("Invalid dense baseline or n")
        key=(x["arm_id"],x["task"],x["split"])
        if key in seen:raise ValueError("Duplicate arm-task-split")
        seen.add(key)
        group=(x["model"],x["budget_T"],x["split"])
        data[group][x["arm_id"]].append(x)
    outputs={}
    for group,armgroup in data.items():
        dense_by_task={}
        for recs in armgroup.values():
            for row in recs:
                key=row["task"]
                if key in dense_by_task and abs(dense_by_task[key]-row["dense_score"])>1e-9:
                    raise ValueError("Dense mismatch across candidate arms")
                dense_by_task[key]=row["dense_score"]
        expected_tasks=set(dense_by_task)
        ranking=[]
        for arm, recs in armgroup.items():
            tasks={rec["task"] for rec in recs}
            if tasks!=expected_tasks:
                raise ValueError("Arms have different task panels: refuse to rank")
            ratio={r["task"]:r["score"]/r["dense_score"] for r in recs}
            macro=100*sum(ratio.values())/len(ratio)
            ranking.append({"arm":arm,"macro_relative_pct":macro,"per_task":ratio,
                            "task_count":len(ratio),"min_n":min(r["n"] for r in recs)})
        outputs["|".join(map(str,group))]=sorted(
            ranking,key=lambda x:(-x["macro_relative_pct"],x["arm"]))
    return outputs

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    a=p.parse_args()
    records=[json.loads(z) for z in Path(a.input).read_text().splitlines() if z.strip()]
    if not records:raise SystemExit("No dev results")
    print(json.dumps(rank_rows(records),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
