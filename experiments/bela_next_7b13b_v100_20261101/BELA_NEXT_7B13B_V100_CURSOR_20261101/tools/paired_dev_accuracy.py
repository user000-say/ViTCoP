"""Developer-panel paired correctness analysis (true harness correctness only).

Input one JSONL record per question with boolean correctness fields:
{"id":"task/video-or-image-id/question-id","group":"OCR",
 "dense_correct":true,"baseline_correct":true,"candidate_correct":false}
The correctness fields MUST be computed using original task scorer (not exact
string match); this script cannot decide what's correct.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

def exact_mc_nemar(n_baseline_only,n_candidate_only):
    n=n_baseline_only+n_candidate_only
    if n==0:return 1.0
    k=min(n_baseline_only,n_candidate_only)
    vals=[math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)
          -n*math.log(2.) for i in range(k+1)]
    return min(1.,2*math.fsum(math.exp(z) for z in vals))

def score_pair(rows):
    if not rows:raise ValueError("No paired dev questions")
    n=len(rows)
    losses=sum(r["dense_correct"] and not r["candidate_correct"] for r in rows)
    b_only=sum(r["baseline_correct"] and not r["candidate_correct"] for r in rows)
    c_only=sum(not r["baseline_correct"] and r["candidate_correct"] for r in rows)
    b=sum(r["baseline_correct"] for r in rows)
    c=sum(r["candidate_correct"] for r in rows)
    return {"n":n,"baseline_accuracy":100*b/n,"candidate_accuracy":100*c/n,
            "delta_pp":100*(c-b)/n,"candidate_wrong_dense_right":losses,
            "baseline_only":b_only,"candidate_only":c_only,
            "mcnemar_exact_two_sided_p":exact_mc_nemar(b_only,c_only)}

def score_dataset(rows):
    seen=set(); by=defaultdict(list)
    for r in rows:
        for f in ("id","group","dense_correct","baseline_correct","candidate_correct"):
            if f not in r:raise ValueError("Missing "+f)
        if r["id"] in seen:raise ValueError("Repeated sample_id "+str(r["id"]))
        seen.add(r["id"])
        for k in ("dense_correct","baseline_correct","candidate_correct"):
            if type(r[k]) is not bool:raise ValueError(k+" must be bool from official scorer")
        by["all"].append(r)
        by[r["group"]].append(r)
    return {g:score_pair(items) for g,items in sorted(by.items())}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--paired-jsonl",required=True)
    a=p.parse_args()
    rows=[json.loads(z) for z in Path(a.paired_jsonl).read_text().splitlines() if z.strip()]
    print(json.dumps(score_dataset(rows),ensure_ascii=False,indent=2))
if __name__=="__main__":main()
