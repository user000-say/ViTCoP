#!/usr/bin/env python3
"""Diagnostic POPE scorer (NOT a replacement for the benchmark's official scorer).

--bela and --dense should contain JSONL with question_id, label and model output.
Cannot infer missing labels or silently treat unparseable outputs as "no".
Reports class confusion, yes-rate, invalid count and optional paired flips.
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path

YES=re.compile(r"^\s*(yes|yeah|yep)\b", re.I)
NO=re.compile(r"^\s*(no|nope)\b", re.I)

def classify(text):
    if text is None: return None
    s=str(text).strip()
    if YES.match(s): return 1
    if NO.match(s): return 0
    return None

def key(row):
    for k in ("question_id","questionId","id"):
        if k in row: return str(row[k])
    raise ValueError("Missing stable question_id/id; cannot pair predictions")

def load(path):
    data={}
    for lineno,line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        row=json.loads(line)
        qid=key(row)
        if qid in data: raise ValueError(f"Duplicate question ID {qid} at {lineno}")
        label=next((row[k] for k in ("label","ground_truth","gt") if k in row),None)
        answer=next((row[k] for k in ("prediction","model_answer","pred","text","output") if k in row),None)
        if label is None:
            raise ValueError(f"Missing ground-truth label on {qid}; join original POPE questions first")
        data[qid]=(classify(label),classify(answer),
                   str(row.get("category",row.get("type","all"))))
        if data[qid][0] is None: raise ValueError(f"Unparseable ground-truth label: {qid}")
    return data

def metrics(rows):
    n=len(rows)
    tp=fp=tn=fn=invalid=0
    for label,pred,_group in rows:
        if pred is None:
            invalid+=1
        elif label==1 and pred==1: tp+=1
        elif label==0 and pred==1: fp+=1
        elif label==0 and pred==0: tn+=1
        else: fn+=1
    p=tp/(tp+fp) if tp+fp else 0.
    r=tp/(tp+fn) if tp+fn else 0.
    f1=2*p*r/(p+r) if p+r else 0.
    valid=n-invalid
    return {"n":n,"valid":valid,"invalid":invalid,"TP":tp,"FP":fp,"TN":tn,"FN":fn,
            "accuracy_on_valid":(tp+tn)/valid if valid else None,
            "precision":p,"recall":r,"f1_on_valid":f1,
            "yes_rate_on_valid":(tp+fp)/valid if valid else None}

def report(data):
    groups=defaultdict(list)
    for val in data.values():
        groups["all"].append(val)
        if val[2]!="all": groups[val[2]].append(val)
    return {k:metrics(v) for k,v in sorted(groups.items())}

def pair_flips(dense,bela):
    common=set(dense)&set(bela)
    if not common: raise ValueError("No overlapping question IDs")
    out={"shared":len(common),"dense_correct_bela_wrong":0,
         "dense_wrong_bela_correct":0, "both_wrong":0, "both_correct":0,
         "invalid_pairs":0, "label_mismatch":0}
    for id in common:
        y1,p1,_=dense[id]
        y2,p2,_=bela[id]
        if y1!=y2:
            out["label_mismatch"]+=1;continue
        if p1 is None or p2 is None:
            out["invalid_pairs"]+=1;continue
        a=p1==y1; b=p2==y2
        if a and b: out["both_correct"]+=1
        elif a and not b: out["dense_correct_bela_wrong"]+=1
        elif not a and b: out["dense_wrong_bela_correct"]+=1
        else: out["both_wrong"]+=1
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bela",required=True)
    ap.add_argument("--dense")
    ap.add_argument("--out")
    args=ap.parse_args()
    bela=load(args.bela)
    result={"warning":"Diagnostic parser; compare with OFFICIAL POPE F1 scorer",
            "bela":report(bela)}
    if args.dense:
        dense=load(args.dense)
        result["dense"]=report(dense)
        result["paired"]=pair_flips(dense,bela)
    text=json.dumps(result,ensure_ascii=False,indent=2)
    if args.out: Path(args.out).write_text(text,encoding="utf-8")
    print(text)
if __name__=="__main__":
    main()
