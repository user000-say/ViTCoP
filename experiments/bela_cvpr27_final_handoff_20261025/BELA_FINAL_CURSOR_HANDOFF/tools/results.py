"""Paired evaluation of official prediction/scoring exports; no replacement judge.
JSONL: sample_id, image_id, correct (boolean), optionally subset,
gold_yes_no, pred_yes_no. All task correctness must come from same official evaluator.
"""
import json, random, math
from pathlib import Path

def read_jsonl(path):
    data={}
    for line_no,line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():continue
        item=json.loads(line)
        if not isinstance(item.get("sample_id"),str) or not isinstance(item.get("image_id"),str):
            raise ValueError(f"{path}:{line_no}: missing ID")
        if type(item.get("correct")) is not bool:
            raise ValueError(f"{path}:{line_no}: 'correct' must be true or false")
        key=item["sample_id"]
        if key in data:raise ValueError(f"duplicated sample_id={key}")
        data[key]=item
    if not data:raise ValueError("empty scoring data")
    return data

def compare(base,new,replicates=1000,seed=2601025):
    if set(base)!=set(new):
        raise ValueError("different sample ID sets; do NOT evaluate different samples")
    ids=sorted(base)
    cluster={}
    cnt={"both_correct":0,"both_wrong":0,"baseline_only":0,"candidate_only":0}
    for i in ids:
        a,b=base[i],new[i]
        if a["image_id"]!=b["image_id"] or a.get("subset")!=b.get("subset"):
            raise ValueError("image IDs or benchmark subsets differ")
        if a["correct"] and b["correct"]:cnt["both_correct"]+=1
        elif not a["correct"] and not b["correct"]:cnt["both_wrong"]+=1
        elif a["correct"]:cnt["baseline_only"]+=1
        else:cnt["candidate_only"]+=1
        cluster.setdefault(a["image_id"],[]).append(int(b["correct"])-int(a["correct"]))
    if replicates<=0:raise ValueError("replicates must be positive")
    groups=list(cluster.values())
    rng=random.Random(seed)
    samples=[]
    for _ in range(replicates):
        selected=[groups[rng.randrange(len(groups))] for _ in groups]
        diffs=[v for g in selected for v in g]
        samples.append(100*sum(diffs)/len(diffs))
    samples.sort()
    n=len(samples)
    return {"n":len(ids),"image_clusters":len(groups),
      "baseline_accuracy":100*sum(base[i]["correct"] for i in ids)/len(ids),
      "candidate_accuracy":100*sum(new[i]["correct"] for i in ids)/len(ids),
      "delta_pp":100*(cnt["candidate_only"]-cnt["baseline_only"])/len(ids),
      "paired_counts":cnt,
      "cluster_bootstrap_95pp":[samples[int((n-1)*0.025)],samples[int((n-1)*0.975)]]}

def _conf(rows):
    m={"TP":0,"FP":0,"TN":0,"FN":0}
    for r in rows:
        y,p=r.get("gold_yes_no"),r.get("pred_yes_no")
        if y not in ("yes","no") or p not in ("yes","no"):
            raise ValueError("POPE gold/pred must be official parsed yes/no")
        m["TP" if y=="yes" and p=="yes" else
          "FN" if y=="yes" else
          "FP" if p=="yes" else "TN"]+=1
    tp,fp,tn,fn=(m[k] for k in ("TP","FP","TN","FN"))
    prec=tp/(tp+fp) if tp+fp else 0
    rec=tp/(tp+fn) if tp+fn else 0
    m.update(precision=prec,recall=rec,f1=2*prec*rec/(prec+rec) if prec+rec else 0,
             accuracy=(tp+tn)/(tp+fp+tn+fn))
    return m

def pope_report(samples):
    by={}
    for v in samples.values():
        subset=v.get("subset")
        if subset not in ("random","popular","adversarial"):
            raise ValueError("POPE subtype is required")
        by.setdefault(subset,[]).append(v)
    return {"overall_micro":_conf(list(samples.values())),
            "by_subset":{name:_conf(rows) for name,rows in sorted(by.items())}}


def pope_paired_bootstrap(base,new,replicates=500,seed=20261025):
    """Image-cluster bootstrap of POPE F1 change (percentage points), not accuracy."""
    if set(base)!=set(new):
        raise ValueError("POPE sample ID sets must match")
    if type(replicates)!=int or replicates<=0:
        raise ValueError("positive integer replicates required")
    ids=sorted(base)
    byimage={}
    for i in ids:
        if base[i]["image_id"]!=new[i]["image_id"]:
            raise ValueError("misaligned image IDs")
        if base[i].get("gold_yes_no")!=new[i].get("gold_yes_no"):
            raise ValueError("mismatched gold labels")
        if base[i].get("subset")!=new[i].get("subset"):
            raise ValueError("mismatched POPE subsets")
        byimage.setdefault(base[i]["image_id"],[]).append(i)
    rng=random.Random(seed)
    chunks=list(byimage.values())
    point=100*(pope_report(new)["overall_micro"]["f1"]-
               pope_report(base)["overall_micro"]["f1"])
    samples=[]
    for _ in range(replicates):
        sampled=[chunks[rng.randrange(len(chunks))] for __ in chunks]
        rows=[k for chunk in sampled for k in chunk]
        fa=_conf([base[k] for k in rows])["f1"]
        fb=_conf([new[k] for k in rows])["f1"]
        samples.append(100*(fb-fa))
    samples.sort()
    return {"delta_f1_pp":point,
            "image_cluster_bootstrap_95pp":[
                samples[int((replicates-1)*0.025)],
                samples[int((replicates-1)*0.975)]
            ],"n_clusters":len(chunks),"n_samples":len(ids)}

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("baseline");p.add_argument("candidate")
    p.add_argument("--pope",action="store_true")
    p.add_argument("--bootstrap",type=int,default=1000)
    args=p.parse_args()
    base,new=read_jsonl(args.baseline),read_jsonl(args.candidate)
    out=compare(base,new,replicates=args.bootstrap)
    if args.pope:
        out["baseline_pope"]=pope_report(base)
        out["candidate_pope"]=pope_report(new)
        out["paired_pope_f1"]=pope_paired_bootstrap(base,new,replicates=args.bootstrap)
    print(json.dumps(out,ensure_ascii=False,indent=2))
