"""Register ONLY officially scored runs. CSV scores stay blank until measured."""
from __future__ import annotations
import argparse, csv, os, tempfile
from pathlib import Path
from tools.budget_planner import load_specs

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/"configs/results_pending.csv"

def record_score(*, model, tier, arm, benchmark, value, n_questions,
                 git_commit, processor_revision, dataset_hash, scorer_revision,
                 run_dir, filename=PATH, overwrite=False):
    if not all([git_commit,processor_revision,dataset_hash,scorer_revision,run_dir]):
        raise ValueError("Required source provenance missing")
    if not isinstance(value,(int,float)) or value!=value or value in (float('inf'),-float('inf')):
        raise ValueError("Invalid score")
    limit=1000 if benchmark=="OCRBench" else 2800 if benchmark=="MME" else 100
    if not 0<=value<=limit:
        raise ValueError(f"Score outside plausible {benchmark} range")
    rows=list(csv.DictReader(open(filename,encoding="utf-8",newline="")))
    matches=[i for i,r in enumerate(rows) if (r["model"],r["budget"],r["arm"],r["benchmark"])
             ==(model,tier,arm,benchmark)]
    if len(matches)!=1: raise ValueError("No unique manifest row")
    i=matches[0]
    if rows[i]["status"]=="MEASURED" and not overwrite:
        raise ValueError("Refusing to overwrite an existing measured score")
    rows[i].update(score=str(value),n_questions=str(n_questions),status="MEASURED",
                   git_commit=git_commit,processor_revision=processor_revision,
                   dataset_hash=dataset_hash,scorer_revision=scorer_revision,run_dir=run_dir,
                   notes="Official scorer; user supplied measured value")
    tmp=Path(str(filename)+".tmp")
    with tmp.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    os.replace(tmp,filename)


def main():
    p=argparse.ArgumentParser(description="Record a real run (cannot compute benchmarks itself)")
    for name in ("model","tier","arm","benchmark","git-commit","processor-revision",
                 "dataset-hash","scorer-revision","run-dir"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--score",type=float,required=True)
    p.add_argument("--n-questions",type=int,required=True)
    a=vars(p.parse_args())
    if a["n_questions"]<=0:raise ValueError("n_questions > 0 required")
    record_score(model=a["model"],tier=a["tier"],arm=a["arm"],
        benchmark=a["benchmark"],value=a["score"],n_questions=a["n_questions"],
        git_commit=a["git_commit"],processor_revision=a["processor_revision"],
        dataset_hash=a["dataset_hash"],scorer_revision=a["scorer_revision"],
        run_dir=a["run_dir"])
    print("MEASURED row registered; cross-backbone comparison still requires same-model DENSE/native controls")


if __name__=="__main__":
    main()
