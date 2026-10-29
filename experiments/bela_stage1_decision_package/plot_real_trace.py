#!/usr/bin/env python3
"""Plot real per-patch scores and selection masks; never manufacture a heatmap.

Input: a real exported trace with patch IDs mapped in row-major grid order.
This chart is a token-selection diagnostic, NOT an object localization map.
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from bela_eval import read_jsonl, check_ids

def render(row, height, width, output):
    n=int(row["n_patches"])
    if height*width != n:
        raise ValueError("height*width must match actual patch count")
    original=set(check_ids(row["native_ids"],n,"native_ids"))
    final=set(check_ids(row["final_ids"],n,"final_ids"))
    if "query_scores" not in row:
        raise ValueError("real query_scores[N] must be exported; no synthetic fills")
    scores=np.asarray(row["query_scores"], dtype=np.float64)
    if scores.shape!=(n,) or not np.all(np.isfinite(scores)):
        raise ValueError("query_scores must be a finite [N] array")
    mask0=np.array([int(i in original) for i in range(n)]).reshape(height,width)
    mask1=np.array([int(i in final) for i in range(n)]).reshape(height,width)
    change=np.array([int(i in final)-int(i in original) for i in range(n)]).reshape(height,width)
    fig, axs=plt.subplots(1,4,figsize=(14,3.4),constrained_layout=True)
    arrays=[scores.reshape(height,width), mask0, mask1,change]
    titles=["Raw query proxy", "Native COVER IDs", "After QA correction", "Net token changes"]
    for ax, array, title in zip(axs,arrays,titles):
        ax.imshow(array, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Real trace: "+str(row["sample_id"])+" | selection diagnostic",fontsize=10)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--trace",required=True)
    p.add_argument("--sample-id",required=True)
    p.add_argument("--height",type=int,required=True)
    p.add_argument("--width",type=int,required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    rows=[r for r in read_jsonl(a.trace) if str(r.get("sample_id"))==a.sample_id]
    if len(rows)!=1: raise ValueError("sample ID must resolve to exactly one real trace")
    render(rows[0],a.height,a.width,a.out)
if __name__=="__main__": main()
