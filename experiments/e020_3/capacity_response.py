"""Finite-grid budget headroom, independent of the new aggregation operator.

This is a hindsight diagnostic using real outcomes, NEVER an online policy
or an official R4 upper bound. It uses additive utilities only.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def hindsight(utilities, costs):
    u, c = np.asarray(utilities, float), np.asarray(costs, float)
    if u.ndim != 2 or u.shape[1] != 3 or not len(u) or not np.isfinite(u).all():
        raise ValueError("complete real [N,3] additive utilities required")
    if c.shape != (3,) or not np.isfinite(c).all() or not c[0] < c[1] < c[2]:
        raise ValueError("three increasing costs required")
    if not np.isclose(c[1]-c[0], c[2]-c[1]):
        raise ValueError("equal budget steps required for this exact DP")
    n = len(u)
    # Choice 0/1/2 spends 0/1/2 increments over the low budget. A total
    # <=N increments matches or beats the fixed middle's total cost.
    dp = np.full(n+1, -np.inf); dp[0] = 0.
    back = np.full((n,n+1), -1, np.int8)
    for i, row in enumerate(u):
        new = np.full(n+1, -np.inf)
        for choice in range(3):
            limit = n+1-choice
            score = dp[:limit] + row[choice]
            take = score > new[choice:]
            new[choice:][take] = score[take]
            back[i, choice:][take] = choice
        dp = new
    budget = int(np.argmax(dp))
    value = float(dp[budget]/n)
    chosen = np.empty(n, np.int64)
    for i in range(n-1,-1,-1):
        choice = int(back[i,budget])
        if choice < 0:
            raise RuntimeError("invalid DP traceback")
        chosen[i] = choice
        budget -= choice
    assert budget == 0
    assert float(c[chosen].mean()) <= c[1]+1e-8
    assert value >= u[:,1].mean()-1e-10
    return dict(diagnostic_only=True, choice=chosen.tolist(),
                mean_utility=value, fixed_middle_utility=float(u[:,1].mean()),
                gain_over_middle=float(value-u[:,1].mean()),
                mean_cost=float(c[chosen].mean()), fixed_middle_cost=float(c[1]))


def permute_budgets(ids, choices, seed=17):
    ids, choice = np.asarray(ids,str), np.asarray(choices)
    if ids.ndim!=1 or choice.shape!=ids.shape or len(set(ids.tolist()))!=len(ids):
        raise ValueError("stable unique sample IDs required")
    order=np.argsort(ids,kind="stable")
    out=choice.copy()
    out[order]=choice[order][np.random.default_rng(seed).permutation(len(ids))]
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("real_panel_npz",help="utilities[N,3], costs[3], sample_ids[N]")
    p.add_argument("output")
    a=p.parse_args()
    with np.load(a.real_panel_npz,allow_pickle=False) as d:
        result=hindsight(d["utilities"],d["costs"])
        result["sample_ids"]=d["sample_ids"].astype(str).tolist()
    Path(a.output).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("choice","sample_ids")}))
