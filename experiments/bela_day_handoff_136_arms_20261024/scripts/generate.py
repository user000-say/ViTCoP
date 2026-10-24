"""CPU-only deterministic exact-W experimental candidate generator."""
import csv
from pathlib import Path

LAYERS=(4,8,20)
TARGETS={1024:[60,72,84,96],2048:[128,160,172,192],4096:[224,256,288,320]}
def metrics(a,b,c,layers=LAYERS):
    if len(layers)!=3 or sum(layers)!=32 or not all(type(v)==int and v>0 for v in (a,b,c,*layers)):
        raise ValueError("invalid positive integer segments")
    if not a>=b>=c: raise ValueError("invalid monotonic token counts")
    return sum(k*n for k,n in zip((a,b,c),layers)),sum(k*k*n for k,n in zip((a,b,c),layers))
def solutions(W,a):
    out=[]
    for b in range(1,a+1):
        rest=W-4*a-8*b
        if rest%20:continue
        c=rest//20
        if 8<=c<=b:out.append((a,b,c))
    return out
def spread(rows,n=8):
    if len(rows)<=n:return rows
    return [rows[round(i*(len(rows)-1)/(n-1))] for i in range(n)]
def make():
    rows=[]
    def add(arm,phase,W,a,b,c,selector="CURRENT",note=""):
        w,q=metrics(a,b,c)
        assert w==W
        rows.append(dict(arm=arm,phase=phase,budget=W,k0=a,kmid=b,kdeep=c,
           cut_after_1=4,cut_after_2=12,selector=selector,w_check=w,q_proxy=q,
           needs_stage1_check=int(a not in [60,128,160,172]),
           status="NOT_RUN",note=note))
    for W,entries in TARGETS.items():
        for a in entries:
            for _,b,c in spread(solutions(W,a),8):
                add(f"S_{W}_{a}_{b}_{c}","PILOT",W,a,b,c,note="exact W, schedule sweep")
    for name,W,a,b,c in [
        ("A0",2048,172,90,32),("A2",2048,172,120,20),
        ("A3",2048,172,130,16),("B1",1808,172,90,20),
        ("B2",2288,172,120,32)]:
        add(name,"REUSE",W,a,b,c,note="reported; do not rerun unless audit fails")
        rows[-1]["status"]="REPORTED"
    for W,a in [(1024,96),(2048,172),(4096,256)]:
        for _,b,c in spread(solutions(W,a),3):
            for selector in ("QA_TOPK","RDRIFT","RANDOM_1","RANDOM_2"):
                add(f"I_{W}_{a}_{b}_{c}_{selector}","GATED",W,a,b,c,selector,
                    "only after cut2 selector/callsite audit")
    assert len({x["arm"] for x in rows})==len(rows)
    return rows
if __name__=="__main__":
    rows=make()
    with Path("queue.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    from collections import Counter
    print("total",len(rows),"phases",dict(Counter(r["phase"] for r in rows)),
          "budgets",dict(Counter(r["budget"] for r in rows)))
