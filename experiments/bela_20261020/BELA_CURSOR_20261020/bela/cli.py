"""CLI: python -m bela.cli --target 2048 --entry 172 --boundaries 4,12 --final 32"""
import argparse,json
from .core import find_exact_plans
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--target',type=int,required=True)
    p.add_argument('--entry',type=int,required=True)
    p.add_argument('--blocks',type=int,default=32)
    p.add_argument('--boundaries',required=True)
    p.add_argument('--final',type=int,required=True)
    p.add_argument('--limit',type=int,default=10)
    a=p.parse_args()
    plans=find_exact_plans(blocks=a.blocks,entry=a.entry,target=a.target,
         boundaries=tuple(map(int,a.boundaries.split(','))),
         allowed_counts=range(1,a.entry+1),final_keep=a.final,limit=a.limit)
    print(json.dumps({"target":a.target,"solutions":[x.to_dict() for x in plans],
         "exact":bool(plans)},indent=2))
    if not plans: raise SystemExit(2)
if __name__=='__main__':main()
