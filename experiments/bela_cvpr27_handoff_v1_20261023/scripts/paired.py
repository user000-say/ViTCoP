"""Paired correctness / POPE analysis; NO homebrew scoring. Use official evaluator outputs."""
import argparse, json, random
from pathlib import Path

class PairError(ValueError): pass

def load(path):
    rows={}
    for i,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        if not line.strip(): continue
        try: row=json.loads(line)
        except ValueError as e: raise PairError(f'{path}:{i}: bad JSON') from e
        if any(k not in row for k in ('sample_id','image_id','correct')): raise PairError('missing required fields')
        if not isinstance(row['sample_id'],str) or not row['sample_id'] or not isinstance(row['image_id'],str) or not row['image_id']:
            raise PairError('invalid sample/image IDs')
        if type(row['correct']) is not bool: raise PairError('correct must be JSON boolean')
        if row['sample_id'] in rows: raise PairError('duplicate sample ID')
        rows[row['sample_id']]=row
    if not rows: raise PairError('empty predictions')
    return rows

def percentile(items,q):
    x=sorted(items); at=(len(x)-1)*q; i=int(at); j=min(i+1,len(x)-1)
    return x[i]+(x[j]-x[i])*(at-i)

def compare(a,b,seed=2718,bootstrap=500):
    if set(a)!=set(b): raise PairError('sample ID mismatch: forbid partial join')
    if type(seed) is not int or type(bootstrap) is not int or bootstrap<1: raise PairError('invalid sampling')
    kinds={'both_correct':0,'both_wrong':0,'baseline_only':0,'candidate_only':0}
    groups={}
    for sid in sorted(a):
        x,y=a[sid],b[sid]
        if x['image_id']!=y['image_id']: raise PairError('image mismatch')
        for field in ('gold_binary','subset'):
            if x.get(field)!=y.get(field): raise PairError(field+' mismatch')
        cx,cy=x['correct'],y['correct']
        if cx and cy: kinds['both_correct']+=1
        elif not cx and not cy: kinds['both_wrong']+=1
        elif cx: kinds['baseline_only']+=1
        else: kinds['candidate_only']+=1
        groups.setdefault(x['image_id'],[]).append(int(cy)-int(cx))
    rng=random.Random(seed); byimage=list(groups.values()); draws=[]
    for _ in range(bootstrap):
        chosen=[byimage[rng.randrange(len(byimage))] for _ in byimage]
        values=[d for group in chosen for d in group]
        draws.append(100*sum(values)/len(values))
    abase=100*sum(x['correct'] for x in a.values())/len(a)
    bcand=100*sum(x['correct'] for x in b.values())/len(b)
    return {'n':len(a),'unique_images':len(groups),'a_accuracy_pct':abase,
            'b_accuracy_pct':bcand,'delta_pp':bcand-abase,
            'paired':kinds,'cluster_bootstrap_95pp':[percentile(draws,.025),percentile(draws,.975)],
            'bootstrap_n':bootstrap,'seed':seed}

def pope(rows):
    result={}
    for sid,row in rows.items():
        if not all(k in row for k in ('gold_binary','pred_binary','subset')): raise PairError('POPE row missing fields')
        g,p,subset=row['gold_binary'],row['pred_binary'],row['subset']
        if g not in ('yes','no') or p not in ('yes','no'): raise PairError('invalid POPE yes/no')
        if subset not in ('random','popular','adversarial'): raise PairError('unknown POPE subset')
        m=result.setdefault(subset,{'TP':0,'TN':0,'FP':0,'FN':0})
        label='TP' if g=='yes' and p=='yes' else 'TN' if g=='no' and p=='no' else 'FP' if g=='no' else 'FN'
        m[label]+=1
    def scores(m):
        tp,tn,fp,fn=(m[k] for k in ('TP','TN','FP','FN'))
        precision=tp/(tp+fp) if (tp+fp) else 0
        recall=tp/(tp+fn) if (tp+fn) else 0
        f1=2*precision*recall/(precision+recall) if precision+recall else 0
        return {**m,'P':precision,'R':recall,'F1':f1,'ACC':(tp+tn)/(tp+tn+fp+fn)}
    total={k:sum(m[k] for m in result.values()) for k in ('TP','TN','FP','FN')}
    return {'micro':scores(total),'subsets':{k:scores(v) for k,v in sorted(result.items())}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('a');ap.add_argument('b')
    ap.add_argument('--pope',action='store_true'); ap.add_argument('--seed',type=int,default=2718)
    args=ap.parse_args(); a=load(args.a);b=load(args.b)
    out=compare(a,b,seed=args.seed)
    if args.pope: out['pope_a']=pope(a); out['pope_b']=pope(b)
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
