"""Offline K60 coverage/saliency tradeoff plot from actual diagnostic rows.

Use elva.low_budget_cover.diagnostics on real saved metric/scores/selected
rows, then attach schema/identity/source fields. No ACC is inferred here.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from elva.low_budget_cover import POLICY


def summarize(rows, *, allow_fixture=False):
    if not rows or any(r.get('schema') != 'elva_r3_cover_audit' or type(r.get('fixture')) is not bool for r in rows):
        raise ValueError('actual diagnostic schema and explicit fixture flag required')
    if len({r['fixture'] for r in rows}) != 1 or (rows[0]['fixture'] and not allow_fixture):
        raise ValueError('fixture input requires --allow-fixture; never mix real and synthetic cases')
    signature=lambda r:(r.get('source_hash'),r.get('guide'),r.get('beta'),r.get('work'),r.get('policy'))
    if len({signature(r) for r in rows}) != 1 or rows[0].get('work') != 1024 or rows[0].get('policy') != POLICY:
        raise ValueError('one fixed guide/beta/source at W1024 required per comparison')
    if not rows[0].get('source_hash') or rows[0].get('guide') not in ('pen','l9','dual'):
        raise ValueError('real source identity and explicit guide required')
    groups={}
    for r in rows:
        b=r.get('slots');image=r.get('image_key')
        if (type(b) is not int or b not in (0,6,12,18) or not isinstance(image,str) or not image
                or not r.get('config_hash') or type(r.get('actual_changed')) is not int
                or not 0 <= r['actual_changed'] <= b or r.get('protected_count') != 60-b
                or r.get('candidate_count') != 576):
            raise ValueError('K60 protected/full-pool metadata and actual changed counts required')
        values=(r.get('kept_mass'),r.get('topk_mass'),r.get('kept_coverage',{}).get('weighted'),
                r.get('topk_coverage',{}).get('weighted'),r.get('kept_coverage',{}).get('uniform'))
        if any(type(v) not in (int,float) or not math.isfinite(v) or not -1e-5 <= v <= 1+1e-5 for v in values):
            raise ValueError('finite diagnostic masses/coverage in [0,1] required; exclude zero-saliency cases explicitly')
        group=groups.setdefault(b,{})
        if image in group:raise ValueError('image counted twice; aggregate crops with an explicit protocol first')
        group[image]=r
    image_sets=[set(g) for g in groups.values()]
    if len(groups)<2 or any(ids != image_sets[0] for ids in image_sets[1:]):
        raise ValueError('paired identical image panel and at least two slot levels required')
    result=[]
    for b,group in sorted(groups.items()):
        records=list(group.values())
        result.append(dict(slots=b,images=len(records),
            saliency_mass=float(np.mean([r['kept_mass'] for r in records])),
            saliency_mass_loss_vs_topk=float(np.mean([r['topk_mass']-r['kept_mass'] for r in records])),
            weighted_coverage=float(np.mean([r['kept_coverage']['weighted'] for r in records])),
            coverage_gain_vs_topk=float(np.mean([r['kept_coverage']['weighted']-r['topk_coverage']['weighted'] for r in records])),
            uniform_coverage=float(np.mean([r['kept_coverage']['uniform'] for r in records])),
            actual_changed=float(np.mean([r['actual_changed'] for r in records])),
            zero_change_images=sum(r['actual_changed']==0 for r in records)))
    return result


def render(rows, output, *, allow_fixture=False):
    values=summarize(rows,allow_fixture=allow_fixture)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    with (out/'coverage_tradeoff.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    x=[r['slots'] for r in values]
    for ax,key,title in zip(axes,('saliency_mass_loss_vs_topk','coverage_gain_vs_topk','actual_changed'),
                            ('Saliency mass loss vs Top60','Feature coverage gain vs Top60','Actual additions surviving at K60')):
        ax.plot(x,[r[key] for r in values],marker='o',color='#175f91')
        ax.set(xlabel='Coverage slots b',title=title,xticks=x);ax.grid(alpha=.2)
    if rows[0]['fixture']:
        fig.suptitle('SYNTHETIC FIXTURE — NOT VLM RESULTS',fontsize=10)
        fig.text(.5,.5,'SYNTHETIC FIXTURE — NOT VLM RESULTS',ha='center',rotation=15,alpha=.25,fontsize=22)
    fig.savefig(out/'coverage_tradeoff.png',dpi=170);fig.savefig(out/'coverage_tradeoff.svg');plt.close(fig)
    (out/'summary.json').write_text(json.dumps(dict(fixture=rows[0]['fixture'],rows=values),indent=2)+'\n')
    return values


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',required=True);p.add_argument('--out',required=True)
    p.add_argument('--allow-fixture',action='store_true')
    a=p.parse_args()
    rows=[json.loads(line) for line in Path(a.records).read_text().splitlines() if line.strip()]
    render(rows,a.out,allow_fixture=a.allow_fixture)
