"""Keep previous hypotheses; append current-event and direction controls."""
import hashlib
import json
from pathlib import Path
from make_matrix import build, operator


def expanded():
    rows = build()
    for profile in ('SHIFT', 'SHIFT_K100', 'FINAL11'):
        for metric in ('current_hidden', 'current_key_mean_head'):
            for events in (('first',), ('final',), ('first', 'final')):
                cfg = dict(entry='COVER', profile_alias=profile, first_selector='RDRIFT',
                    final_selector='ORIGINAL', seed=0,
                    operator={**operator(events), 'affinity_mode':'current_rectangular',
                              'metric_source':metric, 'prefix_only':False})
                digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
                rows.append(dict(id='I_'+digest, block='I', **cfg))
    for profile in ('SHIFT', 'SHIFT_K100'):
        for events in (('first',), ('first', 'final')):
            cfg = dict(entry='COVER', profile_alias=profile, first_selector='RDRIFT',
                final_selector='ORIGINAL', seed=0,
                operator={**operator(events), 'affinity_mode':'current_rectangular',
                    'metric_source':'current_hidden', 'prefix_only':True})
            digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
            rows.append(dict(id='J_'+digest, block='J', **cfg))
    keys = [json.dumps({k:v for k,v in r.items() if k not in ('id','block')}, sort_keys=True) for r in rows]
    assert len(set(keys)) == len(rows)
    return rows


if __name__ == '__main__':
    rows = expanded()
    summary = dict(configs=len(rows), blocks={b:sum(r['block']==b for r in rows) for b in 'ABCDEFGHIJ'},
                   maximum_four_task_screen_jobs=4*len(rows))
    Path(__file__).with_name('matrix.json').write_text(json.dumps(dict(schema='e020_4', **summary, rows=rows), indent=2)+'\n')
    print(json.dumps(summary))
