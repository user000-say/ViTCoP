"""Ten predeclared boundary-random configurations: not full random pruning.

Five distinct sample-stable seeds at BASE SHIFT and COVER NEXT. This tests the
observed weak-selector repair without treating four settings as four RNG seeds.
"""
import argparse
import copy
import json
from pathlib import Path
from build_matrix import build,from_configs
from core_matrix import identify


def build_replicates(runtime,phase='screen',seeds=(20,21,22,23,24)):
    old=build(runtime,phase,'independent');lookup={c['config_id']:c for c in old['configurations']}
    configs=[]
    for arm,schedule in [('BASE','FIRST_SHIFT'),('COVER','FIRST_NEXT_WMATCH')]:
        for seed in seeds:
            c=copy.deepcopy(lookup[arm+'__'+schedule+'__reference'])
            c.update(selector_name='boundary_random_g4r2_seed'+str(seed),
                     selector={'kind':'random','group_size':4,'rounds':2,'shortlist':16,'seed':seed})
            configs.append(identify(c))
    return from_configs(runtime,configs,phase)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runtime');p.add_argument('--out',required=True)
    p.add_argument('--phase',default='screen');a=p.parse_args()
    r=build_replicates(json.loads(Path(a.runtime).read_text()),a.phase)
    Path(a.out).write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r['counts']))
