"""Merge compatible disjoint shards; reject config drift and duplicate IDs."""
import argparse
import json
from pathlib import Path
import numpy as np
from controller_numpy import load_dataset


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    ap.add_argument('shards',nargs='+')
    args=ap.parse_args()
    if Path(args.output).exists():
        raise ValueError('output already exists')
    parts=[load_dataset(p) for p in args.shards]
    def manifest(p):
        m=json.loads(str(p['manifest']))
        return {k:v for k,v in m.items() if k not in ('shard_id','shard_count')}
    first=parts[0]
    if any(manifest(p)!=manifest(first) or int(p['stride'])!=int(first['stride']) for p in parts):
        raise ValueError('incompatible manifests: do not merge different methods/schedules')
    row_keys=['base','history','retained','early_nll','late_nll','extra_ms','image_ids','sample_ids']
    merged={k:np.concatenate([p[k] for p in parts],axis=0) for k in row_keys}
    for key in ['schema','kind','stride']:
        merged[key]=first[key]
    merged['manifest']=json.dumps(manifest(first),ensure_ascii=False)
    if len(set(map(str,merged['sample_ids'])))!=len(merged['sample_ids']):
        raise ValueError('duplicate sample IDs across shards')
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**merged)
    load_dataset(args.output)
    print(json.dumps({'output':args.output,'samples':len(merged['sample_ids'])}))
