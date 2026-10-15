"""Generic paired runner; project adapter is mandatory, never auto-guessed.

Example after Cursor implements an adapter:
python collect_pairs.py --adapter my_adapter:build --config frozen.json \
    --output pairs_gpu0.npz --limit 256
Adapter handles disjoint image-level sharding from its config.
"""
import argparse
import importlib
import json
from pathlib import Path
import numpy as np
from adapter_contract import collect_pair
from controller_numpy import SCHEMA, load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', required=True, help='project_module:factory')
    ap.add_argument('--config', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--limit', type=int, default=256)
    args = ap.parse_args()
    if args.limit < 1:
        raise ValueError('limit must be positive')
    module, factory = args.adapter.split(':')
    config = json.loads(Path(args.config).read_text())
    adapter = getattr(importlib.import_module(module), factory)(config)
    manifest = adapter.manifest()
    if manifest.get('data_split') not in ('train_only', 'development_only'):
        raise ValueError('controller collection must use training/development sources only')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix('.jsonl').exists():
        raise ValueError('choose fresh output paths; collection logs are never overwritten')
    rows = []
    with output.with_suffix('.jsonl').open('x') as log:
        for index, sample in enumerate(adapter.iter_samples()):
            if index >= args.limit:
                break
            pair = collect_pair(adapter, sample)
            f = pair['features']
            row = {k:np.asarray(f[k], dtype=float) for k in ['base','history','retained']}
            row.update({k:f[k] for k in ['extra_ms','image_id','sample_id','stride']})
            row.update(early_nll=pair['early']['answer_nll'], late_nll=pair['late']['answer_nll'])
            rows.append(row)
            # Durable within a run; preserves expensive branches if later rows fail.
            record = {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in row.items()}
            record['prefix_fingerprint'] = pair['early']['prefix_fingerprint']
            record['early_prediction'] = pair['early']['prediction']
            record['late_prediction'] = pair['late']['prediction']
            log.write(json.dumps(record, ensure_ascii=False) + '\n')
            log.flush()
    if not rows:
        raise ValueError('adapter produced no samples')
    stride = int(rows[0]['stride'])
    if any(int(r['stride']) != stride for r in rows):
        raise ValueError('summary layout changed between samples')
    payload = {k:np.stack([r[k] for r in rows]) for k in ['base','history','retained']}
    payload.update({k:np.array([r[k] for r in rows]) for k in ['early_nll','late_nll','extra_ms']})
    payload.update(image_ids=np.array([str(r['image_id']) for r in rows]),
                   sample_ids=np.array([str(r['sample_id']) for r in rows]),
                   schema=SCHEMA, kind='real', stride=stride,
                   manifest=json.dumps(manifest, ensure_ascii=False))
    np.savez_compressed(output, **payload)
    load_dataset(output)  # Fail closed on invalid schema/IDs/finite values.
    print(json.dumps({'output':str(output),'paired_samples':len(rows)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
