#!/usr/bin/env python3
"""Reproducibility manifest: abort when actual HF package source differs from snapshot.

No GPU, model download or model weight necessary. --runtime requires installed HF.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bela_next_forward.apply_llava_next import _arms


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--runtime',action='store_true')
    ap.add_argument('--arm',default=os.environ.get('BELA_NEXT_ARM','NEXT7_T160_A0_PAPER'))
    ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    arms=_arms()
    if args.arm not in arms or arms[args.arm]['model']!='next7':
        raise SystemExit(f'Unknown/not-NeXT7 arm: {args.arm}')
    report=dict(arm=args.arm,arm_config=arms[args.arm], python=platform.python_version(),
                torch=None, transformers=None, runtime_hash_checks=None,
                source_sha256={str(x.relative_to(ROOT)):sha(x) for x in sorted((ROOT/'bela_next_forward').glob('*.py'))},
                known_limitations=['batch1','single image','HF5.20.2','greedy only','no GPU verified in this artifact'])
    if args.runtime:
        import torch
        import transformers
        from bela_next_forward.apply_llava_next import _verify_runtime
        _verify_runtime()
        report['torch']=torch.__version__
        report['transformers']=transformers.__version__
        report['runtime_hash_checks']='PASS' if os.environ.get('BELA_REQUIRE_RUNTIME_HASH','1')!='0' else 'DISABLED_NOT_ELIGIBLE'
        if report['runtime_hash_checks']!='PASS':
            raise RuntimeError('PRODUCTION_RUN_DISALLOWS_BELA_REQUIRE_RUNTIME_HASH=0')
    else:
        report['runtime_hash_checks']='NOT_RUN'
    data=json.dumps(report,ensure_ascii=False,indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(data+'\n',encoding='utf8')
    print(data)

if __name__=='__main__':main()
