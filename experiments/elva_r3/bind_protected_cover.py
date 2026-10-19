"""Add the supplied independent K60 selector binding without claiming parity.

Run from the unpacked project directory so absolute source hashes refer to
the server files, not to the file-delivery workspace. No model is downloaded.
"""
import argparse
import json
from pathlib import Path
from elva.low_budget_cover import make_spec


def bind(input_path, output_path):
    value = json.loads(Path(input_path).read_text())
    covers = value.setdefault('native_cover',{})
    if 'protected_coverage' in covers and covers['protected_coverage'].get('implementation_parity_verified'):
        raise ValueError('verified binding already exists; preserve it or explicitly use a new binding file')
    covers['protected_coverage'] = make_spec()
    Path(output_path).write_text(json.dumps(value,indent=2)+'\n')
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();bind(a.binding,a.out)
    print('Source bound. implementation_parity_verified remains false until server NumPy/Torch and real CLIP preflight pass.')
