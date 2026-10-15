"""Strict four-task R4; no swapped MME/MMBench or silently missing task."""
import argparse
import json

DENOM = {'GQA':61.9, 'POPE':85.8, 'MME':1852., 'MMBench':64.1}


def r4(metrics):
    if set(metrics) != set(DENOM):
        raise ValueError('exactly GQA, POPE, MME, MMBench required')
    import math
    if any(not math.isfinite(float(v)) for v in metrics.values()):
        raise ValueError('metrics must be finite')
    return sum(float(metrics[k]) / DENOM[k] for k in DENOM) * 25


def factorial(arms):
    expected = {'COVER_ONE','COVER_TWO','BASE_ONE','BASE_TWO'}
    if set(arms) != expected:
        raise ValueError('four named arms required')
    cover = arms['COVER_TWO'] - arms['COVER_ONE']
    base = arms['BASE_TWO'] - arms['BASE_ONE']
    return {'cover_TWO_minus_ONE_pp':cover, 'base_TWO_minus_ONE_pp':base,
            'interaction_pp':cover - base,
            'COVER_minus_BASE_at_ONE_pp':arms['COVER_ONE'] - arms['BASE_ONE'],
            'COVER_minus_BASE_at_TWO_pp':arms['COVER_TWO'] - arms['BASE_TWO']}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--metrics-json', help='arm -> four raw task metrics')
    args = ap.parse_args()
    if args.metrics_json:
        with open(args.metrics_json) as f:
            values = {k:r4(v) for k,v in json.load(f).items()}
        label = 'computed from supplied raw task metrics'
    else:
        values = {'COVER_ONE':95.21,'COVER_TWO':96.10,'BASE_ONE':95.22,'BASE_TWO':96.10}
        label = 'user rounded aggregate values ONLY; no statistical significance test'
    print(json.dumps({'source':label, 'R4_percent':values,
                      'factorial':factorial(values)}, indent=2))
