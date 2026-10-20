"""CLI: prepare experiments and validate real prefill traces/official summaries.

Example:
 python -m elva.cli plans plans/schedules.json
 python -m elva.cli verify-trace plans/schedules.json E94_K80_48_16 path/to/per_sample_trace.json
 python -m elva.cli official-r4 results_examples/official_scores.json
"""
import argparse
import json
from pathlib import Path
from .core import Schedule, assert_real_trace, r4_score


def plans(path):
    content = json.loads(Path(path).read_text(encoding='utf-8'))
    schedules = [Schedule.from_dict(x) for x in content['schedules']]
    if len({x.name for x in schedules}) != len(schedules):
        raise ValueError('duplicate schedule name')
    w_targets = content['budgets']
    for s in schedules:
        s.verify_budget(int(w_targets[str(s.budget)]))
    return schedules


def main():
    p = argparse.ArgumentParser(description='ELVA preregistered plan + identity contract')
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('plans'); a.add_argument('file')
    a = sub.add_parser('verify-trace')
    a.add_argument('file'); a.add_argument('name'); a.add_argument('trace_file')
    a = sub.add_parser('official-r4'); a.add_argument('file')
    args = p.parse_args()
    if args.cmd == 'official-r4':
        data = json.loads(Path(args.file).read_text(encoding='utf-8'))
        r4 = r4_score(**data)
        print(json.dumps(dict(r4=r4, reciprocal_sum=r4/25.), indent=2))
        return
    schedules = plans(args.file)
    if args.cmd == 'plans':
        print(json.dumps([s.to_dict() for s in schedules], ensure_ascii=False, indent=2))
        return
    matches = [s for s in schedules if s.name == args.name]
    if len(matches) != 1:
        raise ValueError('unknown schedule')
    # Trace files must be produced from actual host-decoder execution, not this script.
    obj = json.loads(Path(args.trace_file).read_text(encoding='utf-8'))
    items = obj if isinstance(obj,list) and obj and isinstance(obj[0],list) else [obj]
    for item in items:
        assert_real_trace(item, matches[0])
    print(f'PASS: {len(items)} physical prefill trace(s), W={matches[0].budget}')

if __name__ == '__main__':
    main()
