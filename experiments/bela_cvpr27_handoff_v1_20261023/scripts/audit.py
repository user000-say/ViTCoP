"""32-layer visual token schedule & actual runtime trace audit; stdlib only."""
import argparse, json
from pathlib import Path

class AuditError(ValueError): pass

def is_int(v): return type(v) is int

def expand(segments):
    if not isinstance(segments,list) or not segments: raise AuditError('empty schedule')
    counts=[]
    for row in segments:
        if not isinstance(row,(list,tuple)) or len(row)!=2: raise AuditError('stage must be [layers,k]')
        n,k=row
        if not is_int(n) or n<=0 or not is_int(k) or k<=0: raise AuditError('positive integers required')
        counts += [k]*n
    return check(counts) and counts

def check(counts):
    if not isinstance(counts,list) or len(counts)!=32 or any(not is_int(x) or x<=0 for x in counts):
        raise AuditError('exactly 32 positive integer token counts required')
    if any(b>a for a,b in zip(counts,counts[1:])): raise AuditError('visual count increased')
    return True

def summary(counts):
    check(counts)
    cuts=[dict(after_layer_index=i-1, before=counts[i-1], after=counts[i])
          for i in range(1,32) if counts[i-1]!=counts[i]]
    return dict(W=sum(counts), Q=sum(x*x for x in counts), mean=sum(counts)/32,
                events=cuts, n_events=len(cuts))

def check_ids(x):
    if not isinstance(x,list) or any(not is_int(i) or i<0 for i in x): raise AuditError('invalid patch ids')
    if len(set(x))!=len(x): raise AuditError('duplicate patch ids')
    return x

def validate_record(row, expected=None):
    required=('run_id','sample_id','image_id','tokens_per_layer','cut_events')
    if any(k not in row for k in required): raise AuditError('missing trace fields')
    if any(not isinstance(row[k],str) or not row[k] for k in required[:3]):
        raise AuditError('empty run/sample/image identifier')
    actual=row['tokens_per_layer']; stats=summary(actual)
    if expected is not None and actual!=expected: raise AuditError('observed layer schedule != claimed schedule')
    events=row['cut_events']
    if not isinstance(events,list) or len(events)!=stats['n_events']:
        raise AuditError('cut event count != token transitions')
    prev=None
    for event,exp in zip(events,stats['events']):
        if not isinstance(event,dict) or event.get('after_layer_index')!=exp['after_layer_index']:
            raise AuditError('cut layer off-by-one or unsorted')
        pre=check_ids(event.get('patch_ids_before'))
        post=check_ids(event.get('patch_ids_after'))
        if len(pre)!=exp['before'] or len(post)!=exp['after']: raise AuditError('cut size mismatch')
        if not set(post).issubset(pre): raise AuditError('patch ID reintroduced')
        if prev is not None and set(pre)!=set(prev): raise AuditError('inconsistent patch ancestry')
        prev=post
        for name in ('selector_module','selector_function','selector_callsite'):
            if not isinstance(event.get(name),str) or not event[name].strip():
                raise AuditError('missing score/selector provenance: '+name)
    return stats

def validate_jsonl(path, expected=None):
    run=None; records=0; ids=set(); template=None
    for number,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        if not line.strip(): continue
        try:
            row=json.loads(line)
            stats=validate_record(row,expected)
        except (ValueError,TypeError) as e: raise AuditError(f'line {number}: {e}') from e
        key=(row['run_id'],row['sample_id'])
        if key in ids: raise AuditError('duplicate run/sample ID')
        ids.add(key)
        if run is None: run=row['run_id']; template=stats
        if row['run_id']!=run or stats!=template: raise AuditError('mixed runs/schedules in trace')
        records+=1
    if not records: raise AuditError('empty trace')
    return {'run_id':run,'n_samples':records,'observed':template}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('cmd',choices=['schedules','trace'])
    ap.add_argument('path')
    ap.add_argument('--schedule',default='config/schedules.json')
    ap.add_argument('--arm',default=None)
    args=ap.parse_args()
    config_path=args.path if args.cmd=='schedules' else args.schedule
    arms=json.loads(Path(config_path).read_text(encoding='utf-8'))['arms']
    if args.cmd=='schedules':
        out={k:{'status':v['status'],**summary(expand(v['stages']))} for k,v in arms.items()}
    else:
        expected=expand(arms[args.arm]['stages']) if args.arm else None
        out=validate_jsonl(args.path,expected)
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
