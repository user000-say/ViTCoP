"""Real subprocess queue for an existing project bridge; no model guessing.

Bridge factory(config) -> object exposing command(job, outdir) -> argv list,
read_result(job, outdir) -> result dict; optional cached_result(job).
Results require actual metric, matching identities and an existing records file.
CUDA_VISIBLE_DEVICES is isolated per worker. GPU3 is excluded by default.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def validate_result(result,job):
    for key,value in [('job_hash',job['job_hash']),('metric_key',job['metric_key']),
                      ('panel_ids_hash',job['panel']['ids_hash'])]:
        if result.get(key)!=value:raise ValueError('result identity mismatch: '+key)
    if not math.isfinite(float(result['metric_value'])):
        raise ValueError('nonfinite metric')
    if not result.get('records_path') or not Path(result['records_path']).is_file():
        raise ValueError('missing real prediction/scoring records')
    selector=job.get('config',{}).get('selector',{})
    if selector.get('kind')=='continuation' and result.get('scorer_sha256')!=selector.get('model_sha256'):
        raise ValueError('learned ACC must identify the scorer ACTUALLY loaded by the model hook')
    return result


def run(matrix,bridge,out,gpus=('0','2','4'),seconds=21600,reserve=1800,
        retries=1,allow_synthetic=False,stop_event=None,drain_event=None):
    if matrix.get('schema')!='e020_matrix_v1':raise ValueError('wrong matrix schema')
    if matrix['kind']!='real' and not allow_synthetic:raise ValueError('synthetic matrix is test-only')
    if not gpus or len(set(gpus))!=len(gpus) or seconds<=reserve or reserve<0:
        raise ValueError('invalid GPU list/deadline')
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=True)
    matrix_hash=hashlib.sha256(json.dumps([j['job_hash'] for j in matrix['jobs']]).encode()).hexdigest()
    state_path=out/'state.json'; results={}; states={}; attempts={}
    if state_path.exists():
        old=json.loads(state_path.read_text())
        if old['runtime_hash']!=matrix['runtime_hash'] or old['phase']!=matrix['phase'] or old.get('matrix_hash')!=matrix_hash:
            raise ValueError('output already belongs to another runtime/phase')
        results=old.get('results',{}); attempts=old.get('attempts',{})
    jobs={j['job_id']:j for j in matrix['jobs']}
    if len(jobs)!=len(matrix['jobs']):raise ValueError('duplicate job IDs in matrix')
    visiting=set();visited=set()
    def check_dependency(jid):
        if jid in visiting:raise ValueError('cyclic job dependency')
        if jid in visited:return
        if jid not in jobs:raise ValueError('job dependency absent from matrix: '+jid)
        visiting.add(jid)
        for parent in jobs[jid].get('depends_on',[]):check_dependency(parent)
        visiting.remove(jid);visited.add(jid)
    for jid in jobs:check_dependency(jid)
    for jid,job in jobs.items():
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',jid):raise ValueError('unsafe job ID')
        if jid in results:
            validate_result(results[jid],job); states[jid]='done'
        else:states[jid]='failed' if attempts.get(jid,0)>retries else 'pending'
    start=time.monotonic(); deadline=start+seconds; active={}; estimates={}
    def save():
        data={'runtime_hash':matrix['runtime_hash'],'matrix_hash':matrix_hash,'phase':matrix['phase'],
              'states':states,'attempts':attempts,'results':results,
              'elapsed_seconds':time.monotonic()-start}
        tmp=state_path.with_suffix('.tmp'); tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2))
        tmp.replace(state_path)
    def stop_all():
        for info in active.values():
            if info['proc'].poll() is None:
                try:os.killpg(info['proc'].pid,signal.SIGTERM)
                except ProcessLookupError:pass
        until=time.monotonic()+2
        while time.monotonic()<until and any(i['proc'].poll() is None for i in active.values()):
            time.sleep(.05)
        for jid,info in active.items():
            if info['proc'].poll() is None:
                try:os.killpg(info['proc'].pid,signal.SIGKILL)
                except ProcessLookupError:pass
            info['log'].close(); states[jid]='interrupted'
    try:
        while True:
            now=time.monotonic()
            if now>=deadline or (stop_event is not None and stop_event.is_set()):break
            # Collect completions; one failed method does not block other rows.
            for jid,info in list(active.items()):
                code=info['proc'].poll()
                if code is None:continue
                info['log'].close(); del active[jid]
                try:
                    if code!=0:raise RuntimeError('child exit '+str(code))
                    result=validate_result(bridge.read_result(jobs[jid],str(info['folder'])),jobs[jid])
                    result['queue_wall_seconds']=time.monotonic()-info['start']
                    result['gpu_assignment']=info['gpu']; results[jid]=result; states[jid]='done'
                    estimates[jobs[jid]['task']]=result['queue_wall_seconds']
                except Exception as error:
                    (info['folder']/'queue_error.txt').write_text(repr(error))
                    states[jid]='pending' if attempts[jid]<=retries else 'failed'
                print(json.dumps({'job':jid,'status':states[jid]},ensure_ascii=False),flush=True)
                save()
            busy={i['gpu'] for i in active.values()}
            launched=False
            for gpu in gpus:
                if gpu in busy:continue
                remaining=deadline-time.monotonic()-reserve
                if remaining<=0:continue
                if drain_event is not None and drain_event.is_set():continue
                available=[j for j in matrix['jobs'] if states[j['job_id']]=='pending'
                           and attempts.get(j['job_id'],0)<=retries
                           and all(states[p]=='done' for p in j.get('depends_on',[]))
                           and estimates.get(j['task'],j.get('estimated_seconds',180))<=remaining]
                if not available:continue
                job=available[0]; jid=job['job_id']
                # Full prior outputs may be reused only through identity-checked
                # project records, never by copying a rounded chat table.
                if hasattr(bridge,'cached_result'):
                    try:
                        cached=bridge.cached_result(job)
                        if cached is not None:
                            results[jid]=validate_result(cached,job); states[jid]='done'; save(); launched=True; continue
                    except Exception as error:
                        (out/(jid+'__cache_rejected.txt')).write_text(repr(error))
                attempts[jid]=attempts.get(jid,0)+1
                folder=out/jid/('attempt_'+str(attempts[jid])); folder.mkdir(parents=True,exist_ok=True)
                (folder/'job.json').write_text(json.dumps(job,ensure_ascii=False,indent=2))
                try:
                    argv=bridge.command(job,str(folder))
                    if not isinstance(argv,list) or not argv or not all(isinstance(a,str) and '\0' not in a for a in argv):
                        raise ValueError('bridge.command must return argv list; shell is never used')
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
                    logfile=(folder/'process.log').open('w')
                    try:proc=subprocess.Popen(argv,stdout=logfile,stderr=subprocess.STDOUT,
                                               env=env,start_new_session=True,
                                               cwd=getattr(bridge,'working_directory',None))
                    except Exception:logfile.close();raise
                    active[jid]={'proc':proc,'log':logfile,'gpu':gpu,'folder':folder,'start':time.monotonic()}
                    states[jid]='running'; launched=True
                except Exception as error:
                    (folder/'queue_error.txt').write_text(repr(error))
                    states[jid]='pending' if attempts[jid]<=retries else 'failed'
                save()
            remaining=deadline-time.monotonic()-reserve
            eligible=any(states[j['job_id']]=='pending' and attempts.get(j['job_id'],0)<=retries
                         and all(states[p]=='done' for p in j.get('depends_on',[]))
                         and not (drain_event is not None and drain_event.is_set())
                         and estimates.get(j['task'],j.get('estimated_seconds',180))<=remaining
                         for j in matrix['jobs'])
            if not active and not eligible:break
            time.sleep(.1)
    finally:
        for jid,job in jobs.items():
            if states[jid]=='pending' and any(states[p] in ('failed','blocked_dependency') for p in job.get('depends_on',[])):
                states[jid]='blocked_dependency'
        stop_all();save()
    return json.loads(state_path.read_text())


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('matrix');ap.add_argument('--bridge',required=True,help='module:factory')
    ap.add_argument('--bridge-config',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--gpus',default='0,2,4');ap.add_argument('--seconds',type=float,default=21600)
    ap.add_argument('--reserve',type=float,default=1800);ap.add_argument('--retries',type=int,default=1)
    ap.add_argument('--allow-synthetic',action='store_true')
    args=ap.parse_args();module,factory=args.bridge.split(':')
    bridge=getattr(importlib.import_module(module),factory)(json.loads(Path(args.bridge_config).read_text()))
    matrix=json.loads(Path(args.matrix).read_text())
    result=run(matrix,bridge,args.out,args.gpus.split(','),args.seconds,args.reserve,args.retries,args.allow_synthetic)
    print(json.dumps({'done':sum(v=='done' for v in result['states'].values()),
                      'total':len(result['states']),'elapsed_seconds':result['elapsed_seconds']}))
