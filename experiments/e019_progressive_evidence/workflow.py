"""One shared wall-clock budget: screen -> confirm + GPU3 timing -> reserves.

All selectors are fixed before screen; independent confirmation is never used
to tune alpha/lambda. Extra stress panels must be predeclared in runtime JSON.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
from pathlib import Path
import threading
import time
from build_matrix import build,job_digest,runtime_digest
from promote import promote
from run_matrix import run
from summarize import aggregate


def extra_matrix(confirm,runtime,kind):
    jobs=[]
    if kind=='timing':
        panel=runtime.get('timing_panel')
        if not panel:return None
        specs=[{'task':'TTFT','metric_key':'TTFT_cuda_mean_ms','panel':panel,'tag':'1024'}]
    else:
        specs=runtime.get('stress_panels',[])
        if not specs:return None
    for spec in specs:
        if not spec['panel'].get('path') or not spec['panel'].get('ids_hash'):
            raise ValueError('freeze real extra-panel path and sample identity before execution')
    for config in confirm['configurations']:
        for spec in specs:
            job={'phase':kind,'kind':runtime['kind'],'task':spec['task'],
                 'config':config,'panel':spec['panel'],'source':runtime['source'],
                 'metric_key':spec['metric_key'],
                 'estimated_seconds':spec.get('estimated_seconds',300),'tag':spec['tag']}
            job['job_id']=config['config_id']+'__'+kind+'_'+spec['tag']
            job['job_hash']=job_digest(job);jobs.append(job)
    return {'schema':'e019_matrix_v1','phase':kind,'kind':runtime['kind'],
            'runtime_hash':runtime_digest(runtime),'configurations':confirm['configurations'],'jobs':jobs,
            'counts':{'configurations':len(confirm['configurations']),'task_jobs':len(jobs)}}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('runtime');ap.add_argument('--bridge',required=True)
    ap.add_argument('--bridge-config',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--hours',type=float,default=6);ap.add_argument('--screen-seconds',type=float,default=7200)
    ap.add_argument('--reserve-seconds',type=float,default=900);ap.add_argument('--top',type=int,default=4)
    ap.add_argument('--gpus',default='0,1,2');ap.add_argument('--timing-gpu',default='3')
    ap.add_argument('--allow-synthetic',action='store_true');args=ap.parse_args()
    runtime=json.loads(Path(args.runtime).read_text());bridge_config=json.loads(Path(args.bridge_config).read_text())
    module,factory=args.bridge.split(':');make=getattr(importlib.import_module(module),factory)
    gpus=args.gpus.split(',')
    if args.timing_gpu in gpus:raise ValueError('timing GPU must be exclusive')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();deadline=start+args.hours*3600
    def remaining():return max(0,deadline-time.monotonic())
    def write(name,value):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
    screen=build(runtime,'screen');write('screen_matrix.json',screen)
    screen_time=min(args.screen_seconds,remaining()-args.reserve_seconds)
    if screen_time<=0:raise ValueError('wall-clock budget too small')
    screen_state=run(screen,make(bridge_config),out/'screen',gpus,screen_time,0,
                     allow_synthetic=args.allow_synthetic)
    write('screen_summary.json',aggregate(screen,screen_state))
    try:confirm=promote(screen,screen_state,runtime,args.top)
    except ValueError as error:
        write('promotion_unavailable.json',{'error':str(error)})
        confirm=build(runtime,'confirm')
        anchors={a+'__'+s+'__reference' for a,s in [('BASE','EARLY'),('COVER','EARLY'),('COVER','LATE'),('BASE','ONE')]}
        confirm['configurations']=[c for c in confirm['configurations'] if c['config_id'] in anchors]
        confirm['jobs']=[j for j in confirm['jobs'] if j['config']['config_id'] in anchors]
        confirm['counts']={'configurations':len(confirm['configurations']),'task_jobs':len(confirm['jobs'])}
    write('confirm_matrix.json',confirm)
    timing=extra_matrix(confirm,runtime,'timing');stress=extra_matrix(confirm,runtime,'stress')
    stop=threading.Event();pool=ThreadPoolExecutor(max_workers=1);future=None
    try:
        if timing and remaining()>args.reserve_seconds:
            write('timing_matrix.json',timing)
            future=pool.submit(run,timing,make(bridge_config),out/'timing',(args.timing_gpu,),
                               remaining(),args.reserve_seconds,1,args.allow_synthetic,stop)
        if remaining()>args.reserve_seconds:
            result=run(confirm,make(bridge_config),out/'confirm',gpus,remaining(),args.reserve_seconds,
                       allow_synthetic=args.allow_synthetic,stop_event=stop)
            write('confirm_summary.json',aggregate(confirm,result))
        if stress and remaining()>args.reserve_seconds:
            write('stress_matrix.json',stress)
            run(stress,make(bridge_config),out/'stress',gpus,remaining(),args.reserve_seconds,
                allow_synthetic=args.allow_synthetic,stop_event=stop)
        if future:future.result()
    finally:
        stop.set();pool.shutdown(wait=True)
        write('workflow_status.json',{'elapsed_seconds':time.monotonic()-start,
              'wall_clock_limit_seconds':args.hours*3600,'screen_jobs':len(screen['jobs']),
              'confirm_jobs':len(confirm['jobs']),'timing_jobs':len(timing['jobs']) if timing else 0,
              'stress_jobs':len(stress['jobs']) if stress else 0,
              'note':'See per-phase state.json for done/failed/pending; no missing metrics are imputed.'})


if __name__=='__main__':main()
