"""A bounded six-hour batch with controls, real counterfactual data and fitting.

Independent jobs and collection start concurrently on different GPUs. No human
approval between rows. One failed new method does not block the other axes.
Only the actual project runner evaluates models; no model backend is simulated.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from build_matrix import build, job_digest, runtime_digest
from run_matrix import run
from summarize import aggregate,promote
from fit_models import train


def collection(runtime,out,gpu,seconds,stop_event=None):
    spec=runtime.get('collection',{});command=spec.get('command')
    shard=Path(spec.get('output_shard',str(out/'pairs.npz')))
    if spec.get('reuse_completed_shard') and shard.is_file():return [str(shard)],{'reused':True}
    if not command:return [],{'error':'No collection adapter command exported; controls continue.'}
    shard.parent.mkdir(parents=True,exist_ok=True)
    argv=[a.format_map({'out':str(shard.resolve())}) for a in command]
    logfile=(out/'collection.log').open('w')
    try:
        proc=subprocess.Popen(argv,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),
                              cwd=spec.get('working_directory'),stdout=logfile,stderr=subprocess.STDOUT,
                              start_new_session=True)
        until=time.monotonic()+max(seconds,1)
        while proc.poll() is None and time.monotonic()<until and not (stop_event and stop_event.is_set()):
            time.sleep(.1)
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        code=proc.returncode
        if shard.is_file():return [str(shard)],{'exit_code':code,'partial_or_complete_shard':True}
        return [],{'exit_code':code,'error':'No paired-action shard; learned branch unavailable.'}
    finally:logfile.close()


def extra_matrix(base,runtime,kind):
    specs=runtime.get('stress_panels',[]) if kind=='stress' else ([runtime['timing_panel']] if runtime.get('timing_panel') else [])
    if not specs:return None
    jobs=[]
    for c in base['configurations']:
        for spec in specs:
            job={'phase':kind,'kind':runtime['kind'],'task':spec.get('task','TTFT'),
                 'metric_key':spec.get('metric_key','TTFT_cuda_mean_ms'),
                 'panel':spec['panel'],'source':runtime['source'],'config':c,
                 'estimated_seconds':spec.get('estimated_seconds',300)}
            if kind=='timing':
                job['timing_protocol']={'paired_reference_config_id':'COVER__FIRST_SHIFT__reference',
                                        'interleave':'ABBA','samples':spec.get('n',1024),
                                        'include_feature_building_and_selector':True,
                                        'report':['cuda_mean','cuda_median','cuda_p90','host_wall','paired_delta']}
            job['job_id']=c['config_id']+'__'+kind+'_'+spec.get('tag',job['task'])
            job['job_hash']=job_digest(job);jobs.append(job)
    return {'schema':'e020_matrix_v1','kind':runtime['kind'],'phase':kind,
            'runtime_hash':runtime_digest(runtime),'configurations':base['configurations'],'jobs':jobs,
            'counts':{'configurations':len(base['configurations']),'task_jobs':len(jobs)}}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('runtime');ap.add_argument('--bridge',default='project_bridge:make')
    ap.add_argument('--bridge-config',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--gpus',default='0,2,4');ap.add_argument('--timing-gpu',default='6')
    ap.add_argument('--hours',type=float,default=6);ap.add_argument('--independent-seconds',type=float,default=3600)
    ap.add_argument('--collection-seconds',type=float,default=3600);ap.add_argument('--learned-screen-seconds',type=float,default=5400)
    ap.add_argument('--reserve-seconds',type=float,default=900);ap.add_argument('--skip-initial-timing',action='store_true')
    ap.add_argument('--allow-synthetic',action='store_true');a=ap.parse_args()
    runtime=json.loads(Path(a.runtime).read_text());bc=json.loads(Path(a.bridge_config).read_text())
    mod,fac=a.bridge.split(':');factory=getattr(importlib.import_module(mod),fac)
    gpus=a.gpus.split(',')
    if len(gpus)!=3 or len(set(gpus+[a.timing_gpu]))!=4:raise ValueError('three ACC GPUs and one exclusive timing GPU required')
    if a.hours*3600<=a.reserve_seconds:raise ValueError('wall budget too small')
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+a.hours*3600;stop=threading.Event()
    def remaining():return max(deadline-time.monotonic(),0.)
    def write(name,value):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
    matrices=[];states=[];timing_future=None
    independent=build(runtime,'screen','independent');write('independent_matrix.json',independent)
    pool=ThreadPoolExecutor(max_workers=3)
    try:
        cf=pool.submit(collection,runtime,out,gpus[2],min(a.collection_seconds,remaining()-a.reserve_seconds),stop)
        independent_future=pool.submit(run,independent,factory(bc),out/'independent',tuple(gpus[:2]),
                                       min(a.independent_seconds,remaining()-a.reserve_seconds),0,1,a.allow_synthetic,stop)
        initial=[c for c in independent['configurations'] if c['selector_name']=='reference' and
                 c['schedule_name'] in ('FIRST_SHIFT','EARLY','THREE_WMATCH')]
        if not a.skip_initial_timing:
            base={**independent,'configurations':initial};timing=extra_matrix(base,runtime,'timing')
            if timing:
                write('initial_timing_matrix.json',timing)
                timing_future=pool.submit(run,timing,factory(bc),out/'initial_timing',(a.timing_gpu,),
                                         remaining(),a.reserve_seconds,1,a.allow_synthetic,stop)
        files,report=cf.result();write('collection_status.json',report)
        if files:
            try:
                forbidden=runtime.get('forbidden_evaluation_image_ids',[])
                if isinstance(forbidden,str):forbidden=json.loads(Path(forbidden).read_text())
                fitted=train(files,runtime['model_directory'],epochs=runtime.get('fit_epochs',160),
                             forbidden_images=forbidden)
                write('fitting_report.json',fitted)
                from diagnostics import diagnose
                write('heldout_mechanism_diagnostics.json',diagnose(files,runtime['model_directory']))
            except Exception as error:write('fitting_unavailable.json',{'error':repr(error)})
        s=independent_future.result();matrices.append(independent);states.append(s)
        write('independent_summary.json',aggregate(independent,s))
        if remaining()>a.reserve_seconds:
            try:learned=build(runtime,'screen','learned',allow_missing_models=True)
            except Exception as error:
                write('learned_unavailable.json',{'error':repr(error)})
                learned={**independent,'configurations':[],'jobs':[],
                         'counts':{'configurations':0,'task_jobs':0}}
            write('learned_matrix.json',learned)
            if learned['jobs']:
                seconds=min(a.learned_screen_seconds,remaining()-a.reserve_seconds)
                s=run(learned,factory(bc),out/'learned',tuple(gpus),seconds,0,1,a.allow_synthetic,stop)
                matrices.append(learned);states.append(s);write('learned_summary.json',aggregate(learned,s))
        confirm=promote(runtime,matrices,states,top=4);write('confirm_matrix.json',confirm)
        # Timing uses a single physical device throughout; never compare raw
        # means collected on different cards as if they were paired measurements.
        if timing_future:timing_future.result();timing_future=None
        timing=extra_matrix(confirm,runtime,'timing')
        if timing and remaining()>a.reserve_seconds:
            write('final_timing_matrix.json',timing)
            timing_future=pool.submit(run,timing,factory(bc),out/'final_timing',(a.timing_gpu,),
                                     remaining(),a.reserve_seconds,1,a.allow_synthetic,stop)
        if remaining()>a.reserve_seconds:
            s=run(confirm,factory(bc),out/'confirm',tuple(gpus),remaining(),a.reserve_seconds,1,a.allow_synthetic,stop)
            write('confirm_summary.json',aggregate(confirm,s))
        stress=extra_matrix(confirm,runtime,'stress')
        if stress and remaining()>a.reserve_seconds:
            write('stress_matrix.json',stress)
            run(stress,factory(bc),out/'stress',tuple(gpus),remaining(),a.reserve_seconds,1,a.allow_synthetic,stop)
        if timing_future:timing_future.result()
    finally:
        stop.set();pool.shutdown(wait=True)
        write('workflow_status.json',{'wall_clock_limit_hours':a.hours,'remaining_seconds':remaining(),
              'screen_capacity':sum(len(m['jobs']) for m in matrices),
              'note':'Only per-phase done records count as completed experiments; missing metrics are not imputed.'})


if __name__=='__main__':main()
