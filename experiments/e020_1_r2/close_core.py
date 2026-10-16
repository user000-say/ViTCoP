"""Finish the EXISTING E020 dependency chain before a new algorithm search.

Optional sidecar runs pending E019/E020 controls on two GPUs during collection,
then drains: no new sidecar launches, running jobs finish without being killed.
The third ACC GPU collects pairs. Fourth GPU measures fixed baselines in parallel.
No learned jobs are claimed tested merely because an independent queue finished.
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
from pathlib import Path
import threading
import time
from build_matrix import build
from core_matrix import build_core,confirm_matrix
from core_progress import inspect
from diagnostics import diagnose
from fit_models import train
from run_matrix import run
from summarize import aggregate
from workflow import collection,extra_matrix
from measurement_anchors import external_anchors


def main():
    p=argparse.ArgumentParser();p.add_argument('runtime');p.add_argument('--bridge',default='project_bridge:make')
    p.add_argument('--bridge-config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--gpus',default='0,2,4');p.add_argument('--timing-gpu',default='6');p.add_argument('--hours',type=float,default=6)
    p.add_argument('--collection-seconds',type=float,default=5400);p.add_argument('--screen-seconds',type=float,default=5400)
    p.add_argument('--reserve-seconds',type=float,default=900);p.add_argument('--sidecar-matrix')
    p.add_argument('--sidecar-bridge');p.add_argument('--sidecar-bridge-config');p.add_argument('--skip-timing',action='store_true')
    p.add_argument('--allow-synthetic',action='store_true');a=p.parse_args()
    runtime=json.loads(Path(a.runtime).read_text());bc=json.loads(Path(a.bridge_config).read_text())
    mod,fac=a.bridge.split(':');factory=getattr(importlib.import_module(mod),fac)
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    gpus=a.gpus.split(',');stop=threading.Event();drain=threading.Event()
    if len(gpus)!=3 or len(set(gpus+[a.timing_gpu]))!=4:raise ValueError('three ACC GPUs plus one exclusive timing GPU required')
    if a.hours*3600<=a.reserve_seconds:raise ValueError('insufficient wall budget')
    if runtime.get('kind')!='real' and not a.allow_synthetic:raise ValueError('real runtime required')
    if runtime.get('kind')=='real' and not runtime.get('schedules',{}).get('FIRST_NEXT_WMATCH'):
        raise ValueError('export the ACTUAL already-run NEXT event list; do not replace it with a reconstructed floor/round choice')
    deadline=time.monotonic()+a.hours*3600
    remaining=lambda:max(0.,deadline-time.monotonic())
    def write(name,data):
        target=out/name;tmp=target.with_suffix('.tmp')
        tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(target)
    declared=list(runtime.get('pair_shards',[]))
    output=runtime.get('collection',{}).get('output_shard')
    if output:declared.append(output)
    files=list(dict.fromkeys(str(Path(f).resolve()) for f in declared if Path(f).is_file()))
    status=inspect(runtime,files);write('core_progress.json',status)
    if not status['primary_models_ready'] and not files and not runtime.get('collection',{}).get('command'):
        write('core_blocked.json',{'core_hypothesis_tested':False,'reason':'No real paired shard and no collection command. Map the existing teacher-forced backend through replay_adapter; do not launch another independent sweep as a substitute.'})
        print('CORE NOT TESTED: configure collection.command or provide real pair_shards.',flush=True)
        return 2
    pool=ThreadPoolExecutor(max_workers=2);side=None;timing=None;matrix=state=None
    try:
        if a.sidecar_matrix:
            m=json.loads(Path(a.sidecar_matrix).read_text())
            if m.get('schema')=='e019_matrix_v1':m={**m,'schema':'e020_matrix_v1'}
            smod,sfac=(a.sidecar_bridge or a.bridge).split(':');sf=getattr(importlib.import_module(smod),sfac)
            sc=json.loads(Path(a.sidecar_bridge_config).read_text()) if a.sidecar_bridge_config else bc
            side=pool.submit(run,m,sf(sc),out/'sidecar',tuple(gpus[:2]),remaining(),a.reserve_seconds,1,a.allow_synthetic,stop,drain)
        if not a.skip_timing:
            refs=build(runtime,'confirm','independent')
            names={'COVER__FIRST_NEXT_WMATCH__reference','COVER__FIRST_SHIFT__reference',
                   'BASE__FIRST_SHIFT__reference','BASE__FIRST_KPLUS8__reference',
                   'COVER__EARLY__reference','BASE__EARLY__reference'}
            refs['configurations']=[c for c in refs['configurations'] if c['config_id'] in names]+external_anchors(runtime)
            tm=extra_matrix(refs,runtime,'timing')
            if tm:
                write('initial_timing_matrix.json',tm)
                timing=pool.submit(run,tm,factory(bc),out/'initial_timing',(a.timing_gpu,),remaining(),a.reserve_seconds,1,a.allow_synthetic,stop)
        if not status['primary_models_ready']:
            def fit_available():
                forbidden=runtime.get('forbidden_evaluation_image_ids',[])
                if isinstance(forbidden,str):forbidden=json.loads(Path(forbidden).read_text())
                try:
                    fitted=train(files,runtime['model_directory'],epochs=runtime.get('fit_epochs',160),
                                 forbidden_images=forbidden,modes=('current','no_context','shuffled','full'),
                                 allow_synthetic=a.allow_synthetic)
                    write('fitting_report.json',fitted)
                except Exception as e:write('fit_error.json',{'error':repr(e)})
            if files:fit_available()
            status=inspect(runtime,files)
            if not status['primary_models_ready'] and runtime.get('collection',{}).get('command') and remaining()>a.reserve_seconds:
                # A partial/old shard must not permanently prevent collection.
                # Preserve it and write a new attempt rather than overwrite it.
                cr=copy.deepcopy(runtime)
                if files:cr['collection']['output_shard']=str(out/('additional_pairs_'+str(time.time_ns())+'.npz'))
                new_files,report=collection(cr,out,gpus[2],min(a.collection_seconds,remaining()-a.reserve_seconds),stop)
                files=list(dict.fromkeys(files+new_files));write('collection_status.json',report)
                if files:fit_available()
        status=inspect(runtime,files);write('core_progress.json',status)
        if not status['primary_models_ready']:
            print('CORE NOT TESTED: no valid calibrated current scorer. See complete validation catalogues, no-op and source checks.',flush=True)
            return 2
        if files:
            try:write('heldout_mechanism_group1.json',diagnose(files,runtime['model_directory'],
                      allow_synthetic=a.allow_synthetic,group_size=1,shortlist=16))
            except Exception as e:write('mechanism_diagnostic_unavailable.json',{'error':repr(e)})
        # Finish owned sidecar jobs, then make all three ACC GPUs available.
        drain.set()
        if side:side.result()
        matrix=build_core(runtime,'screen',allow_missing_models=True);write('core_screen_matrix.json',matrix)
        if remaining()>a.reserve_seconds:
            state=run(matrix,factory(bc),out/'core_screen',tuple(gpus),min(a.screen_seconds,remaining()-a.reserve_seconds),0,1,a.allow_synthetic,stop)
            write('core_screen_summary.json',aggregate(matrix,state))
            write('core_progress.json',inspect(runtime,files,state,matrix))
            cm=confirm_matrix(runtime,matrix,state);write('core_confirm_matrix.json',cm)
            if timing:timing.result();timing=None
            tm=extra_matrix(cm,runtime,'timing') if not a.skip_timing else None
            if tm and remaining()>a.reserve_seconds:
                write('confirm_timing_matrix.json',tm)
                timing=pool.submit(run,tm,factory(bc),out/'confirm_timing',(a.timing_gpu,),remaining(),a.reserve_seconds,1,a.allow_synthetic,stop)
            if remaining()>a.reserve_seconds:
                cs=run(cm,factory(bc),out/'core_confirm',tuple(gpus),remaining(),a.reserve_seconds,1,a.allow_synthetic,stop)
                write('core_confirm_summary.json',aggregate(cm,cs))
                write('confirm_progress.json',inspect(runtime,files,cs,cm))
            if a.sidecar_matrix and remaining()>a.reserve_seconds:
                # Resume the same original sidecar manifest/state after core ACC.
                run(m,sf(sc),out/'sidecar',tuple(gpus),remaining(),a.reserve_seconds,1,a.allow_synthetic,stop)
            if timing:timing.result();timing=None
        return 0 if state and inspect(runtime,files,state,matrix)['core_hypothesis_tested'] else 2
    finally:
        drain.set();stop.set();pool.shutdown(wait=True)
        write('final_core_progress.json',inspect(runtime,files,state,matrix))


if __name__=='__main__':raise SystemExit(main())
