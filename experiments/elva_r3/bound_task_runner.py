"""CLI entrypoint for the existing source-bound VLM generation/scorer loop.

The project callback returns actual records and official metrics. No model,
dataset, cache semantics, native selector or score is guessed here.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
from elva.cover_binding import check_sources
from elva.workflow import validate_binding, blocked_reason
from native_result_io import finish_task


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','binding','task','out','run-key'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    binding=json.loads(Path(args.binding).read_text())
    validate_binding(binding)
    reason=blocked_reason(config,binding)
    if reason:raise RuntimeError(reason)
    project_root=binding.get('project_root')
    if project_root:
        sys.path.insert(0,str(Path(project_root).resolve()))
    entry=binding.get('task_entrypoint','')
    if not isinstance(entry,str) or entry.count(':')!=1:
        raise RuntimeError('bind actual project module:function task_entrypoint')
    module,name=entry.split(':')
    function=getattr(importlib.import_module(module),name)
    check_sources({'source_files':binding.get('task_source_files',{})},function)
    value=function(config=config,binding=binding,task=args.task,
                   run_key=args.run_key,output_dir=str(Path(args.out).parent))
    if not isinstance(value,dict) or set(('records','official_metrics'))-set(value):
        raise ValueError('project callback must return actual per-sample records and official_metrics')
    finish_task(args.out,records=value['records'],official_metrics=value['official_metrics'],
                config=config,binding=binding,task=args.task,run_key=args.run_key,fixture=False)


if __name__=='__main__':main()
