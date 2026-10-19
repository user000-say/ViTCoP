"""CPU checks for the R3 control plane and unchanged R2 numerical modules."""
import compileall
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import re
from make_catalog import write_catalog, MAIN, SHAPES
from tail_pool_equivalence import verify as verify_pool


def main():
    root=Path(__file__).resolve().parent
    out=root/'validation';out.mkdir(exist_ok=True)
    write_catalog(root/'configs')
    assert compileall.compile_dir(str(root/'elva'),quiet=1)
    assert compileall.compile_dir(str(root/'tests'),quiet=1)
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.discover(str(root/'tests')))
    (out/'r3_tests.txt').write_text(stream.getvalue())
    env=dict(os.environ,PYTHONPATH=str(root))
    legacy=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=root/'legacy_reference',
                          env=env,capture_output=True,text=True)
    (out/'r2_regression.txt').write_text(legacy.stdout+legacy.stderr)
    legacy_log=legacy.stdout+legacy.stderr
    legacy_count=int(re.search(r'Ran (\d+) tests?',legacy_log).group(1))
    legacy_skip=re.search(r'OK \(skipped=(\d+)\)',legacy_log)
    legacy_skipped=int(legacy_skip.group(1)) if legacy_skip else 0
    budgets=[dict(budget=w,counts=p.counts,lengths=p.lengths,events=p.events,trace=p.trace,
                  work=p.work,compression_exact=str(p.compression(576))) for w,shapes in SHAPES.items() for p in shapes]
    for row in budgets:
        assert row['budget']==row['work']==sum(row['trace'])
    (out/'budgets.json').write_text(json.dumps(budgets,indent=2)+'\n')
    pool_check=verify_pool()
    (out/'tail_pool_equivalence.json').write_text(json.dumps(pool_check,indent=2)+'\n')
    catalog=write_catalog(root/'configs')
    core=[c for c in catalog['configs'] if c['phase']=='stage1_core']
    summary=dict(r3_tests=result.testsRun,r3_passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
                 r3_failures=len(result.failures),r3_errors=len(result.errors),r3_skipped=len(result.skipped),
                 r2_regression_exit=legacy.returncode,core_configs=len(core),core_four_task_jobs=4*len(core),
                 r2_tests=legacy_count,r2_skipped=legacy_skipped,r2_passed=legacy_count-legacy_skipped if legacy.returncode==0 else None,
                 optional_configs=sum(c['phase']=='stage1_optional' for c in catalog['configs']),budgets=[2048,1024],
                 high_entry=MAIN[2048].counts[0],low_entry=MAIN[1024].counts[0],
                 fixture_jobs_only=True,pretrained_vlm_accuracy_run=False,gpu_ttft_run=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    if not result.wasSuccessful() or legacy.returncode:
        print(stream.getvalue());print(legacy.stderr);raise SystemExit(1)


if __name__=='__main__':main()
