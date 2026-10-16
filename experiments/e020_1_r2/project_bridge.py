"""Generic real-command bridge: no fabricated model/evaluator implementation.

The existing successful ACC runner reads the supplied job.json and writes
result.json. Cursor maps that single command, and adds the actual selector hook.
No command string or shell interpolation is executed.
"""
import json
from pathlib import Path


class Bridge:
    def __init__(self,config):
        self.config=config;self.working_directory=config.get('working_directory')

    def command(self,job,outdir):
        key='TTFT' if job['task']=='TTFT' else ('STRESS' if job['phase']=='stress' else 'ACC')
        template=self.config['commands'].get(key)
        if not isinstance(template,list) or not template or not all(isinstance(a,str) for a in template):
            raise ValueError('map the existing successful '+key+' runner argv in bridge config')
        values={'job_json':str(Path(outdir)/'job.json'),'outdir':outdir,
                'result_json':str(Path(outdir)/'result.json')}
        return [a.format_map(values) for a in template]

    def read_result(self,job,outdir):
        return json.loads((Path(outdir)/'result.json').read_text())

    def cached_result(self,job):
        # Index is produced by the project from real raw predictions and exact
        # inputs. Rounded chat tables cannot supply an identity-checked result.
        index=self.config.get('cache_index')
        if not index:return None
        mapping=json.loads(Path(index).read_text())
        result=mapping.get(job['job_hash'])
        return json.loads(Path(result).read_text()) if isinstance(result,str) else result


def make(config):return Bridge(config)
