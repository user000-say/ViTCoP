"""Aggregate only complete four-task configurations; never impute missing jobs."""
import argparse
import csv
import json
from pathlib import Path

DENOM={'GQA':61.9,'POPE':85.8,'MME':1852.,'MMBench':64.1}


def aggregate(matrix,state):
    output=[]
    for config in matrix['configurations']:
        jobs=[j for j in matrix['jobs'] if j['config']['config_id']==config['config_id']]
        if len(jobs)!=4:continue
        if not all(state.get('states',{}).get(j['job_id'])=='done' for j in jobs):continue
        values={j['task']:float(state['results'][j['job_id']]['metric_value']) for j in jobs}
        if set(values)!=set(DENOM):continue
        row={'config_id':config['config_id'],'stage1':config['stage1'],
             'schedule':config['schedule_name'],'selector':config['selector_name'],
             'visual_work':config['budget']['visual_token_layers'],
             'relative_work_difference':config['budget']['relative_difference'],
             'R4':25*sum(values[t]/DENOM[t] for t in DENOM),**values}
        output.append(row)
    baseline={r['stage1']:r for r in output if r['schedule']=='EARLY' and r['selector']=='reference'}
    for row in output:
        if row['stage1'] in baseline:
            b=baseline[row['stage1']]
            row['delta_R4_vs_EARLY']=row['R4']-b['R4']
            row['worst_normalized_task_delta']=min(100*(row[t]-b[t])/DENOM[t] for t in DENOM)
            row['delta_POPE']=row['POPE']-b['POPE']
            row['delta_MME']=row['MME']-b['MME']
    return output


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('matrix');ap.add_argument('state');ap.add_argument('--out',required=True)
    args=ap.parse_args()
    rows=aggregate(json.loads(Path(args.matrix).read_text()),json.loads(Path(args.state).read_text()))
    Path(args.out).write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    if rows:
        keys=sorted({k for row in rows for k in row})
        with Path(args.out).with_suffix('.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    print(json.dumps({'complete_configurations':len(rows),'phase':json.loads(Path(args.matrix).read_text())['phase'],
                      'warning':'screen-panel R4 is for development selection, not a full-benchmark result'}))
