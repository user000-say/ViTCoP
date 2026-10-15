"""Reproduce arithmetic from user's rounded aggregates; no new GPU results."""
import json
from pathlib import Path
from summarize import DENOM

ROWS={
 'COVER_ONE':[57.08,85.37,1784.8,59.45],
 'BASE_ONE':[56.76,85.04,1792.2,59.79],
 'COVER_EARLY':[57.57,84.42,1767.5,62.54],
 'BASE_EARLY':[57.18,84.03,1769.9,63.14],
 'COVER_LATE':[57.35,84.40,1764.9,63.23],
 'BASE_LATE':[57.27,83.93,1750.3,62.97],
}
TTFT={'COVER':{'ONE':117.3,'EARLY':118.8,'LATE':119.6},
      'BASE':{'ONE':115.9,'EARLY':117.9,'LATE':118.2}}


def report():
    metrics={key:dict(zip(DENOM,values)) for key,values in ROWS.items()}
    r4={key:25*sum(values[t]/DENOM[t] for t in DENOM) for key,values in metrics.items()}
    changes={}
    for arm in ['COVER','BASE']:
        for early,late in [('ONE','EARLY'),('EARLY','LATE')]:
            contributions={t:25*(metrics[arm+'_'+late][t]-metrics[arm+'_'+early][t])/DENOM[t] for t in DENOM}
            changes[arm+'_'+early+'_to_'+late]={'task_contributions_pp':contributions,
                'R4_change_pp':sum(contributions.values()),
                'TTFT_mean_change_ms':TTFT[arm][late]-TTFT[arm][early]}
    return {'source':'user-reported rounded task metrics, n=1024 TTFT cuda means',
            'R4_recomputed':r4,'changes':changes,'visual_work':{'EARLY':2284,'LATE':2564},
            'relative_extra_visual_work':2564/2284-1,
            'note':'No confidence intervals, per-sample oracle, or learned policy effects can be inferred from these aggregates.'}


if __name__=='__main__':print(json.dumps(report(),ensure_ascii=False,indent=2))
