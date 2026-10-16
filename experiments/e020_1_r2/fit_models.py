"""Image-disjoint fitting; no validation/test actions used as training labels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from features import FEATURE_SCHEMA
from scorer_numpy import fit, image_split
from policy_protocol import action_identity, POLICY_SCHEMA, DEFAULT_POLICY, fingerprint
from policy_calibration import calibrate


def train(files,out,*,epochs=160,hidden=32,forbidden_images=(),modes=('current','full','no_context','shuffled'),
          min_calibration_images=12,allow_synthetic=False):
    pairs,labels,records,noise=[],[],[],[]; seen={}; source_hashes=set()
    forbidden=set(map(str,forbidden_images));kinds=set()
    for file in files:
        with np.load(file,allow_pickle=False) as d:
            kind=str(d['kind']);kinds.add(kind)
            if str(d['schema']) != FEATURE_SCHEMA or not (kind=='real' or allow_synthetic and kind=='synthetic'):
                raise ValueError('only real, correct-schema collection shards may train models')
            if not len(d['pairs'])==len(d['labels'])==len(d['records']):
                raise ValueError('shard feature/label/record counts differ')
            for row in d.get('noop_records',[]):
                nr=json.loads(str(row))
                if nr.get('actually_recomputed'):noise.append(float(nr['nll_difference']))
            for x,y,r in zip(d['pairs'],d['labels'],d['records']):
                rec=json.loads(str(r))
                key=action_identity(rec)
                signature=fingerprint({'features':x.tolist(),'label':float(y)})
                if key in seen:
                    if seen[key]!=signature:raise ValueError('conflicting duplicate action features or labels')
                    continue
                if rec['label_kind'] != 'policy': raise ValueError('do not mix mechanism diagnostics into policy targets')
                if kind=='real' and rec.get('feature_origin')!='prompt_only_pre_event':
                    raise ValueError('real features must come from prompt-only pre-event capture')
                if rec.get('policy_schema')!=POLICY_SCHEMA:
                    raise ValueError('refit requires the revised candidate catalogue records')
                if (rec.get('group_size'),rec.get('candidate_limit'),rec.get('proposal_shortlist'))!=(1,16,16):
                    raise ValueError('main one-step fit cannot mix group sizes or search contracts')
                if str(rec['image_id']) in forbidden:
                    raise ValueError('collection overlaps held-out evaluation image IDs')
                seen[key]=signature;pairs.append(x);labels.append(y);records.append(rec);source_hashes.add(rec['source_hash'])
    if not pairs: raise ValueError('no paired actions collected')
    # source_hash should identify checkpoint, prompt, feature definition AND
    # cache/answer normalization policy, but NOT arm/depth or collection shard.
    if len(source_hashes) != 1: raise ValueError('incompatible model/protocol sources across shards')
    if len(kinds)!=1:raise ValueError('cannot mix synthetic and real shards')
    x,y=np.asarray(pairs),np.asarray(labels);image_ids=np.asarray([str(r['image_id']) for r in records])
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    data_hash=fingerprint({'records':records,'pairs':x.tolist(),'labels':y.tolist()})
    reports={}
    train_sel,validation,test=image_split(image_ids)
    train_images=sorted(set(image_ids[train_sel].tolist()))
    for mode in modes:
        if mode=='full' and not all(r.get('feature_availability',{}).get('history',False) for r in records):
            reports[mode]={'status':'UNAVAILABLE_MISSING_HISTORY',
                           'reason':'current-first fit continues; full requires actual history features'}
            continue
        model,report=fit(x,y,image_ids,mode=mode,hidden=hidden,epochs=epochs)
        calibration=calibrate(model,x,y,image_ids,records,validation,noise,min_images=min_calibration_images)
        metadata={'mode':mode,'source_hash':next(iter(source_hashes)),'dataset_hash':data_hash,
                  'training_kind':next(iter(kinds)),
                  'trained_image_ids':train_images,'parameter_count':int(model.w.size+model.b.size+model.v.size),
                  'target':'baseline_answer_mean_nll_minus_swap_answer_mean_nll',
                  'label_count':len(y),'policy_schema':POLICY_SCHEMA,'deployment_policy':DEFAULT_POLICY,
                  'selection_image_ids':sorted(set(image_ids[validation].tolist())),
                  'diagnostic_test_image_ids':sorted(set(image_ids[test].tolist())),
                  'decision_threshold':calibration['decision_threshold'],
                  'accept_enabled':calibration['accept_enabled'],'calibration':calibration}
        model.save(out/(mode+'.npz'),metadata);report['calibration']=calibration;reports[mode]=report
    (out/'training_report.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2))
    return reports


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('files',nargs='+');ap.add_argument('--out',required=True)
    ap.add_argument('--epochs',type=int,default=160);ap.add_argument('--hidden',type=int,default=32)
    ap.add_argument('--forbidden-image-ids')
    ap.add_argument('--modes',default='current,full,no_context,shuffled')
    ap.add_argument('--min-calibration-images',type=int,default=12)
    a=ap.parse_args();forbidden=json.loads(Path(a.forbidden_image_ids).read_text()) if a.forbidden_image_ids else []
    print(json.dumps(train(a.files,a.out,epochs=a.epochs,hidden=a.hidden,forbidden_images=forbidden,
                           modes=a.modes.split(','),min_calibration_images=a.min_calibration_images)))
