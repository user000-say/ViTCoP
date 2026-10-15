"""Image-disjoint fitting; no validation/test actions used as training labels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from features import FEATURE_SCHEMA
from scorer_numpy import fit, image_split


def train(files,out,*,epochs=160,hidden=32,forbidden_images=()):
    pairs,labels,records=[],[],[]; seen=set(); source_hashes=set()
    for file in files:
        with np.load(file,allow_pickle=False) as d:
            if str(d['schema']) != FEATURE_SCHEMA or str(d['kind']) != 'real':
                raise ValueError('only real, correct-schema collection shards may train models')
            for x,y,r in zip(d['pairs'],d['labels'],d['records']):
                rec=json.loads(str(r))
                key=(rec['prefix_hash'],tuple(rec['baseline_patch_ids']),tuple(rec['donor_patch_ids']),tuple(rec['incoming_patch_ids']))
                if key in seen: continue
                if rec['label_kind'] != 'policy': raise ValueError('do not mix mechanism diagnostics into policy targets')
                if str(rec['image_id']) in set(map(str,forbidden_images)):
                    raise ValueError('collection overlaps held-out evaluation image IDs')
                seen.add(key);pairs.append(x);labels.append(y);records.append(rec);source_hashes.add(rec['source_hash'])
    if not pairs: raise ValueError('no paired actions collected')
    # source_hash should identify checkpoint, prompt, feature definition AND
    # cache/answer normalization policy, but NOT arm/depth or collection shard.
    if len(source_hashes) != 1: raise ValueError('incompatible model/protocol sources across shards')
    x,y=np.asarray(pairs),np.asarray(labels);image_ids=np.asarray([str(r['image_id']) for r in records])
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    data_hash=hashlib.sha256(x.tobytes()+y.tobytes()+'|'.join(image_ids).encode()).hexdigest()
    reports={}
    train_sel,_,_=image_split(image_ids)
    train_images=sorted(set(image_ids[train_sel].tolist()))
    for mode in ('full','current','no_context','shuffled'):
        model,report=fit(x,y,image_ids,mode=mode,hidden=hidden,epochs=epochs)
        metadata={'mode':mode,'source_hash':next(iter(source_hashes)),'dataset_hash':data_hash,
                  'training_kind':'real',
                  'trained_image_ids':train_images,'parameter_count':int(model.w.size+model.b.size+model.v.size),
                  'target':'baseline_answer_mean_nll_minus_swap_answer_mean_nll',
                  'label_count':len(y)}
        model.save(out/(mode+'.npz'),metadata);reports[mode]=report
    (out/'training_report.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2))
    return reports


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('files',nargs='+');ap.add_argument('--out',required=True)
    ap.add_argument('--epochs',type=int,default=160);ap.add_argument('--hidden',type=int,default=32)
    ap.add_argument('--forbidden-image-ids')
    a=ap.parse_args();forbidden=json.loads(Path(a.forbidden_image_ids).read_text()) if a.forbidden_image_ids else []
    print(json.dumps(train(a.files,a.out,epochs=a.epochs,hidden=a.hidden,forbidden_images=forbidden)))
