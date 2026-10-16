"""Shared small value network; signed pair score is antisymmetric by design."""
import hashlib
import json
from pathlib import Path
import numpy as np
from features import INPUT_DIM, FEATURE_SCHEMA, feature_mask


def image_split(ids, salt='e020-image-split-v1'):
    r = np.asarray([int(hashlib.sha256((salt+str(s)).encode()).hexdigest()[:12], 16)/16**12 for s in ids])
    return r < .65, (r >= .65) & (r < .82), r >= .82


class PairScorer:
    def __init__(self, mean, std, mask, w, b, v, target_scale=1.):
        self.mean, self.std, self.mask = map(np.asarray, (mean, std, mask))
        self.w, self.b, self.v = map(np.asarray, (w, b, v))
        self.target_scale = float(target_scale)
        if (self.mean.shape!=(INPUT_DIM,) or self.std.shape!=(INPUT_DIM,) or self.mask.shape!=(INPUT_DIM,)
            or self.w.ndim!=2 or self.w.shape[0]!=INPUT_DIM or self.b.shape!=(self.w.shape[1],)
            or self.v.shape!=self.b.shape or np.any(self.std<=0) or self.target_scale<=0
            or not all(np.isfinite(a).all() for a in (self.mean,self.std,self.mask,self.w,self.b,self.v))
            or not np.isfinite(self.target_scale)):
            raise ValueError('invalid scorer dimensions, normalization or weights')

    def predict(self, pairs):
        pairs = np.asarray(pairs, dtype=float)
        if pairs.ndim != 3 or pairs.shape[1:] != (2, INPUT_DIM) or not np.isfinite(pairs).all():
            raise ValueError('pair inputs must be finite [M,2,50]')
        z = (pairs-self.mean)/self.std*self.mask
        value = np.tanh(z@self.w+self.b)@self.v
        return (value[:, 0]-value[:, 1])*self.target_scale

    def save(self, path, metadata):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, mean=self.mean, std=self.std, mask=self.mask,
                            w=self.w, b=self.b, v=self.v, target_scale=self.target_scale,
                            metadata=json.dumps({'schema':FEATURE_SCHEMA, **metadata}, sort_keys=True))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(str(d['metadata']))
            if meta['schema'] != FEATURE_SCHEMA:
                raise ValueError('model feature schema mismatch')
            return cls(*(d[k].copy() for k in ('mean','std','mask','w','b','v')), float(d['target_scale'])), meta


def fit(pairs, labels, image_ids, *, mode='full', hidden=32, epochs=160,
        batch_size=256, lr=.003, decay=.0001, seed=20):
    x, y = np.asarray(pairs, dtype=float), np.asarray(labels, dtype=float)
    if x.ndim != 3 or x.shape[1:] != (2, INPUT_DIM) or y.shape != (len(x),) or len(image_ids) != len(x):
        raise ValueError('dataset shape mismatch')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or hidden < 1 or epochs < 1:
        raise ValueError('invalid training data or hyperparameters')
    train, val, test = image_split(image_ids)
    if min(train.sum(), val.sum(), test.sum()) < 4:
        raise ValueError('need >=4 paired actions in each IMAGE-disjoint split')
    rng = np.random.default_rng(seed)
    mu = x[train].reshape(-1, INPUT_DIM).mean(0)
    sd = np.maximum(x[train].reshape(-1, INPUT_DIM).std(0), .05)
    mask = feature_mask(mode)
    scale = max(float(np.sqrt(np.mean(y[train]**2))), 1e-4)
    z = (x-mu)/sd*mask
    target = y/scale
    if mode == 'shuffled':
        target = target.copy(); target[train] = rng.permutation(target[train])
    w = rng.normal(0, .08, (INPUT_DIM, hidden)); b = np.zeros(hidden); v = rng.normal(0, .08, hidden)
    params = [w, b, v]; moment = [np.zeros_like(a) for a in params]; variance = [a.copy() for a in moment]
    step, best, best_val, best_epoch = 0, None, float('inf'), 0
    indices = np.flatnonzero(train)
    for epoch in range(epochs):
        for start in range(0, len(indices), batch_size):
            if start == 0:
                order = rng.permutation(indices)
            ix = order[start:start+batch_size]; zz = z[ix]
            h = np.tanh(zz@w+b); pred = (h[:,0]-h[:,1])@v
            error = pred-target[ix]
            # Huber loss limits a few catastrophic suffix labels dominating fit.
            d = np.clip(error, -2., 2.)/len(ix)
            dh = np.stack([d[:,None]*v, -d[:,None]*v], axis=1)*(1-h*h)
            gradients = [np.einsum('nsi,nsj->ij', zz, dh)+decay*w,
                         dh.sum((0,1)), (h[:,0]-h[:,1]).T@d+decay*v]
            step += 1
            for p, g, m, vv in zip(params, gradients, moment, variance):
                m *= .9; m += .1*g; vv *= .999; vv += .001*g*g
                p -= lr*(m/(1-.9**step))/(np.sqrt(vv/(1-.999**step))+1e-8)
        model = PairScorer(mu, sd, mask, w, b, v, scale)
        vl = float(np.mean((model.predict(x[val])-y[val])**2))
        if vl < best_val:
            best_val, best_epoch = vl, epoch
            best = [p.copy() for p in params]
    model = PairScorer(mu, sd, mask, *best, scale)
    report = {'mode':mode, 'hidden':hidden, 'best_epoch':best_epoch,
              'train_images':len(set(np.asarray(image_ids)[train].tolist())),
              'val_images':len(set(np.asarray(image_ids)[val].tolist())),
              'test_images':len(set(np.asarray(image_ids)[test].tolist()))}
    for name, sel in [('train',train),('validation',val),('test',test)]:
        p = model.predict(x[sel]); yy = y[sel]; nz = np.abs(yy) > 1e-6
        report[name] = {'n':int(sel.sum()), 'rmse':float(np.sqrt(np.mean((p-yy)**2))),
                        'zero_predictor_rmse':float(np.sqrt(np.mean(yy**2))),
                        'sign_accuracy_nonzero':float(np.mean((p[nz]>0)==(yy[nz]>0))) if nz.any() else None,
                        'mean_real_gain_predicted_positive':float(yy[p>0].mean()) if (p>0).any() else None}
    return model, report
