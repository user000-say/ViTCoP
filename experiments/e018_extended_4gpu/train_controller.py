"""Fit matched-capacity controls. Reports are counterfactual NLL replay, not R4."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from controller_numpy import (SCHEMA, inputs, fit, group_splits, choose_defer,
                              calibrate_lambda, load_dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--fraction', type=float, default=.5)
    ap.add_argument('--allow-synthetic', action='store_true')
    args = ap.parse_args()
    data = load_dataset(args.data, allow_synthetic=args.allow_synthetic)
    train, val, test = group_splits(data['image_ids'])
    if test.sum() < 2:
        raise ValueError('not enough image-group-heldout examples')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    delta = data['early_nll'] - data['late_nll']
    cost, stride = data['extra_ms'], int(data['stride'])
    digest = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    report = {'kind': str(data['kind']), 'evaluation': 'offline paired NLL replay only',
              'n': {'train':int(train.sum()), 'val':int(val.sum()), 'test':int(test.sum())},
              'dataset_sha256': digest, 'seed': args.seed, 'modes': {}}
    for mode in ['zero', 'geometry', 'retained', 'history']:
        x = inputs(data['base'], data['history'], data['retained'], mode, stride)
        model = fit(x, delta, train, val, hidden=args.hidden, steps=args.steps, seed=args.seed)
        pred = model.predict(x)
        lam = calibrate_lambda(pred[val], cost[val], args.fraction)
        action = choose_defer(pred[test], cost[test], lam)
        early, late = data['early_nll'][test], data['late_nll'][test]
        realized = np.where(action, late, early)
        objective_early = early
        objective_late = late + lam * cost[test]
        rng = np.random.default_rng(args.seed)
        random_losses = [float(np.where(rng.permutation(action), late, early).mean())
                         for _ in range(30)]
        same_count_oracle = np.zeros(len(action), dtype=bool)
        # Exact NLL oracle for the same NUMBER of late actions; positive/negative
        # targets are kept signed. Same count is NOT identical measured latency.
        count = int(action.sum())
        if count:
            same_count_oracle[np.argsort(-(early - late), kind='stable')[:count]] = True
        # Shuffle preserves action counts, NOT necessarily latency: rerun and
        # cost-match using prompt-length buckets before an online comparison.
        report['modes'][mode] = {
            'lambda_from_val': lam, 'heldout_delay_fraction': float(action.mean()),
            'mean_extra_ms_from_LUT': float((action * cost[test]).mean()),
            'delta_mse': float(np.mean((pred[test] - delta[test])**2)),
            'early_nll': float(early.mean()), 'late_nll': float(late.mean()),
            'policy_nll': float(realized.mean()),
            'best_fixed_nll': float(min(early.mean(), late.mean())),
            'unbudgeted_oracle_nll': float(np.minimum(early, late).mean()),
            'same_delay_count_oracle_nll': float(np.where(same_count_oracle, late, early).mean()),
            'same_delay_count_oracle_extra_ms': float((same_count_oracle * cost[test]).mean()),
            'policy_objective': float((realized + lam * cost[test] * action).mean()),
            'best_fixed_objective': float(min(objective_early.mean(), objective_late.mean())),
            'oracle_objective': float(np.minimum(objective_early, objective_late).mean()),
            'action_shuffled_nll_mean': float(np.mean(random_losses)),
        }
        metadata = {'schema': SCHEMA, 'mode': mode, 'stride': stride,
                    'lambda': lam, 'kind': str(data['kind']), 'hidden': args.hidden,
                    'dataset_sha256': digest, 'seed': args.seed,
                    'manifest': json.loads(str(data['manifest']))}
        model.save(out / (mode + '.npz'), metadata)
        np.savez_compressed(out / (mode + '_heldout_replay.npz'),
                            sample_ids=data['sample_ids'][test], pred=pred[test],
                            defer=action, delta=delta[test], extra_ms=cost[test])
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
