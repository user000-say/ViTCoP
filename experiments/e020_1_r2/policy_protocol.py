"""Explicit one-step deployment contract and conditional decision identities.

A prefix hash alone is not a policy state: the same prefix may feed different
future schedules or baseline sets. No answer/loss is part of this identity.
"""
import hashlib
import json
import numpy as np

POLICY_SCHEMA = 'e0201_one_step_catalogue_v1'
DEFAULT_POLICY = {'rounds': 1, 'group_size': 1, 'shortlist': 16,
                  'candidate_limit': 16, 'proposal_version': 'affine_rank_grid_v1'}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def catalogue_indices(total, limit=None):
    """Data-independent subset of the rank-grid, identical in NumPy and Torch.

    Fixed integer priorities avoid NumPy/CUDA RNG differences. Limiting the
    catalogue is a compute choice, not a claim to cover all profitable swaps.
    """
    if limit is None or total <= limit:
        return np.arange(total, dtype=np.int64)
    if type(limit) is not int or limit < 1:
        raise ValueError('candidate_limit must be a positive integer or None')
    ix = np.arange(total, dtype=np.int64)
    priority = (ix * 2654435761 + 1013904223) % 4294967296
    return np.sort(np.argsort(priority, kind='stable')[:limit])


def state_identity(record):
    keys = ('source_hash', 'image_id', 'question_id', 'stage1', 'schedule_name',
            'prefix_hash', 'after', 'depth', 'next_after', 'final_k',
            'baseline_patch_ids')
    identity = {k: record.get(k) for k in keys}
    # A missing field is retained, never conflated with a different present one.
    return fingerprint(identity)


def action_identity(record):
    return (state_identity(record), tuple(record['donor_patch_ids']),
            tuple(record['incoming_patch_ids']))


def validate_model_policy(metadata):
    if metadata.get('policy_schema') != POLICY_SCHEMA:
        raise ValueError('model lacks the revised deployment/calibration contract; refit from real pairs')
    policy = metadata.get('deployment_policy')
    if policy != DEFAULT_POLICY:
        raise ValueError('checkpoint uses a different candidate/rollout contract')
    calibration = metadata.get('calibration', {})
    if calibration.get('status') not in ('CALIBRATED_ACCEPT', 'CALIBRATED_ABSTAIN'):
        raise ValueError('complete validation catalogues required before real ACC')
    threshold = metadata.get('decision_threshold')
    if threshold is None or not np.isfinite(threshold):
        raise ValueError('finite validation-only threshold required')
    if type(metadata.get('accept_enabled')) is not bool:
        raise ValueError('explicit calibrated acceptance state required')
    return policy
