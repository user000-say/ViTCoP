"""Project-side collection interface. No guessed decoder/cache monkeypatch.

Cursor implements this protocol against the actual repository. These validation
helpers run, but they cannot verify that a project truly cloned its KV cache.
"""
from typing import Protocol
import numpy as np


class ProjectAdapter(Protocol):
    def prepare_prefix(self, sample):
        """Frozen prompt-only state JUST BEFORE the final pruning decision.

        Must include cache, hidden states, attention mask, position IDs, current
        patch IDs, prefix fingerprint and immutable past pruning actions.
        """
        ...

    def features(self, prefix):
        """Return base, history, retained, and cost-LUT key BEFORE either branch.

        history is the original encoder patch content dropped by Stage1.
        retained is the corresponding Stage1-retained content, same projection.
        Freeze projections with checkpoint metadata and compute summaries once.
        """
        ...

    def clone_prefix(self, prefix):
        """Deep independent branch, including mutable cache and all masks."""
        ...

    def run_branch(self, prefix_clone, sample, action):
        """action = early or late. Return prompt-prefix fingerprint, generated
        prediction and answer-only token-normalized teacher-forced NLL.
        Ground truth may be used ONLY for offline NLL, not features/actions.
        Same reference answer tokens and frozen downstream policy in both runs.
        """
        ...


def collect_pair(adapter: ProjectAdapter, sample):
    prefix = adapter.prepare_prefix(sample)
    features = adapter.features(prefix)
    early = adapter.run_branch(adapter.clone_prefix(prefix), sample, 'early')
    late = adapter.run_branch(adapter.clone_prefix(prefix), sample, 'late')
    if not early['prefix_fingerprint'] or early['prefix_fingerprint'] != late['prefix_fingerprint']:
        raise ValueError('counterfactual branches did not share the same prefix')
    if not early['answer_token_ids'] or early['answer_token_ids'] != late['answer_token_ids']:
        raise ValueError('teacher-forced answers differ between branches')
    nll = np.array([early['answer_nll'], late['answer_nll']], dtype=float)
    if not np.isfinite(nll).all() or np.any(nll < 0):
        raise ValueError('invalid answer NLL')
    return {'features':features, 'early':early, 'late':late,
            'delta':float(nll[0] - nll[1])}
