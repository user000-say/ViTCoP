"""Unit tests for release effect computation."""

import torch
import pytest

from stage2_release.release_effect import ReleaseEffect


@pytest.fixture
def setup_tensors():
    torch.manual_seed(42)
    h, t, d_h, n, m = 4, 3, 32, 200, 100
    d_model = h * d_h
    q_rows = torch.randn(h, t, d_h)
    k_all = torch.randn(h, n, d_h)
    v_all = torch.randn(h, n, d_h)
    vis_idx = torch.arange(50, 150)
    o_proj = torch.randn(d_model, d_model)
    h_rows = torch.randn(t, d_model)
    return q_rows, k_all, v_all, vis_idx, o_proj, h_rows


def test_release_effect_shape(setup_tensors):
    q, k, v, vis_idx, o_proj, h = setup_tensors
    calculator = ReleaseEffect(k_grid=(64, 32, 16, 0))
    output = calculator.compute(q, k, v, vis_idx, o_proj, h, tau=0.05)
    assert output.chosen_k <= len(vis_idx)
    assert len(output.keep_indices) == output.chosen_k
    assert output.token_contributions.shape == (len(vis_idx),)
    # Grid keys always present; +1 key if fallback keeps all (m).
    assert len(output.epsilon_per_k) in (4, 5)
    assert output.chosen_k in output.epsilon_per_k


def test_epsilon_monotonicity(setup_tensors):
    q, k, v, vis_idx, o_proj, h = setup_tensors
    calculator = ReleaseEffect(k_grid=(96, 64, 32, 16, 0))
    output = calculator.compute(q, k, v, vis_idx, o_proj, h, tau=1.0)
    sorted_ks = sorted(output.epsilon_per_k.keys(), reverse=True)
    epsilons = [output.epsilon_per_k[kk] for kk in sorted_ks]
    for i in range(len(epsilons) - 1):
        assert epsilons[i] <= epsilons[i + 1] + 1e-5, (
            f"ε not monotonic: K={sorted_ks[i]} -> {epsilons[i]}, "
            f"K={sorted_ks[i+1]} -> {epsilons[i+1]}"
        )


def test_numerical_accuracy_vs_recompute(setup_tensors):
    """Verify closed-form Δo matches actual attention recomputation."""
    q, k, v, vis_idx, o_proj, h = setup_tensors
    heads, t, d_h = q.shape
    k_test = 50
    m = len(vis_idx)

    calculator = ReleaseEffect(k_grid=(k_test,))
    output = calculator.compute(q, k, v, vis_idx, o_proj, h, tau=1.0)
    epsilon_ours = output.epsilon_per_k[k_test]

    keep_local = set(output.keep_indices.tolist())
    remove_local = sorted(set(range(m)) - keep_local)

    # Full attention + projected output (same W_O convention as module).
    scores_full = torch.einsum("htd,hnd->htn", q, k) / (d_h ** 0.5)
    attn_full = torch.softmax(scores_full, dim=-1)
    o_full = torch.einsum("htn,hnd->htd", attn_full, v)
    o_full_cat = o_full.permute(1, 0, 2).reshape(t, heads * d_h)
    o_full_proj = o_full_cat @ o_proj.t()

    mask_keep = torch.ones(k.shape[1], dtype=torch.bool)
    mask_keep[vis_idx[remove_local]] = False
    k_pruned = k[:, mask_keep]
    v_pruned = v[:, mask_keep]
    scores_pruned = torch.einsum("htd,hnd->htn", q, k_pruned) / (d_h ** 0.5)
    attn_pruned = torch.softmax(scores_pruned, dim=-1)
    o_pruned = torch.einsum("htn,hnd->htd", attn_pruned, v_pruned)
    o_pruned_cat = o_pruned.permute(1, 0, 2).reshape(t, heads * d_h)
    o_pruned_proj = o_pruned_cat @ o_proj.t()

    delta_actual = o_pruned_proj - o_full_proj
    h_norm = h.norm(dim=-1).clamp_min(1e-6)
    epsilon_actual = (delta_actual.norm(dim=-1) / h_norm).mean().item()

    rel_error = abs(epsilon_ours - epsilon_actual) / (epsilon_actual + 1e-8)
    assert rel_error < 0.01, (
        f"Closed-form ε={epsilon_ours:.6f} vs recomputed={epsilon_actual:.6f}, "
        f"rel_err={rel_error:.6f}"
    )


def test_threshold_selection(setup_tensors):
    q, k, v, vis_idx, o_proj, h = setup_tensors
    tau = 0.05
    calculator = ReleaseEffect(k_grid=(96, 64, 48, 32, 16, 0))
    output = calculator.compute(q, k, v, vis_idx, o_proj, h, tau=tau)

    epsilon_chosen = output.epsilon_per_k[output.chosen_k]
    assert epsilon_chosen <= tau + 1e-6

    sorted_ks = sorted(output.epsilon_per_k.keys())
    chosen_idx = sorted_ks.index(output.chosen_k)
    for kk in sorted_ks[:chosen_idx]:
        assert output.epsilon_per_k[kk] > tau
