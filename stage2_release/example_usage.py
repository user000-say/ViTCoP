"""Minimal Stage2 release demo (CPU-safe)."""

from __future__ import annotations

import torch

from stage2_release import create_checkpoint_manager
from stage2_release.checkpoint_manager import ReleaseHistory
from stage2_release.release_effect import ReleaseEffect


def main() -> None:
    torch.manual_seed(0)
    h, t, d_h, n = 4, 2, 16, 128
    d_model = h * d_h
    q = torch.randn(h, t, d_h)
    k = torch.randn(h, n, d_h)
    v = torch.randn(h, n, d_h)
    vis_idx = torch.arange(16, 112)
    o_proj = torch.randn(d_model, d_model)
    hh = torch.randn(t, d_model)

    calc = ReleaseEffect(k_grid=(64, 32, 16, 0))
    out = calc.compute(q, k, v, vis_idx, o_proj, hh, tau=0.05)
    print(f"single-shot: K={out.chosen_k} keep={len(out.keep_indices)}")
    print("ε(K):", {kk: round(vv, 5) for kk, vv in out.epsilon_per_k.items()})

    manager = create_checkpoint_manager([8, 16, 24], base_tau=0.05, num_layers=32)
    history = ReleaseHistory()
    current = vis_idx
    for layer in (8, 16, 24):
        current, out = manager.release_at_checkpoint(
            layer,
            torch.randn(h, t, d_h),
            torch.randn(h, n, d_h),
            torch.randn(h, n, d_h),
            current,
            o_proj,
            hh,
            history,
        )
        print(f"L{layer}: remain={len(current)} K={out.chosen_k}")
    print("summary:", history.get_summary())


if __name__ == "__main__":
    main()
