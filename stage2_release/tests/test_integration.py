"""Integration test simulating progressive release."""

import torch

from stage2_release import create_checkpoint_manager
from stage2_release.checkpoint_manager import ReleaseHistory


def test_full_pipeline():
    torch.manual_seed(123)
    h, t, d_h, n = 4, 3, 32, 200
    d_model = h * d_h

    manager = create_checkpoint_manager(
        checkpoint_layers=[8, 16, 24],
        base_tau=0.05,
        num_layers=32,
    )
    current_vis_idx = torch.arange(50, 150)
    history = ReleaseHistory()

    for layer in [8, 16, 24]:
        q = torch.randn(h, t, d_h)
        k = torch.randn(h, n, d_h)
        v = torch.randn(h, n, d_h)
        o_proj = torch.randn(d_model, d_model)
        hh = torch.randn(t, d_model)
        new_vis_idx, output = manager.release_at_checkpoint(
            layer, q, k, v, current_vis_idx, o_proj, hh, history
        )
        current_vis_idx = new_vis_idx
        print(
            f"Layer {layer}: {len(new_vis_idx)} tokens remain, "
            f"ε={output.epsilon_per_k[output.chosen_k]:.4f}"
        )

    assert len(history.checkpoints) == 3
    assert history.k_values[0] >= history.k_values[1] >= history.k_values[2]
    print("Release trajectory:", history.k_values)
    print("Total released:", history.get_summary()["total_released"])
