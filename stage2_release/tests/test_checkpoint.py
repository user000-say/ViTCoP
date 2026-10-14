"""Unit tests for checkpoint manager."""

from stage2_release.checkpoint_manager import (
    CheckpointConfig,
    CheckpointManager,
    ReleaseHistory,
    create_checkpoint_manager,
)


def test_checkpoint_manager_creation():
    configs = [
        CheckpointConfig(layer=8, tau=0.04),
        CheckpointConfig(layer=16, tau=0.05),
        CheckpointConfig(layer=24, tau=0.06),
    ]
    manager = CheckpointManager(configs, num_layers=32)
    assert len(manager.configs) == 3
    assert manager.is_checkpoint_layer(8)
    assert manager.is_checkpoint_layer(16)
    assert not manager.is_checkpoint_layer(10)


def test_release_history():
    history = ReleaseHistory()
    history.add_checkpoint(8, 96, 0.03, 4)
    history.add_checkpoint(16, 64, 0.05, 32)
    history.add_checkpoint(24, 32, 0.07, 32)
    summary = history.get_summary()
    assert summary["total_checkpoints"] == 3
    assert summary["final_k"] == 32
    assert summary["total_released"] == 68


def test_factory_creation():
    manager = create_checkpoint_manager(
        checkpoint_layers=[8, 16, 24],
        base_tau=0.05,
        num_layers=32,
    )
    assert len(manager.configs) == 3
    assert all(cfg.tau >= 0.05 for cfg in manager.configs)
    taus = [cfg.tau for cfg in manager.configs]
    assert taus[0] < taus[1] < taus[2]
