"""Multi-checkpoint progressive release manager."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .release_effect import ReleaseEffect, ReleaseEffectOutput


@dataclass
class CheckpointConfig:
    """Configuration for a single checkpoint."""

    layer: int
    tau: float
    k_grid: Tuple[int, ...] = (96, 64, 48, 32, 16, 0)


@dataclass
class ReleaseHistory:
    """Tracks release decisions across checkpoints."""

    checkpoints: List[int] = field(default_factory=list)
    k_values: List[int] = field(default_factory=list)
    epsilon_values: List[float] = field(default_factory=list)
    num_released: List[int] = field(default_factory=list)

    def add_checkpoint(self, layer: int, k: int, epsilon: float, released: int):
        self.checkpoints.append(layer)
        self.k_values.append(k)
        self.epsilon_values.append(epsilon)
        self.num_released.append(released)

    def get_summary(self) -> Dict:
        return {
            "total_checkpoints": len(self.checkpoints),
            "final_k": self.k_values[-1] if self.k_values else None,
            "total_released": sum(self.num_released),
            "checkpoint_layers": self.checkpoints,
            "k_trajectory": self.k_values,
        }


class CheckpointManager:
    """Manages progressive visual token release across multiple checkpoints."""

    def __init__(
        self,
        checkpoint_configs: List[CheckpointConfig],
        num_layers: int = 32,
        device: str = "cuda",
    ):
        self.configs = sorted(checkpoint_configs, key=lambda c: c.layer)
        self.num_layers = num_layers
        self.device = device
        for cfg in self.configs:
            if not 0 <= cfg.layer < num_layers:
                raise ValueError(
                    f"Checkpoint layer {cfg.layer} out of range [0, {num_layers})"
                )
        self.calculators = {
            cfg.layer: ReleaseEffect(k_grid=cfg.k_grid) for cfg in self.configs
        }

    def is_checkpoint_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.calculators

    def get_checkpoint_config(self, layer_idx: int) -> Optional[CheckpointConfig]:
        for cfg in self.configs:
            if cfg.layer == layer_idx:
                return cfg
        return None

    @torch.no_grad()
    def release_at_checkpoint(
        self,
        layer_idx: int,
        q_rows: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        current_vis_idx: torch.Tensor,
        o_proj_weight: torch.Tensor,
        h_rows: torch.Tensor,
        history: ReleaseHistory,
    ) -> Tuple[torch.Tensor, ReleaseEffectOutput]:
        if not self.is_checkpoint_layer(layer_idx):
            raise AssertionError(f"Layer {layer_idx} is not a checkpoint")

        cfg = self.get_checkpoint_config(layer_idx)
        assert cfg is not None
        calculator = self.calculators[layer_idx]

        output = calculator.compute(
            q_rows=q_rows,
            k_all=k_cache,
            v_all=v_cache,
            vis_idx=current_vis_idx,
            o_proj_weight=o_proj_weight,
            h_rows=h_rows,
            tau=cfg.tau,
        )

        new_vis_idx = current_vis_idx[output.keep_indices]
        num_released = int(current_vis_idx.numel()) - int(output.chosen_k)
        epsilon_chosen = output.epsilon_per_k[output.chosen_k]
        history.add_checkpoint(layer_idx, output.chosen_k, epsilon_chosen, num_released)
        return new_vis_idx, output

    def create_default_configs(
        self,
        checkpoint_layers: List[int] = None,
        base_tau: float = 0.05,
    ) -> List[CheckpointConfig]:
        if checkpoint_layers is None:
            checkpoint_layers = [8, 16, 24]
        configs = []
        for layer in checkpoint_layers:
            tau = base_tau * (1.0 + 0.1 * (layer / self.num_layers))
            configs.append(CheckpointConfig(layer=layer, tau=tau))
        return configs


def create_checkpoint_manager(
    checkpoint_layers: List[int] = None,
    base_tau: float = 0.05,
    num_layers: int = 32,
    device: str = "cuda",
) -> CheckpointManager:
    """Factory for a checkpoint manager with default settings."""
    if checkpoint_layers is None:
        checkpoint_layers = [8, 16, 24]
    manager = CheckpointManager([], num_layers=num_layers, device=device)
    configs = manager.create_default_configs(checkpoint_layers, base_tau)
    manager.configs = configs
    manager.calculators = {
        cfg.layer: ReleaseEffect(k_grid=cfg.k_grid) for cfg in configs
    }
    return manager
