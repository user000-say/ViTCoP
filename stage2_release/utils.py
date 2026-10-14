"""Utility functions for Stage2 release."""

import torch
from typing import Tuple, Optional


def extract_question_rows(
    input_ids: torch.Tensor,
    question_start_pos: int,
    question_end_pos: int,
    include_last_prompt: bool = True,
) -> Tuple[int, int]:
    """Extract question token span indices."""
    if include_last_prompt and question_start_pos > 0:
        start_idx = question_start_pos - 1
    else:
        start_idx = question_start_pos
    return start_idx, question_end_pos


def prepare_attention_mask(
    seq_length: int,
    question_positions: Tuple[int, int],
    visual_positions: torch.Tensor,
    device: str = "cuda",
) -> Optional[torch.Tensor]:
    """Prepare additive attention mask for question rows."""
    q_start, q_end = question_positions
    t = q_end - q_start

    if visual_positions.numel() == 0 or int(visual_positions.max()) < q_start:
        return None

    mask = torch.zeros(t, seq_length, device=device)
    for i, pos in enumerate(range(q_start, q_end)):
        mask[i, pos + 1 :] = float("-inf")
    return mask


def compute_row_weights(
    question_token_ids: torch.Tensor,
    special_token_ids: Optional[set] = None,
    weight_scheme: str = "uniform",
) -> torch.Tensor:
    """Compute importance weights for question rows."""
    t = len(question_token_ids)

    if weight_scheme == "uniform":
        return torch.ones(t) / t

    if weight_scheme == "content_only":
        if special_token_ids is None:
            special_token_ids = set()
        weights = torch.ones(t)
        for i, tok in enumerate(question_token_ids):
            if tok.item() in special_token_ids:
                weights[i] = 0.1
        return weights / weights.sum()

    raise ValueError(f"Unknown weight scheme: {weight_scheme}")


def get_model_layer_hook(model, layer_idx: int):
    """Get hook point for a specific model layer."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise ValueError(f"Unknown model architecture: {type(model)}")
