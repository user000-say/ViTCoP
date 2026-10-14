"""Learn-then-Test (LTT) calibration for risk control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import numpy as np
from scipy.stats import binom


@dataclass
class CalibrationConfig:
    """Configuration for LTT calibration."""

    epsilon: float = 0.02
    delta: float = 0.1
    lambda_grid: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.lambda_grid is None:
            self.lambda_grid = np.logspace(-2, 0, 30)


class LTTCalibrator:
    """Learn-then-Test calibrator for visual token release thresholds."""

    def __init__(self, config: CalibrationConfig):
        self.config = config

    def calibrate(
        self,
        inference_fn: Callable,
        calibration_data: List,
        reference_outputs: List[str],
        get_tau_from_lambda: Callable[[float], List[float]],
    ) -> Tuple[float, dict]:
        n = len(calibration_data)
        if n != len(reference_outputs):
            raise ValueError("Data and reference must match")

        epsilon = self.config.epsilon
        delta = self.config.delta
        lambda_grid = np.asarray(self.config.lambda_grid)

        risks = []
        disagreements = []
        for lam in lambda_grid:
            tau_list = get_tau_from_lambda(float(lam))
            pruned_outputs = inference_fn(calibration_data, tau_list)
            disagree = sum(
                p.strip() != r.strip()
                for p, r in zip(pruned_outputs, reference_outputs)
            )
            disagreements.append(disagree)
            risks.append(disagree / n)

        # Prefer largest λ that still passes the binomial gate.
        lambda_star = None
        for i in range(len(lambda_grid) - 1, -1, -1):
            k = disagreements[i]
            p_value = binom.cdf(k, n, epsilon)
            if p_value <= delta:
                lambda_star = float(lambda_grid[i])
                break

        if lambda_star is None:
            lambda_star = float(lambda_grid[0])
            print(
                f"Warning: No λ satisfies risk ≤ {epsilon} with conf {1 - delta}. "
                f"Using most conservative λ={lambda_star:.4f}"
            )

        diagnostics = {
            "lambda_star": float(lambda_star),
            "lambda_grid": lambda_grid.tolist(),
            "empirical_risks": risks,
            "disagreements": disagreements,
            "n_calibration": n,
            "epsilon": epsilon,
            "delta": delta,
            "tau_star": get_tau_from_lambda(lambda_star),
        }
        return lambda_star, diagnostics

    @staticmethod
    def linear_tau_schedule(lambda_val: float, base_taus: List[float]) -> List[float]:
        return [lambda_val * b for b in base_taus]


def create_calibrator(
    epsilon: float = 0.02,
    delta: float = 0.1,
    lambda_min: float = 0.01,
    lambda_max: float = 1.0,
    num_lambdas: int = 30,
) -> LTTCalibrator:
    config = CalibrationConfig(
        epsilon=epsilon,
        delta=delta,
        lambda_grid=np.logspace(
            np.log10(lambda_min), np.log10(lambda_max), num_lambdas
        ),
    )
    return LTTCalibrator(config)
