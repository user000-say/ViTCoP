"""Unit tests for LTT calibration."""

from stage2_release.calibration import CalibrationConfig, LTTCalibrator


def test_ltt_selects_largest_passing_lambda():
    cfg = CalibrationConfig(epsilon=0.5, delta=0.5, lambda_grid=None)
    cfg.lambda_grid = __import__("numpy").array([0.01, 0.1, 1.0])
    cal = LTTCalibrator(cfg)

    data = ["a", "b", "c", "d"]
    refs = ["x", "x", "x", "x"]

    def get_tau(lam: float):
        return [lam]

    def inference_fn(calibration_data, tau_list):
        # Always agree → risk 0 → all λ pass → pick largest.
        return list(refs)

    lam_star, diag = cal.calibrate(inference_fn, data, refs, get_tau)
    assert lam_star == 1.0
    assert diag["n_calibration"] == 4
