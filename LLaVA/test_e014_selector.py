"""CPU-only E014 selector smoke tests; no model files or GPU required."""
import os
import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from llava.model.multimodal_encoder.e014_config import E014Config, E014_BUDGETS
from llava.model.multimodal_encoder.e014_stage1_support import (
    _adaptive_budget, _cover_gain, _eviction_risk, e014_choose,
)


class E014SelectorTest(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(14014)
        self.tokens = torch.randn(576, 128, generator=g)
        self.attn = torch.rand(576, generator=g).add_(0.01)

    def select(self, variant, b=16, lam=1.0):
        cfg = E014Config(True, variant, b, lam, 0.0, False, 1)
        return e014_choose(self.tokens, self.attn, cfg, 0)

    def verify(self, x, selected_b):
        selected, original, added, removed, st = x
        self.assertEqual(int(selected.numel()), 172)
        self.assertEqual(int(selected.unique().numel()), 172)
        self.assertTrue(bool(torch.equal(selected, selected.sort().values)))
        self.assertEqual(int(added.numel()), selected_b)
        self.assertEqual(int(removed.numel()), selected_b)
        self.assertEqual(st["budget"], selected_b)
        self.assertTrue(bool(torch.isin(added, original).logical_not().all()))
        self.assertTrue(bool(torch.isin(removed, original).all()))
        expect = torch.cat((original[~torch.isin(original, removed)], added)).sort().values
        self.assertTrue(bool(torch.equal(selected, expect)))

    def test_base_and_fixed_controls(self):
        self.verify(self.select("BASE"), 0)
        for b in E014_BUDGETS:
            for variant in ("FIXED_COVER_PROXY", "FIXED_RISK"):
                self.verify(self.select(variant, b), b)

    def test_adaptive_valid_and_deterministic(self):
        for variant in ("AEE", "ADAPTIVE_B_ONLY"):
            s1 = self.select(variant)
            s2 = self.select(variant)
            self.assertIn(s1[-1]["budget"], E014_BUDGETS)
            self.verify(s1, s1[-1]["budget"])
            self.assertTrue(bool(torch.equal(s1[0], s2[0])))

    def test_exchange_objective_interior_optimum(self):
        # First 16 candidates have high utility, later candidates low utility.
        gain = torch.cat((torch.ones(16), torch.full((388,), .01)))
        risk = torch.full((172,), 0.05)
        b, _ = _adaptive_budget(
            gain, risk, torch.ones(576),
            E014Config(True, "AEE", 16, 1.0, 0.0, False, 1),
        )
        self.assertEqual(b, 16)

    def test_eviction_ignores_diagonal_self_similarity(self):
        t = torch.zeros(576, 4)
        # Construct orthogonal core patches; cosine with self =1 must NOT count.
        t[:172, 0] = 1
        t[0] = torch.tensor((0., 1., 0., 0.))
        t[1] = torch.tensor((0., 0., 1., 0.))
        t[2] = torch.tensor((0., 0., 0., 1.))
        a = torch.ones(576)
        risk = _eviction_risk(t, a, torch.arange(172))
        self.assertAlmostEqual(float(risk[0]), 1.0, places=4)
        self.assertAlmostEqual(float(risk[1]), 1.0, places=4)

    def test_cover_formula_positive_clamp(self):
        tok = torch.eye(4)
        attn = torch.tensor((1., 2., 3., 4.))
        score = _cover_gain(tok, attn, torch.arange(4))
        self.assertEqual(score.shape, (4,))
        self.assertTrue(bool((score >= 0).all()))

    def test_config_validation(self):
        old = os.environ.get("E014_VARIANT")
        try:
            os.environ["E014_VARIANT"] = "NO_SUCH_VARIANT"
            with self.assertRaises(ValueError):
                E014Config.from_env()
        finally:
            if old is None:
                os.environ.pop("E014_VARIANT", None)
            else:
                os.environ["E014_VARIANT"] = old


if __name__ == "__main__":
    unittest.main()
