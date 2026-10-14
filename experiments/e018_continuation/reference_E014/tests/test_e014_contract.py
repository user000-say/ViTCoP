"""E014 extra CPU invariants, particularly E013 exact-comparator fidelity."""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import llava.model.multimodal_encoder.e014_stage1_support as aee
from llava.model.multimodal_encoder.e014_config import E014Config


class E014ContractTest(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(14015)
        self.x = torch.randn(576, 64, generator=g)
        self.attn = torch.rand(576, generator=g).add(0.01)

    def test_zero_swap_fixed_arms_do_not_compute_gain_or_risk(self):
        with patch.object(aee, "_cover_gain", side_effect=AssertionError("gain called")):
            with patch.object(aee, "_eviction_risk", side_effect=AssertionError("risk called")):
                for arm in ("FIXED_COVER_PROXY", "FIXED_RISK"):
                    cfg = E014Config(enabled=True, variant=arm, b=0)
                    selected, core, added, removed, stat = aee.e014_choose(self.x, self.attn, cfg)
                    self.assertTrue(torch.equal(selected, core))
                    self.assertEqual((added.numel(), removed.numel(), stat["budget"]), (0, 0, 0))

    def test_original_helper_exact_output_order_is_unchanged(self):
        expected_call = []
        core = self.attn.topk(172).indices.sort().values
        residual = torch.arange(576)[~torch.isin(torch.arange(576), core)]
        legacy_selected = torch.cat((core[16:], residual[:16])).flip(0)
        def fake_legacy(tokens, attention, variant, pick, b, seed, index):
            expected_call.append((variant, pick, b, seed, index))
            return legacy_selected.clone(), core[:0], core.clone()

        fake_module = types.ModuleType("llava.model.multimodal_encoder.e013_stage1_support")
        fake_module.e013_choose = fake_legacy

        with patch.dict(sys.modules, {fake_module.__name__: fake_module}):
            cfg = E014Config(True, "FIXED_COVER", 16, seed=123)
            selected, original, added, removed, stats = aee.e014_choose(
                self.x, self.attn, cfg, request_index=77
            )
        self.assertTrue(torch.equal(selected, legacy_selected))
        self.assertTrue(torch.equal(original, core))
        self.assertEqual((added.numel(), removed.numel()), (16, 16))
        self.assertEqual(expected_call, [("REPLACE", "COVER", 16, 123, 77)])
        self.assertEqual(stats["score_type"], "E013_EXACT")

    def test_legacy_helper_cannot_silently_return_wrong_exchange_budget(self):
        core = self.attn.topk(172).indices.sort().values
        fake_module = types.ModuleType("llava.model.multimodal_encoder.e013_stage1_support")
        fake_module.e013_choose = lambda *args: (core.clone(), core[:0], core.clone())
        with patch.dict(sys.modules, {fake_module.__name__: fake_module}):
            with self.assertRaisesRegex(RuntimeError, "cardinality"):
                aee.e014_choose(
                    self.x, self.attn, E014Config(True, "FIXED_COVER", 16)
                )

    def test_reject_non_576_clip_geometry(self):
        with self.assertRaisesRegex(ValueError, "576"):
            aee.e014_choose(self.x[:575], self.attn[:575], E014Config(True, "BASE", 0))

    def test_adaptive_zero_utility_prefers_no_exchange(self):
        gain = torch.zeros(404)
        risk = torch.zeros(172)
        b, _ = aee._adaptive_budget(gain, risk, torch.ones(576), E014Config(True, "AEE", 16))
        self.assertEqual(b, 0)

    def test_strong_budget_penalty_forces_zero_exchange(self):
        gain = torch.ones(404)
        risk = torch.zeros(172)
        cfg = E014Config(True, "AEE", 16, lam=1., penalty=2.0)
        b, _ = aee._adaptive_budget(gain, risk, torch.ones(576), cfg)
        self.assertEqual(b, 0)


if __name__ == "__main__":
    unittest.main()
