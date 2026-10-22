import sys
import unittest
from pathlib import Path
import numpy as np
from scripts.analyze_pope import classify, load, report, pair_flips

HERE=Path(__file__).resolve().parents[1]
class PopeTest(unittest.TestCase):
    def test_strict_parser(self):
        self.assertEqual(classify(" Yes, it is"),1)
        self.assertEqual(classify("NO."),0)
        self.assertIsNone(classify("There is a bicycle"))
        self.assertIsNone(classify("maybe"))
        self.assertIsNone(classify(""))

    def test_paired(self):
        d=load(HERE/"examples/pope_dense.jsonl")
        b=load(HERE/"examples/pope_bela.jsonl")
        stats=pair_flips(d,b)
        self.assertEqual(stats["shared"],4)
        self.assertEqual(stats["dense_correct_bela_wrong"],1)
        self.assertEqual(stats["dense_wrong_bela_correct"],1)
        self.assertEqual(report(b)["all"]["invalid"],0)

class TorchAdapterTest(unittest.TestCase):
    def test_cpu_matches_numpy(self):
        try:
            import torch
            from bela_next.stage1_torch_adapter import torch_coverage_difficulty
        except ImportError:
            self.skipTest("torch not installed")
        from bela_next.signals import coverage_difficulty
        x=np.random.default_rng(4).normal(size=(128,24)).astype("float32")
        order=np.arange(96)
        score_np=coverage_difficulty(x,order)["difficulty"]
        score_t=torch_coverage_difficulty(torch.tensor(x),torch.tensor(order))
        self.assertAlmostEqual(score_t,score_np,places=5)

    def test_stage1_selection_contract(self):
        try:
            import torch
            from bela_next.stage1_torch_adapter import dynamic_budget_stage1
        except ImportError:
            self.skipTest("torch not installed")
        feats=torch.randn(576,8)
        selected=dynamic_budget_stage1(lambda k:torch.arange(k),feats,torch.arange(172),"easy")
        self.assertEqual(selected.numel(),96)
        with self.assertRaises(ValueError):
            dynamic_budget_stage1(lambda k:torch.zeros(k),feats,torch.arange(172),"easy")

if __name__=="__main__":
    unittest.main()
