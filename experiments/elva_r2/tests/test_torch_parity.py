"""Run on server before ACC; absence of PyTorch is an explicit skip."""
import importlib.util
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is absent in this workspace")
class TorchParityTests(unittest.TestCase):
    def test_numpy_parity_and_payload_protection(self):
        import torch
        from elva.evidence import select_stage1 as ref, question_visual_mass as mass_ref
        from elva.torch_ops import select_stage1, question_visual_mass
        r = np.random.default_rng(4)
        x, e = r.normal(size=(12, 6)), r.normal(size=(12, 6))
        a, b, ids = r.random(12), r.random(12), np.arange(12)*3
        for guide, beta in [("pen", .5), ("l9", .5), ("dual", 0), ("dual", .5), ("dual", 1)]:
            options = dict(guide=guide, beta=beta)
            reference = ref(x, a, ids, 4, early=e, early_scores=b, early_ids=ids, **options)
            out = select_stage1(torch.tensor(x), torch.tensor(a), torch.tensor(ids), 4,
                                early=torch.tensor(e), early_scores=torch.tensor(b), early_ids=torch.tensor(ids), **options)
            np.testing.assert_array_equal(out["keep"].numpy(), reference["keep"])
            np.testing.assert_array_equal(out["payload"].numpy(), reference["payload"])
            np.testing.assert_allclose(out["metric"].numpy(), reference["metric"], atol=1e-6)
        z = r.normal(size=(2, 3, 15))
        mask = np.ones_like(z, bool)
        cols = np.array([1, 3, 7])
        actual = question_visual_mass(torch.tensor(z), torch.tensor(mask), torch.tensor(cols)).numpy()
        np.testing.assert_allclose(actual, mass_ref(z, mask, cols), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
