import unittest
import torch
from bela.stage2_icer import ICERConfig, select_icer, singleton_perturbation, set_perturbation

class ICERTests(unittest.TestCase):
    def test_zero_swaps_is_exact_topk(self):
        q=torch.tensor([.1,.8,.3,.7])
        f=torch.eye(4)
        ids=torch.tensor([10,11,12,13])
        r=select_icer(q,f,ids,2,config=ICERConfig(max_swaps=0))
        self.assertEqual(r.keep_original_ids.tolist(), [11,13])

    def test_budget_and_identity_preserved(self):
        q=torch.tensor([.1,.8,.3,.7,.2])
        f=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.],[.7,.7]])
        ids=torch.tensor([20,21,22,23,24])
        r=select_icer(q,f,ids,2,config=ICERConfig(max_swaps=2, donor_pool=5, drop_pool=2,
                                                  min_qmass_ratio=.5, min_coverage_gain=0))
        self.assertEqual(len(r.keep_original_ids),2)
        self.assertEqual(len(set(r.keep_original_ids.tolist())),2)
        self.assertEqual(r.keep_slots.tolist(), sorted(r.keep_slots.tolist()))

    def test_final_boundary_ignores_drift(self):
        q=torch.tensor([.9,.8,.1,.2])
        f=torch.eye(4)
        ids=torch.arange(4)
        drift=torch.tensor([0.,0.,10.,9.])
        cfg=ICERConfig(max_swaps=1, donor_pool=4, drop_pool=2, min_qmass_ratio=.1)
        a=select_icer(q,f,ids,2,positive_drift=drift,final_boundary=True,config=cfg)
        b=select_icer(q,f,ids,2,positive_drift=torch.zeros_like(drift),final_boundary=True,config=cfg)
        self.assertEqual(a.keep_original_ids.tolist(), b.keep_original_ids.tolist())

    def test_singleton_and_set_perturbation_finite(self):
        # 1 head, 2 receivers, 4 keys; attention includes text and visual keys.
        a=torch.tensor([[[.1,.2,.3,.4],[.2,.1,.4,.3]]])
        v=torch.tensor([[[1.,0.],[0.,1.],[1.,1.],[2.,0.]]])
        s=singleton_perturbation(a,v,torch.tensor([2,3]))
        d=set_perturbation(a,v,torch.tensor([2]))
        self.assertEqual(s.shape, (2,))
        self.assertTrue(torch.isfinite(s).all())
        self.assertTrue(torch.isfinite(d))

    def test_attention_must_include_all_keys(self):
        a=torch.tensor([[[.2,.2]]])
        v=torch.ones(1,2,3)
        with self.assertRaises(ValueError):
            singleton_perturbation(a,v,torch.tensor([0]))

if __name__=="__main__":
    unittest.main()
