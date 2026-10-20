import unittest
import numpy as np
from bela.core import *
class CoreTests(unittest.TestCase):
    def test_budget_and_trace(self):
        p=Plan(32,172,(Event(3,116),Event(13,32)))
        self.assertEqual(p.work,2284)
        s=LifetimeState(p,list(range(172)))
        for b,n in enumerate(p.layer_counts(),1):
            s.record_block(b,n)
            if b in (3,13):
                slots,ids=s.prune(b,np.arange(n,dtype=float))
                self.assertEqual(len(ids),116 if b==3 else 32)
        self.assertEqual(s.finish()['work'],2284)
    def test_exact_2048(self):
        plans=find_exact_plans(blocks=32,entry=172,target=2048,
            boundaries=[4,12],allowed_counts=range(16,173),final_keep=32)
        self.assertTrue(plans)
        self.assertTrue(all(p.work==2048 and p.events[-1].keep==32 for p in plans))
    def test_exact_1024(self):
        plans=find_exact_plans(blocks=32,entry=60,target=1024,
            boundaries=[3,13],allowed_counts=range(12,61),final_keep=16)
        self.assertTrue(plans)
    def test_invalid(self):
        for args in [(32,172,(Event(3,116),Event(3,32))),
                     (32,172,(Event(33,32),)),
                     (32,172,(Event(3,173),))]:
            with self.assertRaises(ValueError): Plan(*args)
    def test_ties_identity(self):
        result=stable_keep([1.,2.,2.,2.],[9,8,3,5],2)
        self.assertEqual(result.tolist(),[2,3])
        self.assertEqual(stable_keep([1.,2.,2.,2.],[9,8,3,5],2,[9]).tolist(),[0,2])
    def test_edge_all_keep(self):
        p=Plan(2,3,())
        s=LifetimeState(p,[9,3,5])
        s.record_block(1,3);s.record_block(2,3)
        self.assertEqual(s.finish()['counts'],[3,3])
    def test_reject_trace(self):
        p=Plan(2,3,(Event(1,2),))
        s=LifetimeState(p,[0,1,2])
        with self.assertRaises(RuntimeError):s.record_block(1,2)
        s.record_block(1,3)
        s.prune(1,[1.,2.,3.])
        with self.assertRaises(ValueError):s.prune(1,[1.,2.])
        s.record_block(2,2);self.assertEqual(s.finish()['work'],5)
    def test_score(self):
        got=lifetime_scores(current=[.5,1.],older=[.1,.9])
        np.testing.assert_allclose(got,[.46,.64])
        with self.assertRaises(ValueError):lifetime_scores(current=[.2],older=[float('nan')])
    def test_swap(self):
        out=equal_budget_swap(list(range(8)),list(range(8)),[7,6],[0,1],[2,3],5)
        self.assertEqual(len(out['A_interim_ids']),5)
        self.assertEqual(len(out['B_interim_ids']),5)
        self.assertTrue(set([0,1,6,7]).issubset(out['A_interim_ids']))
        self.assertTrue(set([2,3,6,7]).issubset(out['B_interim_ids']))
        self.assertEqual(out['final_ids'],[6,7])
    def test_swap_bad(self):
        with self.assertRaises(ValueError):
            equal_budget_swap([0,1,2],[1,2,3],[2],[0],[0],2)
    def test_w0(self):
        self.assertEqual(Plan(32,576,()).work,18432)
        self.assertEqual(1-2048/18432,8/9)
        self.assertEqual(1-1024/18432,17/18)
if __name__=="__main__": unittest.main()
