import unittest
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from elva.low_budget_cover import numpy_keep, torch_keep, diagnostics, callback, make_spec
from elva.cover_binding import NativeCover, select_checked
from make_catalog import build_catalog


class CoverageTests(unittest.TestCase):
    def test_protected_core_and_actual_additions_reach_output(self):
        # FIXTURE: top 4 are duplicated, a lower-score fifth row covers another direction.
        x = np.array([[1,0],[1,0],[1,0],[1,0],[0,1]], dtype=np.float32)
        s = np.array([5,4,3,2,1], dtype=np.float32)
        keep = numpy_keep(x,s,3,1)
        np.testing.assert_array_equal(keep,[0,1,4])
        audit = diagnostics(x,s,keep,3,1)
        self.assertEqual(audit['actual_changed'],1)
        self.assertEqual(audit['added_local'],[4])
        self.assertGreater(audit['kept_coverage']['weighted'],audit['topk_coverage']['weighted'])
        self.assertLess(audit['kept_mass'],audit['topk_mass'])  # explicit tradeoff

    def test_full_pool_can_select_low_rank_evidence(self):
        x = np.zeros((576,2),dtype=np.float32);x[:,0]=1;x[-1]=[0,1]
        s = np.linspace(1,.01,576,dtype=np.float32)
        keep = numpy_keep(x,s,60,12)
        self.assertIn(575,keep)  # a Top180 candidate restriction would forbid this
        self.assertTrue(set(range(48)).issubset(keep))
        self.assertEqual(len(set(keep)),60)

    def test_zero_affinity_ties_are_stable_topk_and_no_forced_change(self):
        x=np.zeros((80,3),dtype=np.float32)
        for s in (np.zeros(80,dtype=np.float32), np.arange(80,dtype=np.float32)):
            for b in (0,6,12,18,60):
                keep=numpy_keep(x,s,60,b)
                np.testing.assert_array_equal(keep,np.sort(np.argsort(-s,kind='stable')[:60]))

    def test_budget_extremes_and_determinism(self):
        rng=np.random.default_rng(82)
        for n in (1,5,60,80):
            x=rng.normal(size=(n,5)).astype(np.float32);s=rng.random(n).astype(np.float32)
            for k in (1,n):
                for b in (0,k):
                    keep=numpy_keep(x,s,k,b)
                    self.assertEqual(len(keep),k)
                    self.assertEqual(len(np.unique(keep)),k)
                    np.testing.assert_array_equal(keep,numpy_keep(x,s,k,b))
                    self.assertTrue(set(np.argsort(-s,kind='stable')[:k-b]).issubset(keep))

    def test_invalid_inputs_fail(self):
        x=np.eye(3,dtype=np.float32);s=np.ones(3,dtype=np.float32)
        for k,b in ((0,0),(3,4),(3,-1),(True,0)):
            with self.assertRaises(ValueError):numpy_keep(x,s,k,b)
        with self.assertRaises(ValueError):numpy_keep(x,np.array([1,np.nan,1]),2,1)
        with self.assertRaises(ValueError):numpy_keep(x,np.array([-1,1,1]),2,1)

    def test_real_callback_raw_pen_ids_and_not_original_cover_audit(self):
        rng=np.random.default_rng(6)
        x=rng.normal(size=(576,8)).astype(np.float32)
        s=rng.random(576).astype(np.float32)
        ids=np.arange(576,dtype=np.int64)*7+6
        c=next(c for c in build_catalog() if 'W1024_V4_' in c['id'])
        out=select_checked(x,s,ids,c,native_cover=NativeCover(callback,make_spec()))
        np.testing.assert_array_equal(out['payload'],x[out['keep']])
        np.testing.assert_array_equal(out['ids'],ids[out['keep']])
        self.assertEqual(out['stage1_audit']['cover_calls'],1)
        self.assertEqual(out['stage1_audit']['native_calls'],0)
        self.assertEqual(out['stage1_audit']['origin'],'r3_protected_control')
        self.assertEqual(out['stage1_audit']['replacement_slots'],12)
        core=np.argsort(-s,kind='stable')[:48]
        self.assertTrue(set(core).issubset(out['keep']))

    def test_only_one_bounded_coverage_knob_and_explicit_optional_controls(self):
        configs=build_catalog()
        core=[c for c in configs if c['phase']=='stage1_core' and c['method']['budget']==1024]
        covered=[c for c in core if c['method']['stage1']['cover']]
        self.assertEqual(len(covered),2)
        self.assertTrue(all(c['method']['stage1']['coverage_slots']==12 for c in covered))
        extra=[c for c in configs if c['phase']=='stage1_optional' and c['method']['stage1'].get('coverage_slots') in (6,18)]
        self.assertEqual(len(extra),4)
        self.assertEqual(len([c for c in configs if c['method']['stage1']['cover_family']=='parameterized_native']),2)

    def test_refinement_keeps_winning_beta_and_preserves_winner_in_stage2(self):
        base=next(c for c in build_catalog() if c['method']['budget']==1024 and c['method']['stage1'].get('beta')==.25)
        refined=build_catalog(coverage_base=base)
        variants=[c for c in refined if c['method']['stage1'].get('coverage_slots') in (6,18)]
        self.assertEqual(len(variants),2)
        self.assertTrue(all(c['method']['stage1']['beta']==.25 for c in variants))
        winner=variants[0]
        decision={'budgets':{'1024':{'stage1':winner['method']['stage1'],'winner_config':winner}}}
        rebuilt=build_catalog(decisions=decision)
        self.assertTrue(any(c==winner for c in rebuilt))
        self.assertTrue(all(c['method']['stage1']['beta']==.25 for c in rebuilt if c['phase']=='stage2'))


@unittest.skipIf(torch is None,'PyTorch absent locally; run parity on the existing server environment')
class CoverageTorchTests(unittest.TestCase):
    def test_numpy_parity_and_no_payload_change(self):
        rng=np.random.default_rng(33)
        for n,k in ((5,3),(80,60),(576,60)):
            x=rng.normal(size=(n,13)).astype(np.float32);s=rng.random(n).astype(np.float32)
            for b in (0,min(6,k),min(12,k)):
                np.testing.assert_array_equal(torch_keep(torch.from_numpy(x),torch.from_numpy(s),k,b).numpy(),
                                              numpy_keep(x,s,k,b))
        x=torch.zeros(80,3);s=torch.zeros(80)
        torch.testing.assert_close(torch_keep(x,s,60,12),torch.arange(60))


if __name__=='__main__':unittest.main()
