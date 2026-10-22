import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from bela_next.budget import (
    Schedule, PROFILES, PolicyProfile, candidate_schedules,
    schedule_for_cut, verify_monotone_survivors,
)
from bela_next.policy import QuantilePolicy, plan_for_image, audit_plans
from bela_next.signals import coverage_difficulty, coverage_difficulty_from_sets, alignment_drift
from bela_next.stage2_adapter import PruningContract, select_existing_score

class BudgetTest(unittest.TestCase):
    def test_historical_baselines(self):
        self.assertEqual(Schedule(32,172,4,90,12,32).W,2048)
        self.assertEqual(Schedule(32,60,10,28,20,12).W,1024)
        self.assertEqual(Schedule(32,128,12,37,20,18).W,2048)
        self.assertEqual(Schedule(32,64,12,17,20,10).W,1024)

    def test_profiles(self):
        for name,W in [("easy",1536),("normal",2048),("hard",2560)]:
            self.assertEqual(PROFILES[name].schedule().W,W)
            self.assertEqual(len(PROFILES[name].schedule().counts()),32)
            self.assertLess(PROFILES[name].schedule().W,18432)

    def test_new_second_cuts(self):
        for profile in PROFILES.values():
            for c in candidate_schedules(profile):
                self.assertLessEqual(c.W,profile.target_W)
                self.assertGreaterEqual(c.kmid,c.kfinal)
                self.assertEqual(c.cut1,4)
                self.assertIn(c.cut2,(12,16,20))

    def test_bad_schedules(self):
        for args in [(32,40,4,50,12,32),(32,40,12,40,12,32),
                     (32,40,0,40,12,32),(32,577,4,64,12,32)]:
            with self.assertRaises(ValueError):
                Schedule(*args)
        with self.assertRaises(TypeError):
            Schedule(32,40.,4,36,12,32)
        with self.assertRaises(ValueError):
            schedule_for_cut(PolicyProfile("bad",32,1000),12)
        with self.assertRaises(ValueError):
            schedule_for_cut(PROFILES["normal"],32)

    def test_ids_no_resurrection(self):
        self.assertTrue(verify_monotone_survivors([0,1,2],[0,2],2))
        with self.assertRaises(ValueError):
            verify_monotone_survivors([0,1,2],[2,0],2)
        for ids, n in [([0,3],2),([1,1],2),([1],2)]:
            with self.assertRaises(ValueError):
                verify_monotone_survivors([0,1,2],ids,n)

class DifficultyTest(unittest.TestCase):
    def test_duplicates_nonfinite_bad_k(self):
        X=np.eye(128)
        for order in [[0,0]+list(range(2,97)),list(range(4))]:
            with self.assertRaises(ValueError):
                coverage_difficulty(X,order)
        x=X.copy();x[0,0]=np.nan
        with self.assertRaises(ValueError):
            coverage_difficulty(x,list(range(96)))

    def test_identical_vs_distinct(self):
        identical=np.ones((128,32))
        distinct=np.eye(128)
        a=coverage_difficulty(identical,np.arange(96))
        b=coverage_difficulty(distinct,np.arange(96))
        self.assertLess(a["difficulty"],1e-10)
        self.assertGreater(b["difficulty"],0.4)
        vals=list(b["residual"].values())
        self.assertEqual(vals,sorted(vals,reverse=True))

    def test_relabel_invariance(self):
        X=np.random.default_rng(3).normal(size=(128,16))
        order=np.arange(96)
        baseline=coverage_difficulty(X,order)["difficulty"]
        perm=np.random.default_rng(13).permutation(128)
        inv=np.argsort(perm)
        shifted=coverage_difficulty(X[perm],inv[order])["difficulty"]
        self.assertAlmostEqual(baseline,shifted,places=11)


    def test_existing_K_sets_match_greedy_prefix(self):
        x=np.random.default_rng(19).normal(size=(128,16))
        greedy=np.arange(96)
        actual={k:np.sort(greedy[:k]) for k in (16,32,64,96)}
        from_sets=coverage_difficulty_from_sets([x,x],actual)
        from_greedy=coverage_difficulty(x,greedy)
        self.assertAlmostEqual(from_sets["difficulty"],from_greedy["difficulty"],places=11)

    def test_non_nested_actual_sets_and_missing_K(self):
        x=np.eye(128)
        rng=np.random.default_rng(3)
        actual={k:rng.choice(128,size=k,replace=False) for k in (16,32,64,96)}
        result=coverage_difficulty_from_sets([x,x],actual)
        self.assertGreater(result["difficulty"],0)
        with self.assertRaises(ValueError):
            coverage_difficulty_from_sets([x,x],{16:actual[16]})
        with self.assertRaises(ValueError):
            coverage_difficulty_from_sets([x,np.eye(127)],actual)

    def test_drift_top_sets_and_ties(self):
        self.assertEqual(alignment_drift([4,5,6,7],[1,2,3,4],[1,2,3,4])["churn"],0.)
        self.assertEqual(alignment_drift([4,5,6,7],[4,3,2,1],[1,2,3,4])["churn"],1.)
        self.assertFalse(alignment_drift([1,2],[0,0],[1,0])["valid"])
        with self.assertRaises(ValueError):
            alignment_drift([1,1],[.1,.2],[.1,.2])
        with self.assertRaises(ValueError):
            alignment_drift([1,2],[.1,float("nan")],[.1,.2])

class PolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy=QuantilePolicy.calibrate([.08,.13,.18,.22,.29,.34,.41,.48])

    def test_frozen_profiles_and_budget(self):
        plans=[
            plan_for_image(self.policy,.10),
            plan_for_image(self.policy,.26),
            plan_for_image(self.policy,.44)
        ]
        self.assertEqual([p.profile for p in plans],["easy","normal","hard"])
        self.assertEqual([p.schedule.W for p in plans],[1536,2048,2560])
        self.assertEqual(audit_plans(plans)["mean_realized_W"],2048.0)

    def test_stage2_keeps_budget(self):
        for drift,expected_cut in [(0.1,20),(.35,20),(.9,12)]:
            p=plan_for_image(self.policy,.26,"adaptive",drift)
            self.assertEqual(p.schedule.cut2,expected_cut)
            self.assertEqual(p.schedule.W,2048)
        with self.assertRaises(ValueError):
            plan_for_image(self.policy,.3,"adaptive",None)
        with self.assertRaises(ValueError):
            plan_for_image(self.policy,.3,"adaptive",1.4)

    def test_no_information_falls_back_to_fixed(self):
        p=QuantilePolicy.calibrate([.2]*8)
        self.assertEqual(p.choose_profile(.2),"normal")
        self.assertEqual(p.choose_profile(.9),"normal")

    def test_no_hidden_test_calibration(self):
        with self.assertRaises(ValueError):
            QuantilePolicy.calibrate(np.arange(10),source_split="test")
        with self.assertRaises(ValueError):
            QuantilePolicy.calibrate([1,2,3],source_split="dev")

    def test_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"policy.json"
            self.policy.save(path)
            self.assertEqual(self.policy,QuantilePolicy.load(path))

class Stage2ContractTest(unittest.TestCase):
    def test_topk_ties(self):
        chosen=select_existing_score([4,1,7,3],[2.,2.,1.,9.],2)
        self.assertEqual(chosen,(1,3))

    def test_events_and_decode_safety(self):
        s=Schedule(32,8,4,5,12,2)
        tracker=PruningContract(s,tuple(range(8)))
        with self.assertRaises(RuntimeError):
            tracker.step_after_block(3,[0,1,2,3,4])
        tracker.step_after_block(4,[0,1,2,3,4])
        tracker.step_after_block(12,[1,4])
        self.assertEqual(tracker.validate_end_prefill()["final_count"],2)
        with self.assertRaises(RuntimeError):
            tracker.step_after_block(20,[1,4])

    def test_misalign_prune(self):
        s=Schedule(32,8,4,5,12,2)
        tracker=PruningContract(s,tuple(range(8)))
        with self.assertRaises(ValueError):
            tracker.step_after_block(4,[0,1,2,3,9])
        self.assertEqual(tracker.next_event,0)

if __name__=="__main__":
    unittest.main()
