#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import bela_eval as ev
import joint_baseline as jb
import query_anchor_history_reference as qa
import plot_real_trace as pr


class ReferenceBehavior(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(26)
        self.x=torch.randn(96,24)
        self.q=torch.linspace(-3,3,96)
        self.ids=torch.arange(24)

    def run_selector(self, ids, b, scores=None):
        return qa.select_question_anchored(native_keep_ids=ids,
            visual_metric=self.x,
            raw_query_scores=self.q if scores is None else scores,
            config=qa.AnchorConfig(max_swaps=b), record=True)

    def test_b8_and_b16_and_repeated_pass(self):
        b8=self.run_selector(self.ids,8)
        b16=self.run_selector(self.ids,16)
        second=self.run_selector(b8.keep_ids,8)
        self.assertEqual(b8.swaps,8)
        self.assertEqual(b16.swaps,16)
        self.assertEqual(second.swaps,8)
        self.assertEqual(b16.keep_ids.tolist(),second.keep_ids.tolist())
        self.assertEqual(len(set(b16.keep_ids.tolist())),24)

    def test_one_vs_two_vs_three_passes_with_equal_total_budget(self):
        # This checks a frozen synthetic scoring trace, NOT semantic utility or GPU quality.
        import torch
        torch.manual_seed(20271029)
        x=torch.randn(144,32)
        q=torch.linspace(-3,3,144)
        native=torch.arange(32)

        def sequential(caps):
            ids=native
            accepted=[]
            for b in caps:
                result=qa.select_question_anchored(native_keep_ids=ids,
                    visual_metric=x, raw_query_scores=q,
                    config=qa.AnchorConfig(max_swaps=b), record=True)
                ids=result.keep_ids
                accepted.append(result.swaps)
            return set(ids.tolist()), accepted

        once,n1=sequential([8])
        twice,n2=sequential([4,4])
        thrice,n3=sequential([3,3,2])
        self.assertEqual(n1,[8])
        self.assertEqual(n2,[4,4])
        self.assertEqual(n3,[3,3,2])
        self.assertEqual(once,twice)
        self.assertEqual(once,thrice)

    def test_zero_capacity_empty_signal_and_flat(self):
        self.assertEqual(self.run_selector(self.ids,0).swaps,0)
        self.assertEqual(self.run_selector(self.ids,8,torch.zeros(96)).swaps,0)
        no_signal=qa.select_question_anchored(native_keep_ids=self.ids,
            visual_metric=self.x, config=qa.AnchorConfig(max_swaps=8))
        self.assertEqual(no_signal.swaps,0)
        full=qa.select_question_anchored(native_keep_ids=torch.arange(96),
            visual_metric=self.x, config=qa.AnchorConfig(max_swaps=8))
        self.assertEqual(full.swaps,0)
        empty=qa.select_question_anchored(native_keep_ids=torch.empty(0,dtype=torch.long),
            visual_metric=self.x, config=qa.AnchorConfig(max_swaps=8))
        self.assertEqual(empty.swaps,0)

    def test_invalid_ids(self):
        with self.assertRaises(ValueError):
            self.run_selector(torch.tensor([0,0,1]),8)
        with self.assertRaises(ValueError):
            self.run_selector(torch.tensor([-1,1]),8)
        with self.assertRaises(ValueError):
            self.run_selector(torch.tensor([0,100]),8)

    def test_individual_accepted_step_guards(self):
        r=self.run_selector(self.ids,8)
        for x in r.records:
            self.assertGreaterEqual(x["query_gain"],.08-1e-6)
            self.assertGreaterEqual(x["donor_novelty"],.05-1e-6)
            self.assertGreaterEqual(x["coverage_delta"],-.005-1e-6)
            self.assertGreaterEqual(x["query_gain"]*x["donor_novelty"]+0.5*x["coverage_delta"],.0015-1e-5)

    def test_trace_replay_checks_real_reference_actions(self):
        r=self.run_selector(self.ids,8)
        trace={"sample_id":"test:1","n_patches":96,
               "native_ids":self.ids.tolist(),"final_ids":r.keep_ids.tolist(),
               "swaps":r.swaps,"max_swaps":8,"records":r.records,
               "max_coverage_loss":.005,"min_query_gain":.08}
        report=ev.audit_trace([trace])
        self.assertEqual(report["mean_accepted_swaps"],8)
        self.assertEqual(report["fraction_at_hard_cap"],1)
        bad=dict(trace)
        bad["final_ids"]=[0]*24
        with self.assertRaises(ValueError): ev.audit_trace([bad])
        bad2=dict(trace)
        bad2["records"]=trace["records"][:-1]
        with self.assertRaises(ValueError): ev.audit_trace([bad2])


class JointBaselineTests(unittest.TestCase):
    def test_capacity_and_guard(self):
        np.random.seed(22)
        x=np.random.randn(18,4); x/=np.linalg.norm(x,axis=1,keepdims=True)
        sim=np.clip(x@x.T,0,1)
        q=np.linspace(0,1,18)
        native=np.array([0,1,2,3,4])
        unrestricted=jb.joint_greedy(sim,q,5,.2)
        restricted=jb.joint_greedy(sim,q,5,.2,native,2)
        self.assertEqual(len(unrestricted),5)
        self.assertEqual(len(restricted),5)
        self.assertEqual(len(set(restricted)-set(native)),2)
        self.assertGreaterEqual(len(set(restricted)&set(native)),3)
        self.assertGreaterEqual(jb.representation_coverage(sim,restricted),0)
        self.assertEqual(len(jb.joint_greedy(sim,q,0,.2)),0)

    def test_bad_shape_and_out_of_range(self):
        with self.assertRaises(ValueError):
            jb.joint_greedy(np.ones((2,2)),np.ones(3),2,0)
        with self.assertRaises(ValueError):
            jb.joint_greedy(np.ones((3,3)),np.ones(3),2,.2,[0,1,1],1)


class EvaluationTests(unittest.TestCase):
    def test_pope_metrics_and_duplicate_question_id(self):
        a=[{"sample_id":f"pope:{i}","question_id":"REPEATED","image_id":str(i//2),
            "gold":gold,"pred":pred} for i,(gold,pred) in enumerate(
                [("yes","yes"),("yes","no"),("no","yes"),("no","no")])]
        b=[dict(r) for r in a]
        b[1]["pred"]="yes"
        before=ev.summary(a,"pope"); after=ev.summary(b,"pope")
        self.assertEqual(before["tp"],1)
        self.assertEqual(before["fp"],1)
        self.assertAlmostEqual(before["f1"],50)
        self.assertGreater(after["f1"],before["f1"])
        report=ev.paired_compare(a,b,"pope",n_boot=300,seed=7)
        self.assertEqual(report["independent_image_clusters"],2)
        self.assertEqual(report["flips"]["no->yes"],1)
        self.assertEqual(report["n"],4)
        self.assertEqual(report["paired_cluster_bootstrap_ci95_pp"],
                         ev.paired_compare(a,b,"pope",n_boot=300,seed=7)
                         ["paired_cluster_bootstrap_ci95_pp"])

    def test_official_gqa_correct_field(self):
        a=[{"sample_id":f"s{i}","image_id":"img1" if i<2 else "img2",
            "correct":ok} for i,ok in enumerate([True,False,True])]
        b=[dict(x) for x in a]
        b[1]["correct"]=True
        rep=ev.paired_compare(a,b,"gqa",n_boot=100,seed=1)
        self.assertAlmostEqual(rep["baseline"]["accuracy"],200/3)
        self.assertAlmostEqual(rep["delta_pp"],100/3)
        self.assertEqual(rep["baseline"]["scoring_basis"],"official_correct")
        with self.assertRaises(ValueError):
            ev.paired_compare(a,a[:-1],"gqa",n_boot=10)

    def test_duplicate_id_and_mismatched_gold(self):
        a=[{"sample_id":"s","gold":"yes","pred":"yes","image_id":"image"}]
        with self.assertRaises(ValueError): ev.unique_index(a+a)
        b=[dict(a[0],gold="no")]
        with self.assertRaises(ValueError): ev.paired_compare(a,b,"pope",n_boot=10)

    def test_heatmap_requires_real_query_trace(self):
        row={"sample_id":"synthetic-unit-test","n_patches":4,
             "native_ids":[0,1],"final_ids":[0,2],
             "query_scores":[.1,.2,.4,.3]}
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"diag.png"
            pr.render(row,2,2,str(out))
            self.assertTrue(out.exists())
            with self.assertRaises(ValueError):
                pr.render(dict(row,query_scores=[]),2,2,str(out))


if __name__=="__main__":
    unittest.main(verbosity=2)
