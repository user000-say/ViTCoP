import itertools
import importlib
import json
import subprocess
import tempfile
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recycle import recycle, cosine_cache, AffinityCache
from fast_selectors import pooled_qk, key_energy, topk, random_keep
from costs import PROFILES, token_trace, work_report, r4, pareto_ids, amdahl_hypothesis
from make_matrix import build, operator
from runtime import RecyclingState, input_seed
from capacity_response import hindsight, permute_budgets


class Modules(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[1., 2.], [3., 4.], [10., 20.], [14., 24.]], np.float32)
        self.f = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        self.a = cosine_cache(self.f)
        self.xy = np.array([[0, 0], [0, 1], [10, 0], [10, 1]], float)

    def test_disabled_is_bit_exact_and_preserves_keep_order(self):
        k = np.array([2, 0])
        result = recycle(self.x, k, alpha=0)
        self.assertEqual(result.hidden.tobytes(), self.x[k].tobytes())
        np.testing.assert_array_equal(result.keep, k)

    def test_same_cardinality_every_strength(self):
        for alpha in (0., .25, .5, 1.):
            out = recycle(self.x, [0, 2], self.a, self.xy, alpha=alpha)
            self.assertEqual(out.hidden.shape, (2, 2))
            np.testing.assert_array_equal(out.keep, [0, 2])

    def test_known_weighted_cluster_mean(self):
        out = recycle(self.x, [0, 2], self.a, self.xy, weights=[1, 3, 2, 2], alpha=1)
        np.testing.assert_allclose(out.hidden, [[2.5, 3.5], [12., 22.]])
        np.testing.assert_array_equal(out.receiver, [0, 1])

    def test_bounded_perturbation_formula(self):
        full = recycle(self.x, [0, 2], self.a, self.xy, alpha=1)
        half = recycle(self.x, [0, 2], self.a, self.xy, alpha=.5)
        np.testing.assert_allclose(half.hidden, self.x[[0, 2]] + .5 * (full.hidden-self.x[[0, 2]]))

    def test_weight_scale_invariance(self):
        a = recycle(self.x, [0, 2], self.a, self.xy, weights=[1, 2, 3, 4])
        b = recycle(self.x, [0, 2], self.a, self.xy, weights=[10, 20, 30, 40])
        np.testing.assert_allclose(a.hidden, b.hidden)

    def test_algebraic_first_moment_not_future_attention(self):
        w = np.array([1., 2., 3., 4.])
        out = recycle(self.x, [0, 2], self.a, self.xy, weights=w, alpha=1)
        coarse = (out.hidden * (out.anchor_mass + out.donor_mass)[:, None]).sum(0)
        np.testing.assert_allclose(coarse, (self.x * w[:, None]).sum(0))

    def test_group_gate_blocks_cross_image_merge(self):
        out = recycle(self.x, [0, 2], np.ones((4,4)), self.xy,
                      groups=[0, 1, 0, 1], radius=None)
        self.assertEqual(out.audit["assigned"], 0)
        np.testing.assert_array_equal(out.hidden, self.x[[0, 2]])

    def test_spatial_gate_blocks_far_similar_features(self):
        out = recycle(self.x, [0], self.a, self.xy, radius=.1)
        self.assertEqual(out.audit["assigned"], 0)
        np.testing.assert_array_equal(out.hidden, self.x[[0]])

    def test_semantic_gate_rejects_unrelated_state(self):
        a = np.eye(4)
        out = recycle(self.x, [0, 2], a, self.xy, radius=None)
        self.assertEqual(out.audit["assigned"], 0)

    def test_no_donor_returns_exact_gather(self):
        k = np.array([3, 1, 0, 2])
        out = recycle(self.x, k)
        np.testing.assert_array_equal(out.hidden, self.x[k])

    def test_zero_features_have_finite_affinity_and_no_default_merge(self):
        a = cosine_cache(np.zeros((4, 5)))
        self.assertTrue(np.isfinite(a).all())
        out = recycle(self.x, [0, 2], a, self.xy)
        self.assertEqual(out.audit["assigned"], 0)

    def test_zero_query_weights_are_logged(self):
        out = recycle(self.x, [0, 2], self.a, self.xy, weights=np.zeros(4))
        self.assertTrue(out.audit["uniform_weight_fallback"])
        self.assertTrue(np.isfinite(out.hidden).all())

    def test_zero_mass_cluster_has_no_update(self):
        out = recycle(self.x, [0, 2], self.a, self.xy, weights=[0,0,1,1])
        np.testing.assert_array_equal(out.hidden[0], self.x[0])

    def test_norm_restore_and_cancellation_safe(self):
        out = recycle(self.x, [0, 2], self.a, self.xy, preserve_norm=True)
        np.testing.assert_allclose(np.linalg.norm(out.hidden,axis=1), np.linalg.norm(self.x[[0,2]],axis=1),rtol=1e-6)
        z = recycle(np.array([[1.,0.],[-1.,0.]]), [0], np.ones((2,2)),
                    radius=None, alpha=1, preserve_norm=True)
        np.testing.assert_array_equal(z.hidden, [[1.,0.]])

    def test_dtype_preserved_and_source_immutable(self):
        x = self.x.astype(np.float16); old=x.copy()
        out = recycle(x, [0, 2], self.a, self.xy)
        self.assertEqual(out.hidden.dtype, x.dtype)
        np.testing.assert_array_equal(x, old)

    def test_random_recipient_supports_no_geometry(self):
        a = recycle(self.x, [0, 2], np.ones((4,4)), radius=None, assignment="random", seed=8)
        b = recycle(self.x, [0, 2], np.ones((4,4)), radius=None, assignment="random", seed=8)
        np.testing.assert_array_equal(a.receiver, b.receiver)
        np.testing.assert_array_equal(a.hidden, b.hidden)

    def test_random_and_semantic_have_same_acceptance_set(self):
        for mode in ("semantic", "random"):
            out = recycle(self.x, [0, 2], self.a, self.xy, assignment=mode)
            self.assertEqual(out.audit["assigned"], 2)

    def test_invalid_inputs(self):
        for keep in ([0,0], [-1], [4], [], [0.5]):
            with self.subTest(keep=keep), self.assertRaises(ValueError):
                recycle(self.x, keep, self.a, self.xy)
        for alpha in (-1., 2., np.nan):
            with self.assertRaises(ValueError):
                recycle(self.x, [0,2], self.a, self.xy, alpha=alpha)
        for weights in ([1,-1,1,1], [np.nan]*4, [1,2]):
            with self.assertRaises(ValueError):
                recycle(self.x, [0,2], self.a, self.xy, weights=weights)

    def test_cache_mapping_and_cross_input_rejection(self):
        c=AffinityCache("imageA:coverhash",self.f,self.xy)
        view=c.view("imageA:coverhash",np.array([3,1,2]))
        np.testing.assert_array_equal(view["affinity"],self.a[np.ix_([3,1,2],[3,1,2])])
        with self.assertRaises(ValueError): c.view("imageB",np.array([0,1]))
        with self.assertRaises(ValueError): c.view("imageA:coverhash",np.array([0,0]))

    def test_two_event_identity_tracking(self):
        state=RecyclingState("sample",self.f,self.xy,[10,20,30,40],operator(("first","final")))
        first,ids=state.compress("first",self.x,np.array([2,0,3]))
        np.testing.assert_array_equal(ids,[30,10,40])
        final,ids=state.compress("final",first.hidden,np.array([1,0]))
        np.testing.assert_array_equal(ids,[10,30])
        self.assertEqual(final.hidden.shape,(2,2))
        with self.assertRaises(ValueError): state.compress("final",final.hidden,np.array([0]))

    def test_query_feature_failure_does_not_consume_event(self):
        s=RecyclingState("sample",self.f,self.xy,[10,20,30,40],operator(("first",),variant="query_weights"))
        with self.assertRaises(ValueError): s.compress("first",self.x,np.array([0,2]))
        out,_=s.compress("first",self.x,np.array([0,2]),np.ones(4))
        self.assertEqual(len(out.hidden),2)

    def test_off_runtime_requires_no_feature_cache(self):
        s=RecyclingState("sample",None,None,[10,20,30,40],operator(()))
        self.assertIsNone(s.cache)
        out,_=s.compress("first",self.x,np.array([2,0]))
        self.assertEqual(out.hidden.tobytes(),self.x[[2,0]].tobytes())

    def test_input_seed_is_stable_and_sample_specific(self):
        self.assertEqual(input_seed("a",4),input_seed("a",4))
        self.assertNotEqual(input_seed("a",4),input_seed("b",4))

    def test_pooled_qk_matches_explicit_raw_logit_mean(self):
        rng=np.random.default_rng(1)
        q=rng.normal(size=(3,5,7)); k=rng.normal(size=(3,8,7))
        full=np.einsum("hrd,hnd->hrn",q,k)/np.sqrt(7)
        np.testing.assert_allclose(pooled_qk(q,k,causal_bounds=(10,9)),full.mean((0,1)))
        w=np.array([1,2,3,4,5],float);w/=w.sum()
        np.testing.assert_allclose(pooled_qk(q,k,w,causal_bounds=(10,9)),np.einsum("hrn,r->n",full,w)/3)

    def test_pooled_qk_rejects_unverified_or_future_reader(self):
        q=np.ones((2,3,4)); k=np.ones((2,5,4))
        for bounds in (None,(3,5),(5,5),(-1,-2)):
            with self.assertRaises(ValueError): pooled_qk(q,k,causal_bounds=bounds)

    def test_key_energy_is_norm_ordering(self):
        k=np.random.default_rng(2).normal(size=(3,8,7))
        np.testing.assert_allclose(key_energy(k),np.linalg.norm(k.transpose(1,0,2).reshape(8,-1),axis=1)**2)

    def test_topk_optimal_and_stable(self):
        scores=np.array([.1,.7,.2,.7])
        got=topk(scores,2)
        self.assertAlmostEqual(scores[got].sum(),max(scores[list(c)].sum() for c in itertools.combinations(range(4),2)))
        np.testing.assert_array_equal(topk(np.zeros(5),2),[0,1])

    def test_random_cardinality_identity_and_seed(self):
        for k in (1,32,172):
            a=random_keep(172,k,5);b=random_keep(172,k,5)
            np.testing.assert_array_equal(a,b)
            self.assertEqual(len(np.unique(a)),k)
            self.assertTrue(np.all(a[:-1]<a[1:]))

    def test_all_profile_costs_and_inverse(self):
        expected=dict(SHIFT=2284,SHIFT_K108=2204,SHIFT_K100=2124,SHIFT_K132=2444,FINAL11=2116,FINAL15_WMATCH=2284)
        for name,ev in PROFILES.items():
            report=work_report(ev)
            self.assertEqual(report["visual_token_layers"],expected[name])
            a,k=ev[0]; b,final=ev[1]
            self.assertEqual((expected[name]-a*172-(32-b)*final)/(b-a),k)

    def test_trace_does_not_confuse_layer_indices(self):
        with self.assertRaises(ValueError): token_trace(32,172,[(0,116),(13,32)])
        with self.assertRaises(ValueError): token_trace(32,172,[(3,173)])
        t=token_trace(32,172,PROFILES["SHIFT"])
        np.testing.assert_array_equal(t,[172]*3+[116]*10+[32]*19)

    def test_r4_named_metrics_and_inverse(self):
        metrics=dict(GQA=61.9,POPE=85.8,MME=1852.,MMBench=64.1)
        self.assertEqual(r4(metrics),100.)
        metrics={k:v*.9 for k,v in metrics.items()}
        self.assertAlmostEqual(r4(metrics),90.)
        with self.assertRaises(ValueError): r4(dict(GQA=61.9,POPE=85.8,MME=1852.,MMB=64.1))

    def test_pareto_requires_real_matching_conditions(self):
        rows=[dict(id="a",R4=96.,ttft_ms=100.,provenance_sha256="x"),
              dict(id="b",R4=97.,ttft_ms=100.,provenance_sha256="x"),
              dict(id="c",R4=96.5,ttft_ms=99.,provenance_sha256="x")]
        self.assertEqual(set(pareto_ids(rows)),{"b","c"})
        rows[0]["provenance_sha256"]="y"
        with self.assertRaises(ValueError): pareto_ids(rows)

    def test_amdahl_labels_hypothesis_and_accounts_overhead(self):
        x=amdahl_hypothesis(100.,60.,.9,2.)
        self.assertTrue(x["hypothetical_only"])
        self.assertEqual(x["estimated_ms"],98.)
        self.assertEqual(x["savings_ms"],2.)

    def test_matrix_has_unique_meaningful_configurations(self):
        rows=build()
        self.assertEqual(len(rows),79)
        self.assertEqual(len({r["id"] for r in rows}),79)
        self.assertEqual({b:sum(r["block"]==b for r in rows) for b in "ABCDEFGH"},
                         dict(A=8,B=18,C=12,D=12,E=12,F=6,G=6,H=5))

    def test_capacity_dp_matches_full_enumeration(self):
        rng=np.random.default_rng(7)
        for n in range(1,7):
            u=rng.normal(size=(n,3))
            actual=hindsight(u,[2124,2284,2444])
            expected=max(sum(u[i,k] for i,k in enumerate(choices))/n
                         for choices in itertools.product(range(3),repeat=n) if sum(choices)<=n)
            self.assertAlmostEqual(actual["mean_utility"],expected)
            self.assertLessEqual(actual["mean_cost"],2284)

    def test_capacity_includes_all_middle_option(self):
        u=np.tile([0.,1.,0.],(5,1))
        got=hindsight(u,[1,2,3])
        self.assertEqual(got["choice"],[1]*5)
        self.assertEqual(got["gain_over_middle"],0.)

    def test_budget_shuffle_is_order_independent(self):
        ids=np.array(["b","d","a","c"]);choice=np.array([0,2,1,0])
        a=permute_budgets(ids,choice,9);b=permute_budgets(ids[::-1],choice[::-1],9)
        self.assertEqual(dict(zip(ids,a)),dict(zip(ids[::-1],b)))
        np.testing.assert_array_equal(np.sort(a),np.sort(choice))

    def test_public_package_import(self):
        sys.path.insert(0,str(ROOT.parent))
        package=importlib.import_module("e020_3")
        self.assertTrue(callable(package.recycle))
        self.assertTrue(callable(importlib.import_module("e020_3.runtime").RecyclingState))

    def test_capacity_cli_runs_and_labels_diagnostic(self):
        with tempfile.TemporaryDirectory() as folder:
            d=Path(folder)
            np.savez(d/"panel.npz",utilities=np.array([[0.,1.,1.],[1.,1.,0.]]),
                     costs=np.array([2124,2284,2444]),sample_ids=np.array(["a","b"]))
            run=subprocess.run([sys.executable,str(ROOT/"capacity_response.py"),str(d/"panel.npz"),str(d/"out.json")],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            data=json.loads((d/"out.json").read_text())
            self.assertTrue(data["diagnostic_only"])
            self.assertEqual(data["sample_ids"],["a","b"])

    def test_convex_bounds_random_stress(self):
        rng=np.random.default_rng(4)
        for n in (2,9,32,172):
            x=rng.normal(size=(n,11)).astype(np.float32)
            f=rng.normal(size=(n,7));aff=cosine_cache(f)
            keep=np.sort(rng.choice(n,max(1,n//3),replace=False))
            for alpha in (.25,.5,1.):
                y=recycle(x,keep,aff,radius=None,cosine_floor=-1,alpha=alpha).hidden
                self.assertTrue(np.isfinite(y).all())
                self.assertTrue(np.all(y>=x.min(0)-1e-6))
                self.assertTrue(np.all(y<=x.max(0)+1e-6))

    def test_torch_parity_when_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch absent locally; real CPU/CUDA parity is required on server")
        from torch_backend import recycle as tr, pooled_qk as tq, cosine_cache as tc
        for device in ["cpu"]+(["cuda"] if torch.cuda.is_available() else []):
            x=torch.tensor(self.x,device=device);xy=torch.tensor(self.xy,device=device)
            a=tc(torch.tensor(self.f,device=device));keep=torch.tensor([0,2],device=device)
            for alpha in (0.,.25,.5,1.):
                for norm in (False,True):
                    y,_=tr(x,keep,a,xy,alpha=alpha,preserve_norm=norm,debug=True)
                    expected=recycle(self.x,[0,2],self.a,self.xy,alpha=alpha,preserve_norm=norm)
                    np.testing.assert_allclose(y.cpu().numpy(),expected.hidden,rtol=1e-5,atol=1e-6)
            q=np.random.default_rng(5).normal(size=(2,3,4));k=np.random.default_rng(6).normal(size=(2,7,4))
            np.testing.assert_allclose(tq(torch.tensor(q,device=device),torch.tensor(k,device=device),causal_bounds=(10,9)).cpu().numpy(),pooled_qk(q,k,causal_bounds=(10,9)),rtol=1e-5,atol=1e-6)
            from torch_backend import inject_visual
            original=torch.arange(24.,device=device).reshape(1,6,4)
            changed=inject_visual(original,torch.tensor([1,4],device=device),torch.ones((2,4),device=device),inplace=False)
            self.assertTrue(torch.equal(changed[:,[0,2,3,5]],original[:,[0,2,3,5]]))
            self.assertTrue(torch.equal(original,torch.arange(24.,device=device).reshape(1,6,4)))
            from runtime_torch import RecyclingState as TS
            s=TS("sample",torch.tensor(self.f,device=device),xy,torch.tensor([10,20,30,40],device=device),operator(("first","final")),debug=True)
            a,ids,_=s.compress("first",x,keep)
            b,ids,_=s.compress("final",a,torch.tensor([1],device=device))
            self.assertEqual(ids.cpu().tolist(),[30])


if __name__=="__main__":
    unittest.main(verbosity=2)
