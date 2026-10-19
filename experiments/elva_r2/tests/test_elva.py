import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

from elva.budget import Schedule, Trace, feasible_suffixes, choose_suffix
from elva.evidence import (unit, select_stage1, question_visual_mass, rank_drift,
                           average_ranks, compact_sequence)
from elva.counterfactual import Swap, run_swap, answer_nll, ESTIMAND
from elva.toy_adapter import ToyAdapter
from elva.learning import (PairModel, pair_features, split_image, fit_pairs,
                           choose_swap, calibrate_threshold)
from elva.integration import BoundRuntime
from elva.capture import LayerCapture
from make_catalog import build_catalog
from run_queue import digest, validate_result, run_process, conditional_ready, run as run_queue
from elva.benchmark import paired_ttft, summarize
from elva.policy import propose, feature_catalog, decide, collect_catalog


class BudgetTests(unittest.TestCase):
    def test_requested_plateau_and_inverse(self):
        for counts, budget in [((60, 28, 12), 1024), ((90, 42, 18), 1536), ((120, 56, 24), 2048)]:
            p = Schedule(counts, (10, 10, 12))
            self.assertEqual(p.work, budget)
            self.assertEqual(p.events, ((10, counts[1]), (20, counts[2])))
            self.assertEqual(sum(p.trace), budget)
            self.assertEqual((1-p.compression(576))*576*32, budget)
            self.assertEqual(p.trace[9], counts[0])
            self.assertEqual(p.trace[10], counts[1])
            self.assertEqual(p.trace[20], counts[2])

    def test_naive_request_is_over_budget(self):
        self.assertEqual(Schedule((60, 30, 16), (10, 11, 11)).work, 1106)
        self.assertNotEqual(1106, 1024)

    def test_invalid_schedules(self):
        for counts, lengths in [((10, 10), (2, 2)), ((10, 20), (2, 2)), ((10,), (0,)),
                                ((0,), (2,)), ((True,), (2,)), ((10,), (2.5,)), ((), ())]:
            with self.subTest(counts=counts, lengths=lengths), self.assertRaises(ValueError):
                Schedule(counts, lengths)

    def test_square_cost_is_not_work(self):
        a, b = Schedule((60, 28, 12), (10, 10, 12)), Schedule((60, 32, 12), (10, 8, 14))
        self.assertEqual(a.work, b.work)
        self.assertNotEqual(a.proxies(7), b.proxies(7))

    def test_trace_records_before_block(self):
        p = Schedule((5, 2), (2, 3))
        trace = Trace(p.work, p.depth)
        for block, k in enumerate(p.trace, 1):
            trace.before_block(block, k)
        p.validate_trace(trace.finish())
        with self.assertRaises(ValueError):
            trace.before_block(5, 2)

    def test_trace_rejects_resurrection_and_shortfall(self):
        trace = Trace(20, 4)
        trace.before_block(1, 4)
        with self.assertRaises(ValueError):
            trace.before_block(2, 5)
        with self.assertRaises(ValueError):
            trace.finish()

    def test_suffix_solver_against_exhaustive_integer_enumeration(self):
        # Independent brute-force baseline over ALL small catalogs.
        cases = 0
        for depth in range(3, 8):
            for completed in range(1, depth-1):
                boundaries = tuple(range(completed+1, depth))
                for current in range(2, 7):
                    spent = completed*current
                    for budget in range(spent+1, spent+(depth-completed)*current+1):
                        finals = tuple(range(1, current))
                        actual = feasible_suffixes(depth=depth, completed=completed, spent=spent,
                                                  current_k=current, budget=budget,
                                                  boundaries=boundaries, finals=finals)
                        actual_set = {(p.counts, p.lengths) for p in actual}
                        brute = set()
                        for boundary, middle, final in itertools.product(boundaries, range(1, current+1), finals):
                            d1, d2 = boundary-completed, depth-boundary
                            if final < middle and spent+middle*d1+final*d2 == budget:
                                brute.add(((middle, final), (d1, d2)))
                        self.assertEqual(actual_set, brute)
                        cases += 1
        self.assertEqual(cases, 1000)

    def test_suffix_choice_abstains_and_rejects_cost_mismatch(self):
        catalog = feasible_suffixes(depth=32, completed=10, spent=600, current_k=60,
                                    budget=1024, boundaries=(18, 20, 26), finals=(12,))
        self.assertEqual(len(catalog), 3)
        self.assertEqual(choose_suffix(catalog, [0, -.1, .02], threshold=.05), 0)
        self.assertEqual(choose_suffix(catalog, [0, -.1, .2], threshold=.05), 2)
        with self.assertRaises(ValueError):
            choose_suffix([catalog[0], Schedule((5,), (2,))], [0, 1])
        with self.assertRaises(ValueError):
            choose_suffix(catalog, [0, float("nan"), 2])


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        r = np.random.default_rng(3)
        self.pen, self.early = r.normal(size=(8, 5)), r.normal(size=(8, 5))
        self.ids = np.arange(8)*7+100
        self.a, self.b = np.arange(8)/40, np.arange(8)[::-1]/40

    def select(self, **kw):
        return select_stage1(self.pen, self.a, self.ids, 3, early=self.early,
                             early_scores=self.b, early_ids=self.ids, **kw)

    def test_payload_and_projector_width_protected(self):
        for guide in ("pen", "l9", "dual"):
            out = self.select(guide=guide)
            np.testing.assert_array_equal(out["payload"], self.pen[out["keep"]])
            self.assertEqual(out["payload"].shape, (3, 5))

    def test_dual_kernel_is_weighted_cosine(self):
        beta = .3
        out = self.select(guide="dual", beta=beta)
        p = self.pen/np.linalg.norm(self.pen, axis=1, keepdims=True)
        e = self.early/np.linalg.norm(self.early, axis=1, keepdims=True)
        np.testing.assert_allclose(out["metric"]@out["metric"].T,
                                   (1-beta)*(p@p.T)+beta*(e@e.T), atol=1e-12)

    def test_dual_endpoints_match_single_layer(self):
        for beta, single in [(0., "pen"), (1., "l9")]:
            dual, one = self.select(guide="dual", beta=beta), self.select(guide=single)
            np.testing.assert_array_equal(dual["keep"], one["keep"])
            np.testing.assert_array_equal(dual["metric"], one["metric"])

    def test_geometry_and_score_factor_ablation(self):
        geometry_only = self.select(guide="dual", score_guide="pen")
        np.testing.assert_array_equal(geometry_only["scores"], self.a)
        score_only = self.select(guide="pen", score_guide="dual")
        np.testing.assert_array_equal(score_only["metric"], self.pen)
        np.testing.assert_array_equal(score_only["scores"], (self.a+self.b)/2)

    def test_original_cover_required(self):
        with self.assertRaises(RuntimeError):
            self.select(use_cover=True)

    def test_mutating_callback_cannot_change_original_payload(self):
        before = self.pen.copy()
        def malicious(**kwargs):
            kwargs["metric_features"][:] = 0
            kwargs["scores"][:] = 0
            kwargs["patch_ids"][:] = 999
            return np.array([0, 3, 6])
        out = self.select(use_cover=True, cover=malicious)
        np.testing.assert_array_equal(self.pen, before)
        np.testing.assert_array_equal(out["payload"], before[[0, 3, 6]])

    def test_zero_vector_finite_and_frozen_ids(self):
        self.pen[0] = 0
        out = self.select(guide="dual", frozen_keep=[0, 4, 7])
        self.assertTrue(np.isfinite(out["metric"]).all())
        np.testing.assert_array_equal(out["ids"], self.ids[[0, 4, 7]])

    def test_tiny_normalization_matches_documented_epsilon(self):
        result = unit([[0., 0.], [1e-30, 0.]])
        np.testing.assert_allclose(result, [[0, 0], [1e-18, 0]], atol=0)

    def test_alignment_nonfinite_and_duplicate_keep_rejected(self):
        with self.assertRaises(ValueError):
            select_stage1(self.pen, self.a, self.ids, 3, guide="l9", early=self.early,
                          early_scores=self.b, early_ids=self.ids[::-1])
        with self.assertRaises(ValueError):
            self.select(frozen_keep=[0, 0, 2])
        self.pen[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.select()

    def test_causal_normalization_all_visible_keys(self):
        z = np.zeros((2, 4))
        mask = np.array([[1, 1, 0, 0], [1, 1, 1, 1]], bool)
        mass = question_visual_mass(z, mask, [0, 1])
        np.testing.assert_allclose(mass, [.375, .375])
        self.assertLess(mass.sum(), 1)

    def test_attention_mask_padding_extremes(self):
        z = np.array([[10000, -10000, np.nan]])
        np.testing.assert_allclose(question_visual_mass(z, np.array([[1, 1, 0]], bool), [0]), [1])
        with self.assertRaises(ValueError):
            question_visual_mass(z, np.zeros_like(z, dtype=bool), [0])
        with self.assertRaises(ValueError):
            question_visual_mass(z, np.array([[1, 1, 0]]), [0])

    def test_rank_ties_and_history_original_id_alignment(self):
        np.testing.assert_allclose(average_ranks([5, 5, 5]), [.5, .5, .5])
        a = rank_drift([3, 1, 2], [20, 50, 10], [1, 2, 99, 3], [50, 10, 99, 20])
        b = rank_drift([3, 1, 2], [20, 50, 10], [3, 1, 2], [20, 50, 10])
        np.testing.assert_array_equal(a, b)
        with self.assertRaises(ValueError):
            rank_drift([3, 1], [20, 50], [1], [20])

    def test_compaction_keeps_text_and_original_positions(self):
        h = np.arange(24).reshape(6, 4).astype(float)
        compact, p, ids, rows = compact_sequence(h, [2, 5, 7, 9, 12, 20], [-1, 88, 4, 23, -1, -1], [4])
        np.testing.assert_array_equal(rows, [0, 2, 4, 5])
        np.testing.assert_array_equal(p, [2, 7, 12, 20])
        np.testing.assert_array_equal(compact, h[rows])
        with self.assertRaises(ValueError):
            compact_sequence(h, [2, 5, 7, 9, 12, 20], [-1, 88, 4, 23, -1, -1], [999])


class CounterfactualTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ToyAdapter()
        self.prefix = self.adapter.capture(2)
        self.swap = Swap(2, 5, (11, 14, 17), (20, 23), (26, 29), (11, 14))

    def execute(self, mode="native", swap=None):
        return run_swap(self.adapter, self.prefix, swap or self.swap,
                        image_key="fixture-image", question_key="fixture-question", mode=mode)

    def test_actual_causal_decoder_branches_and_work(self):
        out = self.execute()
        self.assertEqual(out["trace"], [12, 12, 5, 5, 5, 2, 2, 2])
        self.assertEqual(out["work"], 45)
        self.assertTrue(math.isfinite(out["delta"]))
        self.assertEqual(out["final_ids"], [11, 14])

    def test_swap_sign_reverses(self):
        reverse = Swap(2, 5, self.swap.common, self.swap.group_b, self.swap.group_a, self.swap.final)
        self.assertAlmostEqual(self.execute()["delta"], -self.execute(swap=reverse)["delta"], places=12)

    def test_support_alignment_changes_cache_estimuand(self):
        native, aligned = self.execute(), self.execute("aligned_support")
        self.assertNotAlmostEqual(native["nll_a"], aligned["nll_a"], places=10)
        self.assertEqual(aligned["cache_mode"], "aligned_support")

    def test_repeated_noop_actual_forwards_are_identical(self):
        values = []
        for _ in range(4):
            state = self.adapter.fork(self.prefix)
            self.adapter.prune(state, self.swap.common+self.swap.group_a)
            self.adapter.advance(state, 5)
            self.adapter.prune(state, self.swap.final)
            values.append(self.adapter.score_suffix(state)["nll"])
        self.assertEqual(max(values)-min(values), 0)

    def test_shared_prefix_immutable(self):
        before = self.adapter.inspect(self.prefix)
        self.execute()
        self.assertEqual(before, self.adapter.inspect(self.prefix))

    def test_aliasing_fork_is_rejected(self):
        self.adapter.fork = lambda state: state
        with self.assertRaises(ValueError):
            self.execute()

    def test_invalid_sets_and_unknown_group(self):
        for args in [(2, 2, (1,), (2,), (3,), (1,)),
                     (2, 5, (1,), (1,), (3,), (1,)),
                     (2, 5, (1,), (2,), (3, 4), (1,)),
                     (2, 5, (1,), (2,), (3,), (7,))]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                Swap(*args)
        unknown = Swap(2, 5, (11, 14), (999,), (20,), (11,))
        with self.assertRaises(ValueError):
            self.execute(swap=unknown)

    def test_answer_nll_shift_not_same_position(self):
        logits = np.log([[.8, .2], [.6, .4], [.1, .9]])
        actual = answer_nll(logits, [0, 1, 0], np.array([0, 1, 1], bool))
        self.assertAlmostEqual(actual, (-math.log(.2)-math.log(.6))/2)
        with self.assertRaises(ValueError):
            answer_nll(logits, [0, 1, 0], np.zeros(3, bool))


class LearningTests(unittest.TestCase):
    def test_shared_policy_catalog_no_resurrection_and_limit(self):
        ids = np.arange(60)*3+19
        scores = np.arange(60)/60
        catalog = propose(scores, ids, np.arange(28), 12)
        self.assertEqual(len(catalog.swaps), 15)
        self.assertTrue(set(catalog.final).issubset(catalog.baseline))
        for common, inserted, removed in catalog.swaps:
            self.assertEqual(len(common)+len(inserted), 28)
            self.assertEqual(len(inserted), len(removed))
            self.assertTrue(set(catalog.final).issubset(common))
        features = np.c_[scores, scores**2]
        matrix = feature_catalog(catalog, features, begin=10, end=20, depth=32)
        model = PairModel(np.ones(8), np.zeros(8), "native", tuple("abcdefgh"), catalog.rule_hash)
        self.assertEqual(decide(catalog, features, model, begin=10, end=20, depth=32, threshold=0)["action"], 0)
        with self.assertRaises(ValueError):
            feature_catalog(catalog, features, begin=10, end=21, depth=32)

    def test_policy_collects_real_fixture_forwards_and_blocks_training(self):
        adapter = ToyAdapter()
        prefix = adapter.capture(2)
        ids = np.array(adapter.inspect(prefix)["active_ids"])
        scores = np.arange(12)/12
        catalog = propose(scores, ids, np.arange(7), 2, group_size=2, limit=4, begin=2, end=5, depth=8)
        x = scores[:, None]
        rows = collect_catalog(adapter, prefix, catalog, ids, x, begin=2, end=5, depth=8,
                               image_key="fixture", question_key="question", feature_names=list("abcd"))
        self.assertEqual(len(rows), 3)
        self.assertEqual({row["work"] for row in rows}, {51})
        self.assertTrue(all(row["fixture"] for row in rows))
        with self.assertRaises(ValueError):
            fit_pairs(rows, feature_names=list("abcd"), catalog_hash=catalog.rule_hash, cache_mode="native")

    def records(self):
        keys = [f"test-generated-{i}" for i in range(100) if split_image(f"test-generated-{i}") == "train"]
        result = []
        for i in range(20):
            x = np.array([i/20, (i % 3)-1.])
            result.append(dict(schema=ESTIMAND, cache_mode="native", catalog_hash="catalog",
                               feature_names=["a", "b"], feature_cutoff=10, begin=10,
                               image_key=keys[i % 5], x=x.tolist(), delta=float(x@np.array([2., -.3]))))
        return result

    def test_pair_features_context_and_antisymmetry(self):
        x = np.arange(30).reshape(10, 3)/10
        a = pair_features(x, [0, 1], [2, 3], [4, 5], begin=10, end=20, depth=32)
        b = pair_features(x, [0, 1], [4, 5], [2, 3], begin=10, end=20, depth=32)
        np.testing.assert_array_equal(a, -b)
        changed = pair_features(x, [6, 7], [2, 3], [4, 5], begin=10, end=20, depth=32)
        self.assertFalse(np.allclose(a, changed))

    def test_grouped_split_stable(self):
        self.assertEqual(split_image("canonical-image"), split_image("canonical-image"))
        self.assertEqual(set(split_image(f"img-{i}") for i in range(100)), {"train", "validation", "test"})

    def test_fit_save_load_and_prediction(self):
        records = self.records()
        model = fit_pairs(records, feature_names=["a", "b"], catalog_hash="catalog", cache_mode="native", ridge=1e-6)
        x = np.array([r["x"] for r in records])
        np.testing.assert_allclose(model.predict(x), [r["delta"] for r in records], atol=1e-6)
        np.testing.assert_allclose(model.predict(-x), -model.predict(x))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"head.json"
            model.save(path)
            loaded = PairModel.load(path, feature_names=["a", "b"], catalog_hash="catalog", cache_mode="native")
            np.testing.assert_array_equal(model.predict(x), loaded.predict(x))
            with self.assertRaises(ValueError):
                PairModel.load(path, feature_names=["a", "b"], catalog_hash="old", cache_mode="native")

    def test_old_schema_future_features_and_fixture_rejected(self):
        for key, value in [("schema", "old_e020"), ("feature_cutoff", 11), ("fixture", True), ("cache_mode", "aligned_support")]:
            records = self.records()
            records[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                fit_pairs(records, feature_names=["a", "b"], catalog_hash="catalog", cache_mode="native")

    def test_validation_images_cannot_enter_fit(self):
        records = self.records()
        records[0]["image_key"] = next(f"held-{i}" for i in range(100) if split_image(f"held-{i}") != "train")
        with self.assertRaises(ValueError):
            fit_pairs(records, feature_names=["a", "b"], catalog_hash="catalog", cache_mode="native")

    def test_candidate_max_calibration_and_all_noop(self):
        p = np.array([[0, .8, .2], [0, .1, .6]])
        good = calibrate_threshold(p, [[0, .4, -.3], [0, -.1, .2]], [0, 0, 0])
        bad = calibrate_threshold(p, [[0, -.4, -.3], [0, -.1, -.2]], [0, 0, 0])
        self.assertEqual(good["status"], "CANDIDATE")
        self.assertEqual(bad["status"], "NO_SUPPORTED_DECISIONS")
        self.assertEqual(bad["accepted"], 0)
        self.assertEqual(choose_swap([0, .1], threshold=.2), 0)


class IntegrationAndCatalogTests(unittest.TestCase):
    def test_queue_runs_resumes_and_recovers_bad_result(self):
        # Mock runner lives only in a temporary directory and labels its
        # metrics/binding as fixtures. No model score is produced.
        source = '''import sys,json,hashlib,pathlib
cfg,task,out,key,binding=sys.argv[1:]
c=json.loads(pathlib.Path(cfg).read_text()); b=json.loads(pathlib.Path(binding).read_text())
m=c['method']; s=m['schedule']; trace=[k for k,d in zip(s['counts'],s['lengths']) for _ in range(d)]
h=lambda v: hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
value=dict(run_key=key,config_hash=c['config_hash'],task=task,backend=b['backend'],cache_mode=m['cache_mode'],samples=2,scorer_sha=b['scorer_sha'],metrics={'fixture_metric':1.},binding_hash=h(b),trace_audit=dict(expected=trace,observed_samples=2,mismatch_count=0))
pathlib.Path(out).write_text(json.dumps(value))
'''
        configs = build_catalog()[:2]
        binding = dict(source_commit="FIXTURE", model_sha="FIXTURE", processor_sha="FIXTURE",
                       scorer_sha="FIXTURE", prompt_sha="FIXTURE", tokenizer_sha="FIXTURE",
                       dataset_sha="FIXTURE", backend="FIXTURE", cache_mode="native",
                       native_selectors={"fixture": "FIXTURE"}, original_tokens=576, depth=32)
        with tempfile.TemporaryDirectory() as d:
            runner, output = Path(d)/"runner.py", Path(d)/"out"
            runner.write_text(source)
            command = [sys.executable, str(runner), "{config}", "{task}", "{out}", "{run_key}", "{binding}"]
            args = (configs, binding, command, output, ["0", "1"], ["FIXTURE_TASK"], .01)
            self.assertEqual([s["status"] for s in run_queue(*args)], ["DONE", "DONE"])
            self.assertEqual([s["status"] for s in run_queue(*args)], ["REUSED_VERIFIED", "REUSED_VERIFIED"])
            path = next(output.glob("*/result.json"))
            value = json.loads(path.read_text())
            value["trace_audit"]["mismatch_count"] = 1
            path.write_text(json.dumps(value))
            self.assertEqual(sorted(s["status"] for s in run_queue(*args)), ["FAILED", "REUSED_VERIFIED"])
            self.assertEqual(sorted(s["status"] for s in run_queue(*args)), ["DONE", "REUSED_VERIFIED"])

    def test_conditional_queue_requires_new_real_evidence(self):
        config = next(c for c in build_catalog() if c["method"]["stage2"]["first_selector"] == "TRANSIENT_PAIR_CURRENT")
        self.assertFalse(conditional_ready(config, {}))
        binding = dict(transient_adapter_ready=True, transient_gate=dict(
            schema=ESTIMAND, status="READY", cache_mode="native", weights_sha="test-only",
            catalog_hash="test-only", real_labels=True, heldout_gain_over_noise=True, generation_checked=True))
        self.assertTrue(conditional_ready(config, binding))
        binding["transient_gate"]["schema"] = "old_e020"
        self.assertFalse(conditional_ready(config, binding))

    def test_first_token_callback_timing_and_missing_token(self):
        rows = paired_ttft(["fixture0", "fixture1"], ("a", "b"), lambda s, m: s+m,
                           lambda: None, rounds=2, warmup=0)
        self.assertEqual(len(rows), 8)
        self.assertEqual(summarize(rows)["a"]["observations"], 4)
        with self.assertRaises(ValueError):
            paired_ttft([0], ("a", "b"), lambda s, m: None, lambda: None, warmup=0)

    def test_capture_completed_block_indices_and_removal(self):
        class Handle:
            def __init__(self, hooks, fn): self.hooks, self.fn = hooks, fn
            def remove(self): self.hooks.remove(self.fn)
        class Block:
            def __init__(self, i): self.i, self.hooks = i, []
            def register_forward_hook(self, fn):
                self.hooks.append(fn)
                return Handle(self.hooks, fn)
            def __call__(self, x):
                y = x+self.i
                for fn in self.hooks: fn(self, (x,), (y,))
                return y
        blocks = [Block(i) for i in range(1, 13)]
        hidden = [np.zeros((1, 5, 3))]
        with LayerCapture(blocks) as capture:
            for block in blocks: hidden.append(block(hidden[-1]))
            early, pen = capture.get()
            np.testing.assert_array_equal(early, hidden[9])
            np.testing.assert_array_equal(pen, hidden[-2])
        self.assertTrue(all(not block.hooks for block in blocks))

    def test_capture_invalid_layer_rejected(self):
        with self.assertRaises(ValueError):
            LayerCapture([None]*8, early_completed=9)

    def test_catalog_distinct_and_all_budgets_exact(self):
        configs = build_catalog()
        self.assertEqual(len(configs), len({c["config_hash"] for c in configs}))
        self.assertEqual(len(configs), 83)
        self.assertEqual(sum(not c["method"]["conditional"] for c in configs), 74)
        for c in configs:
            p = Schedule(**c["method"]["schedule"])
            self.assertEqual(p.work, c["method"]["budget"])
            self.assertEqual(p.depth, 32)

    def test_native_runtime_callbacks_and_positions(self):
        state = dict(ids=[101, 303, 202, 404, 505], protected=[1, 10])
        def inspect(s):
            return dict(active_ids=s["ids"], protected_positions=s["protected"], backend="bound")
        def select(s, *, k, **kwargs):
            return s["ids"][:k]
        def compact(s, *, keep_ids):
            return dict(ids=list(keep_ids), protected=s["protected"].copy())
        p = Schedule((5, 2), (2, 3))
        rt = BoundRuntime(p, select, compact, inspect)
        for block in range(1, 6):
            rt.before_block(block, state)
            state = rt.after_block(block, state, selector="SOURCE_BOUND")
        self.assertEqual(rt.finish(), p.trace)

    def test_runtime_rejects_backend_change(self):
        p = Schedule((2, 1), (1, 1))
        state = dict(ids=[1, 2], backend="flash")
        inspect = lambda s: dict(active_ids=s["ids"], protected_positions=[7], backend=s["backend"])
        rt = BoundRuntime(p, lambda *a, **k: [1], lambda *a, **k: dict(ids=[1], backend="eager"), inspect)
        rt.before_block(1, state)
        with self.assertRaises(ValueError):
            rt.after_block(1, state, selector="native")

    def test_result_reuse_requires_trace_and_provenance(self):
        config = build_catalog()[0]
        binding = dict(backend="fixture", scorer_sha="fixture")
        schedule = Schedule(**config["method"]["schedule"])
        value = dict(run_key="key", config_hash=config["config_hash"], task="GQA", backend="fixture",
                     cache_mode="native", samples=2, scorer_sha="fixture", metrics={"fixture_only": 1.},
                     binding_hash=digest(binding), trace_audit=dict(expected=list(schedule.trace), observed_samples=2, mismatch_count=0))
        validate_result(value, config, binding, "GQA", "key")
        value["trace_audit"]["mismatch_count"] = 1
        with self.assertRaises(ValueError):
            validate_result(value, config, binding, "GQA", "key")
        value["trace_audit"]["mismatch_count"] = 0
        value["metrics"] = {"fixture_only": float("nan")}
        with self.assertRaises(ValueError):
            validate_result(value, config, binding, "GQA", "key")

    def test_timeout_process_group_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            marker = Path(d)/"should-not-exist"
            child = "import time,pathlib; time.sleep(1); pathlib.Path("+repr(str(marker))+").touch()"
            parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',"+repr(child)+"]); time.sleep(9)"
            with (Path(d)/"log").open("w") as log:
                with self.assertRaises(subprocess.TimeoutExpired):
                    run_process([sys.executable, "-c", parent], env=None, log=log, timeout=.15)
            # Parent and child receive group termination; wait through the
            # child's hypothetical write deadline using a short test subprocess.
            subprocess.run([sys.executable, "-c", "import time; time.sleep(1.1)"], check=True)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
