import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from check_plan import budget, permanent_survivors_feasible, validate
from paired_analysis import align, metric, paired_report
from reference_selector import (bounded_admission, cut_exchange, disagreement,
    initialize_anchors, matched_random_admission, matched_random_cut, rank01,
    shuffle_entry_risk)
from run_queue import gate_value, jobs_for_phase, matching_result


class BudgetChecks(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "config/experiments.json").read_text())

    def test_exact_anchors_and_cut_indices(self):
        l2 = budget(self.config["schedules"]["L2"])
        e3 = budget(self.config["schedules"]["E3"])
        self.assertEqual((l2["W"], l2["Q_visual_only"], l2["cuts_after_1based_blocks"]),
                         (1024, 52664, [12, 20]))
        self.assertEqual((e3["W"], e3["Q_visual_only"], e3["cuts_after_1based_blocks"], e3["final_K"]),
                         (2048, 241856, [4, 12, 22], 16))

    def test_permanent_76_survivors_are_infeasible(self):
        self.assertFalse(permanent_survivors_feasible(self.config["schedules"]["E3"], 76))
        self.assertTrue(permanent_survivors_feasible(self.config["schedules"]["E3"], 16))

    def test_invalid_growth_and_layer_counts_rejected(self):
        for s in ([[10, 16], [12, 16]], [[32, 31]], [[32.0, 32]]):
            with self.assertRaises(ValueError):
                budget(s)

    def test_every_defined_arm_matches_declared_budget(self):
        self.assertEqual(validate(self.config)["status"], "PASS")


class SelectorChecks(unittest.TestCase):
    def setUp(self):
        self.n = 32
        self.c = np.eye(self.n)
        self.native = np.arange(self.n, 0, -1, dtype=float)
        self.query = np.arange(self.n, dtype=float)
        self.anchors = {"ids": [31], "weights": {"31": 1.0}, "capacity": 2}

    def test_ties_do_not_create_fake_disagreement(self):
        x = np.ones(32)
        np.testing.assert_allclose(rank01(x), 0.5)
        np.testing.assert_array_equal(disagreement(x, x), np.zeros(32))

    def test_admission_and_random_control_match_donors_and_capacity(self):
        base = list(range(16))
        g9 = np.arange(32, dtype=float)
        late = g9[::-1]
        plan = bounded_admission(base, g9, late, self.native, self.c, 2)
        control = matched_random_admission(base, plan, 32, 17)
        self.assertEqual(len(set(plan["ids"])), 16)
        self.assertEqual(plan["actual_count"], 2)
        self.assertEqual(control["removed"], plan["removed"])
        self.assertEqual(control["actual_count"], plan["actual_count"])
        self.assertTrue(set(control["added"]).isdisjoint(base))

    def test_no_admission_when_no_source_signal(self):
        plan = bounded_admission(range(16), np.ones(32), np.ones(32), self.native, self.c, 2)
        self.assertEqual(plan["actual_count"], 0)
        self.assertEqual(plan["ids"], list(range(16)))

    def test_positive_proxy_swap_does_not_increase_K(self):
        result = cut_exchange(range(32), range(16), self.native, self.query,
                              self.anchors, self.c, self.c)
        self.assertEqual(len(result["ids"]), 16)
        self.assertIn(31, result["ids"])
        self.assertLessEqual(len(result["swaps"]), 2)
        self.assertGreater(result["proxy_after"], result["proxy_before"])

    def test_deleted_anchor_cannot_be_resurrected(self):
        result = cut_exchange(range(31), range(16), self.native, self.query,
                              self.anchors, self.c, self.c)
        self.assertNotIn(31, result["ids"])
        self.assertEqual(result["swaps"], [])

    def test_flat_query_has_no_new_query_conditioned_action(self):
        result = cut_exchange(range(32), range(16), self.native, np.ones(32),
                              self.anchors, self.c, self.c)
        self.assertEqual(result["swaps"], [])

    def test_no_query_gate_is_not_native_score_reoptimization(self):
        # Even if an adapter's native set is not raw-score Top-K, zero query
        # evidence must not trigger an unrelated native-score-only rewrite.
        keep = list(range(8, 24))
        result = cut_exchange(range(32), keep, self.native, np.ones(32),
                              self.anchors, self.c, self.c)
        self.assertEqual(result["ids"], keep)

    def test_source_only_control_removes_only_explicit_query_gate(self):
        result = cut_exchange(range(32), range(16), self.native, np.ones(32),
                              self.anchors, self.c, self.c, condition_on_query=False)
        self.assertGreater(len(result["swaps"]), 0)
        self.assertEqual(len(result["ids"]), 16)

    def test_fractional_admission_and_overcapacity_anchors_are_rejected(self):
        with self.assertRaises(ValueError):
            bounded_admission(range(16), self.native, self.query, self.native, self.c, 1.5)
        state = {"ids":[30,31],"weights":{"30":1.0,"31":1.0},"capacity":1}
        with self.assertRaises(ValueError):
            cut_exchange(range(32), range(16), self.native, self.query, state, self.c, self.c)

    def test_lambda_zero_preserves_native_and_order(self):
        alive = list(reversed(range(32)))
        result = cut_exchange(alive, range(16), self.native, self.query,
                              self.anchors, self.c, self.c, weight=0.0)
        self.assertEqual(result["ids"], list(reversed(range(16))))

    def test_shared_prefix_random_cut_has_same_net_donors(self):
        actual = cut_exchange(range(32), range(16), self.native, self.query,
                              self.anchors, self.c, self.c)
        control = matched_random_cut(range(32), range(16), actual, 32, 17)
        self.assertEqual(control["actual_count"], actual["net_changed_count"])
        self.assertEqual(set(control["removed"]), set(range(16)) - set(actual["ids"]))
        self.assertEqual(len(control["ids"]), 16)

    def test_shuffle_preserves_weight_multiset_and_anchor_cap(self):
        d = np.linspace(0, 1, 32)
        shuffled = shuffle_entry_risk(range(16), d, 17)
        np.testing.assert_allclose(np.sort(shuffled[:16]), np.sort(d[:16]))
        np.testing.assert_allclose(shuffled[16:], d[16:])
        state = initialize_anchors(range(16), shuffled, 2)
        self.assertLessEqual(len(state["ids"]), 2)

    def test_randomized_subset_and_quota_invariants(self):
        rng = np.random.default_rng(8)
        for n, k in [(20, 10), (32, 16), (48, 24)]:
            for _ in range(12):
                a = rng.random((n, n)); a = (a + a.T) / 2
                np.fill_diagonal(a, 1)
                native, query = rng.random(n), rng.random(n)
                keep = np.argsort(-native)[:k].tolist()
                state = initialize_anchors(range(n), rng.random(n), 2)
                result = cut_exchange(range(n), keep, native, query, state, a, a)
                self.assertEqual(len(result["ids"]), k)
                self.assertEqual(len(set(result["ids"])), k)
                self.assertTrue(set(result["ids"]).issubset(range(n)))
                self.assertLessEqual(len(result["swaps"]), min(2, k // 8))
                self.assertGreaterEqual(result["proxy_after"] + 1e-12, result["proxy_before"])


class AnalysisChecks(unittest.TestCase):
    def test_textvqa_soft_score_is_not_binarized(self):
        a = [{"task": "textvqa_val_ocr", "sample_id": "1", "image_id": "x", "score": .2}]
        b = [dict(a[0], score=.5)]
        report = paired_report(a, b, "textvqa_val_ocr", reps=10)
        self.assertAlmostEqual(report["delta"], 30)
        self.assertNotIn("binary_transitions", report)

    def test_pope_recomputes_nonlinear_f1_by_image_cluster(self):
        a = [{"task": "pope", "sample_id": str(i), "image_id": "same-image", "split": "random", "label": y, "pred_label": pred}
             for i, (y, pred) in enumerate([(1, 1), (0, 1)])]
        b = [dict(a[0]), dict(a[1], pred_label=0)]
        report = paired_report(a, b, "pope", reps=20, pope_reduction="pooled")
        self.assertEqual(report["n_images"], 1)
        self.assertAlmostEqual(report["delta"], 100 / 3)
        self.assertAlmostEqual(report["cluster_bootstrap_95ci"][0], report["delta"])

    def test_pope_reduction_must_be_explicit(self):
        rows = [{"task": "pope", "sample_id": "x", "image_id": "i", "split": "random", "label": 1, "pred_label": 1},
                {"task": "pope", "sample_id": "y", "image_id": "j", "split": "popular", "label": 1, "pred_label": 0}]
        self.assertAlmostEqual(metric(rows, "pope", "pooled"), 200 / 3)
        self.assertEqual(metric(rows, "pope", "macro"), 50)
        with self.assertRaises(ValueError):
            metric(rows, "pope")

    def test_missing_pairs_and_null_pope_labels_fail_closed(self):
        row = {"task": "gqa", "sample_id": "1", "image_id": "i", "score": 1}
        with self.assertRaises(ValueError):
            align([row], [])
        with self.assertRaises(ValueError):
            metric([{"label": 0, "pred_label": None}], "pope", "pooled")

    def test_mme_cannot_be_replaced_with_row_mean(self):
        with self.assertRaises(ValueError):
            metric([{"score": 1}], "mme")


class QueueChecks(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "config/experiments.json").read_text())
        self.gates = json.loads((ROOT / "config/gates.example.json").read_text())

    def test_unknown_or_unevidenced_gates_do_not_release_jobs(self):
        self.assertFalse(gate_value(self.gates, "runtime_bound"))
        self.gates["gates"]["runtime_bound"] = {"value": True, "evidence": []}
        self.assertFalse(gate_value(self.gates, "runtime_bound"))

    def test_low_budget_has_five_arms_and_deterministic_control_is_not_repeated(self):
        jobs = jobs_for_phase(self.config, self.gates, "pilot_low")
        self.assertEqual(len(jobs), 20)
        self.assertEqual(len({j["arm"] for j in jobs}), 5)
        combined = jobs_for_phase(self.config, self.gates, "pilot")
        self.assertEqual(len(combined), 32)
        self.assertEqual({j["gpu"] for j in combined}, {0, 2, 4, 7})
        self.gates["winners"]["1024"] = "AS_1024"
        controls = jobs_for_phase(self.config, self.gates, "controls")
        self.assertEqual(len([j for j in controls if j["arm"].startswith("SOURCE_ONLY")]), 4)

    def test_wrong_budget_winner_is_rejected(self):
        self.gates["winners"]["1024"] = "AS_2048"
        with self.assertRaises(ValueError):
            jobs_for_phase(self.config, self.gates, "confirm")

    def test_cache_requires_current_ids_and_predictions_hash(self):
        job = jobs_for_phase(self.config, self.gates, "pilot_low")[0]
        with tempfile.TemporaryDirectory() as t:
            p = Path(t); predictions = b'{"synthetic":true}\n'
            (p / "predictions.jsonl").write_bytes(predictions)
            meta = {"arm":job["arm"],"task":job["task"],"actual_W":1024,"seed":17,
                    "run_tag":"synthetic","spec_hash":job["spec_hash"],"sample_count":1,
                    "ids_sha256":"correct-ids","scorer_hash":"s","checkpoint_hash":"c",
                    "latency_fix_commit":"fixed","complete":True,
                    "predictions_sha256":hashlib.sha256(predictions).hexdigest()}
            (p / "run_meta.json").write_text(json.dumps(meta))
            (p / "metrics.json").write_text('{"status":"complete"}')
            self.assertTrue(matching_result(p,job,"synthetic","correct-ids"))
            self.assertFalse(matching_result(p,job,"synthetic","changed-ids"))
            (p / "predictions.jsonl").write_text('tampered')
            self.assertFalse(matching_result(p,job,"synthetic","correct-ids"))

    def test_queue_dry_run_needs_no_cuda_and_does_not_launch_adapter(self):
        p = subprocess.run([sys.executable, str(ROOT / "tools/run_queue.py"), "--phase", "pilot_low"],
                           capture_output=True, text=True, check=True)
        result = json.loads(p.stdout)
        self.assertEqual(result["mode"], "dry_run")
        self.assertEqual(result["jobs_ready"], 0)
        self.assertEqual(result["jobs_blocked"], 20)


if __name__ == "__main__":
    unittest.main()
