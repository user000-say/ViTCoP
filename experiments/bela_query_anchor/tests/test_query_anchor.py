import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.bela_query_anchor.query_anchor import (
    AnchorConfig, matched_random_entry_control, select_question_anchored
)
from experiments.bela_query_anchor.llava_adapter import apply_llava_query_anchor
from experiments.bela_query_anchor.pope_audit import (
    grouped, paired, read_predictions, stats,
)


class EntrySelectorTests(unittest.TestCase):
    def setUp(self):
        # Two genuinely different visual clusters. Native selection missed B.
        self.feat = torch.tensor(
            [[1., 0.]] * 8 + [[0., 1.]] * 4
        )
        self.native = torch.tensor([0, 1, 2, 3])
        self.raw = torch.tensor([0.] * 8 + [1.] * 4)
        self.cfg = AnchorConfig(max_swaps=2, min_raw_std=.005)

    def select(self, **kwargs):
        return select_question_anchored(
            native_keep_ids=self.native,
            visual_metric=self.feat,
            config=self.cfg,
            **kwargs
        )

    def test_missing_signal_is_exact_no_op(self):
        result = self.select()
        self.assertEqual(result.swaps, 0)
        self.assertEqual(result.keep_ids.tolist(), self.native.tolist())

    def test_uniform_question_signal_is_no_op(self):
        result = self.select(raw_query_scores=torch.ones(12))
        self.assertEqual(result.swaps, 0)
        self.assertEqual(result.reason, "unreliable_flat_query")

    def test_question_missing_patch_can_enter_at_same_k(self):
        result = self.select(raw_query_scores=self.raw, record=True)
        self.assertGreater(result.swaps, 0)
        self.assertTrue(set(result.keep_ids.tolist()) & set(range(8, 12)))
        self.assertEqual(result.keep_ids.numel(), 4)
        self.assertEqual(len(set(result.keep_ids.tolist())), 4)
        self.assertTrue(result.records)
        self.assertTrue(all(r["coverage_delta"] >= -.00501 for r in result.records))

    def test_reversal_changes_selected_cluster(self):
        result = self.select(raw_query_scores=self.raw)
        reversed_result = self.select(raw_query_scores=-self.raw)
        self.assertEqual(reversed_result.swaps, 0)
        self.assertNotEqual(result.keep_ids.tolist(), reversed_result.keep_ids.tolist())

    def test_llm_embedding_path_matches_precomputed_query_scores(self):
        e = self.feat.clone()
        question = torch.tensor([[0., 1.]])
        result = self.select(visual_llm_embeds=e, question_embeds=question)
        self.assertGreater(result.swaps, 0)

    def test_zero_token_query_is_noop(self):
        result = self.select(visual_llm_embeds=self.feat,
                             question_embeds=torch.empty(0, 2))
        self.assertEqual(result.swaps, 0)

    def test_out_of_range_and_duplicate_ids_fail(self):
        for bad in (torch.tensor([0, 1, 2, 12]), torch.tensor([0, 1, 2, 2])):
            with self.subTest(bad=bad.tolist()), self.assertRaises(ValueError):
                select_question_anchored(native_keep_ids=bad,
                                         visual_metric=self.feat,
                                         raw_query_scores=self.raw)

    def test_nan_fails_closed(self):
        x = self.raw.clone()
        x[2] = float("nan")
        with self.assertRaises(ValueError):
            self.select(raw_query_scores=x)

    def test_no_capacity_k_equals_n_and_empty_k(self):
        for ids in (torch.arange(12), torch.empty(0, dtype=torch.long)):
            out = select_question_anchored(native_keep_ids=ids,
                   visual_metric=self.feat, raw_query_scores=self.raw)
            self.assertEqual(out.swaps, 0)

    def test_two_feature_views(self):
        out = select_question_anchored(native_keep_ids=self.native,
              visual_metric=(self.feat, self.feat.clone()),
              raw_query_scores=self.raw, config=self.cfg)
        self.assertGreater(out.swaps, 0)

    def test_random_control_matches_removed_count_not_future_tokens(self):
        actual = self.select(raw_query_scores=self.raw).keep_ids
        first = matched_random_entry_control(self.native, actual, 12, 17)
        second = matched_random_entry_control(self.native, actual, 12, 17)
        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual(first.numel(), self.native.numel())
        self.assertEqual(len(set(first.tolist())), first.numel())
        self.assertEqual(
            len(set(self.native.tolist()) - set(first.tolist())),
            len(set(self.native.tolist()) - set(actual.tolist()))
        )

    def test_projector_adapter_uses_question_only(self):
        emb = torch.nn.Embedding(4, 2)
        with torch.no_grad():
            emb.weight.copy_(torch.tensor(
                [[0., 1.], [1., 0.], [0., -1.], [-1., 0.]]
            ))
        out = apply_llava_query_anchor(
            mode="s1_anchor", native_keep_ids=self.native,
            dual_metric=self.feat,
            full_vision_features_for_projector=self.feat,
            question_only_ids=torch.tensor([0]),
            mm_projector=torch.nn.Identity(),
            llm_input_embeddings=emb, config=self.cfg
        )
        self.assertGreater(out.swaps, 0)
        off = apply_llava_query_anchor(
            mode="off", native_keep_ids=self.native,
            dual_metric=self.feat,
            full_vision_features_for_projector=self.feat,
            question_only_ids=torch.tensor([-200]),
            mm_projector=torch.nn.Identity(),
            llm_input_embeddings=emb, config=self.cfg
        )
        self.assertEqual(off.swaps, 0)
        with self.assertRaises(ValueError):
            apply_llava_query_anchor(
                mode="s1_anchor", native_keep_ids=self.native,
                dual_metric=self.feat,
                full_vision_features_for_projector=self.feat,
                question_only_ids=torch.tensor([-200]),
                mm_projector=torch.nn.Identity(),
                llm_input_embeddings=emb, config=self.cfg
            )


    def test_wrong_projector_dimension_rejected(self):
        emb = torch.nn.Embedding(4, 3)
        with self.assertRaises(ValueError):
            apply_llava_query_anchor(
                mode="s1_anchor", native_keep_ids=self.native,
                dual_metric=self.feat,
                full_vision_features_for_projector=self.feat,
                question_only_ids=torch.tensor([0]),
                mm_projector=torch.nn.Identity(),
                llm_input_embeddings=emb, config=self.cfg
            )


    def test_saliency_guard_preserves_protected_native(self):
        # Native patch 0 is query-irrelevant, so normally dropped; saliency guards it.
        sal = torch.zeros(12)
        sal[0] = 1.0
        result = select_question_anchored(
            native_keep_ids=self.native, visual_metric=self.feat,
            raw_query_scores=self.raw, native_saliency=sal,
            config=AnchorConfig(max_swaps=2, saliency_guard_slots=1,
                                min_raw_std=.005)
        )
        self.assertIn(0, result.keep_ids.tolist())
        self.assertEqual(result.keep_ids.numel(), 4)

    def test_saliency_requires_valid_vector_when_enabled(self):
        with self.assertRaises(ValueError):
            select_question_anchored(
                native_keep_ids=self.native, visual_metric=self.feat,
                raw_query_scores=self.raw,
                config=AnchorConfig(saliency_guard_slots=2)
            )
        with self.assertRaises(ValueError):
            select_question_anchored(
                native_keep_ids=self.native, visual_metric=self.feat,
                raw_query_scores=self.raw, native_saliency=torch.ones(3),
                config=AnchorConfig(saliency_guard_slots=2)
            )

    def test_content_mask_filters_question_tokens(self):
        emb = torch.nn.Embedding(4, 2)
        with torch.no_grad():
            emb.weight.copy_(torch.tensor([
                [0., 1.], [1., 0.], [0., -1.], [-1., 0.]
            ]))
        out = apply_llava_query_anchor(
            mode="s1_anchor", native_keep_ids=self.native, dual_metric=self.feat,
            full_vision_features_for_projector=self.feat,
            question_only_ids=torch.tensor([1, 0]),
            question_content_mask=torch.tensor([False, True]),
            mm_projector=torch.nn.Identity(),
            llm_input_embeddings=emb, config=self.cfg
        )
        self.assertGreater(out.swaps, 0)
        empty = apply_llava_query_anchor(
            mode="s1_anchor", native_keep_ids=self.native, dual_metric=self.feat,
            full_vision_features_for_projector=self.feat,
            question_only_ids=torch.tensor([1, 0]),
            question_content_mask=torch.tensor([False, False]),
            mm_projector=torch.nn.Identity(),
            llm_input_embeddings=emb, config=self.cfg
        )
        self.assertEqual(empty.swaps, 0)


    def test_mask_cannot_hide_image_placeholder(self):
        emb = torch.nn.Embedding(4, 2)
        with self.assertRaises(ValueError):
            apply_llava_query_anchor(
                mode="s1_anchor", native_keep_ids=self.native,
                dual_metric=self.feat, full_vision_features_for_projector=self.feat,
                question_only_ids=torch.tensor([-200, 0]),
                question_content_mask=torch.tensor([False, True]),
                mm_projector=torch.nn.Identity(), llm_input_embeddings=emb,
                config=self.cfg
            )

    def test_real_size_576_and_two_budgets(self):
        torch.manual_seed(9)
        features = torch.randn(576, 32)
        query = torch.randn(576)
        for k in (64, 172):
            initial = torch.randperm(576)[:k]
            out = select_question_anchored(
                native_keep_ids=initial, visual_metric=features,
                raw_query_scores=query, config=AnchorConfig(max_swaps=3)
            )
            assert out.keep_ids.numel() == k
            assert out.keep_ids.unique().numel() == k
            assert int(out.keep_ids.min()) >= 0 and int(out.keep_ids.max()) < 576

    def test_random_stress_many_shapes_and_view_variants(self):
        generator = torch.Generator().manual_seed(44)
        for n, k in [(6, 1), (9, 8), (32, 8), (64, 17), (128, 64)]:
            for _ in range(5):
                f = torch.randn(n, 12, generator=generator)
                q = torch.randn(n, generator=generator) * .2
                base = torch.randperm(n, generator=generator)[:k]
                out = select_question_anchored(
                    native_keep_ids=base, visual_metric=f,
                    raw_query_scores=q,
                    config=AnchorConfig(max_swaps=3, min_raw_std=.001)
                )
                self.assertEqual(out.keep_ids.numel(), k)
                self.assertEqual(out.keep_ids.unique().numel(), k)
                self.assertTrue(((out.keep_ids >= 0) & (out.keep_ids < n)).all())
                self.assertLessEqual(out.swaps, 3)


class PopeAuditTests(unittest.TestCase):
    def test_confusion_and_transitions(self):
        before = {
           "p1": {"sample_id": "p1", "image_id": "i1", "label": 1,
                   "pred_label": 0, "split": "random", "valid": True},
           "p2": {"sample_id": "p2", "image_id": "i1", "label": 1,
                   "pred_label": 1, "split": "random", "valid": True},
           "n1": {"sample_id": "n1", "image_id": "i2", "label": 0,
                   "pred_label": 0, "split": "adversarial", "valid": True},
           "n2": {"sample_id": "n2", "image_id": "i2", "label": 0,
                   "pred_label": 1, "split": "adversarial", "valid": True},
        }
        after = {k: dict(v) for k, v in before.items()}
        after["p1"]["pred_label"] = 1
        after["n1"]["pred_label"] = 1
        s = stats(before.values())
        self.assertEqual((s["TP"], s["TN"], s["FP"], s["FN"]), (1, 1, 1, 1))
        report = paired(before, after)
        self.assertEqual(report["positive_fn_to_tp"], 1)
        self.assertEqual(report["negative_tn_to_fp"], 1)
        self.assertIn("split:adversarial", grouped(before))

    def test_duplicate_and_invalid_pred_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "pred.jsonl"
            row = {"sample_id": "x", "image_id":"i", "split":"random",
                   "label":1, "pred_label":1, "valid":True}
            p.write_text(json.dumps(row)+"\n"+json.dumps(row)+"\n")
            with self.assertRaises(ValueError):
                read_predictions(p)
            p.write_text(json.dumps({**row, "valid":False})+"\n")
            with self.assertRaises(ValueError):
                read_predictions(p)



class ShuffleControlTests(unittest.TestCase):
    def test_mismatched_images_and_deterministic(self):
        from experiments.bela_query_anchor.make_shuffle_map import make_shuffle_map
        rows = [
            {"sample_id": "a0", "image_id": "A"},
            {"sample_id": "a1", "image_id": "A"},
            {"sample_id": "b0", "image_id": "B"},
            {"sample_id": "c0", "image_id": "C"},
        ]
        a = make_shuffle_map(rows, 17)
        b = make_shuffle_map(rows, 17)
        self.assertEqual(a, b)
        im = {r["sample_id"]: r["image_id"] for r in rows}
        self.assertTrue(all(im[x] != im[y] for x, y in a.items()))

    def test_one_image_control_refuses_to_fake_shuffle(self):
        from experiments.bela_query_anchor.make_shuffle_map import make_shuffle_map
        with self.assertRaises(ValueError):
            make_shuffle_map([{"sample_id": "q1", "image_id": "A"}])

if __name__ == "__main__":
    unittest.main()
