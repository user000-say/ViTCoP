import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"visual_audit"))
from make_fixture import create
from stage1_visual_audit import (validate_record, load_record, keep_mask, pair_metrics,
                                audit, render_sweep, matched_random_keep)


class VisualAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        create(self.root)
        self.path = self.root/"record_FIXTURE_ONLY.json"
        self.raw = json.loads(self.path.read_text())
        self.raw["image_path"] = str(self.root/"fixture_crop.png")

    def tearDown(self):
        self.temp.cleanup()

    def validate(self, value=None):
        return validate_record(value or self.raw, allow_fixture=True)

    def test_original_ids_are_not_row_indices(self):
        record = self.validate()
        chosen = record["methods"]["pen"]["keep_ids"]
        expected = [i for i, pid in enumerate(record["patch_ids"]) if pid in chosen]
        np.testing.assert_array_equal(np.flatnonzero(keep_mask(record, "pen")), expected)
        self.assertGreater(max(chosen), len(record["patch_ids"]))

    def test_all_and_disjoint_selected_sets(self):
        raw = copy.deepcopy(self.raw)
        raw["methods"] = [dict(name="a", config_hash="a", keep_ids=raw["patch_ids"][:4]),
                          dict(name="b", config_hash="b", keep_ids=raw["patch_ids"][-4:])]
        record = self.validate(raw)
        result = pair_metrics(record, "a", "b")
        self.assertEqual((result["common"], result["union"], result["jaccard"]), (0, 8, 0))
        self.assertEqual(pair_metrics(record, "a", "a")["jaccard"], 1)

    def test_random_supplement_matches_changed_count_and_common_support(self):
        ids = np.array([70, 4, 92, 33, 11, 55])
        baseline, candidate = [70, 4, 92], [4, 11, 55]
        keep, info = matched_random_keep(ids, baseline, candidate, seed=3)
        self.assertEqual(len(keep), len(baseline))
        self.assertEqual(info["changed"], 2)
        self.assertEqual(set(keep)&set(baseline), {4})
        np.testing.assert_array_equal(keep, ids[np.isin(ids, keep)])
        np.testing.assert_array_equal(keep, matched_random_keep(ids, baseline, candidate, seed=3)[0])

    def test_random_supplement_no_change_and_full_keep(self):
        for ids in ([99], [99, 7, 18]):
            keep, info = matched_random_keep(ids, ids, ids, seed=0)
            np.testing.assert_array_equal(keep, ids)
            self.assertEqual(info["changed"], 0)

    def test_random_supplement_cannot_resurrect_ids_or_change_k(self):
        for base, candidate in (([2, 3], [3]), ([2], [100]), ([2, 2], [2, 3])):
            with self.assertRaises(ValueError):
                matched_random_keep([2, 3, 4], base, candidate, seed=1)

    def test_zero_mass_and_uniform_scores_render_without_division(self):
        for score in (0., 1.):
            raw = copy.deepcopy(self.raw)
            for key in ("guide_scores", "pen_scores"):
                raw[key] = [score]*len(raw["patch_ids"])
            p = self.root/f"scores-{score}.json"
            p.write_text(json.dumps(raw))
            summary = audit([p], self.root/f"out-{score}", baseline="pen", candidate="dual050", allow_fixture=True)
            self.assertEqual(summary["shared_raw_vmax"], 1)
            self.assertEqual(summary["score_zero"], score == 0)

    def test_heatmaps_have_actual_scalar_mapped_faces(self):
        from stage1_visual_audit import _boxes
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        fig, ax = plt.subplots()
        faces = _boxes(ax, [[0, 0, 1, 1], [1, 0, 2, 1]], values=[0., 1.], norm=Normalize(0, 1))
        fig.canvas.draw()
        colors = faces.get_facecolors()
        self.assertEqual(colors.shape, (2, 4))
        self.assertFalse(np.allclose(colors[0], colors[1]))
        plt.close(fig)

    def test_image_coordinates_and_cross_crop_overlap_rejected(self):
        for edit in (lambda r: r["boxes_xyxy"][0].__setitem__(0, -1),
                     lambda r: r["boxes_xyxy"].__setitem__(1, r["boxes_xyxy"][0]),
                     lambda r: r["boxes_xyxy"][0].__setitem__(2, 999)):
            raw = copy.deepcopy(self.raw)
            edit(raw)
            with self.assertRaises(ValueError):
                self.validate(raw)

    def test_duplicate_unknown_ids_mismatched_k_and_nan_rejected(self):
        edits = [lambda r: r["patch_ids"].__setitem__(0, r["patch_ids"][1]),
                 lambda r: r["methods"][0]["keep_ids"].__setitem__(0, 99999),
                 lambda r: r["methods"][0]["keep_ids"].pop(),
                 lambda r: r["pen_scores"].__setitem__(0, float("nan"))]
        for edit in edits:
            raw = copy.deepcopy(self.raw)
            edit(raw)
            with self.assertRaises(ValueError):
                self.validate(raw)

    def test_annotated_region_geometry_is_not_attention_coverage(self):
        raw = copy.deepcopy(self.raw)
        raw["methods"] = [dict(name="a", config_hash="a", keep_ids=raw["patch_ids"][:2]),
                          dict(name="b", config_hash="b", keep_ids=raw["patch_ids"][-2:])]
        row = pair_metrics(self.validate(raw), "a", "b")["annotated_region_coverage"][0]
        self.assertEqual(row["baseline_area_fraction"], .5)
        self.assertEqual(row["candidate_area_fraction"], 0.)

    def test_fixture_source_binding_and_repeated_question_guard(self):
        with self.assertRaises(ValueError):
            load_record(self.path)
        with self.assertRaises(ValueError):
            audit([self.path, self.path], self.root/"duplicate", baseline="pen", candidate="dual050", allow_fixture=True)
        raw = copy.deepcopy(self.raw)
        raw["source"]["pen_completed"] = raw["source"]["vision_depth"]
        with self.assertRaises(ValueError):
            self.validate(raw)

    def test_real_layout_shared_scale_and_sweep_gate(self):
        summary = audit([self.path], self.root/"plots", baseline="pen", candidate="dual050", allow_fixture=True)
        self.assertEqual(summary["image_count"], 1)
        self.assertEqual(summary["shared_raw_vmax"], .2)
        matrices = np.asarray(summary["overlap_matrix"])
        np.testing.assert_array_equal(matrices, matrices.T)
        np.testing.assert_array_equal(matrices.diagonal(), 1)
        for p in (self.root/"plots").rglob("*.png"):
            with Image.open(p) as img:
                self.assertGreater(img.width, 100)
                self.assertGreater(img.height, 100)
        ledger = self.root/"sweep_FIXTURE_ONLY.json"
        with self.assertRaises(ValueError):
            render_sweep(ledger, self.root/"sweep")
        render_sweep(ledger, self.root/"sweep", allow_fixture=True)
        raw = json.loads(ledger.read_text())
        raw["rows"][0]["selection_mode"] = "quota"
        ledger.write_text(json.dumps(raw))
        with self.assertRaises(ValueError):
            render_sweep(ledger, self.root/"bad", allow_fixture=True)


if __name__ == "__main__":
    unittest.main()
