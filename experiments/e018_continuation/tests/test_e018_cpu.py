"""Runnable without torch: budgets, loss targets, grouping, fitting and patch audit."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from e018_tools import contract
from train_e018_controller import train, predict
C = contract()


class Contracts(unittest.TestCase):
    def test_cost_matching_post_block_semantics(self):
        for final, mid, expected in ((32, 102, 2284), (64, 118, 3020), (96, 134, 3756)):
            self.assertEqual(C.token_work(172, ((8, final),)), expected)
            self.assertEqual(C.token_work(172, ((4, mid), (12, final))), expected)
        self.assertEqual(C.token_work(146, ((8, 32),)), 2050)
        self.assertEqual(C.token_work(172, ((2, 133), (8, 32))), 2050)
        self.assertEqual(C.token_work(128, ((6, 46),)), 2046)

    def test_invalid_schedule_rejected(self):
        for events in (((4, 100), (3, 30)), ((4, 100), (12, 110)), ((31, 32),), ((4, 0),)):
            with self.assertRaises(ValueError):
                C.E018Config(events=events).validate(32, 172)

    def test_protocol_excludes_lambda_not_cache_or_source(self):
        s1 = {"name": "FIXED_COVER", "exchange_budget": 32}
        a = C.E018Config(protocol_id="frozen-v1")
        b = C.E018Config(protocol_id="frozen-v1", lam=7)
        c = C.E018Config(protocol_id="frozen-v1", cache_policy="scrub_all")
        self.assertEqual(a.fingerprint(172, 32, s1), b.fingerprint(172, 32, s1))
        self.assertNotEqual(a.fingerprint(172, 32, s1), c.fingerprint(172, 32, s1))
        self.assertNotEqual(a.fingerprint(172, 32, s1), a.fingerprint(172, 32, {**s1, "exchange_budget": 16}))

    def test_image_grouped_finite_sample_quantile(self):
        self.assertEqual(C.group_split("same-image"), C.group_split("same-image"))
        self.assertEqual(C.conformal_upper(list(range(19)), 0.1), 17)
        with self.assertRaises(ValueError):
            C.conformal_upper([1], 0.01)

    def test_real_fit_and_feature_ablation(self):
        rng = np.random.default_rng(15)
        x = rng.normal(size=(512, len(C.FEATURE_NAMES)))
        # Independent synthetic test of training, not experimental/model evidence.
        y = 0.1 * np.tanh(x[:, 2]) - 0.06 * x[:, 7]
        mask = np.ones(x.shape[1])
        w = train(x[:400], y[:400], mask, epochs=180, seed=20)
        error = np.mean((predict(x[400:], w, mask) - y[400:]) ** 2)
        baseline = np.mean((y[400:] - y[:400].mean()) ** 2)
        self.assertLess(error, baseline * 0.2)
        mask[2] = 0
        a, b = x[:10].copy(), x[:10].copy()
        b[:, 2] += 100
        np.testing.assert_allclose(predict(a, w, mask), predict(b, w, mask))

    def test_all_legacy_ast_preserved(self):
        for rel in ("llava/model/language_model/vitcop_modeling_llama.py",
                    "llava/model/multimodal_encoder/clip_encoder.py"):
            original = (ROOT / "reference_E014" / rel).read_text()
            modified = (ROOT / rel).read_text()
            kept, inside = [], False
            for line in modified.splitlines(True):
                if "# E018-BEGIN:" in line:
                    inside = True
                if not inside:
                    kept.append(line)
                if "# E018-END" in line:
                    inside = False
            self.assertFalse(inside)
            self.assertEqual(ast.dump(ast.parse(original)), ast.dump(ast.parse("".join(kept))))

    def test_all_python_syntax(self):
        for path in ROOT.rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path))

    def test_complete_training_cli_and_calibration_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(10)
            rows = []
            for i in range(300):
                gid = "synthetic-training-test-image-%d" % i
                x = rng.normal(size=len(C.FEATURE_NAMES))
                rows.append(dict(schema=C.SCHEMA, fingerprint="synthetic-test-only",
                                 image_id=gid, question_id=1, event=4, split=C.group_split(gid),
                                 feature_names=list(C.FEATURE_NAMES), features=x.tolist(),
                                 utility=float(.08*np.tanh(x[0]))))
            data, output = root / "labels.jsonl", root / "controller.json"
            data.write_text("\n".join(json.dumps(x) for x in rows))
            result = subprocess.run([sys.executable, str(ROOT / "scripts/train_e018_controller.py"),
                                     "--labels", str(data), "--output", str(output), "--epochs", "15"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            model = json.loads(output.read_text())
            self.assertTrue(model["calibrated"])
            self.assertGreater(model["report"]["calibration"]["images"], 9)
            self.assertEqual(len(model["w1"]), len(C.FEATURE_NAMES))
            repeated = subprocess.run([sys.executable, str(ROOT / "scripts/train_e018_controller.py"),
                                       "--labels", str(data), str(data), "--output", str(root / "bad.json")],
                                      capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)

    def test_installer_check_apply_rollback_and_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            manifest = json.loads((ROOT / "BASE_SOURCE_MANIFEST.json").read_text())
            for rel in manifest:
                path = repo / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((ROOT / "reference_E014" / rel).read_bytes())
            for rel in ("llava/model/builder.py", "llava/model/e010_runtime_support.py"):
                (repo / rel).write_text("# dependency placeholder for installer-only test\n")
            cmd = [sys.executable, str(ROOT / "scripts/install_e018.py"), "--repo", str(repo)]
            dry = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(dry.returncode, 0, dry.stderr)
            self.assertFalse((repo / "e018_backups").exists())
            applied = subprocess.run(cmd + ["--apply"], capture_output=True, text=True)
            self.assertEqual(applied.returncode, 0, applied.stderr)
            backup = next((repo / "e018_backups").iterdir())
            rollback = subprocess.run(cmd + ["--rollback", str(backup), "--apply"], capture_output=True, text=True)
            self.assertEqual(rollback.returncode, 0, rollback.stderr)
            for rel in manifest:
                self.assertEqual((repo / rel).read_bytes(), (ROOT / "reference_E014" / rel).read_bytes())
            conflict = repo / "llava/model/language_model/vitcop_modeling_llama.py"
            conflict.write_text(conflict.read_text() + "\n# newer server source\n")
            before = conflict.read_bytes()
            failed = subprocess.run(cmd + ["--apply"], capture_output=True, text=True)
            self.assertEqual(failed.returncode, 2)
            self.assertEqual(conflict.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
