import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from schedule_audit import Schedule, read_experiments, enumerate_mid_for_tail


class ScheduleAuditTests(unittest.TestCase):
    def test_all_matrix_arms_are_valid(self):
        rows = read_experiments(ROOT / "research" / "experiment_matrix.csv")
        self.assertEqual(len(rows), 12)
        for schedule, target, _ in rows:
            with self.subTest(arm=schedule.arm):
                schedule.validate(target)

    def test_known_working_points(self):
        for name, sections, target in [
            ("BELA64", ((172,4),(90,8),(32,20)), 2048),
            ("STAR64", ((128,12),(37,8),(18,12)), 2048),
            ("BELA32", ((60,10),(28,10),(12,12)), 1024),
            ("STAR32", ((64,12),(17,8),(10,12)), 1024),
        ]:
            with self.subTest(name=name):
                s = Schedule(name, sections)
                s.validate(target)
                self.assertEqual(s.visual_token_layers, target)
                self.assertEqual(s.layer_average, target/32)

    def test_tail_enumeration_exact_and_monotone(self):
        candidates = enumerate_mid_for_tail(172,4,8,20,2048,list(range(0,33)))
        by_tail = {s.sections[-1][0]:s.sections[1][0] for s in candidates}
        for tail, mid in [(12,140),(16,130),(20,120),(24,110),(32,90)]:
            self.assertEqual(by_tail[tail], mid)
        self.assertNotIn(31,by_tail)

    def test_ill_formed(self):
        failures = [
            (Schedule("wrong_layers", ((172,3),(90,8),(32,20))),2048),
            (Schedule("wrong_w", ((172,4),(91,8),(32,20))),2048),
            (Schedule("increase", ((80,4),(90,8),(32,20))),2048),
            (Schedule("negative", ((172,4),(-5,8),(32,20))),2048),
            (Schedule("too_big", ((577,4),(90,8),(32,20))),2048),
            (Schedule("zero_segment", ((172,4),(90,0),(32,28))),2048),
        ]
        for s, target in failures:
            with self.subTest(arm=s.arm):
                with self.assertRaises(ValueError):
                    s.validate(target)

    def test_zero_final_allowed_as_diagnostic(self):
        s=Schedule("zero_final",((128,12),(64,8),(0,12)))
        s.validate(2048)  # explicit mathematical diagnostic; model code may not support zero K
        self.assertEqual(s.visual_token_layers,2048)

    def test_impossible_tail_excluded(self):
        self.assertEqual(
            enumerate_mid_for_tail(172,4,8,20,2048,[45]),
            []
        )


if __name__ == "__main__":
    unittest.main()
