import json
from pathlib import Path
import unittest
from bela_next_budget.budget import source_schedule,dynamic_ratio_plan,BudgetSchedule

CFG=json.loads((Path(__file__).resolve().parents[1]/"configs"/"arms.json").read_text())
class BudgetTests(unittest.TestCase):
    def test_all_20_reference_schedules_exact_W_and_nested(self):
        arms=CFG["arm_matrix"]
        self.assertEqual(len(arms),20)
        for row in arms:
            with self.subTest(row=row["arm_id"]):
                s=source_schedule(row)
                self.assertEqual(s.w,s.layers*row["budget_T"])
                self.assertEqual(s.avg,row["budget_T"])
                self.assertTrue(0<s.k2<=s.k1<=s.k0<=2880)
    def test_reference_paper_7b_and_13b_cuts_and_K(self):
        p={(a["model"],a["budget_T"]):a for a in CFG["arm_matrix"] if
           a["arm_id"].endswith("A0_PAPER")}
        self.assertEqual(p["next7",160]["stage_k"],[320,91,46])
        self.assertEqual(p["next13",160]["stage_k"],[320,80,40])
        self.assertEqual(p["next7",320]["cuts_after_layers"],[12,20])
        self.assertEqual(p["next13",320]["cuts_after_layers"],[15,30])
    def test_native_various_image_patches(self):
        for a in CFG["arm_matrix"]:
            if not a["arm_id"].endswith("A0_PAPER"):continue
            orig=source_schedule(a)
            self.assertEqual(dynamic_ratio_plan(orig,2880)["achieved_avg"],a["budget_T"])
            for n in (576,1152,1728,2304,3456):
                with self.subTest(model=a["model"],N=n):
                    out=dynamic_ratio_plan(orig,n,tolerance=18)
                    self.assertTrue(0<out["k2"]<=out["k1"]<=out["k0"]<=n)
                    self.assertLess(abs(out["achieved_avg"]-a["budget_T"]*n/2880),2.1)
    def test_bad_schedule_fails_closed(self):
        with self.assertRaises(ValueError):
            BudgetSchedule("bad",32,(12,20),320,91,47,target_t=160).validate()
        a=source_schedule(CFG["arm_matrix"][0])
        with self.assertRaises(ValueError):
            dynamic_ratio_plan(a,10)
    def test_published_tops_arithmetic_check(self):
        # Appendix table as described; protocol requires rechecking real code
        tops7=12*320+12*160+8*40
        tops13=15*320+15*160+10*40
        self.assertEqual(tops7,6080)
        self.assertEqual(tops13,7600)
        self.assertNotEqual(tops7,32*160)
        self.assertNotEqual(tops13,40*160)
if __name__=="__main__":unittest.main()
