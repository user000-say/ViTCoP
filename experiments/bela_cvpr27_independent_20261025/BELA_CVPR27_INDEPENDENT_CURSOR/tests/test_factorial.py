import unittest
from tools.factorial import analyze,validate,measure,effects

class FactorTests(unittest.TestCase):
 def make_arms(self):
  correct={"BB":[1,0,1,0,0,0],"QB":[1,1,1,0,0,0],
           "BS":[1,0,1,1,0,0],"QS":[1,1,1,1,0,0]}
  out={}
  for arm,v in correct.items():
   out[arm]={}
   for j,b in enumerate(v):
    row=dict(sample_id=f"q{j}",image_id=f"img{j//2}",correct=bool(b),
             gold_yes_no="yes" if j%2==0 else "no",
             pred_yes_no=("yes" if j%2==0 else "no") if b else
                         ("no" if j%2==0 else "yes"),
             subset=("random" if j<2 else "popular" if j<4 else "adversarial"))
    out[arm][f"q{j}"]=row
  return out
 def test_valid(self):
  a=self.make_arms()
  self.assertEqual(len(validate(a,"acc")),6)
 def test_factorial_no_interaction(self):
  a=self.make_arms()
  out=analyze(a,"acc",boot=50,seed=44)
  self.assertAlmostEqual(out["point_effects_pp"]["interaction"],0)
  self.assertAlmostEqual(out["point_effects_pp"]["stage1_given_B_stage2"],100/6)
  self.assertEqual(out["n_image_clusters"],3)
 def test_pope(self):
  a=self.make_arms()
  o=analyze(a,"pope_f1",boot=30,seed=44)
  self.assertEqual(o["mode"],"pope_f1")
  self.assertTrue(all(0<=x<=100 for x in o["arm_scores_percent"].values()))
 def test_unpaired(self):
  a=self.make_arms();a["QB"].pop("q5")
  with self.assertRaises(ValueError):analyze(a,"acc",boot=10)
 def test_mismatched_image(self):
  a=self.make_arms();a["QB"]["q2"]["image_id"]="other"
  with self.assertRaises(ValueError):analyze(a,"acc",boot=10)
 def test_mismatched_gold(self):
  a=self.make_arms();a["QS"]["q1"]["gold_yes_no"]="yes"
  with self.assertRaises(ValueError):analyze(a,"pope_f1",boot=10)
 def test_no_bootstrap(self):
  with self.assertRaises(ValueError):analyze(self.make_arms(),"acc",boot=0)
 def test_bad_mode(self):
  with self.assertRaises(ValueError):analyze(self.make_arms(),"unknown",boot=10)
