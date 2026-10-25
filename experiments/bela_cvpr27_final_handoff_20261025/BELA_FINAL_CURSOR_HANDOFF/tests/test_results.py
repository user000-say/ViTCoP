import unittest
from tools.results import compare, pope_report, pope_paired_bootstrap

class ResultsTests(unittest.TestCase):
 def setup(self):
  a={
   "q1":dict(sample_id="q1",image_id="imgA",correct=True,subset="random",gold_yes_no="yes",pred_yes_no="yes"),
   "q2":dict(sample_id="q2",image_id="imgA",correct=False,subset="random",gold_yes_no="no",pred_yes_no="yes"),
   "q3":dict(sample_id="q3",image_id="imgB",correct=False,subset="popular",gold_yes_no="yes",pred_yes_no="no"),
   "q4":dict(sample_id="q4",image_id="imgC",correct=True,subset="adversarial",gold_yes_no="no",pred_yes_no="no"),
  }
  b={k:dict(v) for k,v in a.items()}
  b["q2"]["correct"]=True;b["q2"]["pred_yes_no"]="no"
  return a,b
 def test_paired(self):
  a,b=self.setup();r=compare(a,b,replicates=90,seed=17)
  self.assertEqual(r["paired_counts"]["candidate_only"],1)
  self.assertEqual(r["paired_counts"]["baseline_only"],0)
  self.assertEqual(r["delta_pp"],25)
  self.assertEqual(r,compare(a,b,replicates=90,seed=17))
 def test_pope_confusion(self):
  a,b=self.setup(); p=pope_report(a)["overall_micro"]
  self.assertEqual([p[k] for k in ("TP","FP","TN","FN")],[1,1,1,1])
  self.assertAlmostEqual(p["f1"],0.5)
  self.assertAlmostEqual(pope_report(b)["overall_micro"]["f1"],2/3)
 def test_pope_f1_bootstrap(self):
  a,b=self.setup()
  r=pope_paired_bootstrap(a,b,replicates=100,seed=1)
  self.assertAlmostEqual(r["delta_f1_pp"],(2/3-.5)*100)
  self.assertEqual(r["n_clusters"],3)
 def test_bad_pair_ids(self):
  a,b=self.setup();b.pop("q2")
  with self.assertRaises(ValueError):compare(a,b)
 def test_bad_gold_labels(self):
  a,b=self.setup();b["q1"]["gold_yes_no"]="no"
  with self.assertRaises(ValueError):pope_paired_bootstrap(a,b)
 def test_bad_subsets(self):
  a,b=self.setup();b["q1"]["subset"]="popular"
  with self.assertRaises(ValueError):compare(a,b)
 def test_bad_image(self):
  a,b=self.setup();b["q1"]["image_id"]="other"
  with self.assertRaises(ValueError):compare(a,b)
 def test_bad_bootstrap(self):
  a,b=self.setup()
  with self.assertRaises(ValueError):compare(a,b,replicates=0)
  with self.assertRaises(ValueError):pope_paired_bootstrap(a,b,replicates=0)
if __name__=="__main__":unittest.main()
