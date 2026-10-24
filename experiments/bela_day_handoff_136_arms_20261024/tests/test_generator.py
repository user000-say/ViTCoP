import unittest
from scripts.generate import metrics,solutions,make
class TestMatrix(unittest.TestCase):
 def test_budget(self):
  for r in make():
   w,q=metrics(r["k0"],r["kmid"],r["kdeep"])
   self.assertEqual((w,q),(r["budget"],r["q_proxy"]))
 def test_sizes(self):
  r=make();self.assertGreaterEqual(len(r),100)
  self.assertEqual(len(r),len(set(x["arm"] for x in r)))
 def test_anchors(self):
  r={x["arm"]:x for x in make()}
  self.assertEqual(r["A2"]["budget"],2048)
  self.assertEqual(r["B1"]["budget"],1808)
  self.assertEqual(r["B2"]["budget"],2288)
 def test_invalid(self):
  with self.assertRaises(ValueError):metrics(80,100,10)
  with self.assertRaises(ValueError):metrics(80,40,0)
  with self.assertRaises(ValueError):metrics(80,40,10,(4,8,19))
  self.assertTrue(solutions(1024,96))
