import unittest
from src.budget import ARMS, compatible_deep, Plan
class Tests(unittest.TestCase):
 def test_budgets(self):
  expected={"A0":2048,"A2":2048,"A3":2048,"B1":1808,"B2":2288,
            "W1024_probe":1024,"W4096_probe":4096,"W1024_historical":1024}
  for name,target in expected.items(): self.assertEqual(ARMS[name].W,target)
 def test_quadratic(self):
  self.assertEqual(ARMS["A0"].Q,203616)
  self.assertEqual(ARMS["A2"].Q,241536)
  self.assertEqual(ARMS["A3"].Q,258656)
 def test_solve(self):
  self.assertEqual(compatible_deep(2048,172,120),20)
  self.assertEqual(compatible_deep(2048,172,130),16)
  self.assertIsNone(compatible_deep(2048,172,121))
 def test_boundary(self):
  with self.assertRaises(ValueError): Plan("bad",((4,120),(8,130),(20,20))).validate()
  with self.assertRaises(ValueError): Plan("bad",((4,172),(8,120),(19,20))).validate()
  with self.assertRaises(ValueError): Plan("bad",((4,172),(8,120),(20,0))).validate()
if __name__=="__main__":unittest.main()
