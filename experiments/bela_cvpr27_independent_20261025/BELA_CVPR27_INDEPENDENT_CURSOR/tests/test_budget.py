import unittest
from tools.budget import Schedule,read_schedule_file,validate_trace

class BudgetTests(unittest.TestCase):
 def test_matrix(self):
  m=read_schedule_file("configs/experiment_matrix.json")
  self.assertEqual(len(m),23)
  for sc,entry in m.values():
   self.assertEqual(sc.W,entry["W"])
   self.assertEqual(len(sc.active_counts),32)
 def test_anchors(self):
  m=read_schedule_file("configs/experiment_matrix.json")
  expected={"F1024_BB":1024,"F2048_BB":2048,"E3_CURRENT":2048,
            "E1_DIAGNOSTIC":2048,"A2_CURRENT":2048,
            "W1024_EARLY":1024,"H1_W4096_REFERENCE":4096}
  for k,w in expected.items():self.assertEqual(m[k][0].W,w)
 def test_E3_cuts(self):
  m=read_schedule_file("configs/experiment_matrix.json")
  self.assertEqual(m["E3_CURRENT"][0].cuts_after,[4,12,22])
  self.assertEqual(m["F1024_BB"][0].cuts_after,[12,20])
 def test_decreasing_only(self):
  for x in [((4,172),(8,120),(20,130)),((32,0),),((31,20),),
            ((12,64),(8,17),(12,True))]:
   with self.assertRaises(ValueError):Schedule("broken",x).validate()
 def mock_trace(self):
  sc=Schedule("E3",((4,172),(8,120),(10,24),(10,16)))
  events=[]
  stages=[list(range(172)),list(range(120)),list(range(24)),list(range(16))]
  for cut,before,after in zip(sc.cuts_after,stages[:-1],stages[1:]):
   events.append({"after_decoder_layer":cut,"original_before":before,"original_after":after,
                  "selector_callsite":f"current.py:{cut}"})
  return sc,{"active_visual_by_layer":sc.active_counts,"cuts":events}
 def test_trace_ok(self):
  a,b=self.mock_trace();self.assertTrue(validate_trace(a,b))
 def test_trace_not_nested(self):
  a,b=self.mock_trace();b["cuts"][2]["original_after"][0]=120
  with self.assertRaises(ValueError):validate_trace(a,b)
 def test_trace_missing_function(self):
  a,b=self.mock_trace();b["cuts"][1]["selector_callsite"]=""
  with self.assertRaises(ValueError):validate_trace(a,b)
 def test_wrong_count(self):
  a,b=self.mock_trace();b["active_visual_by_layer"][12]=99
  with self.assertRaises(ValueError):validate_trace(a,b)
