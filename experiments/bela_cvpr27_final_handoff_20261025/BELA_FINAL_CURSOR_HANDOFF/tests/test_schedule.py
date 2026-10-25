import unittest
from tools.schedule import Schedule,load,validate_trace

class ScheduleTests(unittest.TestCase):
 def test_manifest(self):
  rows=load("config/experiments.json")
  self.assertEqual(len(rows),30)
  self.assertEqual(len({s.name for _,s in rows}),30)
  self.assertTrue(all(s.W==e["W_target"] for e,s in rows))
 def test_equal_budget_and_q(self):
  d={s.name:s for _,s in load("config/experiments.json")}
  self.assertEqual([d[i].W for i in ["A0","A2","E1_RET32_TO12","E2_RET32_TO16","E3_RET24_TO16","E5_DELAY_128_12_20"]],[2048]*6)
  self.assertEqual([d[i].W for i in ["L0_HIST_60_28_12","L2_LATE_64_17_10","L3_MID_80_24_12"]],[1024]*3)
  self.assertEqual(d["B2"].W,2288)
  self.assertEqual(d["B1"].W,1808)
  self.assertEqual(d["H1_W4096"].W,4096)
  self.assertEqual(d["A0"].Q_visual,203616)
  self.assertEqual(d["A2"].Q_visual,241536)
  self.assertEqual(d["E1_RET32_TO12"].Q_visual,243456)
  self.assertEqual(d["E5_DELAY_128_12_20"].Q_visual,211448)
 def test_bad_lengths(self):
  for stages in [((31,32),),((4,64),(8,32),(19,16)),(),((0,64),(32,12)),((32,True),),((32,2.0),)]:
   with self.assertRaises(ValueError):Schedule("bad",stages).check()
 def test_increasing_tokens(self):
  with self.assertRaises(ValueError):
   Schedule("bad",((4,32),(8,64),(20,10))).check()
 def create_trace(self):
  sc=Schedule("E1",((4,172),(8,120),(8,32),(12,12)))
  return sc,{
    "active_visual_tokens_by_decoder_layer":sc.counts,
    "prune_events":[
      dict(after_layer=4,original_patch_ids_before=list(range(172)),
           original_patch_ids_after=list(range(120)),
           score_function_path="a.b",callsite="x.py:10"),
      dict(after_layer=12,original_patch_ids_before=list(range(120)),
           original_patch_ids_after=list(range(32)),
           score_function_path="a.b",callsite="x.py:20"),
      dict(after_layer=20,original_patch_ids_before=list(range(32)),
           original_patch_ids_after=list(range(12)),
           score_function_path="a.b",callsite="x.py:30")
    ]
   }
 def test_trace_valid(self):
  s,r=self.create_trace()
  self.assertTrue(validate_trace(r,s))
 def test_trace_bad_count(self):
  s,r=self.create_trace();r["active_visual_tokens_by_decoder_layer"][11]=22
  with self.assertRaises(ValueError):validate_trace(r,s)
 def test_trace_missing_event(self):
  s,r=self.create_trace();r["prune_events"].pop()
  with self.assertRaises(ValueError):validate_trace(r,s)
 def test_trace_missing_provenance(self):
  s,r=self.create_trace();r["prune_events"][2]["callsite"]=""
  with self.assertRaises(ValueError):validate_trace(r,s)
 def test_trace_reintroduced_patch(self):
  s,r=self.create_trace();r["prune_events"][2]["original_patch_ids_after"][0]=171
  with self.assertRaises(ValueError):validate_trace(r,s)
 def test_trace_duplicate_patch(self):
  s,r=self.create_trace();r["prune_events"][0]["original_patch_ids_after"][0]=1
  with self.assertRaises(ValueError):validate_trace(r,s)
 def test_trace_wrong_layer(self):
  s,r=self.create_trace();r["prune_events"][1]["after_layer"]=13
  with self.assertRaises(ValueError):validate_trace(r,s)

if __name__=="__main__":unittest.main()
