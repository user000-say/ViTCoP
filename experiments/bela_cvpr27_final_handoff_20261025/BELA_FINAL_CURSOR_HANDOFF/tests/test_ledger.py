import unittest
from tools.ledger import validate_file, validate_row,FIELDS
class LedgerTests(unittest.TestCase):
 def test_reported(self):
  data=validate_file("templates/RESULTS_MASTER.csv")
  self.assertEqual(len(data),7)
  self.assertEqual(data[0]["status"],"REPORTED")
 def test_forged_finished(self):
  d={k:"" for k in FIELDS}
  d.update({"arm":"A2","status":"VERIFIED_GPU","gqa":"57.62",
            "pope_f1":"85.44","W_expected":"2048","W_actual":"2048"})
  with self.assertRaises(ValueError):validate_row(d)
 def test_wrong_budget(self):
  d={k:"x" for k in FIELDS}
  d.update({"arm":"trial","status":"VERIFIED_GPU",
   "gqa":"57","pope_f1":"85","mme_total":"1775","mmbench":"63",
   "sciqa":"69","textvqa":"56","ttft_ms":"100","fixed32_ms":"1100",
   "latency_scope":"FULL_IMAGE_TO_FIRST_TOKEN",
   "W_expected":"2048","W_actual":"2088"})
  with self.assertRaises(ValueError):validate_row(d)
 def test_nan_score(self):
  d={k:"" for k in FIELDS};d["status"]="PARTIAL";d["gqa"]="nan"
  with self.assertRaises(ValueError):validate_row(d)
