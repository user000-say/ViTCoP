import copy
import unittest
import numpy as np
from coverage_report import summarize
from elva.low_budget_cover import diagnostics,numpy_keep


def fixture_rows():
    rows=[]
    rng=np.random.default_rng(8)
    for image in ('FIXTURE_A','FIXTURE_B'):
        x=rng.normal(size=(576,4)).astype(np.float32);s=rng.random(576).astype(np.float32)
        for b in (0,6,12,18):
            d=diagnostics(x,s,numpy_keep(x,s,60,b),60,b)
            d.update(schema='elva_r3_cover_audit',fixture=True,image_key=image,config_hash=f'FIXTURE_{b}',
                     source_hash='FIXTURE_SOURCE',guide='pen',beta=None,work=1024)
            rows.append(d)
    return rows


class ReportTests(unittest.TestCase):
    def test_paired_summaries_and_fixture_guard(self):
        rows=fixture_rows()
        with self.assertRaises(ValueError):summarize(rows)
        result=summarize(rows,allow_fixture=True)
        self.assertEqual([r['slots'] for r in result],[0,6,12,18])
        self.assertTrue(all(r['images']==2 for r in result))
        self.assertEqual(result[0]['actual_changed'],0)
        self.assertAlmostEqual(result[0]['saliency_mass_loss_vs_topk'],0)

    def test_mismatched_panel_source_or_guide_and_duplicate_images_rejected(self):
        rows=fixture_rows()
        for altered in (rows[:-1],rows+rows[:1]):
            with self.assertRaises(ValueError):summarize(altered,allow_fixture=True)
        for key,value in (('source_hash','OTHER_FIXTURE'),('guide','dual'),('work',2048),('fixture',False)):
            altered=copy.deepcopy(rows);altered[-1][key]=value
            with self.assertRaises(ValueError):summarize(altered,allow_fixture=True)

    def test_invalid_actual_change_and_nonfinite_values_rejected(self):
        rows=fixture_rows()
        for key,value in (('actual_changed',99),('kept_mass',float('nan')),('protected_count',1)):
            altered=copy.deepcopy(rows);altered[0][key]=value
            with self.assertRaises(ValueError):summarize(altered,allow_fixture=True)


if __name__=='__main__':unittest.main()
