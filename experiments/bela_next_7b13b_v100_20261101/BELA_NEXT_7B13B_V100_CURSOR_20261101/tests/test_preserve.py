import unittest
import torch
from bela_next_budget.preserve import plan_physical_keep

class PreserveTests(unittest.TestCase):
    def test_preserve_text_and_newlines(self):
        # patch positions 0,1,2,4,5,7,8; 3,6 are newlines; 9-12 are text.
        pos=torch.tensor([0,1,2,4,5,7,8],dtype=torch.long)
        selected=torch.tensor([0,3,5],dtype=torch.long)
        crop_ids=torch.tensor([0,0,0,1,1,1,1],dtype=torch.long)
        p=plan_physical_keep(13,pos,selected,
            original_patch_ids=torch.arange(7),crop_ids=crop_ids)
        self.assertEqual(p["kept_patch_count"],3)
        self.assertEqual(p["kept_nonpatch_count"],6)
        self.assertEqual(p["full_keep_indices"].tolist(),[0,3,4,6,7,9,10,11,12])
        self.assertEqual(p["retained_crop_ids"].tolist(),[0,1,1])
    def test_bad_indices_fail(self):
        with self.assertRaises(ValueError):
            plan_physical_keep(10,torch.tensor([1,1]),torch.tensor([0]))
        with self.assertRaises(ValueError):
            plan_physical_keep(10,torch.tensor([1,2]),torch.tensor([0,2]))
if __name__=="__main__":unittest.main()
