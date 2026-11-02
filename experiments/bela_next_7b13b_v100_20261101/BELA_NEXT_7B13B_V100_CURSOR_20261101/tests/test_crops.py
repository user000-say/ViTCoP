import unittest
from bela_next_budget.crops import allocate_crop_quota,crop_quota_from_ids

class CropTests(unittest.TestCase):
    def test_uniform_320_and_640(self):
        for k in (160,320,640):
            q=allocate_crop_quota([576]*5,k,min_per_crop=1)
            self.assertEqual(q,[k//5]*5)
    def test_global_boost(self):
        a=crop_quota_from_ids([576]*5,320,policy="uniform")
        b=crop_quota_from_ids([576]*5,320,policy="thumbnail_boost",global_weight=1.5)
        self.assertEqual(sum(b),320)
        self.assertGreater(b[0],a[0])
        self.assertTrue(all(1<=n<=576 for n in b))
    def test_crops_variable_and_caps(self):
        q=allocate_crop_quota([576,576,125,100,50],600,min_per_crop=1)
        self.assertEqual(sum(q),600)
        self.assertTrue(all(x<=cap for x,cap in zip(q,[576,576,125,100,50])))
        self.assertEqual(allocate_crop_quota([3,4,0],7),[3,4,0])
    def test_infeasible_rejected(self):
        with self.assertRaises(ValueError):
            allocate_crop_quota([5,5],11)
        with self.assertRaises(ValueError):
            allocate_crop_quota([5,0],0,min_per_crop=1)
        with self.assertRaises(ValueError):
            allocate_crop_quota([10,10],10,weights=[1,-1])
if __name__=="__main__":unittest.main()
