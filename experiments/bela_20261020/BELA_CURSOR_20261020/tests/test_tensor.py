import unittest
from bela.tensor_adapter import compose_sequence_keep,gather_prefill_tensor
class TensorTests(unittest.TestCase):
    def test_noncontiguous_visual(self):
        self.assertEqual(compose_sequence_keep(8,[1,3,5,6],[2,0]),[0,1,2,4,5,7])
    def test_reject_duplicate(self):
        with self.assertRaises(ValueError):compose_sequence_keep(8,[1,1],[0])
    def test_reject_slot(self):
        with self.assertRaises(ValueError):compose_sequence_keep(8,[1,3],[2])
    def test_optional_torch(self):
        try: import torch
        except ImportError: self.skipTest("torch not installed")
        h=torch.arange(16,dtype=torch.float32).reshape(1,8,2)
        result,indices=gather_prefill_tensor(h,[1,3,5,6],[2,0])
        self.assertEqual(result.shape,(1,6,2))
        self.assertEqual(result[0,:,0].tolist(),[0,2,4,8,10,14])
if __name__=="__main__":unittest.main()
