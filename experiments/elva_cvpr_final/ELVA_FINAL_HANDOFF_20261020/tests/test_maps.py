import unittest
import tempfile
from pathlib import Path
from elva.plot_patch_maps import make_maps

class TestMaps(unittest.TestCase):
    def test_real_ids_contract(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'a.png'
            make_maps({'pen':[0,3],'L9':[2,7]},str(out),grid_h=2,grid_w=4)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size,100)
            with self.assertRaises(ValueError):
                make_maps({'bad':[0,0]},str(out),grid_h=2,grid_w=4)
            with self.assertRaises(ValueError):
                make_maps({'bad':[8]},str(out),grid_h=2,grid_w=4)
            with self.assertRaises(ValueError):
                make_maps({},str(out),grid_h=2,grid_w=4)

if __name__=='__main__':unittest.main()
