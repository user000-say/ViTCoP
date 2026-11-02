import unittest
from pathlib import Path
import tempfile
import json
from tools.v100_preflight import parse_ids,model_info

class PreflightTests(unittest.TestCase):
    def test_gpu_id_validation(self):
        self.assertEqual(parse_ids("0,2,4,7"),[0,2,4,7])
        for bad in ("0,1,2","0,0,1,2","0,1,2,three","0;rm -rf /,1,2,3"):
            with self.assertRaises(ValueError):
                parse_ids(bad)
    def test_model_layer_config(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/"config.json"
            f.write_text(json.dumps({"model_type":"llava","num_hidden_layers":40,
                "image_grid_pinpoints":[[672,672]]}))
            x=model_info(str(f),40)
            self.assertEqual(x["layers"],40)
            with self.assertRaises(ValueError):
                model_info(str(f),32)
if __name__=="__main__":unittest.main()
