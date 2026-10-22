import unittest
from bela_next.budget import PROFILES
from scripts.audit_trace import check

def record():
    s=PROFILES["normal"].schedule()
    entry=list(range(s.k0))
    return {"sample_id":"synthetic","schedule":s.as_dict(),"entry_patch_ids":entry,
            "events":[
                {"post_block":s.cut1,"selected_patch_ids":entry[:s.kmid]},
                {"post_block":s.cut2,"selected_patch_ids":entry[:s.kfinal]},
            ],
            "observed_visual_counts_per_block":list(s.counts())}

class AuditTest(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(check(record())["observed_W"],2048)

    def test_fake_counts(self):
        x=record()
        x["observed_visual_counts_per_block"]=[32]*32
        with self.assertRaises(ValueError):check(x)

    def test_token_resurrection(self):
        x=record()
        x["events"][1]["selected_patch_ids"][0]=575
        with self.assertRaises(ValueError):check(x)

    def test_no_actual_trace(self):
        x=record()
        del x["observed_visual_counts_per_block"]
        with self.assertRaises(ValueError):check(x)

if __name__=="__main__":unittest.main()
