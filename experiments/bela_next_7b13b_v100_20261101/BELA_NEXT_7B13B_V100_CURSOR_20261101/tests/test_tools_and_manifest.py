import json
from pathlib import Path
import unittest
from tools.print_arm_env import print_arm
from tools.plan_priority_queue import priority
from tools.score_dev_matrix import rank_rows
from tools.paired_dev_accuracy import score_dataset
from tools.verify_run_manifest import run_check

CFG=json.loads((Path(__file__).resolve().parents[1]/"configs"/"arms.json").read_text())
class ToolTests(unittest.TestCase):
    def test_all_priority_in_matrix(self):
        p=priority(CFG,take=3)
        self.assertEqual(len(p),12)
        self.assertEqual(len(set(p)),len(p))
        self.assertTrue(p[0].startswith("NEXT7_T160_"))
    def test_arm_env_disabled_in_unmodified_repo_explicit(self):
        arm=CFG["arm_matrix"][0]
        env=print_arm(CFG,arm)
        self.assertEqual(env["BELA_NEXT_V2_W_LOCK"],str(arm["visual_patch_token_layer_W"]))
        self.assertEqual(env["BELA_NEXT_V2_QA_SWAPS"],"8")
        self.assertEqual(env["BELA_NEXT_V2_STAGE2"],"Q_ATTN_ALL")
    def test_real_trace_validator_synthetic(self):
        arm=next(a for a in CFG["arm_matrix"] if a["arm_id"]=="NEXT7_T160_A0_PAPER")
        a,b=arm["cuts_after_layers"]
        ka,kb,kc=arm["stage_k"]
        rec={"id":"ONLY_SYNTHETIC_EXAMPLE","arm_id":arm["arm_id"],
             "sample_patch_n":2880,"crop_patch_counts":[576]*5,
             "visual_patch_token_counts_per_layer":[ka]*a+[kb]*(b-a)+[kc]*(32-b),
             "prompt_sha256":"hash","dense_prompt_sha256":"hash",
             "pos_and_cache_verified":True,"zero_mask_newline":False}
        got=run_check(CFG,[rec],mode="paper")
        self.assertEqual(got[0]["actual_patch_W"],5120)
        rec_bad=dict(rec,sample_patch_n=2000)
        with self.assertRaises(ValueError):
            run_check(CFG,[rec_bad],mode="paper")
        with self.assertRaises(ValueError):
            run_check(CFG,[dict(rec,pos_and_cache_verified=False)],mode="paper")
    def test_dev_scorer_uses_relative_not_raw_MME(self):
        rec=[]
        for arm in ("A","B"):
            for task,dense,metric in (("GQA",60,59 if arm=="A" else 58),
                                      ("MME",1800,1740 if arm=="A" else 1760)):
                rec.append({"arm_id":arm,"model":"next7","budget_T":160,
                            "task":task,"dense_score":dense,"score":metric,
                            "n":200,"split":"DEV_SCREEN_V1"})
        ranked=rank_rows(rec)
        self.assertEqual(len(ranked),1)
        self.assertEqual(len(next(iter(ranked.values()))),2)
        with self.assertRaises(ValueError):
            rank_rows([dict(x,split="FINAL_TEST") for x in rec])
    def test_paired_true_scorer(self):
        rows=[{"id":str(i),"group":"ocr" if i<2 else "detail",
              "dense_correct":True,"baseline_correct":i%2==0,
              "candidate_correct":i<3} for i in range(4)]
        res=score_dataset(rows)
        self.assertEqual(res["all"]["n"],4)
        self.assertEqual(res["all"]["candidate_only"],1)
        with self.assertRaises(ValueError):
            score_dataset(rows+rows[:1])
if __name__=="__main__":unittest.main()
