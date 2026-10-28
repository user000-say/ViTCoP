import json, tempfile, unittest, sys
from pathlib import Path
import torch

from tools.budget_planner import compute_budget, load_specs, allocate_group_quotas
from tools.model_config_probe import inspect_config
from tools.entry_bridge import choose_postmerge_qa
from qa_cover_original.experiments.bela_query_anchor.query_anchor import (
    select_question_anchored, AnchorConfig,
)

class BudgetTests(unittest.TestCase):
    def test_all_reference_budgets(self):
        specs=load_specs()
        for model,spec in specs.items():
            for tier,entry in spec["budgets"].items():
                b=compute_budget(model,tier,spec["reference_n"])
                if entry["kind"]=="layer_average":
                    self.assertEqual(b.W,entry["target_t"] * b.L,(model,tier,b))
                    self.assertEqual(b.avg_visual_tokens,entry["target_t"])
                else:
                    self.assertEqual(b.W,entry["target_k"] * b.L,(model,tier))
                self.assertEqual(b.counts[0],entry["counts"][0])
                self.assertLessEqual(max(b.counts),b.N)
                self.assertTrue(all(b.counts[i]>=b.counts[i+1] for i in range(len(b.counts)-1)))

    def test_interpolated_dynamic_grid(self):
        for model, spec in load_specs().items():
            for N in (160,512,1024,1296,2880,4100):
                for tier in ("moderate","stress"):
                    b=compute_budget(model,tier,N)
                    self.assertEqual(b.dense_W,N*b.L)
                    self.assertLessEqual(b.W,b.dense_W)
                    self.assertGreater(b.W,0)
                    self.assertTrue(b.reduction_fraction>=0)

    def test_exact_named_budgets(self):
        self.assertEqual(compute_budget("llava-next-7b","stress",2880).counts,(320,91,46))
        self.assertEqual(compute_budget("llava-next-7b","stress",2880).W,5120)
        self.assertEqual(compute_budget("llava-next-13b","stress",2880).W,6400)
        self.assertEqual(compute_budget("qwen3-vl-8b","stress",1296).W,4608)
        self.assertEqual(compute_budget("qwen3-vl-8b","exploratory_94p4",1296).W,2592)
        self.assertEqual(compute_budget("qwen3.5-9b","stress",1024).W,2048)

    def test_groups(self):
        self.assertEqual(allocate_group_quotas(320,[576]*5,1),[64]*5)
        self.assertEqual(sum(allocate_group_quotas(8,[4,4,4,4,4])),8)
        with self.assertRaises(ValueError):
            allocate_group_quotas(40,[5]*5)
        with self.assertRaises(ValueError):
            allocate_group_quotas(2,[5]*5,1)

class ConfigTests(unittest.TestCase):
    def test_qwen_35_mixed_stack(self):
        cfg={"model_type":"qwen3_5","text_config":{
            "num_hidden_layers":32,"layer_types":["linear_attention"]*3+["full_attention"]+
            ["linear_attention"]*3+["full_attention"]+
            ["linear_attention"]*3+["full_attention"]+
            ["linear_attention"]*3+["full_attention"]+
            ["linear_attention"]*3+["full_attention"]+
            ["linear_attention"]*3+["full_attention"]+
            ["linear_attention"]*3+["full_attention"]+
            ["linear_attention"]*3+["full_attention"]},
            "vision_config":{"depth":27}}
        res=inspect_config(cfg)
        self.assertEqual(res["linear_attention_layers"],24)
        self.assertEqual(res["full_attention_layers_1based"],[4,8,12,16,20,24,28,32])

    def test_qwen3vl_deepstack(self):
        cfg={"model_type":"qwen3_vl","text_config":{"num_hidden_layers":36},
             "vision_config":{"deepstack_visual_indexes":[8,16,24],"spatial_merge_size":2}}
        a=inspect_config(cfg)
        self.assertEqual(a["decoder_layers"],36)
        self.assertEqual(len(a["deepstack_indexes"]),3)

class BridgeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2027)
        self.n=60
        f=torch.randn(self.n, 16)
        f[-8:,:]=0
        f[-8:,0]=2.0
        self.features=(f, f*1.02)
        self.native=torch.arange(18)
        self.visual = torch.zeros(self.n, 16)
        self.visual[:,:] = f
        self.question=torch.zeros(1,16)
        self.question[0,0]=1.
        self.coords=torch.stack([torch.zeros(self.n,dtype=torch.long),
                                 torch.arange(self.n)//10, torch.arange(self.n)%10],dim=1)
        self.deepstack=[torch.randn(self.n,32) for _ in range(3)]

    def test_query_entry_keeps_k_and_mrope_deepstack(self):
        out=choose_postmerge_qa(
            model_name="qwen3-vl-8b",native_keep_ids=self.native,
            dual_metric=self.features, projected_full_vision=self.visual,
            question_content_embeds=self.question,mrope_ids=self.coords,
            deepstack_features=self.deepstack, max_swaps_total=8)
        self.assertEqual(len(out.keep_ids),18)
        self.assertEqual(len(out.keep_ids.unique()),18)
        self.assertTrue(torch.equal(out.selected_mrope_ids,self.coords[out.keep_ids]))
        self.assertTrue(all(torch.equal(x,d[out.keep_ids]) for x,d in zip(out.selected_deepstack,self.deepstack)))
        self.assertTrue(out.swaps <= 8)

    def test_off_mode_exact_noop(self):
        out=choose_postmerge_qa(
            model_name="qwen3.5-9b",native_keep_ids=self.native,
            dual_metric=self.features,projected_full_vision=self.visual,
            question_content_embeds=self.question,enabled=False)
        self.assertTrue(torch.equal(out.keep_ids,self.native))
        self.assertEqual(out.swaps,0)

    def test_wrong_deepstack_alignment_is_rejected(self):
        with self.assertRaises(ValueError):
            choose_postmerge_qa(
                model_name="qwen3-vl-8b",native_keep_ids=self.native,
                dual_metric=self.features, projected_full_vision=self.visual,
                question_content_embeds=self.question,
                deepstack_features=[torch.ones(7,16)])

    def test_llava_next_group_counts_maintained(self):
        groups=torch.arange(self.n)//20
        native=torch.cat([torch.arange(0,6),torch.arange(20,26),torch.arange(40,46)])
        out=choose_postmerge_qa(
            model_name="llava-next-7b",native_keep_ids=native,
            dual_metric=self.features,projected_full_vision=self.visual,
            question_content_embeds=self.question,group_ids=groups,
            max_swaps_total=8)
        self.assertEqual(len(out.keep_ids),18)
        for i in range(3):
            self.assertEqual(int(((out.keep_ids//20)==i).sum()),6)
        self.assertLessEqual(out.swaps,8)

    def test_non_llava_groups_fail_closed(self):
        with self.assertRaises(ValueError):
            choose_postmerge_qa(
                model_name="qwen3-vl-8b",native_keep_ids=self.native,
                dual_metric=self.features,projected_full_vision=self.visual,
                question_content_embeds=self.question,
                group_ids=torch.arange(self.n)//20)

    def test_large_group_fails_closed_before_similarity_allocation(self):
        with self.assertRaises(ValueError):
            choose_postmerge_qa(
                model_name="qwen3-vl-8b",native_keep_ids=self.native,
                dual_metric=self.features,projected_full_vision=self.visual,
                question_content_embeds=self.question,max_group_n=20)

    def test_no_query_is_safe_noop(self):
        out=choose_postmerge_qa(
            model_name="qwen3.5-9b",native_keep_ids=self.native,
            dual_metric=self.features,projected_full_vision=None,question_content_embeds=None)
        self.assertTrue(torch.equal(out.keep_ids,self.native))
        self.assertEqual(out.swaps,0)


class ProvenanceTests(unittest.TestCase):
    def test_score_record_only_with_all_real_provenance(self):
        import tempfile, csv
        from tools.record_scores import record_score
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/"results.csv"
            with target.open("w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=[
                    "model","budget","arm","benchmark","score","status","n_questions",
                    "git_commit","processor_revision","dataset_hash",
                    "scorer_revision","run_dir","notes"])
                w.writeheader()
                w.writerow(dict(model="qwen3-vl-8b",budget="stress",
                                arm="DENSE",benchmark="POPE",score="",
                                status="PENDING"))
            with self.assertRaises(ValueError):
                record_score(model="qwen3-vl-8b",tier="stress",arm="DENSE",benchmark="POPE",
                             value=90,n_questions=9000,git_commit="",
                             processor_revision="x",dataset_hash="x",scorer_revision="x",
                             run_dir="x",filename=target)
            record_score(model="qwen3-vl-8b",tier="stress",arm="DENSE",benchmark="POPE",
                         value=90,n_questions=9000,git_commit="abc123",
                         processor_revision="rev",dataset_hash="sha256",scorer_revision="official",
                         run_dir="proof",filename=target)
            self.assertEqual(list(csv.DictReader(target.open()))[0]["status"],"MEASURED")
            with self.assertRaises(ValueError):
                record_score(model="qwen3-vl-8b",tier="stress",arm="DENSE",benchmark="POPE",
                             value=98,n_questions=9000,git_commit="abc123",
                             processor_revision="rev",dataset_hash="sha256",scorer_revision="official",
                             run_dir="proof",filename=target)

    def test_trace_id_checks_are_strict(self):
        from tools.trace_audit import verify_one
        t={"model":"qwen3.5-9b","budget":"stress",
           "observed_n":1024,"entry_ids":list(range(64)),
           "post_cut_ids":[]}
        self.assertEqual(verify_one(t)["W"],2048)
        broken=dict(t,entry_ids=list(range(63))+[63+1024])
        with self.assertRaises(ValueError):
            verify_one(broken)

    def test_synthetic_four_model_families_safe_shapes(self):
        torch.manual_seed(12)
        for model in ("llava-next-7b","llava-next-13b","qwen3-vl-8b","qwen3.5-9b"):
            n=90
            x=torch.randn(n, 12)
            x[80:]=0
            x[80:,0]=2
            native=torch.arange(20)
            vis=x.clone()
            text=torch.zeros((1,12))
            text[0,0]=1
            groups=torch.arange(n)//30 if model.startswith("llava-next") else None
            out=choose_postmerge_qa(
                model_name=model,native_keep_ids=native,dual_metric=(x,x),
                projected_full_vision=vis,question_content_embeds=text,
                group_ids=groups)
            self.assertEqual(out.keep_ids.numel(),20)
            self.assertTrue(out.keep_ids.unique().numel()==20)



if __name__=="__main__":
    unittest.main()
