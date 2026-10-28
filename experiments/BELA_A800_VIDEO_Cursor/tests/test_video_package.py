import unittest
import torch

from bela_video_a800.schedule import budget
from bela_video_a800.video_entry import select_framewise, query_proxy
from bela_video_a800.trace_audit import check_trace
from bela_video_a800.shuffle_control import stable_key, build_mismatch


class VideoScheduleTest(unittest.TestCase):
    def test_paper_tier_32(self):
        b=budget("T32")
        self.assertEqual((b.frames,b.patches_per_frame,b.decoder_layers),(64,169,28))
        self.assertEqual(b.framewise_counts,(64,11,4))
        self.assertEqual(b.cut_layers_after,(12,21))
        self.assertEqual(b.entry_total_k,4096)
        self.assertEqual(b.final_total_k,256)
        self.assertEqual(b.visual_token_layer_W,57280)
        self.assertEqual(b.dense_visual_token_layer_W,302848)
        self.assertAlmostEqual(b.nominal_reduction_fraction, .8108615, places=4)

    def test_paper_tier_16(self):
        b=budget("T16")
        self.assertEqual(b.framewise_counts,(32,5,3))
        self.assertEqual(b.entry_total_k,2048)
        self.assertEqual(b.final_total_k,192)
        self.assertEqual(b.visual_token_layer_W,28800)
        self.assertAlmostEqual(b.nominal_reduction_fraction, .904901, places=4)

    def test_boundaries(self):
        with self.assertRaises(ValueError):budget("T16",frames=65)
        with self.assertRaises(ValueError):budget("T32",patches_per_frame=40)
        b=budget("T32",frames=8)
        self.assertEqual(b.visual_token_layer_W,7160)
        self.assertEqual(len(b.layerwise_frame_k),28)
        self.assertEqual(b.layerwise_frame_k[:12],(64,)*12)
        self.assertEqual(b.layerwise_frame_k[12:21],(11,)*9)
        self.assertEqual(b.layerwise_frame_k[21:],(4,)*7)


class VideoEntryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2027)
        self.frames=3
        self.n=169
        self.features=torch.zeros((3,169,8))
        self.features[:,:161,0]=1
        self.features[:,161:,1]=1
        self.projected=self.features.clone()
        self.text=torch.zeros((2,8))
        self.text[:,1]=1

    def _native(self,k):
        return torch.arange(k).repeat(self.frames,1)

    def test_t32_ids_stay_in_original_frame(self):
        o=select_framewise(
            tier="T32",native_ids=self._native(64),
            dual_views=(self.features,self.features*2),
            projected_full_pooled_vision=self.projected,
            question_content_embeddings=self.text,
        )
        self.assertEqual(o.ids.shape,(3,64))
        self.assertTrue(all(0<=x<=8 for x in o.swap_count))
        self.assertTrue(any(x>0 for x in o.swap_count))
        self.assertTrue(torch.all(o.ids[:,:-1]<=o.ids[:,1:]))
        self.assertEqual(len(o.flattened_visual_only_ids.unique()),3*64)
        for f in range(self.frames):
            orig=set(int(x) for x in o.ids[f])
            global_ids=set(int(x) for x in o.flattened_visual_only_ids[f*64:(f+1)*64])
            self.assertEqual(global_ids,{f*169+x for x in orig})

    def test_t16_random_control_keeps_drop_count(self):
        kw=dict(tier="T16",native_ids=self._native(32),
             dual_views=(self.features,self.features),
             projected_full_pooled_vision=self.projected,
             question_content_embeddings=self.text)
        qa=select_framewise(**kw)
        control=select_framewise(**kw,control="matched_random",seed=17)
        self.assertEqual(qa.ids.shape,(3,32))
        self.assertTrue(all(x<=4 for x in qa.swap_count))
        self.assertEqual(control.swap_count,qa.swap_count)
        for f in range(self.frames):
            before=set(self._native(32)[f].tolist())
            dqa=before-set(qa.ids[f].tolist())
            drandom=before-set(control.ids[f].tolist())
            self.assertEqual(dqa,drandom)
            self.assertEqual(len(control.ids[f].unique()),32)

    def test_off_and_no_question_exact_noop(self):
        native=self._native(64)
        common=dict(tier="T32",native_ids=native,dual_views=(self.features,self.features))
        off=select_framewise(**common,control="off")
        missing=select_framewise(**common)
        self.assertTrue(torch.equal(off.ids,native))
        self.assertTrue(torch.equal(missing.ids,native))
        self.assertEqual(off.swap_count,(0,0,0))

    def test_invalid_ids_and_shapes_fail_closed(self):
        native=self._native(64)
        native[0,0]=170
        with self.assertRaises(ValueError):
            select_framewise(tier="T32",native_ids=native,dual_views=(self.features,))
        with self.assertRaises(ValueError):
            select_framewise(tier="T32",native_ids=self._native(64),
                             dual_views=(self.features[:,:60],))

    def test_flat_question_means_no_swap(self):
        query=torch.zeros((3,169))
        o=select_framewise(tier="T32",native_ids=self._native(64),
                           dual_views=(self.features,),raw_query_scores=query)
        self.assertEqual(o.swap_count,(0,0,0))


class VideoTraceTest(unittest.TestCase):
    def _make_trace(self,tier="T32",f=3,n=169):
        b=budget(tier,frames=f)
        entry,k1,k2=b.framewise_counts
        return {
            "tier":tier,"num_frames":f,"pooled_patches_per_frame":n,
            "entry_frame_patch_ids":[list(range(entry)) for _ in range(f)],
            "after_layer12_frame_patch_ids":[list(range(k1)) for _ in range(f)],
            "after_layer21_frame_patch_ids":[list(range(k2)) for _ in range(f)],
            "frame_timestamps_sec":[float(i) for i in range(f)],
            "observed_visual_k_by_decoder_layer":[f*k for k in b.layerwise_frame_k],
            "position_and_cache_regression_pass":True,
        }

    def test_valid_trace_and_layer_counts(self):
        record=self._make_trace()
        d=check_trace(record,require_physical=True)
        self.assertEqual(d["W"],budget("T32",frames=3).visual_token_layer_W)

    def test_token_resurrection_fails(self):
        record=self._make_trace()
        record["after_layer21_frame_patch_ids"][0][-1]=150
        with self.assertRaises(ValueError):
            check_trace(record)

    def test_wrong_actual_layer_k_fails(self):
        record=self._make_trace()
        record["observed_visual_k_by_decoder_layer"][4]-=1
        with self.assertRaises(ValueError):
            check_trace(record,require_physical=True)

    def test_must_check_cache(self):
        record=self._make_trace()
        del record["position_and_cache_regression_pass"]
        with self.assertRaises(ValueError):
            check_trace(record,require_physical=True)


class VideoMismatchTest(unittest.TestCase):
    def test_duplicate_question_ids_not_used_as_unique_key(self):
        records=[]
        for vi in range(4):
            for qi in range(3):
                records.append(dict(dataset="Video-MME",row_index=vi*3+qi,
                                    video_id=f"video_{vi}",question_id=0,
                                    question_stem=f"Question {vi}_{qi}"))
        result=build_mismatch(records,seed=42)
        self.assertEqual(len(result),12)
        self.assertEqual(len({r["sample_key"] for r in result}),12)
        self.assertTrue(all(str(r["video_id"])!=str(r["anchor_question_source_video_id"]) for r in result))
        self.assertEqual(result,build_mismatch(records,seed=42))

    def test_one_video_fails(self):
        with self.assertRaises(ValueError):
            build_mismatch([dict(dataset="MVBench",row_index=1,video_id="only",
                                 question_stem="Hello")])

if __name__=="__main__":
    unittest.main()
