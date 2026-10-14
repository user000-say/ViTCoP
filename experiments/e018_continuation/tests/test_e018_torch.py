"""Server-only runtime regressions with a small causal RoPE decoder, no weights.

Run after install at complete repo root. This file is NOT claimed executed when
torch/transformers are absent locally. Dense and compressed outputs are compared
to independent direct attention equations, not a copied selector implementation.
"""
import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import torch
import torch.nn as nn
from types import SimpleNamespace
from transformers.cache_utils import DynamicCache
from transformers import LlamaConfig
from llava.model.language_model.vitcop_modeling_llama import LlamaSdpaAttention
from llava.model.e018_stage2_support import Runtime, publish_stage1, question_response, Controller
from llava.model.e018_contract import E018Config, FEATURE_NAMES, SCHEMA
from llava.model.e018_counterfactual import capture_at, continuation_pair
from llava.model.e018_qed_rs_diagnostic import qed_rs


class Block(nn.Module):
    def __init__(self, i, width):
        super().__init__()
        config = LlamaConfig(hidden_size=width, intermediate_size=width*2,
                             num_attention_heads=2, num_key_value_heads=2,
                             max_position_embeddings=128, attention_dropout=0.)
        self.self_attn = LlamaSdpaAttention(config, layer_idx=i)
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 2), nn.Tanh(), nn.Linear(width * 2, width))

    def forward(self, x, attention_mask, position_ids, past_key_value, **kwargs):
        result = self.self_attn(self.norm(x), attention_mask=attention_mask,
                                position_ids=position_ids, past_key_value=past_key_value,
                                use_cache=True, output_attentions=False)
        x = x + result[0]
        return (x + self.mlp(self.norm(x)), past_key_value)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block(i, 8) for i in range(6)])
        self.norm, self.embed_tokens = nn.LayerNorm(8), nn.Embedding(23, 8)
        self.config = SimpleNamespace(output_attentions=False, output_hidden_states=False, use_return_dict=True)
        self._use_sdpa, self._use_flash_attention_2 = True, False


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(121)
        self.core = Tiny().eval()
        self.head = nn.Linear(8, 23, bias=False).eval()
        self.x = torch.randn(1, 12, 8)
        self.cfg = E018Config(True, "COLLECT", ((1, 5), (3, 2)), protocol_id="toy-test")

    def batch(self):
        ids = torch.arange(8)
        publish_stage1(ids, ids[:6], torch.ones(8), "toy", 2)
        return dict(inputs_embeds=self.x, image_shape=8, pre_prompt_length_list=[1],
                    token_length_list=[12], e006_question_span=(9, 11), use_cache=True,
                    return_dict=True)

    def test_mask_full_prefix_not_visual_only(self):
        q, k = torch.randn(1, 2, 7, 4), torch.randn(1, 2, 7, 4)
        qi = torch.tensor([5, 6])
        got = question_response(q, k, None, qi, 1, 3)
        logits = q @ k.transpose(-2, -1) / 2
        mask = torch.triu(torch.ones(7, 7, dtype=torch.bool), diagonal=1)
        probs = logits.masked_fill(mask, -float("inf")).softmax(-1)
        reference = probs.mean(1)[0, qi, 1:4]
        torch.testing.assert_close(got["q_attn"], reference.mean(0))
        torch.testing.assert_close(got["g"], reference.sum(-1))
        self.assertTrue(torch.all(got["g"] < 1))

    def test_rs_same_question_roles_revert_to_relevance(self):
        a = torch.tensor([[.1, .2, .4], [.05, .1, .2], [.2, .4, .8]])
        result = qed_rs(a)
        self.assertLess(float(result["role_diversity"]), 1e-5)
        torch.testing.assert_close(result["score"], result["relevance"], atol=1e-5, rtol=1e-5)

    def test_no_padding_batch_or_empty_question(self):
        r = Runtime(self.core, self.cfg)
        batch = self.batch()
        batch["attention_mask"] = torch.tensor([[1]*11+[0]])
        with self.assertRaises(ValueError):
            r.forward(**batch)
        batch = self.batch()
        batch["e006_question_span"] = (10, 10)
        with self.assertRaises(ValueError):
            r.forward(**batch)

    def test_dense_identity_with_patched_real_sdpa(self):
        r = Runtime(self.core, replace(self.cfg, mode="STATIC", events=()))
        out = r.forward(**self.batch())
        h, cache = self.x.clone(), DynamicCache()
        positions = torch.arange(12)[None]
        for layer in self.core.layers:
            h = layer(h, None, positions, cache)[0]
        reference = self.core.norm(h)
        torch.testing.assert_close(out.last_hidden_state, reference)
        self.assertEqual(r.state.physical_events, 0)

    def test_qed_single_event_selection_and_outputs_match_original_side(self):
        # E006's original side reader computes score on the SAME real Q/K.
        cfg = replace(self.cfg, mode="STATIC", events=((1, 5),), selector="QED")
        r = Runtime(self.core, cfg)
        attn = self.core.layers[1].self_attn
        attn._e006_enabled = True
        attn._e006_selector = "QED"
        attn._e006_visual_start, attn._e006_visual_count = 1, 8
        attn._e006_question_indices = torch.tensor([9, 10])
        attn._e006_target_k, attn._e006_eps, attn._e006_profile = 5, 1e-8, False
        attn._e006_span_source = "explicit"
        # Get old selection before Runtime deliberately disables all legacy readers.
        h, cache = self.x.clone(), DynamicCache()
        for layer in self.core.layers[:2]:
            h = layer(h, None, torch.arange(12)[None], cache)[0]
        old = attn._e006_last
        self.assertIsNotNone(old)
        selected = old["selected_local"].sort().values
        rows = torch.cat((torch.tensor([0]), selected + 1, torch.tensor([9, 10, 11])))
        h = h.index_select(1, rows)
        for layer in self.core.layers[2:]:
            h = layer(h, None, rows[None], cache)[0]
        expected = self.core.norm(h)
        out = r.forward(**self.batch())
        self.assertEqual(r.state.visual_ids.tolist(), selected.tolist())
        torch.testing.assert_close(out.last_hidden_state, expected)

    def test_two_prunes_text_positions_decode_and_cache_policy(self):
        for policy in ("layer_local", "scrub_all"):
            r = Runtime(self.core, replace(self.cfg, cache_policy=policy))
            out = r.forward(**self.batch())
            self.assertEqual(out.last_hidden_state.shape[1], 6)
            self.assertEqual(r.state.question_indices.tolist(), [3, 4])
            positions = r.state.positions.tolist()
            self.assertEqual([p for p in positions if p not in range(1, 9)], [0, 9, 10, 11])
            self.assertEqual(r.state.physical_events, 2)
            c = DynamicCache.from_legacy_cache(out.past_key_values)
            expected = [12, 12, 9, 9, 6, 6] if policy == "layer_local" else [6] * 6
            self.assertEqual([k.shape[2] for k in c.key_cache], expected)
            out = r.forward(input_ids=torch.tensor([[4]]), past_key_values=c, return_dict=True)
            self.assertEqual(r.state.original_next, 13)
            self.assertEqual([k.shape[2] for k in out.past_key_values.key_cache], [v + 1 for v in expected])
            self.assertEqual(r.state.physical_events, 2)
            with self.assertRaises(ValueError):
                r.forward(input_ids=torch.tensor([[5]]), past_key_values=c, position_ids=torch.tensor([[6]]))

    def test_counterfactual_common_prefix_final_ids_and_work(self):
        for policy in ("layer_local", "scrub_all"):
            for layer, expected in ((1, 6), (3, 6)):
                cfg = replace(self.cfg, cache_policy=policy)
                r = Runtime(self.core, cfg)
                snap = capture_at(r, self.batch(), layer)
                frozen = snap["hidden"].clone()
                pair = continuation_pair(self.core, self.head, cfg, snap, [3, 5, 7])
                self.assertEqual(pair["delta_token_work"], expected)
                self.assertAlmostEqual(pair["utility"], pair["now_nll"] - pair["later_nll"])
                self.assertEqual(len(pair["final_ids"]), 2)
                torch.testing.assert_close(snap["hidden"], frozen)
                if policy == "scrub_all":
                    self.assertEqual(pair["now_cache_lengths"], pair["later_cache_lengths"])

    def test_online_both_delay_endpoints_at_most_two_physical_events(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = replace(self.cfg, mode="ONLINE", controller=str(Path(d) / "toy.json"))
            fp = cfg.fingerprint(8, 6, dict(name="toy", exchange_budget=2))
            n = len(FEATURE_NAMES)
            # A synthetic constant controller exclusively for execution tests.
            artifact = dict(schema=SCHEMA, feature_names=list(FEATURE_NAMES), calibrated=True,
                            fingerprint=fp, mean=[0.] * n, scale=[1.] * n, w1=[[0.]] * n,
                            b1=[0.], w2=[[0.]], b2=[1.], q_upper=0.)
            Path(cfg.controller).write_text(json.dumps(artifact))
            r = Runtime(self.core, cfg)
            out = r.forward(**self.batch())
            self.assertEqual([e["action"] for e in r.state.events], ["continue", "continue"])
            self.assertEqual(r.state.physical_events, 2)
            self.assertEqual(r.state.visual_ids.numel(), 2)
            self.assertEqual(r.state.per_layer_n, [8, 8, 8, 8, 5, 5])
            cfg = replace(cfg, max_work=31)
            r = Runtime(self.core, cfg)
            r.forward(**self.batch())
            self.assertEqual(sum(r.state.per_layer_n), 30)

    def test_proxy_controls_and_on_policy_prefix_capture(self):
        r = Runtime(self.core, replace(self.cfg, mode="PROXY", proxy_threshold=0))
        r.forward(**self.batch())
        self.assertEqual(sum(r.state.per_layer_n), 42)
        self.assertEqual(r.state.gather_operations, 2)
        r = Runtime(self.core, replace(self.cfg, mode="PROXY", proxy_threshold=1))
        r.forward(**self.batch())
        self.assertEqual(sum(r.state.per_layer_n), 30)
        self.assertEqual(r.state.gather_operations, 2)
        with tempfile.TemporaryDirectory() as d:
            cfg = replace(self.cfg, controller=str(Path(d) / "rollout.json"))
            n = len(FEATURE_NAMES)
            fp = cfg.fingerprint(8, 6, dict(name="toy", exchange_budget=2))
            artifact = dict(schema=SCHEMA, feature_names=list(FEATURE_NAMES), calibrated=True,
                            fingerprint=fp, mean=[0.] * n, scale=[1.] * n,
                            w1=[[0.]] * n, b1=[0.], w2=[[0.]], b2=[1.], q_upper=0.)
            Path(cfg.controller).write_text(json.dumps(artifact))
            r = Runtime(self.core, cfg)
            snap = capture_at(r, self.batch(), 3)
            self.assertEqual(snap["state"].physical_events, 1)
            self.assertEqual(snap["state"].gather_operations, 1)
            pair = continuation_pair(self.core, self.head, cfg, snap, [4, 5])
            self.assertEqual(pair["delta_token_work"], 6)


if __name__ == "__main__":
    unittest.main()
