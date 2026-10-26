"""Step-1 bottleneck finder: E3 Fixed32 stage + per-layer decode CUDA timings.

Read-only instrumentation via monkeypatch (default). Optional --skip-cache-pos-cat
probes removing decode-time cache_positions torch.cat (bookkeeping only).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path("/home/xiangyue/ViTCoP")
PANEL = REPO / "experiments/e014_latency_fair_256/panel/speed_256.jsonl"
MODEL = REPO / "experiments/e014_latency_fair_256/models_llava"
CFG = REPO / "experiments/bela_cvpr27_final_handoff_20261025/configs/p1_arms/E3_RET24_TO16.json"
OUT = REPO / "logs/bela_decode_profile_20261026"


def setup_env(policy: str) -> None:
    for p in (
        str(REPO),
        str(REPO / "experiments"),
        str(REPO / "experiments/elva_r3"),
        str(REPO / "LLaVA"),
        str(REPO / "e020_lib"),
        str(REPO / "lmms-eval"),
    ):
        if p not in sys.path:
            sys.path.insert(0, p)
    from apply_llava import overlay
    from e014_arms import apply_arm_env

    os.environ["ELVA_R3_CONFIG"] = str(CFG)
    os.environ["E014_ARM"] = "E018_COVER_TWO32"
    overlay(apply_arm_env("E018_COVER_TWO32"))
    os.environ["E018_CACHE_POLICY"] = policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="layer_local", choices=["layer_local", "scrub_all"])
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument(
        "--skip-cache-pos-cat",
        action="store_true",
        help="OPT probe: skip cache_positions torch.cat during decode",
    )
    args = ap.parse_args()
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    setup_env(args.policy)

    import e006_span
    import torch
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.model import e018_stage2_support as e018mod
    from llava.model.builder import load_pretrained_model
    from PIL import Image
    from transformers import LogitsProcessorList
    from transformers.cache_utils import Cache, DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.policy}" + ("_nocat" if args.skip_cache_pos_cat else "")
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with PANEL.open() as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= max(args.n, args.warmups):
                break

    name = get_model_name_from_path(str(MODEL))
    tok, model, proc, _ = load_pretrained_model(
        str(MODEL),
        None,
        name,
        load_8bit=False,
        load_4bit=False,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    prof = {
        "prefill_ms": [],
        "layer_ms_sum": [0.0] * 32,
        "layer_calls": 0,
        "cache_pos_cat_ms": [],
        "norm_ms": [],
        "outside_layers_ms": [],
    }

    _orig_fwd = e018mod.Runtime.forward

    def _fwd2(self, **kwargs):
        past_key_values = kwargs.get("past_key_values")
        legacy = not isinstance(past_key_values, Cache)
        cache = DynamicCache.from_legacy_cache(past_key_values) if legacy else past_key_values
        is_prefill = cache.get_seq_length() == 0
        if is_prefill:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = _orig_fwd(self, **kwargs)
            torch.cuda.synchronize()
            prof["prefill_ms"].append((time.perf_counter() - t0) * 1000.0)
            return out

        m = self.model
        s = self.state
        input_ids = kwargs.get("input_ids")
        inputs_embeds = kwargs.get("inputs_embeds")
        output_hidden_states = kwargs.get("output_hidden_states")
        return_dict = kwargs.get("return_dict")
        position_ids = kwargs.get("position_ids")
        hidden = m.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if s is None or hidden.shape[1] != 1:
            raise ValueError("decode requires state and q_len=1")
        output_hidden_states = bool(
            output_hidden_states if output_hidden_states is not None else m.config.output_hidden_states
        )
        return_dict = m.config.use_return_dict if return_dict is None else return_dict
        history = () if output_hidden_states else None
        pos = torch.tensor([[s.original_next]], device=hidden.device, dtype=torch.long)
        if position_ids is not None and not torch.equal(position_ids.to(pos.device), pos):
            raise ValueError("Decode RoPE must use original position")

        torch.cuda.synchronize()
        step_t0 = time.perf_counter()
        layer_times = []
        cat_ms = 0.0
        for i, layer in enumerate(m.layers):
            if output_hidden_states:
                history += (hidden,)
            layer.self_attn._e018_callback = None
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            output = layer(
                hidden,
                attention_mask=None,
                position_ids=pos,
                past_key_value=cache,
                use_cache=True,
                output_attentions=False,
            )
            hidden = output[0]
            ev1.record()
            torch.cuda.synchronize()
            dt = float(ev0.elapsed_time(ev1))
            layer_times.append(dt)
            prof["layer_ms_sum"][i] += dt
            prof["layer_calls"] += 1
            if not args.skip_cache_pos_cat:
                c0 = time.perf_counter()
                s.cache_positions[i] = torch.cat((s.cache_positions[i], pos[0]))
                cat_ms += (time.perf_counter() - c0) * 1000.0
        s.original_next += 1
        n0 = torch.cuda.Event(enable_timing=True)
        n1 = torch.cuda.Event(enable_timing=True)
        n0.record()
        hidden = m.norm(hidden)
        n1.record()
        torch.cuda.synchronize()
        norm_ms = float(n0.elapsed_time(n1))
        prof["norm_ms"].append(norm_ms)
        if output_hidden_states:
            history += (hidden,)

        step_ms = (time.perf_counter() - step_t0) * 1000.0
        if not hasattr(self, "_prof_decode_steps"):
            self._prof_decode_steps = []
        self._prof_decode_steps.append(step_ms)
        if not args.skip_cache_pos_cat:
            prof["cache_pos_cat_ms"].append(cat_ms)
        prof["outside_layers_ms"].append(step_ms - sum(layer_times) - cat_ms - norm_ms)

        result_cache = cache.to_legacy_cache() if legacy else cache
        if not return_dict:
            return tuple(x for x in (hidden, result_cache, history) if x is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden,
            past_key_values=result_cache,
            hidden_states=history,
            attentions=None,
        )

    e018mod.Runtime.forward = _fwd2  # type: ignore

    class StepCounter:
        def __init__(self):
            self.n = 0

        def __call__(self, input_ids, scores):
            self.n += 1
            return scores

    def prepare(row):
        with Image.open(row["image"]) as src:
            image = src.convert("RGB")
            image_size = tuple(src.size)
        image_tensor = process_images([image], proc, model.config).to(
            device="cuda:0", dtype=torch.float16
        )
        context = str(row.get("eval_context") or row["question"])
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + context)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inp = tokenizer_image_token(prompt, tok, IMAGE_TOKEN_INDEX, return_tensors="pt")
        ids = inp.unsqueeze(0).to("cuda:0")
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        doc = {"question": str(row.get("doc_question") or row["question"])}
        start, end, *_ = e006_span.explicit_span(
            tok, prompt, inp.tolist(), doc, context, IMAGE_TOKEN_INDEX
        )
        return dict(
            input_ids=ids,
            attention_mask=ids.ne(pad_id),
            pad_token_id=pad_id,
            images=image_tensor,
            image_sizes=[image_size],
            span=(int(start), int(end)),
        )

    def run_fixed32(prepared):
        rt = getattr(model.model, "_e018_runtime", None)
        if rt is not None:
            rt._prof_decode_steps = []
        counter = StepCounter()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model.generate(
                prepared["input_ids"],
                images=prepared["images"],
                image_sizes=prepared["image_sizes"],
                attention_mask=prepared["attention_mask"],
                pad_token_id=prepared["pad_token_id"],
                do_sample=False,
                num_beams=1,
                use_cache=True,
                min_new_tokens=32,
                max_new_tokens=32,
                logits_processor=LogitsProcessorList([counter]),
                e006_question_span=prepared["span"],
            )
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1000.0
        if counter.n != 32:
            raise RuntimeError(f"steps {counter.n}")
        dec = list(getattr(model.model._e018_runtime, "_prof_decode_steps", []))
        return wall, dec

    for j in range(args.warmups):
        run_fixed32(prepare(rows[j % len(rows)]))
    prof["prefill_ms"].clear()
    prof["layer_ms_sum"] = [0.0] * 32
    prof["layer_calls"] = 0
    prof["cache_pos_cat_ms"].clear()
    prof["norm_ms"].clear()
    prof["outside_layers_ms"].clear()

    walls = []
    all_decode_steps = []
    for idx in range(args.n):
        wall, dec = run_fixed32(prepare(rows[idx]))
        walls.append(wall)
        all_decode_steps.append(dec)
        print(f"SAMPLE {idx+1}/{args.n} Fixed32={wall:.1f} decode_steps={len(dec)}", flush=True)

    n_decode_layers = max(prof["layer_calls"] // 32, 1)
    layer_avg = [prof["layer_ms_sum"][i] / n_decode_layers for i in range(32)]
    flat_steps = [x for steps in all_decode_steps for x in steps]

    summary = {
        "policy": args.policy,
        "skip_cache_pos_cat": args.skip_cache_pos_cat,
        "n": args.n,
        "fixed32_wall_ms": {
            "mean": statistics.mean(walls),
            "median": statistics.median(walls),
            "min": min(walls),
            "max": max(walls),
        },
        "prefill_ms": {
            "mean": statistics.mean(prof["prefill_ms"]) if prof["prefill_ms"] else None,
            "n": len(prof["prefill_ms"]),
        },
        "decode_step_ms": {
            "mean": statistics.mean(flat_steps) if flat_steps else None,
            "n_steps": len(flat_steps),
        },
        "layer_avg_ms": layer_avg,
        "layer_avg_early_0_3": statistics.mean(layer_avg[0:4]),
        "layer_avg_mid_4_11": statistics.mean(layer_avg[4:12]),
        "layer_avg_late_12_21": statistics.mean(layer_avg[12:22]),
        "layer_avg_tail_22_31": statistics.mean(layer_avg[22:32]),
        "cache_pos_cat_ms_per_step": {
            "mean": statistics.mean(prof["cache_pos_cat_ms"]) if prof["cache_pos_cat_ms"] else 0.0,
        },
        "norm_ms_per_step": {
            "mean": statistics.mean(prof["norm_ms"]) if prof["norm_ms"] else None,
        },
        "outside_layers_ms_per_step": {
            "mean": statistics.mean(prof["outside_layers_ms"]) if prof["outside_layers_ms"] else None,
        },
        "hypothesis_reads": [],
    }
    early = summary["layer_avg_early_0_3"]
    tail = summary["layer_avg_tail_22_31"]
    if early > tail * 1.15:
        summary["hypothesis_reads"].append(
            f"EARLY_LAYERS_SLOWER: early={early:.3f} vs tail={tail:.3f} ms → KV length likely matters"
        )
    else:
        summary["hypothesis_reads"].append(
            f"LAYER_TIME_FLAT: early={early:.3f} vs tail={tail:.3f} ms → not KV-length bound"
        )
    cat = summary["cache_pos_cat_ms_per_step"]["mean"]
    if cat > 0.3:
        summary["hypothesis_reads"].append(
            f"CACHE_POS_CAT_COSTLY: {cat:.2f} ms/step ×31 ≈ {cat * 31:.0f} ms/Fixed32"
        )
    outside = summary["outside_layers_ms_per_step"]["mean"] or 0.0
    if outside > 0.5:
        summary["hypothesis_reads"].append(
            f"OUTSIDE_LAYERS: {outside:.2f} ms/step (python/embed/sync)"
        )
    # HF generate overhead estimate
    pref = summary["prefill_ms"]["mean"] or 0.0
    dec_mean = summary["decode_step_ms"]["mean"] or 0.0
    accounted = pref + dec_mean * 31  # approx: 1 prefill + 31 more decode after first token
    # Fixed32 does 32 decode steps after prefill in our instrumented path
    accounted2 = pref + dec_mean * 32
    summary["accounted_vs_wall"] = {
        "prefill_plus_32_decode_steps": accounted2,
        "fixed32_wall": summary["fixed32_wall_ms"]["mean"],
        "hf_generate_overhead_ms": summary["fixed32_wall_ms"]["mean"] - accounted2,
    }

    path = out_dir / "profile_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print("PROFILE_DONE", path, flush=True)
    print(json.dumps({k: summary[k] for k in summary if k != "layer_avg_ms"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
