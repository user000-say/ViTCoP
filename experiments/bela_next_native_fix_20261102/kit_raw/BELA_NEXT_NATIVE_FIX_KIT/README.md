# BELA Native NeXT-7B Engineering Repair Kit

**Scope:** User's `llava-hf/llava-v1.6-vicuna-7b-hf` / NeXT-7B, original top-performing BELA: dual-depth Native COVER + global QA-COVER B8 + progressive Q_ATTN_ALL, L12/L20; A0–A4 W5120. No STAR-Pro QR selection.

## Contents

| File | Role |
| --- | --- |
| `bela_next_fix/native_layout.py` | Call HF's *own* `pack_image_features` with provenance IDs; unpad/newline mapping; pack dual ViT depth via same geometry. |
| `bela_next_fix/legacy_bridge.py` | Validated A0–A4 config; fail-closed bridge requiring actual legacy Dual COVER and original QA donor-swap. |
| `bela_next_fix/progressive_hooks.py` | PyTorch hooks for physical L12/L20 pruning, Q_ATTN from actual attention weights, 4D causal/rotary slicing and **DynamicCache** trimming. |
| `bela_next_fix/runner.py` | End-to-end NeXT v4.48-style packed prefill and deterministic greedy decoding with preserved Dense position IDs, strict trace. |
| `bela_next_fix/scoring.py` | Optional *audit-only* ScienceQA short letter parser (`B.` -> `B`), no generation-time hacks. |
| `smoke_real_next.py` | Original Dense token equivalence, actual GPU A0–A4 smoke, machine-readable logs. |
| `tests/` | Offline CPU fake-Next/LLM test suite including boundary and negative tests. |
| `docs/CURSOR_EXECUTE_FIRST.md` | Cursor Chinese instructions and no-go gates. |

## Run offline tests

```bash
python -m unittest discover -s tests -v
```

## Important limitations

- The referenced uploaded archives **do not contain** the active V100 repo or legacy 1.5-7B `qa_core` files. This kit cannot honestly contain a finished diff against unknown `apply_llava_next.py` or an invented reproduction of original QA. **Implement two small wrappers against the actual historical functions before live tests.** Without them the adapter raises errors and stops.
- Only pinned-style Hugging Face LLaVA-NeXT `get_image_features -> list[per-image projected crop]` and LlamaDecoderLayer `DynamicCache` are supported. Actual current `transformers` installed on V100 must be verified in real smoke. **Never silently fallback to Dense on version incompatibility.**
- No actual NeXT-7B checkpoint, V100 or six-task dataset was available here; **unit-test success is not a GPU result or accuracy improvement**.
- V100 single-device FP16 may not fit for 7B + vision tower if each card only has 16GB. No unsafe multi-GPU device-map claims.
- This is greedy-only reference generation. Native Dense and adapter must exactly match before metrics. Preserve task-specific output format/scorer and do not stop on first newline.
- `W=5120` means the scheduled **visual Patch** token-layer budget. Native newline, text, image encoding, indexing, prefill/decode KV bytes and real latency are separate costs. Input AnyRes may not yield exactly 2880 packable Patch tokens.
- Alternative scientific claim: fixing engineering distortions can help, but **95%+ relative Dense is NOT guaranteed**.

See `docs/CURSOR_EXECUTE_FIRST.md` for strict integration and evaluation procedure.
