"""Explicit user-question span in the expanded multimodal sequence.

The verified rule is an exact token span of the user question. This is not the
post-visual fallback. A candidate is accepted only when the decoded tokens
equal that question string.
"""
from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence, Tuple

STAGE1_VISUAL = 172
MME_SUFFIX = " Please answer yes or no."
POST_PROMPTS = (
    # longest first (VizWiz nests the short VQA post-prompt)
    "\nWhen the provided information is insufficient, respond with 'Unanswerable'.\nAnswer the question using a single word or phrase.",
    "\nAnswer the question using a single word or phrase.",
    "\nAnswer with the option's letter from the given choices directly.",
    "\nAnswer the question with Yes or No.",
)


def question_candidates(doc: Any, context: str) -> List[Tuple[str, str]]:
    found: List[Tuple[str, str]] = []
    seen = set()

    def add(name: str, text: str) -> None:
        text = str(text)
        if text and text not in seen:
            seen.add(text)
            found.append((name, text))

    raw = ""
    if hasattr(doc, "get"):
        raw = str(doc.get("question") or "")
    if raw:
        stripped = raw.strip()
        add("doc_question", stripped)
        # VizWiz lmms-eval capitalizes the question in doc_to_text.
        add("doc_question_capitalized", stripped.capitalize())
        if MME_SUFFIX in raw:
            add("mme_question_without_yesno_suffix", raw.replace(MME_SUFFIX, "").strip())
    ctx = str(context)
    for post in POST_PROMPTS:
        if ctx.endswith(post):
            body = ctx[: -len(post)]
            add("context_without_post_prompt", body)
            add("context_without_post_prompt_stripped", body.strip())
            break  # longest matching suffix wins
    return found


def _chunk_ids(tokenizer, text: str) -> List[int]:
    ids = tokenizer(text).input_ids
    bos = getattr(tokenizer, "bos_token_id", None)
    if ids and bos is not None and ids[0] == bos:
        ids = ids[1:]
    return list(ids)


def explicit_span(tokenizer, prompt: str, raw_ids: Sequence[int], doc: Any, context: str, image_token_id: int):
    """Return (expanded_start, expanded_end, raw_start, raw_end, definition, question)."""
    global STAGE1_VISUAL
    entry = os.environ.get("ELVA_R3_ENTRY_K")
    if entry:
        STAGE1_VISUAL = int(entry)
    if prompt.count("<image>") != 1:
        raise RuntimeError(f"prompt does not contain exactly one image placeholder: {prompt.count('<image>')}")
    positions = [i for i, tok in enumerate(raw_ids) if tok == image_token_id]
    if len(positions) != 1:
        raise RuntimeError(f"expected one image token, found {len(positions)}")
    image_pos = positions[0]
    after = prompt.split("<image>", 1)[1]
    ids_after = _chunk_ids(tokenizer, after)
    if ids_after != list(raw_ids[image_pos + 1 :]):
        raise RuntimeError("image-tail tokens do not match the prompt split")

    errors = []
    for definition, user_question in question_candidates(doc, context):
        if user_question not in after:
            errors.append(f"{definition}: question text is not in the prompt")
            continue
        q_at = after.index(user_question)
        before_q = after[:q_at]
        through_q = after[: q_at + len(user_question)]
        ids_before = _chunk_ids(tokenizer, before_q)
        ids_through = _chunk_ids(tokenizer, through_q)
        aligned = (
            ids_through == ids_after[: len(ids_through)]
            and ids_before == ids_through[: len(ids_before)]
        )
        if aligned:
            raw_start = image_pos + 1 + len(ids_before)
            raw_end = image_pos + 1 + len(ids_through)
        else:
            raw_start, raw_end = _exact_decode_range(tokenizer, raw_ids, image_pos, user_question)
            if raw_start is None:
                errors.append(f"{definition}: token boundary is not aligned")
                continue
        decoded = tokenizer.decode(list(raw_ids[raw_start:raw_end]))
        if decoded != user_question:
            errors.append(f"{definition}: decoded {decoded!r} != {user_question!r}")
            continue
        shift = STAGE1_VISUAL - 1
        return (
            raw_start + shift,
            raw_end + shift,
            raw_start,
            raw_end,
            definition,
            user_question,
        )
    raise RuntimeError("explicit question span failed; " + " | ".join(errors))


def _exact_decode_range(tokenizer, raw_ids, image_pos: int, user_question: str):
    tail = list(raw_ids[image_pos + 1 :])
    hits = []
    for i in range(len(tail)):
        acc = []
        for j in range(i + 1, len(tail) + 1):
            acc.append(tail[j - 1])
            if tokenizer.decode(acc) == user_question:
                hits.append((image_pos + 1 + i, image_pos + 1 + j))
                break
            if len(tokenizer.decode(acc)) > len(user_question) + 8:
                break
    if len(hits) != 1:
        return None, None
    return hits[0]
