"""Frozen, label-free question-token mask for the QA-COVER pilot.

Pieces are the tokenizer's own decode of each question id. A piece is dropped
only when it is punctuation or a template word. Object nouns are kept.
Ground-truth answers are never read.
"""
from __future__ import annotations

TEMPLATE_PIECES = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "the", "there", "this", "that", "these",
    "those", "to", "with", "image", "picture", "photo", "user", "assistant",
    "yes", "no", "do", "does", "did", "what", "which", "who", "whom", "whose",
    "where", "when", "how", "please", "answer",
})
_PUNCT = set("?.!,;:'\"`()[]{}<>/\\-_")


def piece_is_content(text: str) -> bool:
    token = text.strip().lower().strip("".join(_PUNCT))
    if not token:
        return False
    if all(ch in _PUNCT or ch.isspace() for ch in token):
        return False
    return token not in TEMPLATE_PIECES


def content_mask_from_decodes(pieces) -> list:
    return [piece_is_content(p) for p in pieces]
