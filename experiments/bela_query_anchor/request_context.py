"""Request-local QA-COVER slot.

One in-flight sample only. This is not a sticky LAST_QUESTION cache:
begin_question refuses re-entry, and end_question resets the context var.
"""
from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class QARequest:
    question_ids: torch.Tensor
    content_mask: Optional[torch.Tensor]
    question_text: str
    definition: str
    source: str
    sample_id: str = ""
    image_id: str = ""
    donor_sample_id: Optional[str] = None
    mm_projector: Any = None
    embed_tokens: Any = None
    native_ids: Optional[list] = None
    final_ids: Optional[list] = None
    added_ids: list = field(default_factory=list)
    removed_ids: list = field(default_factory=list)
    swaps: int = 0
    reason: str = ""


_current: ContextVar[Optional[QARequest]] = ContextVar("bela_qa_request", default=None)


def current() -> Optional[QARequest]:
    return _current.get()


def begin_question(
    question_ids: torch.Tensor,
    content_mask: Optional[torch.Tensor],
    *,
    question_text: str,
    definition: str,
    source: str,
    sample_id: str = "",
    image_id: str = "",
    donor_sample_id: Optional[str] = None,
) -> Token:
    if _current.get() is not None:
        raise RuntimeError("QA-COVER request overlap; refuse concurrent or nested samples")
    if question_ids.ndim != 1:
        raise RuntimeError("question_ids must be 1-D")
    req = QARequest(
        question_ids=question_ids.detach().cpu().long(),
        content_mask=None if content_mask is None else content_mask.detach().cpu().bool(),
        question_text=question_text,
        definition=definition,
        source=source,
        sample_id=sample_id,
        image_id=image_id,
        donor_sample_id=donor_sample_id,
    )
    return _current.set(req)


def end_question(token: Token) -> None:
    _current.reset(token)


def attach_modules(mm_projector, embed_tokens) -> None:
    req = _current.get()
    if req is None:
        raise RuntimeError("QA-COVER encode_images ran without a question request")
    if req.mm_projector is not None or req.embed_tokens is not None:
        raise RuntimeError("QA-COVER modules already attached for this request")
    req.mm_projector = mm_projector
    req.embed_tokens = embed_tokens


def detach_modules() -> None:
    req = _current.get()
    if req is None:
        return
    req.mm_projector = None
    req.embed_tokens = None
