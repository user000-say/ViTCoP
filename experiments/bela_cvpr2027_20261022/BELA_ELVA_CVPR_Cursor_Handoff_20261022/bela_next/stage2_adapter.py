"""Safe adapter boundaries for production RDRIFT (no new decoder math).

This module does NOT compute/fake actual attention, hidden-state transfer,
physical token gather, masks, position IDs, or KV-cache compaction.

Use it ONLY once a live-model hook provides REAL original patch identities and
REAL production selector scores at the correct post-block boundary.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from .budget import Schedule, verify_monotone_survivors

def select_existing_score(patch_ids, production_scores, keep_k):
    """Deterministic top-k of ORIGINAL RDRIFT scores; keep input order.

    This is not a replacement for the real scoring implementation. Scores and
    IDs must come from the *same* actual decoder state; no fake placeholders.
    """
    ids=[int(i) for i in patch_ids]
    vals=[float(v) for v in production_scores]
    if len(ids)!=len(vals) or len(ids)!=len(set(ids)):
        raise ValueError("patch IDs and scores mismatch")
    if not all(math.isfinite(x) for x in vals):
        raise ValueError("non-finite production selector scores")
    if not 0<keep_k<=len(ids):
        raise ValueError("invalid keep_k")
    winners=[i for i,score in
             sorted(zip(ids,vals),key=lambda x:(-x[1],x[0]))[:keep_k]]
    chosen=set(winners)
    selected=tuple(i for i in ids if i in chosen)
    verify_monotone_survivors(ids,selected,keep_k)
    return selected

@dataclass
class PruningContract:
    """Validate token-ID events; never call `.step` on decoder TOKEN generation.

    For prefill, after block `cut1`, call step_after_block(cut1, selected_ids);
    similarly for cut2. All actual decoder execution, gather and KV semantics
    are the responsibility of the existing verified server code.
    """
    schedule: Schedule
    entry_ids: tuple
    event_log: list = field(default_factory=list)

    def __post_init__(self):
        self.entry_ids=tuple(int(i) for i in self.entry_ids)
        if len(self.entry_ids)!=self.schedule.k0 or len(set(self.entry_ids))!=len(self.entry_ids):
            raise ValueError("Entry count/uniqueness mismatch")
        self.current_ids=self.entry_ids
        self.next_event=0

    def expected_event(self):
        if self.next_event==0: return (self.schedule.cut1,self.schedule.kmid)
        if self.next_event==1: return (self.schedule.cut2,self.schedule.kfinal)
        return None

    def step_after_block(self, post_block_number, selected_original_ids):
        event=self.expected_event()
        if event is None:
            raise RuntimeError("Unexpected third pruning event")
        expected_layer,expected_count=event
        if post_block_number!=expected_layer:
            raise RuntimeError("Prune must occur at registered post-block boundary")
        verify_monotone_survivors(self.current_ids,selected_original_ids,expected_count)
        selected=tuple(int(i) for i in selected_original_ids)
        self.event_log.append({
            "post_block":post_block_number,
            "input_count":len(self.current_ids),
            "output_count":len(selected),
            "removed_count":len(self.current_ids)-len(selected),
        })
        self.current_ids=selected
        self.next_event+=1
        return selected

    def validate_end_prefill(self):
        if self.next_event!=2 or len(self.current_ids)!=self.schedule.kfinal:
            raise RuntimeError("Incomplete/faulty prefill pruning path")
        return {"expected_W":self.schedule.W,
                "final_count":len(self.current_ids), "events":list(self.event_log)}
