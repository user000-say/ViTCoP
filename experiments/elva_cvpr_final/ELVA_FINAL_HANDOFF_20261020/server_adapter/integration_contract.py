"""Host VLM integration contract. Intentional abstract boundaries: NOT a fake patch.

CURSOR: bind callbacks to the CURRENT R3 runner (not October 10 E014 snapshot):
  - first decode request: Stage1 original pen hidden + original patch IDs
  - Stage1 selected ID set using ORIGINAL pen/L9/dual and COVER/PCOV callbacks
  - AFTER each actual decoder block: measure physical visual count
  - AT registered prune event: invoke ORIGINAL FIRST/FINAL selector
    (exact REF/RDRIFT implementation, not a rewritten approximation)
  - call ORIGINAL hidden, mask, RoPE, cache, position gather
  - save prefill traces, true original patch IDs, generation answers

IMPORTANT: this file deliberately does NOT implement CUDA decoder or mask/KV.
"""
from dataclasses import dataclass
from typing import Callable
from elva.core import Schedule, TokenLedger, assert_real_trace

@dataclass
class PatchContract:
    """These actions must be implemented by the host's REAL, tested VLM code."""
    stage1_select_original: Callable       # (image,question,k,mode,cover_mode) -> original patch IDs
    first_select_original: Callable        # (original IDs, k, current model state, selector_name) -> local keep indices
    final_select_original: Callable        # same contract, FINAL always host original selector
    host_physical_gather: Callable         # (full model state, local keep indices)-> updated full state
    log_true_counts: Callable             # (completed_block, physical_count)-> None


def validate_registered_event(schedule:Schedule, completed_block:int,
                              before:TokenLedger, keep_local:list[int]):
    """Call BEFORE host gather; no state committed unless validation passes."""
    target = schedule.expected_after(completed_block)
    if target is None: raise ValueError('unregistered pruning event')
    chosen_ids = before.next_keep(keep_local, target)
    if target >= len(before.ids): raise ValueError('no pruning at event')
    return chosen_ids


def verify_end_of_prefill(schedule:Schedule, actual_counts:list[int]):
    assert_real_trace(actual_counts, schedule)
    return schedule.budget
