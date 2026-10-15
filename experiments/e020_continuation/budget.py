"""Explicit layer conventions; visual work is not an exact latency model."""
import math


def canonical(events, depth, convention):
    if convention not in ('zero_based_after_block', 'completed_blocks'):
        raise ValueError('export whether indices or completed-block counts are used')
    result = []
    for event in events:
        layer, keep = event
        if type(layer) is not int or type(keep) is not int:
            raise ValueError('integer layer and keep required')
        after = layer + (convention == 'zero_based_after_block')
        if not 0 <= after <= depth or keep < 1:
            raise ValueError('invalid boundary or visual count')
        if result and after <= result[-1][0]:
            raise ValueError('boundaries must increase strictly')
        result.append([after, keep])
    return result


def profile(depth, initial, events, convention='completed_blocks', text_tokens=None):
    if type(depth) is not int or depth < 1 or type(initial) is not int or initial < 1:
        raise ValueError('positive integer model depth and initial count required')
    events = canonical(events, depth, convention)
    last, count, counts = 0, initial, []
    for after, keep in events:
        if keep > count:
            raise ValueError('deleted patches cannot be reinserted')
        counts.extend([count] * (after-last))
        last, count = after, keep
    counts.extend([count] * (depth-last))
    out = {'completed_block_events': events, 'per_block_visual': counts,
           'visual_token_layers': sum(counts), 'final_visual': count,
           'visual_square_sum': sum(c*c for c in counts)}
    if text_tokens is not None:
        if type(text_tokens) is not int or text_tokens < 1:
            raise ValueError('positive text count required')
        out['joint_sequence_square_sum'] = sum((c+text_tokens)**2 for c in counts)
    return out


def matched_two_event(depth, initial, target_work, first_after, last_after, final):
    if not 0 <= first_after < last_after <= depth:
        raise ValueError('invalid allocation interval')
    keep = math.floor((target_work-first_after*initial-(depth-last_after)*final)
                      / (last_after-first_after))
    if not final <= keep <= initial:
        raise ValueError('infeasible matched allocation')
    return [[first_after, keep], [last_after, final]]
