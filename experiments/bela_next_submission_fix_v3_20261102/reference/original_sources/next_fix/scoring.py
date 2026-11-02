"""Audit-only ScienceQA letter parsing; preserve raw answers and Dense scoring."""
import re

# Only match isolated choice letters at the very start of a short-answer field.
# Do not turn explanation text ('Because ...') into choice B.
_choice=re.compile(r'^\s*\(?([A-Ea-e])\)?\s*[\.)]?\s*$')


def choice_letter(raw):
    hit=_choice.fullmatch(raw or '')
    return hit.group(1).upper() if hit else None


def paired_choice_audit(raw_predictions, reference_labels):
    if len(raw_predictions)!=len(reference_labels):
        raise ValueError('Prediction/reference count mismatch')
    return [{'raw':p,'parsed_choice':choice_letter(p),
             'reference':y,'parsed_correct':choice_letter(p)==y.upper()}
            for p,y in zip(raw_predictions,reference_labels)]
