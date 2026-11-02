#!/usr/bin/env python3
"""Portable lmms-eval BELA NeXT-7B entry: no absolute legacy source imports.

Use on a clean copy of the users' transformers 5.20.2 environment. GPU
jobs and the user's existing queue are intentionally NOT managed here.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
if os.environ.get('LMMS_EVAL_ROOT'):
    sys.path.insert(0, str(Path(os.environ['LMMS_EVAL_ROOT']).expanduser().resolve()))
for key, value in (
    ('HF_HUB_OFFLINE', '1'), ('HF_DATASETS_OFFLINE','1'),
    ('TRANSFORMERS_OFFLINE','1'), ('TOKENIZERS_PARALLELISM','false')
):
    os.environ.setdefault(key,value)
if os.environ.get('BELA_NEXT_V2_ENABLE','0').lower() not in ('1','true','yes'):
    raise SystemExit('BELA_NEXT_V2_ENABLE=1 is required: do not label Dense as pruned')
if not os.environ.get('BELA_NEXT_ARM','').strip():
    raise SystemExit('BELA_NEXT_ARM is required')

from lmms_eval.models.llava_hf import LlavaHf
from bela_next_forward.apply_llava_next import apply_patch
_old_init = LlavaHf.__init__


def _installed_init(self, *args, **kwargs):
    _old_init(self,*args,**kwargs)
    # Tokenizer MUST be the same instance used to build original input IDs.
    # Do not let Stage1 silently load a different repo/template.
    tokenizer = getattr(self,'_tokenizer',None)
    if tokenizer is None:
        processor = getattr(self, '_processor',None)
        tokenizer = getattr(processor,'tokenizer',None)
    if tokenizer is None:
        raise RuntimeError('TOKENIZER_MISSING: lmms-eval adapter did not expose the real tokenizer')
    self._model.model._tokenizer = tokenizer
    apply_patch(self._model)
    checksum = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    print(f'[BELA_NEXT] patch={checksum} arm={os.environ["BELA_NEXT_ARM"]} '
          f'processor_pinpoints={getattr(getattr(self, "_image_processor", None), "image_grid_pinpoints", "unknown")} '
          f'packed=True',flush=True)

LlavaHf.__init__ = _installed_init

from lmms_eval.__main__ import cli_evaluate
if __name__ == '__main__':
    cli_evaluate()
