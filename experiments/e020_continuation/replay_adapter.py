"""Full-forward replay collection when native mid-layer resume is unavailable.

Offline only: rerun the same deterministic input, override first-event patch IDs
after the captured checkpoint, and verify the REAL prompt-prefix hash matches.
No shared mutable KV state, and no need to implement a model-specific suffix
forward loop. Extra prefix recomputation is collection cost, never online cost.

Backend factory(config['backend_config']) must expose:
  examples(): iterable of original labelled training/calibration examples;
  run_teacher_forced(example, *, stage1, schedule_name, first_keep_patch_ids,
                     forced_final_ids, capture_prompt_state):
      returns {snapshot: {...}, answer_nll, prefix_hash, cache_policy,
               answer_tokens_hash, answer_token_count, optional correct}.
snapshot contains features/scores/selected/patch_ids/after/depth/next_after/
final_k plus image_id/question_id. Features are captured from prompt positions
only, even when the teacher-forced forward contains answer tokens later.
The backend reuses the project's existing preprocessing, pruning and model.
"""
import copy
import hashlib
import importlib
import json
import numpy as np
from build_matrix import digest


def teacher_forced_nll(logits,labels,ignore_index=-100):
    """Causal shift, same answer token normalization on both branches."""
    import torch
    import torch.nn.functional as F
    if logits.ndim!=3 or labels.shape!=logits.shape[:2] or logits.shape[0]!=1:
        raise ValueError('single-sample causal logits/labels required')
    target=labels[:,1:].reshape(-1);valid=target!=ignore_index
    if not bool(valid.any()):raise ValueError('no labelled answer tokens')
    loss=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),target,
                         ignore_index=ignore_index,reduction='none')[valid].mean()
    ids=target[valid].detach().cpu().numpy().astype('<i8')
    return {'answer_nll':float(loss.detach().cpu()),'answer_tokens_hash':hashlib.sha256(ids.tobytes()).hexdigest(),
            'answer_token_count':len(ids)}


class ReplayAdapter:
    supports_identity_kv_purge=False
    kind='real'

    def __init__(self,backend,config):
        self.backend,self.config=backend,config
        self.source_hash=digest(config['source'])
        self.kind=getattr(backend,'kind','real')
        self.supports_identity_kv_purge=bool(getattr(backend,'supports_identity_kv_purge',False))

    def snapshots(self):
        policies=self.config.get('schedules',['FIRST_SHIFT','EARLY'])
        arms=self.config.get('arms',['COVER','BASE'])
        for number,example in enumerate(self.backend.examples()):
            if number>=self.config.get('max_examples',192):break
            policy=policies[number%len(policies)]
            for arm in arms:
                out=self.backend.run_teacher_forced(example,stage1=arm,schedule_name=policy,
                    first_keep_patch_ids=None,forced_final_ids=None,capture_prompt_state=True)
                s=dict(out['snapshot']);s.update(stage1=arm,schedule_name=policy,source_hash=self.source_hash,
                                               cache_policy=out['cache_policy'])
                s['_example']=example;s['_prefix_hash']=out['prefix_hash']
                s['_baseline_result']={k:v for k,v in out.items() if k!='snapshot'}
                yield s

    def clone(self,state):
        # Replays construct fresh model state. Inputs are required read-only;
        # copied metadata prevents one branch modifying the selected set.
        result={k:copy.deepcopy(v) for k,v in state.items() if k!='_example'}
        result['_example']=state['_example']
        return result

    def prefix_digest(self,state):return state['_prefix_hash']

    def run_suffix(self,state,keep_patch_ids,forced_final_ids=None):
        ids=np.asarray(state['patch_ids']);base=ids[np.asarray(state['selected'],dtype=int)]
        if forced_final_ids is None and np.array_equal(np.asarray(keep_patch_ids),base):
            return copy.deepcopy(state['_baseline_result'])
        out=self.backend.run_teacher_forced(state['_example'],stage1=state['stage1'],
                schedule_name=state['schedule_name'],first_keep_patch_ids=np.asarray(keep_patch_ids).tolist(),
                forced_final_ids=None if forced_final_ids is None else np.asarray(forced_final_ids).tolist(),
                capture_prompt_state=True)
        return {k:v for k,v in out.items() if k!='snapshot'}


def make(config):
    mod,factory=config['backend'].split(':')
    backend=getattr(importlib.import_module(mod),factory)(config.get('backend_config',{}))
    return ReplayAdapter(backend,config)
