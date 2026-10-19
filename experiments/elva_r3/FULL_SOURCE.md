# ELVA R3 完整源码展开稿

入口说明见 CURSOR_HANDOFF_CN.md。此稿逐字展开压缩包中所有Python源码；没有省略号替代函数。真实服务器接线与Torch/GPU验证边界见README及validation。

## bind_protected_cover.py

```python
"""Add the supplied independent K60 selector binding without claiming parity.

Run from the unpacked project directory so absolute source hashes refer to
the server files, not to the file-delivery workspace. No model is downloaded.
"""
import argparse
import json
from pathlib import Path
from elva.low_budget_cover import make_spec


def bind(input_path, output_path):
    value = json.loads(Path(input_path).read_text())
    covers = value.setdefault('native_cover',{})
    if 'protected_coverage' in covers and covers['protected_coverage'].get('implementation_parity_verified'):
        raise ValueError('verified binding already exists; preserve it or explicitly use a new binding file')
    covers['protected_coverage'] = make_spec()
    Path(output_path).write_text(json.dumps(value,indent=2)+'\n')
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();bind(a.binding,a.out)
    print('Source bound. implementation_parity_verified remains false until server NumPy/Torch and real CLIP preflight pass.')

```

## bound_task_runner.py

```python
"""CLI entrypoint for the existing source-bound VLM generation/scorer loop.

The project callback returns actual records and official metrics. No model,
dataset, cache semantics, native selector or score is guessed here.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
from elva.cover_binding import check_sources
from elva.workflow import validate_binding, blocked_reason
from native_result_io import finish_task


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','binding','task','out','run-key'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    binding=json.loads(Path(args.binding).read_text())
    validate_binding(binding)
    reason=blocked_reason(config,binding)
    if reason:raise RuntimeError(reason)
    project_root=binding.get('project_root')
    if project_root:
        sys.path.insert(0,str(Path(project_root).resolve()))
    entry=binding.get('task_entrypoint','')
    if not isinstance(entry,str) or entry.count(':')!=1:
        raise RuntimeError('bind actual project module:function task_entrypoint')
    module,name=entry.split(':')
    function=getattr(importlib.import_module(module),name)
    check_sources({'source_files':binding.get('task_source_files',{})},function)
    value=function(config=config,binding=binding,task=args.task,
                   run_key=args.run_key,output_dir=str(Path(args.out).parent))
    if not isinstance(value,dict) or set(('records','official_metrics'))-set(value):
        raise ValueError('project callback must return actual per-sample records and official_metrics')
    finish_task(args.out,records=value['records'],official_metrics=value['official_metrics'],
                config=config,binding=binding,task=args.task,run_key=args.run_key,fixture=False)


if __name__=='__main__':main()

```

## coverage_report.py

```python
"""Offline K60 coverage/saliency tradeoff plot from actual diagnostic rows.

Use elva.low_budget_cover.diagnostics on real saved metric/scores/selected
rows, then attach schema/identity/source fields. No ACC is inferred here.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from elva.low_budget_cover import POLICY


def summarize(rows, *, allow_fixture=False):
    if not rows or any(r.get('schema') != 'elva_r3_cover_audit' or type(r.get('fixture')) is not bool for r in rows):
        raise ValueError('actual diagnostic schema and explicit fixture flag required')
    if len({r['fixture'] for r in rows}) != 1 or (rows[0]['fixture'] and not allow_fixture):
        raise ValueError('fixture input requires --allow-fixture; never mix real and synthetic cases')
    signature=lambda r:(r.get('source_hash'),r.get('guide'),r.get('beta'),r.get('work'),r.get('policy'))
    if len({signature(r) for r in rows}) != 1 or rows[0].get('work') != 1024 or rows[0].get('policy') != POLICY:
        raise ValueError('one fixed guide/beta/source at W1024 required per comparison')
    if not rows[0].get('source_hash') or rows[0].get('guide') not in ('pen','l9','dual'):
        raise ValueError('real source identity and explicit guide required')
    groups={}
    for r in rows:
        b=r.get('slots');image=r.get('image_key')
        if (type(b) is not int or b not in (0,6,12,18) or not isinstance(image,str) or not image
                or not r.get('config_hash') or type(r.get('actual_changed')) is not int
                or not 0 <= r['actual_changed'] <= b or r.get('protected_count') != 60-b
                or r.get('candidate_count') != 576):
            raise ValueError('K60 protected/full-pool metadata and actual changed counts required')
        values=(r.get('kept_mass'),r.get('topk_mass'),r.get('kept_coverage',{}).get('weighted'),
                r.get('topk_coverage',{}).get('weighted'),r.get('kept_coverage',{}).get('uniform'))
        if any(type(v) not in (int,float) or not math.isfinite(v) or not -1e-5 <= v <= 1+1e-5 for v in values):
            raise ValueError('finite diagnostic masses/coverage in [0,1] required; exclude zero-saliency cases explicitly')
        group=groups.setdefault(b,{})
        if image in group:raise ValueError('image counted twice; aggregate crops with an explicit protocol first')
        group[image]=r
    image_sets=[set(g) for g in groups.values()]
    if len(groups)<2 or any(ids != image_sets[0] for ids in image_sets[1:]):
        raise ValueError('paired identical image panel and at least two slot levels required')
    result=[]
    for b,group in sorted(groups.items()):
        records=list(group.values())
        result.append(dict(slots=b,images=len(records),
            saliency_mass=float(np.mean([r['kept_mass'] for r in records])),
            saliency_mass_loss_vs_topk=float(np.mean([r['topk_mass']-r['kept_mass'] for r in records])),
            weighted_coverage=float(np.mean([r['kept_coverage']['weighted'] for r in records])),
            coverage_gain_vs_topk=float(np.mean([r['kept_coverage']['weighted']-r['topk_coverage']['weighted'] for r in records])),
            uniform_coverage=float(np.mean([r['kept_coverage']['uniform'] for r in records])),
            actual_changed=float(np.mean([r['actual_changed'] for r in records])),
            zero_change_images=sum(r['actual_changed']==0 for r in records)))
    return result


def render(rows, output, *, allow_fixture=False):
    values=summarize(rows,allow_fixture=allow_fixture)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    with (out/'coverage_tradeoff.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    x=[r['slots'] for r in values]
    for ax,key,title in zip(axes,('saliency_mass_loss_vs_topk','coverage_gain_vs_topk','actual_changed'),
                            ('Saliency mass loss vs Top60','Feature coverage gain vs Top60','Actual additions surviving at K60')):
        ax.plot(x,[r[key] for r in values],marker='o',color='#175f91')
        ax.set(xlabel='Coverage slots b',title=title,xticks=x);ax.grid(alpha=.2)
    if rows[0]['fixture']:
        fig.suptitle('SYNTHETIC FIXTURE — NOT VLM RESULTS',fontsize=10)
        fig.text(.5,.5,'SYNTHETIC FIXTURE — NOT VLM RESULTS',ha='center',rotation=15,alpha=.25,fontsize=22)
    fig.savefig(out/'coverage_tradeoff.png',dpi=170);fig.savefig(out/'coverage_tradeoff.svg');plt.close(fig)
    (out/'summary.json').write_text(json.dumps(dict(fixture=rows[0]['fixture'],rows=values),indent=2)+'\n')
    return values


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',required=True);p.add_argument('--out',required=True)
    p.add_argument('--allow-fixture',action='store_true')
    a=p.parse_args()
    rows=[json.loads(line) for line in Path(a.records).read_text().splitlines() if line.strip()]
    render(rows,a.out,allow_fixture=a.allow_fixture)

```

## elva/__init__.py

```python
"""ELVA R2: CPU reference and source-bound VLM integration contracts."""
from .budget import Schedule, Trace, feasible_suffixes, choose_suffix


```

## elva/benchmark.py

```python
"""Paired end-to-end FIRST TOKEN timing; real VLM is a bound callback.

Input preparation belongs inside run_request. Dedicated measurement device.
No CUDA-event-only result is relabeled as user-observable wall TTFT.
"""
import math
import random
import time
import numpy as np


def paired_ttft(samples, methods, run_request, synchronize, *, rounds=3, warmup=3, seed=19):
    if len(methods) != 2 or methods[0] == methods[1] or not samples or type(rounds) is not int or rounds < 1:
        raise ValueError("two distinct methods, samples, positive rounds required")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be nonnegative")
    for i in range(warmup):
        for method in methods:
            run_request(samples[i % len(samples)], method)
    synchronize()
    rng, rows = random.Random(seed), []
    for repeat in range(rounds):
        order = list(range(len(samples)))
        rng.shuffle(order)
        for i in order:
            # Balanced order over rounds/input order, avoiding all-A then all-B.
            local = methods if (i+repeat) % 2 == 0 else methods[::-1]
            for method in local:
                synchronize()
                start = time.perf_counter_ns()
                first_token = run_request(samples[i], method)
                synchronize()
                elapsed = (time.perf_counter_ns()-start)/1e6
                if first_token is None or not math.isfinite(elapsed) or elapsed <= 0:
                    raise ValueError("callback must emit an actual first token")
                rows.append(dict(sample=i, repeat=repeat, method=method, wall_ttft_ms=elapsed,
                                 first_token=str(first_token)))
    return rows


def summarize(rows):
    output = {}
    for method in sorted(set(row["method"] for row in rows)):
        values = np.array([row["wall_ttft_ms"] for row in rows if row["method"] == method], float)
        if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("invalid timing rows")
        output[method] = dict(mean_ms=float(values.mean()), p50_ms=float(np.median(values)),
                              p90_ms=float(np.quantile(values, .9)), observations=len(values))
    return output

```

## elva/budget.py

```python
"""Exact decoder token-layer budgets; events occur AFTER completed blocks.

This module does not estimate task utility or latency. A suffix predictor
must be calibrated for the exact suffix catalog used at deployment.
"""
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
import math


def integer(x, name, minimum=0):
    if isinstance(x, bool) or not isinstance(x, Integral) or x < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(x)


@dataclass(frozen=True)
class Schedule:
    counts: tuple
    lengths: tuple

    def __post_init__(self):
        counts = tuple(integer(k, "count", 1) for k in self.counts)
        lengths = tuple(integer(d, "length", 1) for d in self.lengths)
        if not counts or len(counts) != len(lengths):
            raise ValueError("one positive length per count required")
        if any(a <= b for a, b in zip(counts, counts[1:])):
            raise ValueError("counts must strictly decrease; merge equal stages")
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "lengths", lengths)

    @property
    def depth(self):
        return sum(self.lengths)

    @property
    def trace(self):
        return tuple(k for k, d in zip(self.counts, self.lengths) for _ in range(d))

    @property
    def work(self):
        return sum(k * d for k, d in zip(self.counts, self.lengths))

    @property
    def events(self):
        completed, result = 0, []
        for d, k in zip(self.lengths[:-1], self.counts[1:]):
            completed += d
            result.append((completed, k))
        return tuple(result)

    def compression(self, original_tokens):
        n = integer(original_tokens, "original_tokens", 1)
        if self.counts[0] > n:
            raise ValueError("entry exceeds original token count")
        return 1 - Fraction(self.work, n * self.depth)

    def validate_trace(self, actual):
        actual = tuple(integer(k, "trace count", 1) for k in actual)
        if actual != self.trace:
            raise ValueError("actual pre-block visual counts disagree with schedule")

    def proxies(self, text_tokens):
        t = integer(text_tokens, "text_tokens", 0)
        # Dense causal pair count. Protected structural tokens belong in t.
        pairs = sum((t+k)*(t+k+1)//2 for k in self.trace)
        base_pairs = self.depth * t*(t+1)//2
        return dict(token_layers=self.work, visual_square=sum(k*k for k in self.trace),
                    incremental_causal_pairs=pairs-base_pairs)


@dataclass
class Trace:
    """Record counts BEFORE each block, never counts after a pruning hook."""
    budget: int
    depth: int = 32

    def __post_init__(self):
        self.budget = integer(self.budget, "budget", 1)
        self.depth = integer(self.depth, "depth", 1)
        self.counts = []

    @property
    def spent(self):
        return sum(self.counts)

    def before_block(self, block, visual_count):
        block = integer(block, "block", 1)
        k = integer(visual_count, "visual_count", 1)
        if block != len(self.counts)+1 or block > self.depth:
            raise ValueError("blocks must be recorded once in execution order")
        if self.counts and k > self.counts[-1]:
            raise ValueError("deleted tokens cannot reappear")
        if self.spent+k > self.budget:
            raise ValueError("hard token-layer budget exceeded")
        self.counts.append(k)

    def finish(self):
        if len(self.counts) != self.depth or self.spent != self.budget:
            raise ValueError("incomplete trace or exact budget not met")
        return tuple(self.counts)


def feasible_suffixes(*, depth, completed, spent, current_k, budget,
                      boundaries, finals):
    """Enumerate every exact TWO-STAGE continuation in this finite catalog.

    Prefix is already executed. No token resurrection, hidden borrowing,
    or test-batch budget balancing. No global-optimality claim is made.
    """
    depth = integer(depth, "depth", 1)
    completed = integer(completed, "completed")
    spent = integer(spent, "spent")
    current_k = integer(current_k, "current_k", 1)
    budget = integer(budget, "budget", 1)
    if completed >= depth or spent >= budget:
        raise ValueError("no unspent continuation")
    result = set()
    for boundary in boundaries:
        b = integer(boundary, "boundary", 1)
        if not completed < b < depth:
            raise ValueError("boundary must be inside the remaining depth")
        d1, d2 = b-completed, depth-b
        for final in finals:
            f = integer(final, "final", 1)
            numerator = budget-spent-f*d2
            if numerator % d1:
                continue
            middle = numerator//d1
            if f < middle <= current_k:
                result.add(Schedule((middle, f), (d1, d2)))
    return tuple(sorted(result, key=lambda p: (p.lengths, p.counts)))


def choose_suffix(catalog, predicted_gains, *, anchor=0, threshold=0.):
    """Single-path choice; zero/negative/uncertain gains keep the anchor.

    Gains must be relative to anchor, from a suffix-action estimator.
    A group-survival estimator is NOT a valid substitute.
    """
    if not catalog or len(catalog) != len(predicted_gains):
        raise ValueError("one prediction per feasible suffix required")
    anchor = integer(anchor, "anchor")
    if anchor >= len(catalog) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("invalid anchor/threshold")
    if len({(p.work, p.depth) for p in catalog}) != 1:
        raise ValueError("suffixes must have identical remaining work and depth")
    gains = tuple(float(v) for v in predicted_gains)
    if not all(math.isfinite(v) for v in gains) or gains[anchor] != 0.:
        raise ValueError("finite, anchor-relative predictions required")
    best = max(range(len(gains)), key=lambda i: gains[i])
    return anchor if gains[best] <= threshold else best


```

## elva/capture.py

```python
"""Single-forward layer capture using native module hooks, without attentions.

Pass the actual source-bound ViT block list. Completed block 9 -> blocks[8].
For a D-block CLIP, hidden_states[-2] -> output of completed block D-1.
This is a capture utility, not a model-specific architecture resolver.
"""
from .budget import integer


class LayerCapture:
    def __init__(self, blocks, *, early_completed=9, pen_completed=None):
        self.blocks = list(blocks)
        self.early = integer(early_completed, "early_completed", 1)
        self.pen = len(self.blocks)-1 if pen_completed is None else integer(pen_completed, "pen_completed", 1)
        if not self.blocks or not 1 <= self.early <= len(self.blocks) or not 1 <= self.pen <= len(self.blocks):
            raise ValueError("requested layer is outside encoder")
        self.values, self.handles = {}, []

    def __enter__(self):
        if self.handles:
            raise RuntimeError("capture already attached")
        self.values.clear()
        def hook_for(completed):
            def hook(module, args, output):
                value = output[0] if isinstance(output, (tuple, list)) else output
                if completed in self.values:
                    raise RuntimeError("unexpected repeated forward; clear capture per request")
                # Clone only the requested layers; record its cost in TTFT.
                self.values[completed] = value.detach().clone() if hasattr(value, "detach") else value.copy()
            return hook
        for completed in sorted({self.early, self.pen}):
            self.handles.append(self.blocks[completed-1].register_forward_hook(hook_for(completed)))
        return self

    def get(self):
        if not {self.early, self.pen}.issubset(self.values):
            raise RuntimeError("required layer did not execute")
        return self.values[self.early], self.values[self.pen]

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

```

## elva/clip_stage1.py

```python
"""Actual Stage1 hook for unfused CLIP q_proj/k_proj encoder blocks.

Wrap the existing vision_tower call, select raw pen rows, then call the real
projector. No second CLIP forward, dense attention output, model download,
or backend replacement. This guarded adapter is intentionally not an
architecture auto-detector; bind the actual encoder block list on server.
"""
import math
from .cover_binding import select_checked


class CLIPStage1:
    """One image / one crop / one native CLIP forward per instance.

    Layer numbers mean COMPLETED blocks. With 24 blocks, pen is 23 and L9
    is 9. Score block numbers are separate and explicit: historical final
    attention can therefore be preserved while payload stays at pen.
    Score definition is mean-head CLS key-softmax, then patch-L1. Verify it
    against the existing saliency rule before identifying it as a baseline.
    Fused QKV, non-CLIP positional transforms, masked/padded vision inputs,
    training, batching and anyres concatenation are rejected or unsupported.
    """
    def __init__(self, blocks, config, *, cover=None, pen_completed=None,
                 pen_score_completed=None, l9_completed=9, l9_score_completed=None):
        self.blocks, self.config, self.cover = list(blocks), config, cover
        self.pen = len(self.blocks)-1 if pen_completed is None else pen_completed
        self.early = l9_completed
        self.score_blocks = {'pen': self.pen if pen_score_completed is None else pen_score_completed,
                             'l9': self.early if l9_score_completed is None else l9_score_completed}
        stage = config['method']['stage1']
        sg = stage['guide'] if stage.get('score_guide','same') == 'same' else stage['score_guide']
        beta = stage.get('beta') or 0.
        self.need_early = any(g == 'l9' or (g == 'dual' and beta > 0) for g in (stage['guide'],sg))
        self.feature_blocks = {self.pen} | ({self.early} if self.need_early else set())
        self.needed_scores = {'pen'} | ({'l9'} if self.need_early else set())
        required = self.feature_blocks | {self.score_blocks[n] for n in self.needed_scores}
        if any(type(i) is not int or not 1 <= i <= len(self.blocks) for i in required):
            raise ValueError('explicit completed CLIP block numbers required')
        self.handles, self.features, self.projections = [], {}, {}
        self.used = False

    def __enter__(self):
        import torch
        if self.handles or self.used:
            raise RuntimeError('create a fresh CLIPStage1 for each native vision forward')
        if any(block.training for block in self.blocks):
            raise RuntimeError('evaluation CLIP required; training/dropout is not supported')
        def feature_hook(layer):
            def capture(module,args,output):
                if layer in self.features:
                    raise RuntimeError('multiple forwards in one CLIPStage1 instance')
                value = output[0] if isinstance(output,(tuple,list)) else output
                if not isinstance(value,torch.Tensor) or value.ndim != 3 or value.shape[:2] != (1,577):
                    raise ValueError('expected one CLS plus 576 patches in one image/crop')
                self.features[layer] = value.detach().clone()
            return capture
        def projection_hook(layer,kind):
            def capture(module,args,output):
                key = (layer,kind)
                if key in self.projections:
                    raise RuntimeError('repeated q/k projection; bind an unfused native CLIP path')
                if not isinstance(output,torch.Tensor) or output.ndim != 3 or output.shape[:2] != (1,577):
                    raise ValueError('native CLIP projection layout is not [1,577,D]')
                self.projections[key] = (output[:,:1] if kind == 'q' else output).detach().clone()
            return capture
        def mask_check(module,args,kwargs):
            masks = [kwargs.get('attention_mask'),kwargs.get('causal_attention_mask')]
            masks += list(args[1:3])
            if any(m is not None for m in masks):
                raise ValueError('masked vision attention unsupported; bind native saliency explicitly')
        try:
            for layer in sorted(self.feature_blocks):
                self.handles.append(self.blocks[layer-1].register_forward_hook(feature_hook(layer)))
            for layer in sorted({self.score_blocks[n] for n in self.needed_scores}):
                attn = self.blocks[layer-1].self_attn
                if not hasattr(attn,'q_proj') or not hasattr(attn,'k_proj') or not hasattr(attn,'num_heads'):
                    raise ValueError('explicit unfused CLIP q_proj/k_proj blocks required')
                if getattr(attn,'q_norm',None) is not None or getattr(attn,'k_norm',None) is not None:
                    raise ValueError('post-projection q/k transforms require a source-specific adapter')
                self.handles.append(attn.register_forward_pre_hook(mask_check,with_kwargs=True))
                for kind in ('q','k'):
                    self.handles.append(getattr(attn,kind+'_proj').register_forward_hook(projection_hook(layer,kind)))
            return self
        except BaseException:
            self.__exit__()
            raise

    def _scores(self,name):
        import torch
        layer = self.score_blocks[name]
        if (layer,'q') not in self.projections or (layer,'k') not in self.projections:
            raise RuntimeError('q/k hook did not execute; no silent score substitution')
        attn = self.blocks[layer-1].self_attn
        q,k = self.projections[(layer,'q')],self.projections[(layer,'k')]
        heads = attn.num_heads
        if type(heads) is not int or heads < 1 or q.shape[-1] % heads or k.shape[-1] != q.shape[-1]:
            raise ValueError('invalid native CLIP head layout')
        d = q.shape[-1]//heads
        scale = getattr(attn,'scale',getattr(attn,'scaling',d**-.5))
        if not isinstance(scale,(int,float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError('source-bound scalar CLIP QK scale required')
        q = q.float().reshape(1,1,heads,d).transpose(1,2)
        k = k.float().reshape(1,577,heads,d).transpose(1,2)
        p = ((q*scale) @ k.transpose(-1,-2)).softmax(-1)
        s = p[0,:,0,1:].mean(0)
        return s/s.sum().clamp_min(torch.finfo(s.dtype).tiny)

    def select(self, native_pen_payload, original_patch_ids, *, context=None):
        import torch
        if self.used or not self.handles:
            raise RuntimeError('select exactly once inside the capture context after the native forward')
        if not self.feature_blocks.issubset(self.features):
            raise RuntimeError('the actual native vision forward has not produced required layers')
        pen = self.features[self.pen][0,1:]
        expected = native_pen_payload
        if expected.ndim == 3 and expected.shape[0] == 1:
            expected = expected[0]
        if (expected.shape != pen.shape or expected.device != pen.device
                or not torch.equal(expected,pen.to(dtype=expected.dtype))):
            raise ValueError('native projector payload is not the captured raw pen; inspect source before proceeding')
        # Preserve the native vision wrapper's output dtype, including its
        # existing final cast back to the image tensor dtype.
        pen = expected
        early = self.features[self.early][0,1:] if self.need_early else None
        output = select_checked(pen,self._scores('pen'),original_patch_ids,self.config,
                                early=early,early_scores=self._scores('l9') if self.need_early else None,
                                early_ids=original_patch_ids if self.need_early else None,
                                native_cover=self.cover,context=context,backend='torch')
        output['capture_meta'] = dict(feature_completed_pen=self.pen,feature_completed_l9=self.early,
                                      score_completed=self.score_blocks,early_captured=self.need_early,
                                      score_rule='mean_heads_cls_key_softmax_then_patch_l1',
                                      vision_forwards=1)
        self.used = True
        return output

    def __exit__(self,*args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.features.clear()
        self.projections.clear()


def encode_projected(*, vision_tower, projector, images, blocks, original_patch_ids,
                     config, cover=None, context=None, **layer_numbers):
    """Replace the two lines vision_tower(images) -> projector(features).

    vision_tower must be the EXISTING evaluation wrapper returning raw pen
    [1,576,D] (or [576,D]). The actual processor and remaining VLM are unchanged.
    Returns projected [1,K,D_text] (or [K,D_text]) plus selection/audit tensors.
    The caller passes these projected tokens to its real decoder path.
    """
    import torch
    with torch.no_grad(), CLIPStage1(blocks,config,cover=cover,**layer_numbers) as capture:
        payload = vision_tower(images)
        if not isinstance(payload,torch.Tensor):
            raise ValueError('bind existing vision wrapper returning the original pen payload tensor')
        selected = capture.select(payload,original_patch_ids,context=context)
        input_to_projector = selected['payload'].unsqueeze(0) if payload.ndim == 3 else selected['payload']
        projected = projector(input_to_projector)
    return projected,selected

```

## elva/counterfactual.py

```python
"""True temporary-survival swaps, with native and support-aligned cache modes.

Model-specific execution belongs to a source-bound adapter, never a guessed
monkey patch. Both arms remove the compared groups before the shared suffix.
"""
from dataclasses import dataclass
import math
import numpy as np
from .budget import integer
from .evidence import ids_array


ESTIMAND = "transient_survival_swap_v1"


@dataclass(frozen=True)
class Swap:
    begin: int                 # completed blocks at the shared prefix
    end: int                   # completed blocks when both groups disappear
    common: tuple              # shared support DURING the window
    group_a: tuple
    group_b: tuple
    final: tuple               # identical support AFTER the window

    def __post_init__(self):
        begin, end = integer(self.begin, "begin"), integer(self.end, "end", 1)
        if end <= begin:
            raise ValueError("positive survival window required")
        for name in ("common", "group_a", "group_b", "final"):
            object.__setattr__(self, name, tuple(map(int, ids_array(getattr(self, name)))))
        c, a, b, f = map(set, (self.common, self.group_a, self.group_b, self.final))
        if c&a or c&b or a&b or len(a) != len(b) or not f.issubset(c):
            raise ValueError("disjoint equal groups and final subset of common required")

    @property
    def window_k(self):
        return len(self.common)+len(self.group_a)


def _native_equal(value_a, value_b, message):
    if value_a != value_b:
        raise ValueError(message)


def _validate_view(view, swap, prefix_view):
    required = {"prefix_fingerprint", "completed", "active_ids", "kv_support",
                "protected_positions", "visual_positions"}
    if not required.issubset(view):
        raise ValueError("adapter inspection missing mandatory fields")
    _native_equal(view["prefix_fingerprint"], prefix_view["prefix_fingerprint"], "prefix identity changed")
    _native_equal(view["protected_positions"], prefix_view["protected_positions"], "text/structural positions changed")
    expected_positions = {i: prefix_view["visual_positions"][i] for i in swap.final}
    _native_equal(view["visual_positions"], expected_positions, "original visual positions changed")
    _native_equal(view["completed"], swap.end, "wrong join boundary")
    _native_equal(tuple(sorted(view["active_ids"])), tuple(sorted(swap.final)), "final identity mismatch")


def run_swap(adapter, prefix, swap, *, image_key, question_key, mode="native"):
    """Adapter methods: inspect/fork/prune/advance/align_cache/score_suffix.

    prefix contains prompt-only states. score_suffix computes answer-only
    token-mean NLL and real generation using labels ONLY after decisions.
    Native labels include access to earlier-layer KV during later decoding.
    Aligned-support labels remove extra visual KVs from every computed layer;
    surviving text/visual values intentionally remain branch-specific.
    """
    if mode not in ("native", "aligned_support"):
        raise ValueError("invalid cache intervention")
    before = adapter.inspect(prefix)
    if before["completed"] != swap.begin:
        raise ValueError("prefix is not at the intervention boundary")
    active = set(before["active_ids"])
    if not set(swap.common+swap.group_a+swap.group_b).issubset(active):
        raise ValueError("group contains deleted/unknown token")
    prefix_hash = before["prefix_fingerprint"]
    arms, views = [], []
    for group in (swap.group_a, swap.group_b):
        state = adapter.fork(prefix)
        if state is prefix:
            raise ValueError("fork aliases shared prefix")
        adapter.prune(state, tuple(sorted(swap.common+group)))
        adapter.advance(state, swap.end)
        adapter.prune(state, tuple(sorted(swap.final)))
        if mode == "aligned_support":
            adapter.align_cache(state, tuple(sorted(swap.final)))
        view = adapter.inspect(state)
        _validate_view(view, swap, before)
        views.append(view)
        outcome = adapter.score_suffix(state)
        if not {"nll", "answer_tokens", "trace", "backend", "generated"}.issubset(outcome):
            raise ValueError("incomplete real suffix outcome")
        if not math.isfinite(outcome["nll"]) or outcome["nll"] < 0 or outcome["answer_tokens"] < 1:
            raise ValueError("invalid answer-only NLL")
        arms.append(outcome)
        if adapter.inspect(prefix) != before:
            raise ValueError("branch mutated shared prefix; deep-copy cache/metadata")
    _native_equal(arms[0]["trace"], arms[1]["trace"], "not an equal-work swap")
    _native_equal(arms[0]["backend"], arms[1]["backend"], "attention backend differs")
    _native_equal(arms[0]["answer_tokens"], arms[1]["answer_tokens"], "answer normalization differs")
    if mode == "aligned_support":
        _native_equal(views[0]["kv_support"], views[1]["kv_support"], "cache supports differ at join")
        expected = set(swap.final)
        if any(set(layer) != expected for layer in views[0]["kv_support"]):
            raise ValueError("extra/missing visual cache IDs after support alignment")
    trace = tuple(arms[0]["trace"])
    if len(trace) < swap.end or trace[swap.begin:swap.end] != (swap.window_k,)*(swap.end-swap.begin):
        raise ValueError("adapter trace does not execute the intended survival window")
    return dict(schema=ESTIMAND, image_key=str(image_key), question_key=str(question_key),
                cache_mode=mode, prefix_fingerprint=prefix_hash,
                begin=swap.begin, end=swap.end, final_ids=list(swap.final),
                common=list(swap.common), group_a=list(swap.group_a), group_b=list(swap.group_b),
                work=sum(trace), trace=list(trace), nll_a=arms[0]["nll"], nll_b=arms[1]["nll"],
                delta=arms[1]["nll"]-arms[0]["nll"],
                generated_a=arms[0]["generated"], generated_b=arms[1]["generated"],
                backend=arms[0]["backend"],
                note="positive delta favors group_a; offline label, not online input")


def answer_nll(logits, labels, answer_mask):
    """Standard CAUSAL shift: logits[t] predicts labels[t+1]. No fake EOS.

    Single full-sequence CPU reference. When prompt is compacted, routing
    and masks must refer to the compacted sequence. Token-mean loss only.
    """
    z = np.asarray(logits, dtype=float)
    y, mask = np.asarray(labels), np.asarray(answer_mask)
    if z.ndim != 2 or len(z) < 2 or y.shape != (len(z),) or mask.shape != y.shape:
        raise ValueError("aligned [T,V] logits and [T] labels/mask required")
    if y.dtype.kind not in "iu" or mask.dtype != np.bool_ or not np.isfinite(z).all():
        raise ValueError("integer labels, boolean answer mask and finite logits required")
    valid = mask[1:] & (y[1:] != -100)
    if not valid.any():
        raise ValueError("no answer targets")
    target = y[1:][valid]
    if target.min() < 0 or target.max() >= z.shape[1]:
        raise ValueError("answer label outside vocabulary")
    values = z[:-1][valid]
    shift = values.max(-1)
    lse = shift+np.log(np.exp(values-shift[:, None]).sum(-1))
    return float((lse-values[np.arange(len(target)), target]).mean())


```

## elva/cover_binding.py

```python
"""Source-bound COVER with explicit capacity and no top-k fallback.

The real project supplies one thin callback with the standard keyword
signature. This module never invents the historical COVER formula.
"""
import hashlib
import importlib
import inspect
from pathlib import Path
import numpy as np
from .budget import integer
from .evidence import indices


FAMILIES = ("legacy_native", "parameterized_native", "legacy_pool172_then_topk", "protected_coverage")


def origin(family):
    return ("r3_protected_control" if family == "protected_coverage" else
            "none" if family == "none" else "historical_source_bound")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spec_fingerprint(spec):
    sources = spec.get("source_files", {})
    if not sources or not isinstance(sources, dict):
        raise RuntimeError("BLOCKED_SOURCE: bind original source files and the thin callback")
    actual = {str(Path(path).resolve()): expected for path, expected in sources.items()}
    for path, expected in actual.items():
        if not isinstance(expected, str) or len(expected) != 64 or file_sha(path) != expected:
            raise RuntimeError("BLOCKED_SOURCE: changed or invalid COVER source fingerprint")
    return hashlib.sha256(repr(sorted(actual.items())).encode()).hexdigest()


def check_sources(spec, callback):
    fingerprint = spec_fingerprint(spec)
    source = inspect.getsourcefile(callback)
    files = {str(Path(path).resolve()) for path in spec["source_files"]}
    if source is None or str(Path(source).resolve()) not in files:
        raise RuntimeError("BLOCKED_SOURCE: callback itself must be included in source_files")
    return fingerprint


def load_cover(spec):
    entry = spec.get("entrypoint", "")
    if not isinstance(entry, str) or entry.count(":") != 1:
        raise RuntimeError("BLOCKED_SOURCE: set a real module:function COVER entrypoint")
    module, name = entry.split(":")
    callback = getattr(importlib.import_module(module), name)
    return NativeCover(callback, spec)


def checked_keep(value, n, k):
    if isinstance(value, np.ndarray):
        return indices(value, n, k)
    if not hasattr(value, "device"):
        raise ValueError("native COVER must return NumPy or PyTorch local indices, not a list")
    import torch
    if (not isinstance(value, torch.Tensor) or value.shape != (k,) or value.dtype != torch.long
            or value.unique().numel() != k or int(value.min()) < 0 or int(value.max()) >= n):
        raise ValueError("native COVER must return unique device-local long active-row indices")
    return value


class NativeCover:
    def __init__(self, callback, spec):
        self.callback, self.spec = callback, dict(spec)
        if spec.get("family") not in FAMILIES:
            raise ValueError("explicit COVER implementation family required")
        self.family = spec["family"]
        self.supported = tuple(integer(k, "supported K", 1) for k in spec.get("supported_k", ()))
        if not self.supported or len(set(self.supported)) != len(self.supported):
            raise ValueError("declare unique verified supported K values")
        if self.family == "legacy_native" and self.supported != (172,):
            raise ValueError("unmodified historical COVER is declared at K172 only")
        if self.family == "legacy_pool172_then_topk" and any(k >= 172 for k in self.supported):
            raise ValueError("pooled route is an explicitly separate smaller-K method")
        self.source_sha = check_sources(spec, callback)
        self.events = []
        self.last_native_keep = self.last_requested_keep = None

    def __call__(self, *, metric_features, scores, patch_ids, k, context=None):
        k = integer(k, "requested K", 1)
        if k not in self.supported:
            raise RuntimeError(f"BLOCKED_CAPACITY: {self.family} has no verified K={k} implementation")
        n = len(scores)
        if len(metric_features) != n or len(patch_ids) != n or k > n:
            raise ValueError("aligned full active candidate pool required")
        native_k = 172 if self.family == "legacy_pool172_then_topk" else k
        if native_k > n:
            raise ValueError("native pool exceeds current candidate universe")
        slots = self.spec.get("replacement_slots", {}).get(str(native_k))
        local_context = dict(context or {})
        if self.family == "protected_coverage":
            from .low_budget_cover import POLICY
            slots = local_context.get("coverage_slots", slots)
            if slots not in self.spec.get("allowed_slots", ()) or self.spec.get("coverage_policy") != POLICY:
                raise RuntimeError("BLOCKED_SOURCE: unverified protected coverage configuration")
            local_context["coverage_policy"] = POLICY
        if type(slots) is not int or not 0 <= slots <= native_k:
            raise RuntimeError("BLOCKED_SOURCE: bind actual COVER replacement slots for this native K")
        local_context.update(cover_family=self.family, requested_k=k, native_k=native_k,
                             replacement_slots=slots)
        keep = self.callback(metric_features=metric_features, scores=scores,
                             patch_ids=patch_ids, k=native_k, context=local_context)
        keep = checked_keep(keep, n, native_k)
        self.last_native_keep = keep
        if self.family == "legacy_pool172_then_topk":
            if isinstance(keep, np.ndarray):
                keep = keep[np.argsort(-scores[keep], kind="stable")[:k]]
                keep = np.sort(keep)
            else:
                import torch
                chosen = torch.argsort(scores.index_select(0, keep), descending=True, stable=True)[:k]
                keep = keep.index_select(0, chosen).sort().values
        keep = checked_keep(keep, n, k)
        self.last_requested_keep = keep
        self.events.append(dict(family=self.family, source_sha=self.source_sha,
                                requested_k=k, native_k=native_k, replacement_slots=slots,
                                cover_calls=1, native_calls=int(self.family != "protected_coverage"),
                                origin=origin(self.family), fallback=False, candidate_count=n))
        return keep


def select_checked(pen, pen_scores, ids, config, *, early=None, early_scores=None,
                   early_ids=None, native_cover=None, context=None, backend="numpy"):
    """Actual selection using R2 kernels, with an enforceable COVER call audit.

    Keep the audit per sample. Export tensors/IDs only outside timed paths.
    No automatic native->pooled or COVER->top-k conversion is permitted.
    """
    stage = config["method"]["stage1"]
    k = config["method"]["schedule"]["counts"][0]
    family = stage["cover_family"]
    covered = family != "none"
    if covered and (native_cover is None or native_cover.family != family):
        raise RuntimeError("BLOCKED_SOURCE: required original COVER family is not bound")
    previous = len(native_cover.events) if native_cover is not None else 0
    if backend == "numpy":
        from .evidence import select_stage1
    elif backend == "torch":
        from .torch_ops import select_stage1
    else:
        raise ValueError("backend must be numpy or torch")
    local_context = dict(context or {})
    if covered and family == "protected_coverage":
        from .low_budget_cover import POLICY
        if stage.get("coverage_policy") != POLICY or type(stage.get("coverage_slots")) is not int:
            raise ValueError("explicit protected coverage slots and policy required in method fingerprint")
        local_context["coverage_slots"] = stage["coverage_slots"]
    kwargs = dict(guide=stage["guide"], beta=stage.get("beta") or 0., early=early,
                  early_scores=early_scores, early_ids=early_ids,
                  use_cover=covered, cover=native_cover if covered else None,
                  context=local_context, score_guide=stage.get("score_guide", "same"))
    result = select_stage1(pen, pen_scores, ids, k, **kwargs)
    events = native_cover.events[previous:] if native_cover is not None else []
    if len(events) != int(covered):
        raise RuntimeError("COVER native callback count disagrees with the requested method")
    result["stage1_audit"] = dict(cover_family=family, requested_k=k, cover_calls=len(events),
                                 native_calls=sum(e["native_calls"] for e in events), origin=origin(family),
                                 cover_source_sha=native_cover.source_sha if covered else None,
                                 native_k=events[0]["native_k"] if covered else None,
                                 replacement_slots=events[0]["replacement_slots"] if covered else 0,
                                 fallback=False, payload="pen")
    return result

```

## elva/evidence.py

```python
"""Identity-safe Stage1 selection and attention/rank reference functions.

COVER/QED are intentionally NOT approximated. Bind original source callbacks.
NumPy reference only; moving CUDA tensors to CPU is not a deployment path.
"""
import numpy as np


def indices(value, n, size=None):
    a = np.asarray(value)
    if (a.ndim != 1 or a.dtype.kind not in "iu" or len(a) == 0
            or len(np.unique(a)) != len(a) or a.min() < 0 or a.max() >= n
            or (size is not None and len(a) != size)):
        raise ValueError("unique nonempty integer active-local indices required")
    return a.astype(np.int64, copy=False)


def ids_array(value):
    a = np.asarray(value)
    if a.ndim != 1 or a.dtype.kind not in "iu" or len(a) == 0 or len(np.unique(a)) != len(a) or a.min() < 0:
        raise ValueError("unique nonnegative original IDs required")
    return a.astype(np.int64, copy=False)


def finite(value, ndim):
    a = np.asarray(value)
    if a.ndim != ndim or min(a.shape) == 0 or not np.isfinite(a).all():
        raise ValueError("finite nonempty array required")
    return a


def unit(x):
    x = finite(x, 2).astype(np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x/np.maximum(norm, 1e-12)


def topk(scores, k):
    s = finite(scores, 1)
    if type(k) is not int or not 0 < k <= len(s):
        raise ValueError("invalid K")
    return np.sort(np.argsort(-s, kind="stable")[:k])


def select_stage1(pen, pen_scores, ids, k, *, guide="pen", early=None,
                  early_scores=None, early_ids=None, beta=.5, cover=None,
                  use_cover=False, context=None, frozen_keep=None, score_guide="same"):
    """Guidance changes selection only. Payload is ALWAYS raw pen[keep].

    Layer scores must use the same bound saliency rule and normalization.
    Patch IDs include image/crop identity. Returned local keep is spatially
    ordered; projector input dimension and vector values are unchanged.
    """
    x, a, pid = finite(pen, 2), finite(pen_scores, 1), ids_array(ids)
    if x.dtype.kind != "f" or a.shape != (len(x),) or len(pid) != len(x) or (a < 0).any():
        raise ValueError("floating payload and aligned nonnegative saliency required")
    if type(k) is not int or not 0 < k <= len(x):
        raise ValueError("invalid K")
    if guide not in ("pen", "l9", "dual") or not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("invalid guidance")
    sg = guide if score_guide == "same" else score_guide
    if sg not in ("pen", "l9", "dual"):
        raise ValueError("invalid score guide")
    needs_early = (guide == "l9" or (guide == "dual" and beta > 0)
                   or sg == "l9" or (sg == "dual" and beta > 0))
    if needs_early:
        e, b = finite(early, 2), finite(early_scores, 1)
        eid = ids_array(early_ids)
        if e.shape != x.shape or b.shape != a.shape or not np.array_equal(pid, eid) or (b < 0).any():
            raise ValueError("cross-layer feature/score/ID alignment failed")
    if guide == "pen" or (guide == "dual" and beta == 0):
        metric, scores = x, a
    elif guide == "l9" or beta == 1:
        metric, scores = e, b
    else:
        metric = np.concatenate((np.sqrt(1-beta)*unit(x), np.sqrt(beta)*unit(e)), axis=1)
    scores = a if sg == "pen" or (sg == "dual" and beta == 0) else (
        b if sg == "l9" or beta == 1 else (1-beta)*a+beta*b)
    if frozen_keep is not None:
        keep = indices(frozen_keep, len(x), k)
    elif use_cover:
        if cover is None:
            raise RuntimeError("BLOCKED_SOURCE: bind original COVER")
        keep = indices(cover(metric_features=metric.copy(), scores=scores.copy(),
                             patch_ids=pid.copy(), k=k, context=context), len(x), k)
    else:
        keep = topk(scores, k)
    keep = np.sort(keep)
    return dict(payload=x[keep].copy(), keep=keep, ids=pid[keep].copy(),
                metric=metric.copy(), scores=scores.copy())


def question_visual_mass(logits, allowed, visual_columns):
    """Partial question QK only, AFTER RoPE/scaling; [Q,T] or [H,Q,T].

    Denominator covers ALL causal-visible keys; answer rows are forbidden
    by the caller's prompt-only query routing, not guessed here.
    """
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim not in (2, 3) or min(z.shape) == 0:
        raise ValueError("logits must be [Q,T] or [H,Q,T]")
    raw_mask = np.asarray(allowed)
    if raw_mask.dtype != np.bool_:
        raise ValueError("explicit boolean causal/padding mask required")
    mask = np.broadcast_to(raw_mask, z.shape)
    if not mask.any(-1).all() or not np.isfinite(z[mask]).all():
        raise ValueError("invalid visible row")
    cols = indices(visual_columns, z.shape[-1])
    z = np.where(mask, z, -np.inf)
    p = np.exp(z-z.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return p[..., cols].mean(axis=tuple(range(z.ndim-1)))


def average_ranks(scores):
    """Larger response -> larger rank. Equal scores get equal average rank."""
    s = finite(scores, 1).astype(float)
    order = np.argsort(s, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(s[order]) != 0)+1]
    stops = np.r_[starts[1:], len(s)]
    ranks = np.empty(len(s), dtype=float)
    for start, stop in zip(starts, stops):
        ranks[order[start:stop]] = (start+stop-1)/2
    return ranks/max(1, len(s)-1)


def rank_drift(current, active_ids, previous, previous_ids, weight=.6):
    """User-specified RDRIFT reference, NOT claimed byte-identical to legacy.

    Rank both responses on the SAME current candidate universe. Align history
    by original IDs, never compare post-prune ranks with full-pool ranks.
    """
    now, old = finite(current, 1), finite(previous, 1)
    active, history = ids_array(active_ids), ids_array(previous_ids)
    if len(now) != len(active) or len(old) != len(history) or not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("invalid history/weight")
    lookup = {int(v): i for i, v in enumerate(history)}
    if any(int(v) not in lookup for v in active):
        raise ValueError("missing history; do not silently substitute current")
    matched = old[[lookup[int(v)] for v in active]]
    rn, ro = average_ranks(now), average_ranks(matched)
    return weight*rn+(1-weight)*(rn-ro)


def compact_sequence(hidden, original_positions, visual_ids, keep_ids):
    """CPU reference only: retain ALL text/structural rows (-1), original RoPE positions."""
    h = finite(hidden, 2)
    p, v = np.asarray(original_positions), np.asarray(visual_ids)
    keep = ids_array(keep_ids)
    if p.shape != (len(h),) or v.shape != p.shape or p.dtype.kind not in "iu" or v.dtype.kind not in "iu":
        raise ValueError("aligned sequence positions/visual IDs required")
    if len(np.unique(p)) != len(p) or (np.diff(p) <= 0).any() or (v < -1).any():
        raise ValueError("strict original positions and -1 for protected rows required")
    current = v[v >= 0]
    if len(np.unique(current)) != len(current) or not set(keep).issubset(set(current)):
        raise ValueError("duplicate or resurrected visual IDs")
    rows = np.flatnonzero((v < 0) | np.isin(v, keep))
    return h[rows].copy(), p[rows].copy(), v[rows].copy(), rows

```

## elva/integration.py

```python
"""Thin source-bound runtime contract, independent of a tensor framework.

Native callbacks operate ON DEVICE. This file never converts CUDA tensors
to NumPy. Adapters must compact position_ids/masks and maintain per-layer
cache_position/cache support with the actual model's native semantics.
"""
from dataclasses import dataclass
from .budget import Trace


@dataclass
class BoundRuntime:
    schedule: object
    select_native: object
    compact_native: object
    inspect_native: object

    def __post_init__(self):
        self.trace = Trace(self.schedule.work, self.schedule.depth)
        self.events = dict(self.schedule.events)
        self.audit = []

    def before_block(self, block, state):
        view = self.inspect_native(state)
        self.trace.before_block(block, len(view["active_ids"]))
        expected = self.schedule.trace[block-1]
        if len(view["active_ids"]) != expected:
            raise ValueError("wrong hook location or keep count")

    def after_block(self, completed, state, *, selector, context=None):
        if completed != len(self.trace.counts):
            raise ValueError("pruning must follow the recorded block")
        if completed not in self.events:
            return state
        before = self.inspect_native(state)
        keep = self.select_native(state, k=self.events[completed], selector=selector, context=context)
        # keep uses ORIGINAL globally unique patch IDs, not sequence columns.
        ids = tuple(map(int, keep))
        if len(ids) != self.events[completed] or len(set(ids)) != len(ids) or not set(ids).issubset(before["active_ids"]):
            raise ValueError("selector returned duplicate/unknown IDs or wrong K")
        result = self.compact_native(state, keep_ids=ids)
        after = self.inspect_native(result)
        if (set(after["active_ids"]) != set(ids)
                or before["protected_positions"] != after["protected_positions"]
                or before["backend"] != after["backend"]):
            raise ValueError("compaction changed protected positions/backend/identity")
        if "visual_positions" in before:
            expected_positions = {i: before["visual_positions"][i] for i in ids}
            if after.get("visual_positions") != expected_positions:
                raise ValueError("compaction changed original visual RoPE positions")
        self.audit.append(dict(completed=completed, selector=selector,
                               before=before["active_ids"], after=after["active_ids"]))
        return result

    def finish(self):
        actual = self.trace.finish()
        self.schedule.validate_trace(actual)
        if len(self.audit) != len(self.events):
            raise ValueError("missing or duplicate pruning hook")
        return actual

```

## elva/learning.py

```python
"""Optional diagnostic signed-gain regressor for NEW transient-swap labels.

It is deliberately small. A working synthetic regression is not evidence
that VLM survival gains are predictable. No old E020 weights are accepted.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from .counterfactual import ESTIMAND
from .evidence import finite, indices


def split_image(image_key, salt="elva-r2-frozen-split"):
    """Split by canonical decoded-image identity, not question/filename."""
    if not isinstance(image_key, str) or not image_key:
        raise ValueError("canonical image key required")
    value = int(hashlib.sha256((salt+image_key).encode()).hexdigest()[:16], 16) % 100
    return "train" if value < 70 else "validation" if value < 85 else "test"


def pair_features(token_features, common, group_a, group_b, *, begin, end, depth):
    """Features only from prompt and states available BEFORE this decision.

    Caller supplies aligned observables (current mass, rank drift, ViT CLS,
    layer-specific coverage descriptors, etc.). Do not feed NLL, future
    states, teacher-forced answer queries, or oracle-selected final IDs.
    Reversing groups negates the vector exactly, including context terms.
    """
    x = finite(token_features, 2).astype(float)
    c, a, b = (indices(v, len(x)) for v in (common, group_a, group_b))
    if set(c)&set(a) or set(c)&set(b) or set(a)&set(b) or len(a) != len(b):
        raise ValueError("disjoint equal groups required")
    if not 0 <= begin < end <= depth:
        raise ValueError("invalid observation/window depth")
    delta = x[a].mean(0)-x[b].mean(0)
    variance = x[a].var(0)-x[b].var(0)
    context = x[c].mean(0)
    return np.r_[delta, variance, delta*context, delta*(end-begin)/depth]


@dataclass
class PairModel:
    scale: np.ndarray
    weights: np.ndarray
    cache_mode: str
    feature_names: tuple
    catalog_hash: str
    stage: str = "decoder_transient_swap"

    def predict(self, features):
        x = np.asarray(features, dtype=float)
        if x.ndim not in (1, 2) or x.shape[-1] != len(self.weights) or not np.isfinite(x).all():
            raise ValueError("finite features with trained schema required")
        return (x/self.scale)@self.weights

    def save(self, path):
        value = dict(schema=ESTIMAND, stage=self.stage, cache_mode=self.cache_mode,
                     feature_names=list(self.feature_names), catalog_hash=self.catalog_hash,
                     scale=self.scale.tolist(), weights=self.weights.tolist())
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

    @classmethod
    def load(cls, path, *, cache_mode, catalog_hash, feature_names):
        with open(path, encoding="utf-8") as f:
            value = json.load(f)
        if (value.get("schema") != ESTIMAND or value.get("stage") != "decoder_transient_swap"
                or value.get("cache_mode") != cache_mode or value.get("catalog_hash") != catalog_hash
                or value.get("feature_names") != list(feature_names)):
            raise ValueError("incompatible weights/estimand/catalog; never reuse old E020 head")
        scale, weights = np.asarray(value["scale"], float), np.asarray(value["weights"], float)
        if (scale.ndim != 1 or len(scale) == 0 or weights.shape != scale.shape
                or len(feature_names) != len(scale) or not np.isfinite(scale).all()
                or not np.isfinite(weights).all() or (scale <= 0).any()):
            raise ValueError("invalid model arrays")
        return cls(scale, weights, cache_mode, tuple(feature_names), catalog_hash)


def fit_pairs(records, *, feature_names, catalog_hash, cache_mode, ridge=1.):
    """Train split only, no intercept/mean subtraction; antisymmetry preserved."""
    if not np.isfinite(ridge) or ridge <= 0 or not records:
        raise ValueError("positive ridge and nonempty real records required")
    if (not feature_names or len(set(feature_names)) != len(feature_names)
            or not catalog_hash or cache_mode not in ("native", "aligned_support")):
        raise ValueError("nonempty unique feature schema and valid catalog/cache mode required")
    xs, ys = [], []
    images = set()
    for record in records:
        if (record.get("schema") != ESTIMAND or record.get("cache_mode") != cache_mode
                or record.get("catalog_hash") != catalog_hash
                or record.get("feature_names") != list(feature_names)
                or record.get("feature_cutoff") != record.get("begin")
                or split_image(record["image_key"]) != "train"):
            raise ValueError("invalid schema/cache/catalog/causal cutoff or split leakage")
        if record.get("fixture", False):
            raise ValueError("synthetic fixtures cannot train a deployable VLM model")
        x = np.asarray(record["x"], float)
        y = float(record["delta"])
        if x.shape != (len(feature_names),) or not np.isfinite(x).all() or not np.isfinite(y):
            raise ValueError("nonfinite or mismatched training data")
        xs.append(x)
        ys.append(y)
        images.add(record["image_key"])
    if len(images) < 2:
        raise ValueError("at least two independent training images required")
    x, y = np.asarray(xs), np.asarray(ys)
    scale = np.sqrt((x*x).mean(0))
    scale = np.where(scale > 1e-12, scale, 1.)
    z = x/scale
    weights = np.linalg.solve(z.T@z+ridge*np.eye(z.shape[1]), z.T@y)
    return PairModel(scale, weights, cache_mode, tuple(feature_names), catalog_hash)


def choose_swap(predicted_gains, *, threshold):
    """One complete frozen candidate catalog, index 0 is the no-op."""
    gains = finite(predicted_gains, 1).astype(float)
    if gains[0] != 0 or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("zero no-op gain and calibrated nonnegative threshold required")
    best = int(np.argmax(gains))
    return best if gains[best] > threshold else 0


def calibrate_threshold(predicted, true_gains, noop_differences):
    """VALIDATION images only, equal-work actions incl. true no-op column 0.

    Accepts complete [states, actions] catalogs. Threshold candidates use
    predicted maxima, not individual-action thresholds. Actual VLM generation
    confirmation is a separate requirement even if NLL gain is positive.
    """
    p, y = finite(predicted, 2), finite(true_gains, 2)
    noise = finite(noop_differences, 1)
    if p.shape != y.shape or (p[:, 0] != 0).any() or (y[:, 0] != 0).any():
        raise ValueError("complete matched catalogs with zero no-op column required")
    floor = float(np.quantile(np.abs(noise), .95))
    maxima = np.maximum(0., p.max(1))
    candidates = np.unique(np.r_[floor, maxima[maxima >= floor]])
    best = None
    for threshold in candidates:
        actions = np.array([choose_swap(row, threshold=float(threshold)) for row in p])
        gains = y[np.arange(len(y)), actions]
        score = float(gains.mean())
        key = (score, float(threshold))  # conservative tie: fewer accepted actions
        if best is None or key > best[0]:
            best = key, actions
    return dict(threshold=best[0][1], validation_gain=best[0][0],
                accepted=int((best[1] != 0).sum()), states=len(p),
                noise_floor=floor,
                status="CANDIDATE" if best[0][0] > floor and (best[1] != 0).any() else "NO_SUPPORTED_DECISIONS")

```

## elva/low_budget_cover.py

```python
"""Explicit new control: protect salient patches, fill by residual coverage.

This is NOT the historical E013 COVER. It uses ordinary facility-location
coverage and makes no algorithmic novelty or accuracy claim. All candidates
and demand patches remain eligible; raw pen payload is never modified.
"""
import numpy as np


POLICY = "protected_positive_cosine_half_uniform_half_saliency_v1"


def _dimensions(features, scores, k, slots):
    if (len(features.shape) != 2 or min(features.shape) == 0
            or scores.shape != (features.shape[0],)
            or type(k) is not int or not 0 < k <= len(scores)
            or type(slots) is not int or not 0 <= slots <= k):
        raise ValueError("aligned [N,D] / [N] and 0 <= coverage slots <= K required")


def numpy_keep(features, scores, k, slots):
    """CPU reference; return sorted local rows, with >= K-slots core retained."""
    x = np.asarray(features, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32)
    _dimensions(x, s, k, slots)
    if not np.isfinite(x).all() or not np.isfinite(s).all() or (s < 0).any():
        raise ValueError("finite features and nonnegative saliency required")
    order = np.argsort(-s, kind="stable")
    if slots == 0:
        return np.sort(order[:k]).astype(np.int64)
    # Ranking candidates first makes an exact gain tie favor higher saliency.
    z = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), np.float32(1e-12))
    affinity = np.maximum(z @ z[order].T, np.float32(0.))
    n = len(s)
    # Scaling before summation avoids overflow for large but finite scores.
    relative = s / max(float(s.max()), 1e-30)
    saliency_weights = relative / max(float(relative.sum()), 1e-30)
    if not relative.any():
        saliency_weights = np.full(n, 1/n, dtype=np.float32)
    weights = .5 * (saliency_weights + np.float32(1/n))
    core_size = k-slots
    selected = np.zeros(n, dtype=bool)
    selected[:core_size] = True
    best = affinity[:, :core_size].max(1) if core_size else np.zeros(n, dtype=np.float32)
    for _ in range(slots):
        gains = (weights[:, None] * np.maximum(affinity-best[:, None], 0)).sum(0)
        gains[selected] = -np.inf
        chosen = int(np.argmax(gains))
        selected[chosen] = True
        best = np.maximum(best, affinity[:, chosen])
    return np.sort(order[selected]).astype(np.int64)


def torch_keep(features, scores, k, slots, *, debug=True):
    """CUDA-resident implementation; no .cpu()/.numpy()/.item() in greedy loop.

    FP32 reductions and affinity are used even for BF16/FP16 payloads. Server
    preflight must check NumPy parity and actual input layout/device first.
    """
    import torch
    import torch.nn.functional as F
    _dimensions(features, scores, k, slots)
    if not features.is_floating_point() or not scores.is_floating_point() or features.device != scores.device:
        raise ValueError("floating device-aligned features and saliency required")
    if debug and (not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(scores).all())
                  or bool((scores < 0).any())):
        raise ValueError("finite features and nonnegative saliency required")
    with torch.no_grad():
        s = scores.float()
        order = torch.argsort(s, descending=True, stable=True)
        if slots == 0:
            return order[:k].sort().values
        z = F.normalize(features.float(), dim=-1, eps=1e-12)
        affinity = (z @ z.index_select(0, order).T).clamp_min_(0)
        n = len(s)
        relative = s/s.max().clamp_min(1e-30)
        normalized = relative/relative.sum().clamp_min(1e-30)
        uniform = torch.full_like(s, 1/n)
        saliency_weights = torch.where(relative.sum() > 0, normalized, uniform)
        weights = .5*(saliency_weights+uniform)
        core_size = k-slots
        selected = torch.zeros(n, dtype=torch.bool, device=s.device)
        selected[:core_size] = True
        best = affinity[:, :core_size].amax(1) if core_size else torch.zeros_like(s)
        for _ in range(slots):
            gains = ((affinity-best[:, None]).clamp_min(0)*weights[:, None]).sum(0)
            gains = gains.masked_fill(selected, -torch.inf)
            chosen = gains.argmax().reshape(1)
            selected.scatter_(0, chosen, True)
            best = torch.maximum(best, affinity.index_select(1, chosen).squeeze(1))
        # Boolean nonzero output length is fixed by K; no Python scalar read.
        return order[selected].sort().values


def callback(*, metric_features, scores, patch_ids, k, context):
    """Source-bound standard COVER callback, separate implementation family."""
    if context.get("coverage_policy") != POLICY:
        raise ValueError("explicit protected coverage policy required")
    slots = context["replacement_slots"]
    if len(patch_ids) != len(scores):
        raise ValueError("full aligned original patch IDs required")
    if isinstance(metric_features, np.ndarray):
        return numpy_keep(metric_features, scores, k, slots)
    return torch_keep(metric_features, scores, k, slots)


def make_spec():
    """Executable implementation binding; preflight flag deliberately false."""
    from pathlib import Path
    import hashlib
    source = Path(__file__).resolve()
    return dict(family="protected_coverage", entrypoint="elva.low_budget_cover:callback",
                supported_k=[60], replacement_slots={"60":12}, allowed_slots=[6,12,18],
                coverage_policy=POLICY, source_files={str(source):hashlib.sha256(source.read_bytes()).hexdigest()},
                implementation_parity_verified=False)


def diagnostics(features, scores, keep, k, slots):
    """Offline numeric audit; caller exports real tensors OUTSIDE TTFT timing.

    Coverage is feature-space coverage, not a guarantee of OCR/region evidence.
    Spatial plots require actual crop coordinates in visual_audit separately.
    """
    from .evidence import indices
    x = np.asarray(features, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32)
    _dimensions(x, s, k, slots)
    keep = indices(keep, len(s), k)
    if not np.isfinite(x).all() or not np.isfinite(s).all() or (s < 0).any():
        raise ValueError("finite nonnegative diagnostic evidence required")
    order = np.argsort(-s, kind="stable")
    baseline, core = order[:k], order[:k-slots]
    if not set(core).issubset(set(keep)):
        raise ValueError("protected salient core was not retained")
    z = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    relative = s/max(float(s.max()),1e-30)
    norm = relative/max(float(relative.sum()),1e-30)
    if not relative.any():
        norm = np.full(len(s),1/len(s),dtype=np.float32)
    weights = .5*(norm+1/len(s))
    def coverage(rows):
        best = np.maximum(z @ z[rows].T, 0).max(1)
        return dict(uniform=float(best.mean()), weighted=float(weights @ best))
    added = sorted(set(map(int, keep))-set(map(int, baseline)))
    removed = sorted(set(map(int, baseline))-set(map(int, keep)))
    denominator = float(relative.sum())
    return dict(policy=POLICY, candidate_count=len(s), protected_count=len(core), slots=slots,
                actual_changed=len(added), added_local=added, removed_local=removed,
                topk_mass=float(relative[baseline].sum()/denominator) if denominator else None,
                kept_mass=float(relative[keep].sum()/denominator) if denominator else None,
                topk_coverage=coverage(baseline), kept_coverage=coverage(keep),
                zero_saliency=not bool(relative.any()))

```

## elva/policy.py

```python
"""Single-swap lifetime policy; offline and online share one finite catalog.

Selected groups live only to end, then BOTH possible groups are discarded.
This is distinct from old final-identity replacement. NumPy reference only.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from .counterfactual import Swap, run_swap
from .evidence import finite, ids_array, indices, topk
from .learning import pair_features, choose_swap


def hash_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class Catalog:
    baseline: tuple               # active-local baseline indices
    final: tuple                  # active-local final indices, fixed before choice
    swaps: tuple                  # (common, inserted, removed), active-local
    rule_hash: str                 # shared across images, used by model manifest
    action_hash: str               # particular identities in this sample
    window: tuple
    active_ids: tuple


def propose(scores, active_ids, baseline_keep, final_k, *, group_size=4, limit=16,
            begin=10, end=20, depth=32, proposal_rule="BIND_NATIVE_RULE"):
    """At most limit TOTAL actions incl. no-op; no resurrection or answer use.

    scores is a proposal ranking (native Q_ATTN or RDRIFT), not a utility
    certificate. Final anchors are chosen within baseline NOW, never from a
    future answer-dependent reference. Labels must cover this exact catalog.
    """
    s, pid = finite(scores, 1), ids_array(active_ids)
    if len(s) != len(pid) or type(final_k) is not int or type(group_size) is not int or type(limit) is not int:
        raise ValueError("aligned scores/IDs and integer sizes required")
    base = indices(baseline_keep, len(pid))
    if not 0 < final_k <= len(base) or group_size < 1 or limit < 1:
        raise ValueError("invalid final/group/catalog sizes")
    if not all(type(v) is int for v in (begin, end, depth)) or not 0 <= begin < end <= depth:
        raise ValueError("invalid survival window")
    final = np.sort(base[topk(s[base], final_k)])
    removable = np.setdiff1d(base, final)
    outside = np.setdiff1d(np.arange(len(s)), base)
    # Rank tie-breaking is stable in active sequence order.
    removable = removable[np.argsort(s[removable], kind="stable")]
    outside = outside[np.argsort(-s[outside], kind="stable")]
    chunks = lambda a: [a[i:i+group_size] for i in range(0, len(a)-group_size+1, group_size)]
    candidates = []
    for d in chunks(removable):
        for inserted in chunks(outside):
            common = np.setdiff1d(base, d)
            if len(common) == 0:
                continue
            proxy = float(s[inserted].sum()-s[d].sum())
            candidates.append((proxy, tuple(common), tuple(inserted), tuple(d)))
    candidates.sort(key=lambda a: (-a[0], a[2], a[3]))
    swaps = tuple(row[1:] for row in candidates[:limit-1])
    rule = dict(version="single_transient_swap_catalog_v1", group_size=group_size,
                limit=limit, final_k=final_k, middle_k=len(base),
                begin=begin, end=end, depth=depth, proposal_rule=proposal_rule,
                grouping="stable-score-disjoint-chunks", final="current-baseline-topk")
    base, final = tuple(sorted(map(int, base))), tuple(map(int, final))
    identity = dict(ids=pid.tolist(), baseline=base, final=final, swaps=swaps)
    # Convert NumPy index scalars before canonical JSON hashing.
    swaps = tuple(tuple(tuple(map(int, part)) for part in action) for action in swaps)
    identity["swaps"] = swaps
    return Catalog(base, final, swaps, hash_value(rule), hash_value(identity),
                   (begin, end, depth), tuple(map(int, pid)))


def feature_catalog(catalog, token_features, *, begin, end, depth):
    if (begin, end, depth) != catalog.window:
        raise ValueError("survival window differs from trained catalog")
    x = finite(token_features, 2)
    zero = np.zeros(x.shape[1]*4)
    return np.array([zero]+[pair_features(x, c, inserted, removed,
                                         begin=begin, end=end, depth=depth)
                            for c, inserted, removed in catalog.swaps])


def decide(catalog, token_features, model, *, begin, end, depth, threshold):
    if (model.catalog_hash != catalog.rule_hash or model.stage != "decoder_transient_swap"
            or model.cache_mode != "native"):
        raise ValueError("model/catalog mismatch")
    features = feature_catalog(catalog, token_features, begin=begin, end=end, depth=depth)
    predicted = np.asarray(model.predict(features), float)
    action = choose_swap(predicted, threshold=threshold)
    if action == 0:
        keep = catalog.baseline
    else:
        common, inserted, _ = catalog.swaps[action-1]
        keep = tuple(sorted(common+inserted))
    # Caller must force catalog.final at end. Native re-selection there would
    # change the estimand and invalidate this head's deployment contract.
    return dict(action=action, keep_local=keep,
                keep_ids=tuple(catalog.active_ids[i] for i in keep),
                final_ids=tuple(catalog.active_ids[i] for i in catalog.final),
                predicted_gain=float(predicted[action]), action_hash=catalog.action_hash)


def collect_catalog(adapter, prefix, catalog, active_ids, token_features, *,
                    begin, end, depth, image_key, question_key, feature_names,
                    cache_mode="native"):
    pid = ids_array(active_ids)
    if (tuple(map(int, pid)) != catalog.active_ids
            or tuple(map(int, pid)) != tuple(adapter.inspect(prefix)["active_ids"])
            or len(token_features) != len(pid)):
        raise ValueError("prefix, feature rows and proposal original IDs must have identical order")
    features = feature_catalog(catalog, token_features, begin=begin, end=end, depth=depth)
    if len(feature_names) != features.shape[1]:
        raise ValueError("feature names do not match policy input")
    rows = []
    state_key = hash_value(dict(prefix=adapter.inspect(prefix)["prefix_fingerprint"],
                                image=image_key, question=question_key, begin=begin, end=end,
                                depth=depth, final=[int(pid[i]) for i in catalog.final],
                                action_hash=catalog.action_hash, cache_mode=cache_mode))
    for i, (common, inserted, removed) in enumerate(catalog.swaps, 1):
        original = lambda local: tuple(int(pid[j]) for j in local)
        swap = Swap(begin, end, original(common), original(inserted), original(removed), original(catalog.final))
        row = run_swap(adapter, prefix, swap, image_key=image_key, question_key=question_key, mode=cache_mode)
        row.update(state_key=state_key, catalog_hash=catalog.rule_hash,
                   action_hash=catalog.action_hash, catalog_size=len(catalog.swaps)+1,
                   action_index=i, x=features[i].tolist(), feature_names=list(feature_names),
                   feature_cutoff=begin, fixture=bool(getattr(adapter, "is_fixture", True)))
        rows.append(row)
    return rows

```

## elva/torch_ops.py

```python
"""CUDA-resident Stage1 path. Server parity test is mandatory before ACC.

No attention backend is changed here. Existing COVER stays source-bound.
CPU reference in evidence.py is the tested mathematical specification.
"""
import math
import torch
import torch.nn.functional as F


@torch.no_grad()
def select_stage1(pen, pen_scores, ids, k, *, guide="pen", early=None,
                  early_scores=None, early_ids=None, beta=.5, cover=None,
                  use_cover=False, context=None, frozen_keep=None, debug=True,
                  score_guide="same"):
    if pen.ndim != 2 or min(pen.shape) == 0 or not pen.is_floating_point():
        raise ValueError("floating raw pen [N,D] required")
    n = len(pen)
    if (pen_scores.shape != (n,) or ids.shape != (n,) or ids.dtype != torch.long
            or pen_scores.device != pen.device or ids.device != pen.device
            or not pen_scores.is_floating_point() or type(k) is not int or not 0 < k <= n):
        raise ValueError("aligned source saliency/IDs and valid K required")
    if guide not in ("pen", "l9", "dual") or not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("invalid guide/beta")
    sg = guide if score_guide == "same" else score_guide
    if sg not in ("pen", "l9", "dual"):
        raise ValueError("invalid score guide")
    needs_early = (guide == "l9" or (guide == "dual" and beta > 0)
                   or sg == "l9" or (sg == "dual" and beta > 0))
    if needs_early:
        if (early is None or early.shape != pen.shape or early.device != pen.device
                or not early.is_floating_point() or early_scores is None
                or early_scores.shape != (n,) or early_scores.device != pen.device
                or not early_scores.is_floating_point() or early_ids is None
                or early_ids.shape != ids.shape or early_ids.dtype != torch.long
                or early_ids.device != pen.device):
            raise ValueError("aligned captured early features/scores/IDs required")
    if debug:
        tensors = [pen, pen_scores]+([early, early_scores] if needs_early else [])
        if (any(not bool(torch.isfinite(x).all()) for x in tensors)
                or bool((pen_scores < 0).any()) or ids.unique().numel() != n or bool((ids < 0).any())
                or (needs_early and (not torch.equal(ids, early_ids) or bool((early_scores < 0).any())))):
            raise ValueError("nonfinite/negative/misaligned evidence")
    if guide == "pen" or (guide == "dual" and beta == 0):
        metric, scores = pen, pen_scores
    elif guide == "l9" or beta == 1:
        metric, scores = early, early_scores
    else:
        metric = torch.cat((math.sqrt(1-beta)*F.normalize(pen.float(), dim=-1, eps=1e-12),
                            math.sqrt(beta)*F.normalize(early.float(), dim=-1, eps=1e-12)), dim=-1)
    scores = pen_scores if sg == "pen" or (sg == "dual" and beta == 0) else (
        early_scores if sg == "l9" or beta == 1 else (1-beta)*pen_scores.float()+beta*early_scores.float())
    if frozen_keep is not None:
        keep = frozen_keep
    elif use_cover:
        if cover is None:
            raise RuntimeError("BLOCKED_SOURCE: bind original COVER")
        keep = cover(metric_features=metric.clone(), scores=scores.clone(),
                     patch_ids=ids.clone(), k=k, context=context)
    else:
        keep = torch.argsort(scores, descending=True, stable=True)[:k]
    if not isinstance(keep, torch.Tensor) or keep.shape != (k,) or keep.dtype != torch.long or keep.device != pen.device:
        raise ValueError("native selector must return device-local long [K]")
    if debug and (keep.unique().numel() != k or int(keep.min()) < 0 or int(keep.max()) >= n):
        raise ValueError("invalid/duplicate selected local indices")
    keep = keep.sort().values
    return dict(payload=pen.index_select(0, keep), keep=keep,
                ids=ids.index_select(0, keep), metric=metric, scores=scores)


@torch.no_grad()
def question_visual_mass(logits, allowed, visual_columns):
    if logits.ndim not in (2, 3) or min(logits.shape) == 0 or allowed.dtype != torch.bool:
        raise ValueError("partial [Q,T]/[H,Q,T] logits and boolean visible mask required")
    if allowed.device != logits.device or visual_columns.device != logits.device or visual_columns.dtype != torch.long:
        raise ValueError("device-local masks/column indices required")
    mask = torch.broadcast_to(allowed, logits.shape)
    # Validate once in server preflight; this function deliberately makes its
    # synchronization cost visible in end-to-end timing.
    if (not bool(mask.any(-1).all()) or not bool(torch.isfinite(logits[mask]).all())
            or visual_columns.ndim != 1 or visual_columns.numel() == 0
            or visual_columns.unique().numel() != visual_columns.numel()
            or int(visual_columns.min()) < 0 or int(visual_columns.max()) >= logits.shape[-1]):
        raise ValueError("invalid visible rows/visual indices")
    p = logits.float().masked_fill(~mask, -torch.inf).softmax(-1)
    return p.index_select(-1, visual_columns).mean(dim=tuple(range(logits.ndim-1)))

```

## elva/toy_adapter.py

```python
"""Random causal decoder fixture. No pretrained VLM, no benchmark claim.

Runs prompt prefill and multi-token cached teacher forcing. Visual IDs have
no relation to positions; earlier caches survive normal hidden-row pruning.
"""
from copy import deepcopy
import hashlib
import numpy as np
from .evidence import compact_sequence


class ToyAdapter:
    is_fixture = True
    def __init__(self, depth=8, visual=12, text=3, dim=12, vocab=9, seed=7):
        self.depth, self.vocab = depth, vocab
        r = np.random.default_rng(seed)
        self.weights = r.normal(0, .15, (depth, 3, dim, dim))
        self.output = r.normal(0, .3, (dim, vocab))
        self.embeddings = r.normal(0, .4, (vocab, dim))
        h = r.normal(0, .5, (visual+text, dim))
        ids = np.r_[np.arange(visual)*3+11, [-1]*text]
        positions = np.arange(visual+text)*2
        fingerprint = hashlib.sha256(h.tobytes()+ids.tobytes()).hexdigest()
        self.initial = dict(hidden=h, positions=positions, ids=ids, completed=0,
                            kv=[], trace=[], prefix_fingerprint=fingerprint)
        self.targets = (2, 5, 1)

    def capture(self, completed=2):
        state = self.fork(self.initial)
        self.advance(state, completed)
        state["prefix_fingerprint"] = hashlib.sha256(
            state["hidden"].tobytes()+str(completed).encode()).hexdigest()
        return state

    def inspect(self, state):
        return dict(completed=state["completed"],
                    active_ids=tuple(map(int, state["ids"][state["ids"] >= 0])),
                    protected_positions=tuple(map(int, state["positions"][state["ids"] < 0])),
                    visual_positions={int(i): int(p) for i, p in zip(state["ids"], state["positions"]) if i >= 0},
                    kv_support=tuple(tuple(map(int, c["ids"][c["ids"] >= 0])) for c in state["kv"]),
                    prefix_fingerprint=state["prefix_fingerprint"],
                    state_hash=hashlib.sha256(state["hidden"].tobytes()+b"".join(
                        c["k"].tobytes()+c["v"].tobytes() for c in state["kv"])).hexdigest())

    def fork(self, state):
        return deepcopy(state)

    def prune(self, state, keep):
        h, p, v, _ = compact_sequence(state["hidden"], state["positions"], state["ids"], keep)
        state.update(hidden=h, positions=p, ids=v)

    @staticmethod
    def attention(q, k, v, allowed):
        z = q@k.T/np.sqrt(q.shape[-1])
        z = np.where(allowed, z, -np.inf)
        p = np.exp(z-z.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        return p@v

    def advance(self, state, completed):
        if not state["completed"] <= completed <= self.depth:
            raise ValueError("invalid resume depth")
        for layer in range(state["completed"], completed):
            h, p, ids = state["hidden"], state["positions"], state["ids"]
            q, k, v = (h@w for w in self.weights[layer])
            state["kv"].append(dict(k=k.copy(), v=v.copy(), ids=ids.copy(), positions=p.copy()))
            out = self.attention(q, k, v, p[:, None] >= p[None, :])
            state["hidden"] = h+np.tanh(out)
            state["trace"].append(int((ids >= 0).sum()))
            state["completed"] = layer+1

    def align_cache(self, state, keep):
        for cache in state["kv"]:
            rows = (cache["ids"] < 0) | np.isin(cache["ids"], keep)
            for key in ("k", "v", "ids", "positions"):
                cache[key] = cache[key][rows].copy()

    def _decode(self, state, teacher):
        h = state["hidden"][-1:]
        position = int(state["positions"].max())+1
        losses, generated = [], []
        for i, target in enumerate(self.targets):
            logits = (h@self.output)[0]
            generated.append(int(np.argmax(logits)))
            z = logits-logits.max()
            losses.append(float(np.log(np.exp(z).sum())-z[target]))
            if i == len(self.targets)-1:
                break
            previous = target if teacher else generated[-1]
            h = self.embeddings[previous:previous+1].copy()
            for layer, cache in enumerate(state["kv"]):
                q, k, v = (h@w for w in self.weights[layer])
                cache["k"] = np.concatenate((cache["k"], k))
                cache["v"] = np.concatenate((cache["v"], v))
                cache["ids"] = np.r_[cache["ids"], -1]
                cache["positions"] = np.r_[cache["positions"], position]
                out = self.attention(q, cache["k"], cache["v"],
                                     (cache["positions"] <= position)[None])
                h = h+np.tanh(out)
            position += 1
        return float(np.mean(losses)), generated

    def score_suffix(self, state):
        self.advance(state, self.depth)
        nll, _ = self._decode(self.fork(state), teacher=True)
        _, generated = self._decode(self.fork(state), teacher=False)
        return dict(nll=nll, answer_tokens=len(self.targets),
                    trace=tuple(state["trace"]), backend="numpy_causal_fixture",
                    generated=generated)

```

## elva/workflow.py

```python
"""Per-sample evidence and explicit Stage1 freeze gates for real VLM jobs."""
import json
import math
from pathlib import Path
from .cover_binding import spec_fingerprint, origin
from .budget import Schedule


REQUIRED = ("source_commit", "model_sha", "processor_sha", "tokenizer_sha", "prompt_sha",
            "dataset_sha", "scorer_sha", "backend", "cache_mode", "native_selectors")


def validate_binding(binding, *, allow_fixture=False):
    if binding.get("fixture", True) and not allow_fixture:
        raise ValueError("real server binding required; fixture bindings cannot launch VLM evaluation")
    if any(not binding.get(key) for key in REQUIRED):
        raise ValueError("complete source/model/data/scorer binding required")
    if (binding.get("original_tokens"), binding.get("depth")) != (576, 32):
        raise ValueError("this two-budget catalog is explicitly for 576x32")
    if binding.get("cache_mode") != "native":
        raise ValueError("this primary two-budget comparison uses native KV behavior")
    checks = binding.get("preflight", {})
    if not all(checks.get(key) is True for key in ("off_path_equal", "clip_layers_checked", "pen_payload_equal", "positions_and_kv_checked")):
        raise ValueError("native OFF path, actual CLIP layer and pen payload checks must pass first")


def blocked_reason(config, binding):
    method = config["method"]
    if binding.get("preflight", {}).get("trace_verified", {}).get(str(method["budget"])) is not True:
        return "BLOCKED_TRACE: full VLM pre-block trace not checked for this schedule budget"
    selector = method["stage2"]["first_selector"]
    if not binding["native_selectors"].get(selector):
        return "BLOCKED_SELECTOR: original decoder selector not bound"
    family = method["stage1"]["cover_family"]
    if family == "none":
        return None
    spec = binding.get("native_cover", {}).get(family)
    k = method["schedule"]["counts"][0]
    if not spec or spec.get("family") != family or k not in spec.get("supported_k", []):
        return "BLOCKED_COVER_CAPACITY: no explicit source-bound COVER for requested entry K"
    if family == "protected_coverage":
        if spec.get("implementation_parity_verified") is not True:
            return "BLOCKED_COVER_PARITY: protected-control NumPy/Torch and real CLIP preflight not verified"
        from .low_budget_cover import POLICY
        if (spec.get("coverage_policy") != POLICY or method["stage1"].get("coverage_policy") != POLICY
                or method["stage1"].get("coverage_slots") not in spec.get("allowed_slots", [])):
            return "BLOCKED_COVER_POLICY: unverified protected coverage policy/slots"
    elif spec.get("legacy172_parity_verified") is not True:
        return "BLOCKED_COVER_PARITY: original K172 identity regression not verified"
    try:
        spec_fingerprint(spec)
    except (ValueError, OSError, RuntimeError) as error:
        return f"BLOCKED_COVER_SOURCE: {error}"
    return None


def trace_for(config):
    return list(Schedule(**config["method"]["schedule"]).trace)


def _integer_ids(value, k):
    return (isinstance(value, list) and len(value) == k and all(type(v) is int and v >= 0 for v in value)
            and len(set(value)) == k)


def validate_result(value, config, binding, task, run_key, result_dir, *, allow_fixture=False):
    from make_catalog import digest
    method, stage = config["method"], config["method"]["stage1"]
    if digest(method) != config.get("config_hash"):
        raise ValueError("configuration edited without regenerating its fingerprint")
    if value.get("schema") != "elva_r3_vlm_result" or value.get("inference_kind") != "full_vlm_generate":
        raise ValueError("a CLIP-only pilot or old schema is not an official VLM result")
    if value.get("fixture", True) and not allow_fixture:
        raise ValueError("synthetic results cannot be accepted as real VLM ACC")
    expected = dict(config_hash=config["config_hash"], binding_hash=digest(binding),
                    run_key=run_key, task=task, backend=binding["backend"],
                    cache_mode=method["cache_mode"], scorer_sha=binding["scorer_sha"])
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError("stale configuration, source, task or scorer fingerprint")
    count, metrics = value.get("samples"), value.get("metrics")
    if type(count) is not int or count < 1 or not isinstance(metrics, dict) or not metrics:
        raise ValueError("actual scored sample count and official metrics required")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in metrics.values()):
        raise ValueError("nonfinite official metric")
    declared_count = binding.get("task_samples", {}).get(task)
    if type(declared_count) is not int or count != declared_count:
        raise ValueError("evaluated samples must match the frozen task/panel manifest")
    root = Path(result_dir).resolve()
    filename = value.get("predictions_file")
    if not isinstance(filename, str):
        raise ValueError("per-sample predictions and traces are mandatory")
    predictions = (root/filename).resolve()
    if root not in predictions.parents or not predictions.is_file():
        raise ValueError("predictions file must exist inside this run directory")
    import hashlib
    if hashlib.sha256(predictions.read_bytes()).hexdigest() != value.get("predictions_sha256"):
        raise ValueError("prediction file fingerprint mismatch")
    family = stage["cover_family"]
    covered, k = family != "none", method["schedule"]["counts"][0]
    source_sha = spec_fingerprint(binding["native_cover"][family]) if covered else None
    seen, actual = set(), 0
    with predictions.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = row.get("image_key"), row.get("question_key")
            if any(not isinstance(v, str) or not v for v in key) or key in seen:
                raise ValueError("canonical nonduplicate image/question keys required")
            seen.add(key)
            if (not isinstance(row.get("prediction"), str) or row.get("trace") != trace_for(config)
                    or any(type(v) is not int for v in row["trace"])):
                raise ValueError("missing generation or wrong actual pre-block token trace")
            if not _integer_ids(row.get("stage1_ids"), k):
                raise ValueError("actual Stage1 identities have wrong K or duplicates")
            audit = row.get("stage1_audit", {})
            spec = binding.get("native_cover", {}).get(family, {})
            native_k = 172 if family == "legacy_pool172_then_topk" else k
            slots = (stage.get("coverage_slots") if family == "protected_coverage" else
                     spec.get("replacement_slots", {}).get(str(native_k))) if covered else 0
            required = dict(cover_family=family, requested_k=k, cover_calls=int(covered),
                            native_calls=int(covered and family != "protected_coverage"),
                            origin=origin(family), replacement_slots=slots,
                            fallback=False, payload="pen", cover_source_sha=source_sha)
            if any(audit.get(key) != value for key, value in required.items()):
                raise ValueError("missing native COVER call proof or forbidden fallback")
            expected_native_k = (172 if family == "legacy_pool172_then_topk" else k) if covered else None
            if audit.get("native_k") != expected_native_k:
                raise ValueError("native COVER capacity does not match the declared implementation")
            actual += 1
    if actual != count:
        raise ValueError("prediction rows do not cover all scored samples")


def result_key(config, binding, task):
    from make_catalog import digest
    return digest(dict(method=config["method"], binding=binding, task=task))


def collect_complete(configs, binding, result_root, budget, *, allow_fixture=False):
    complete = []
    for c in configs:
        if c["phase"] != "stage1_core" or c["method"]["budget"] != budget:
            continue
        jobs = []
        for task in ("GQA", "POPE", "MME", "MMBench"):
            key = result_key(c, binding, task)
            folder = Path(result_root)/key
            path = folder/"result.json"
            value = json.loads(path.read_text())
            validate_result(value, c, binding, task, key, folder, allow_fixture=allow_fixture)
            jobs.append(dict(task=task, run_key=key, result_hash=__import__('hashlib').sha256(path.read_bytes()).hexdigest()))
        complete.append(dict(config=c, jobs=jobs))
    if len(complete) != 5:
        raise ValueError("all five matched Stage1 groups and all four tasks must be complete before freeze")
    return complete


def freeze(configs, binding, results, budget, winner, rationale, *, allow_fixture=False):
    from make_catalog import digest
    if budget not in (2048, 1024) or not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("explicit supported budget and evidence-based rationale required")
    complete = collect_complete(configs, binding, results, budget, allow_fixture=allow_fixture)
    selected = next((c for c in configs if c["id"] == winner and c["phase"] in ("stage1_core", "stage1_optional")
                     and c["method"]["budget"] == budget), None)
    if selected is None:
        raise ValueError("winner must be a Stage1 configuration at this budget")
    winner_evidence = []
    for task in ("GQA", "POPE", "MME", "MMBench"):
        key = result_key(selected, binding, task)
        folder = Path(results)/key
        path = folder/"result.json"
        value = json.loads(path.read_text())
        validate_result(value, selected, binding, task, key, folder, allow_fixture=allow_fixture)
        winner_evidence.append(dict(task=task, run_key=key, result_hash=__import__('hashlib').sha256(path.read_bytes()).hexdigest()))
    return dict(schema="elva_r3_stage1_freeze", binding_hash=digest(binding),
                budgets={str(budget):dict(status="FROZEN", stage1=selected["method"]["stage1"],
                         winner=winner, rationale=rationale, core_evidence=complete,
                         winner_config=selected, winner_evidence=winner_evidence)})


def validate_decision(decision, binding, configs, results, budget, *, allow_fixture=False):
    from make_catalog import digest
    if decision.get("schema") != "elva_r3_stage1_freeze" or decision.get("binding_hash") != digest(binding):
        raise ValueError("Stage1 freeze belongs to a different model/data/source binding")
    item = decision.get("budgets", {}).get(str(budget), {})
    if item.get("status") != "FROZEN":
        raise ValueError("Stage1 is not frozen at this budget")
    actual = collect_complete(configs, binding, results, budget, allow_fixture=allow_fixture)
    if item.get("core_evidence") != actual:
        raise ValueError("Stage1 results changed or are incomplete; regenerate the decision")
    winner = next((c for c in configs if c["id"] == item.get("winner")
                   and c["phase"] in ("stage1_core", "stage1_optional") and c["method"]["budget"] == budget), None)
    if not winner or item.get("stage1") != winner["method"]["stage1"] or item.get("winner_config") != winner:
        raise ValueError("frozen Stage1 does not match its recorded result")
    current = freeze(configs, binding, results, budget, winner["id"], item.get("rationale"), allow_fixture=allow_fixture)
    if current["budgets"][str(budget)]["winner_evidence"] != item.get("winner_evidence"):
        raise ValueError("winner results changed after the Stage1 freeze")

```

## freeze_stage1.py

```python
"""Freeze a completed Stage1 choice; preserve all four-task evidence hashes."""
import argparse
import json
from pathlib import Path
from elva.workflow import freeze, validate_binding


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="configs/catalog.json")
    parser.add_argument("--binding", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--budget", type=int, choices=(2048, 1024), required=True)
    parser.add_argument("--winner", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--out", default="stage1_decisions.json")
    args = parser.parse_args()
    load = lambda path: json.loads(Path(path).read_text())
    binding, catalog = load(args.binding), load(args.catalog)
    validate_binding(binding)
    decision = freeze(catalog["configs"], binding, args.results, args.budget, args.winner, args.reason)
    out = Path(args.out)
    if out.exists():
        old = load(out)
        if old.get("schema") != decision["schema"] or old.get("binding_hash") != decision["binding_hash"]:
            raise ValueError("do not merge decisions from different experiments")
        old["budgets"].update(decision["budgets"])
        decision = old
    out.write_text(json.dumps(decision, indent=2, ensure_ascii=False)+"\n")
    print(str(out))

```

## gpu_resources.py

```python
"""Read-only GPU inventory and per-UUID leases. Never kill another job."""
import argparse
import csv
from dataclasses import dataclass
import fcntl
import io
import json
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class GPU:
    index: str
    uuid: str
    used_mib: int
    utilization: int
    pids: tuple

    def available(self, max_used_mib=1024, max_utilization=10):
        return not self.pids and self.used_mib <= max_used_mib and self.utilization <= max_utilization


def parse_inventory(gpus_csv, processes_csv):
    processes = {}
    for row in csv.reader(io.StringIO(processes_csv)):
        if not row:
            continue
        if len(row) != 2:
            raise ValueError("unexpected GPU process query format; cannot assume an empty card")
        uuid, pid = (s.strip() for s in row)
        processes.setdefault(uuid, []).append(int(pid))
    result = []
    for row in csv.reader(io.StringIO(gpus_csv)):
        if not row:
            continue
        if len(row) != 4:
            raise ValueError("unexpected GPU inventory format")
        index, uuid, used, utilization = (s.strip() for s in row)
        if not uuid.startswith("GPU-"):
            raise ValueError("unsupported UUID/MIG mapping; do not guess physical device ownership")
        result.append(GPU(index, uuid, int(used), int(utilization), tuple(processes.get(uuid, ()))))
    if not result or len({g.uuid for g in result}) != len(result):
        raise ValueError("missing or duplicate GPU UUIDs")
    if set(processes)-{g.uuid for g in result}:
        raise ValueError("processes refer to unrecognized UUIDs; inventory is not safe to schedule")
    return result


def inventory():
    def query(argument):
        return subprocess.run(["nvidia-smi", argument, "--format=csv,noheader,nounits"],
                              check=True, capture_output=True, text=True, timeout=15).stdout
    return parse_inventory(query("--query-gpu=index,uuid,memory.used,utilization.gpu"),
                           query("--query-compute-apps=gpu_uuid,pid"))


def respect_visible(gpus, mask):
    if mask is None:
        return list(gpus)
    values = [s.strip() for s in mask.split(",") if s.strip()]
    if not values or values == ["-1"]:
        return []
    known = {g.index for g in gpus}|{g.uuid for g in gpus}
    if set(values)-known:
        raise ValueError("inherited CUDA_VISIBLE_DEVICES cannot be mapped safely to physical UUIDs")
    return [g for g in gpus if g.index in values or g.uuid in values]


def choose(gpus, requested, workers, reserve):
    if workers < 1 or reserve < 0:
        raise ValueError("positive ACC worker count and nonnegative reserve required")
    requested = list(map(str, requested))
    if requested != ["auto"] and len(set(requested)) != len(requested):
        raise ValueError("duplicate requested GPU IDs")
    if requested != ["auto"]:
        known = {g.index for g in gpus}|{g.uuid for g in gpus}
        if set(requested)-known:
            raise ValueError("requested GPU is absent from the current inventory")
    free = [g for g in gpus if g.available() and
            (requested == ["auto"] or g.index in requested or g.uuid in requested)]
    usable = max(0, min(workers, len(free)-reserve))
    return free[:usable], free[usable:]


class Lease:
    def __init__(self, directory, uuid):
        if not uuid.startswith("GPU-") or "/" in uuid:
            raise ValueError("valid physical GPU UUID required")
        self.path = Path(directory)/(uuid+".lock")
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:
            self.stream.close()
            self.stream = None
            raise
        return self

    def __exit__(self, *args):
        if self.stream is not None:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", default=["auto"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--reserve", type=int, default=1)
    args = parser.parse_args()
    devices = inventory()
    selected, rest = choose(devices, args.gpus, args.workers, args.reserve)
    print(json.dumps(dict(inventory=[dict(index=g.index, uuid=g.uuid, used_mib=g.used_mib,
          utilization=g.utilization, pids=g.pids, eligible=g.available()) for g in devices],
          acc=[g.uuid for g in selected], reserved_or_unused=[g.uuid for g in rest]), indent=2))

```

## legacy_reference/make_catalog.py

```python
"""Generate finite, deduplicated development configs. No VLM is run here."""
from pathlib import Path
import hashlib
import json
from collections import Counter
from elva.budget import Schedule


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_catalog():
    result, seen = [], set()

    def add(block, priority, budget, schedule, guide="pen", cover=True,
            selector="Q_ATTN", beta=.5, final_selector="Q_ATTN", seed=0, conditional=False,
            score_guide="same"):
        assert schedule.work == budget and schedule.depth == 32
        method = dict(budget=budget, original_tokens=576, depth=32,
                      schedule=dict(counts=schedule.counts, lengths=schedule.lengths),
                      stage1=dict(guide=guide, cover=cover, beta=beta if guide == "dual" or score_guide == "dual" else None,
                                  score_guide=score_guide, payload="pen", saliency="BIND_ORIGINAL_SAME_RULE"),
                      stage2=dict(first_selector=selector, later_selector=final_selector,
                                  needs_l9_features=selector == "TRANSIENT_PAIR_PLUS_L9",
                                  seed=seed if "RANDOM" in final_selector else None),
                      merge=False, cache_mode="native", conditional=conditional)
        digest = fingerprint(method)
        if digest in seen:
            return
        seen.add(digest)
        result.append(dict(id=f"ELVA_R2_{len(result):03d}", block=block, priority=priority,
                           config_hash=digest, method=method))

    for budget, counts in [(1024, (60, 28, 12)), (1536, (90, 42, 18)), (2048, (120, 56, 24))]:
        k0, middle, final = counts
        long = Schedule(counts, (10, 10, 12))
        for guide, cover in [("pen", False), ("l9", False), ("dual", False), ("pen", True), ("dual", True)]:
            add("A_stage1", 0 if budget == 1536 else 1, budget, long, guide, cover)
        shapes = [long,
                  Schedule((k0, middle, final), (8, 16, 8)),
                  Schedule((k0, 32*budget//1024, final), (10, 8, 14)),
                  Schedule((k0, 20*budget//1024, final), (12, 8, 12))]
        for shape in shapes:
            for selector in ("Q_ATTN", "RDRIFT_NATIVE"):
                add("B_schedule", 1, budget, shape, selector=selector)
        add("J_stage1_stage2_bridge", 2, budget, long, guide="dual", selector="RDRIFT_NATIVE")
        legacy = {1024: (80, 48, 16), 1536: (96, 64, 32), 2048: (160, 96, 32)}[budget]
        for selector in ("Q_ATTN", "RDRIFT_NATIVE"):
            add("C_legacy_recipe", 1, budget, Schedule(legacy, (3, 10, 19)), selector=selector)
            add("H_stage1_transfer", 2, budget, Schedule(legacy, (3, 10, 19)),
                guide="dual", selector=selector)
        for selector in ("QED_NATIVE", "ATTN_COVER_NATIVE"):
            add("D_selector", 1, budget, long, selector=selector)
        for seed in (11, 23, 47):
            add("E_random_final", 2, budget, long, final_selector="RANDOM_FINAL", seed=seed)
        for beta in (.25, .75):
            add("F_beta", 2, budget, long, guide="dual", beta=beta)
        if budget == 1536:
            add("I_stage1_factor", 2, budget, long, guide="dual", score_guide="pen")
            add("I_stage1_factor", 2, budget, long, guide="pen", score_guide="dual")
        # Conditional heads use forced-final identity F, including their own
        # matched anchor. They CANNOT be compared only to natural re-selection.
        for selector in ("FROZEN_FINAL_ANCHOR", "TRANSIENT_PAIR_CURRENT", "TRANSIENT_PAIR_PLUS_L9"):
            add("G_conditional", 3, budget, long, selector=selector, conditional=True)
    return sorted(result, key=lambda c: (c["priority"], c["id"]))


def main():
    root = Path(__file__).resolve().parent
    out = root/"configs"
    out.mkdir(exist_ok=True)
    catalog = build_catalog()
    for config in catalog:
        (out/f"{config['id']}.json").write_text(json.dumps(config, indent=2, ensure_ascii=False)+"\n")
    summary = dict(configs=len(catalog), blocks=dict(Counter(c["block"] for c in catalog)),
                   unconditional=sum(not c["method"]["conditional"] for c in catalog),
                   conditional=sum(c["method"]["conditional"] for c in catalog),
                   four_task_jobs=4*len(catalog),
                   note="development capacity, not completed experiments; native source aliases must be deduplicated")
    (out/"catalog.json").write_text(json.dumps(dict(summary=summary, configs=catalog), indent=2)+"\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()


```

## legacy_reference/run_queue.py

```python
"""Resumable three-worker ACC queue; caller supplies the REAL native runner.

No shell interpolation. GPU IDs are configurable; reserve a separate card
for TTFT. Official scorer output and complete trace are mandatory for DONE.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import threading
import time


REQUIRED_BINDING = ("source_commit", "model_sha", "processor_sha", "scorer_sha",
                    "prompt_sha", "tokenizer_sha", "dataset_sha", "backend", "cache_mode",
                    "native_selectors", "original_tokens", "depth")


def run_process(argv, *, env, log, timeout):
    process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except ProcessLookupError:
            pass
        raise
    if code:
        raise subprocess.CalledProcessError(code, argv)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def conditional_ready(config, binding):
    if not config["method"]["conditional"]:
        return True
    selector = config["method"]["stage2"]["first_selector"]
    if not binding.get("transient_adapter_ready", False):
        return False
    if selector == "FROZEN_FINAL_ANCHOR":
        return True
    gate = binding.get("transient_gate", {})
    return (gate.get("schema") == "transient_survival_swap_v1"
            and gate.get("status") == "READY" and gate.get("cache_mode") == "native"
            and bool(gate.get("weights_sha")) and bool(gate.get("catalog_hash"))
            and gate.get("real_labels") is True and gate.get("heldout_gain_over_noise") is True
            and gate.get("generation_checked") is True)


def validate_result(value, config, binding, task, run_key):
    method = config["method"]
    if value.get("run_key") != run_key or value.get("config_hash") != config["config_hash"]:
        raise ValueError("stale or missing provenance fingerprint")
    if value.get("task") != task or value.get("backend") != binding["backend"] or value.get("cache_mode") != method["cache_mode"]:
        raise ValueError("task/backend/cache mismatch")
    count = value.get("samples")
    if type(count) is not int or count <= 0 or value.get("scorer_sha") != binding["scorer_sha"]:
        raise ValueError("real scored samples and bound scorer required")
    metrics = value.get("metrics")
    if not isinstance(metrics, dict) or not metrics or not all(type(v) in (int, float) and math.isfinite(v) for v in metrics.values()):
        raise ValueError("missing/nonfinite official metrics")
    expected = [k for k, d in zip(method["schedule"]["counts"], method["schedule"]["lengths"]) for _ in range(d)]
    audit = value.get("trace_audit", {})
    if (audit.get("expected") != expected or audit.get("observed_samples") != count
            or audit.get("mismatch_count") != 0):
        raise ValueError("missing/incorrect all-sample trace audit")
    if value.get("binding_hash") != digest(binding):
        raise ValueError("binding changed; do not reuse old results")


def run(catalog, binding, command, output, gpus, tasks, hours, allow_conditional=False):
    if any(key not in binding or not binding[key] for key in REQUIRED_BINDING):
        raise ValueError("complete server binding manifest required")
    if (binding["original_tokens"], binding["depth"]) != (576, 32):
        raise ValueError("this catalog is for 576x32; regenerate budgets for other architectures")
    if not gpus or len(set(gpus)) != len(gpus) or hours <= 0 or not math.isfinite(hours):
        raise ValueError("distinct ACC GPU IDs and finite positive time budget required")
    if not isinstance(command, list) or not command or any(not isinstance(v, str) for v in command):
        raise ValueError("runner command must be an argv array")
    required = ("{config}", "{task}", "{out}", "{run_key}", "{binding}")
    if any(not any(p in token for token in command) for p in required):
        raise ValueError("runner argv missing mandatory routing placeholders")
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out/"binding.json").write_text(json.dumps(binding, indent=2))
    deadline = time.monotonic()+hours*3600
    jobs, states, lock = [], [], threading.Lock()
    for config in catalog:
        if config["method"]["conditional"] and (not allow_conditional or not conditional_ready(config, binding)):
            states.append(dict(config=config["id"], status="DEFERRED_CONDITIONAL"))
            continue
        for task in tasks:
            key = digest(dict(config=config, binding=binding, task=task))
            jobs.append((config, task, key))

    def worker(gpu):
        while time.monotonic() < deadline:
            with lock:
                if not jobs:
                    return
                config, task, key = jobs.pop(0)
            path = out/key
            path.mkdir(exist_ok=True)
            result_path, config_path = path/"result.json", path/"config.json"
            config_path.write_text(json.dumps(config, indent=2))
            status = dict(config=config["id"], task=task, run_key=key, gpu=gpu)
            try:
                if result_path.exists():
                    value = json.loads(result_path.read_text())
                    validate_result(value, config, binding, task, key)
                    status["status"] = "REUSED_VERIFIED"
                else:
                    replacements = {"{config}": str(config_path), "{task}": task,
                                    "{out}": str(result_path), "{run_key}": key,
                                    "{binding}": str(out/"binding.json")}
                    argv = command.copy()
                    for marker, value in replacements.items():
                        argv = [token.replace(marker, value) for token in argv]
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                    remaining = max(.01, deadline-time.monotonic())
                    with (path/"runner.log").open("w") as log:
                        run_process(argv, env=env, log=log, timeout=remaining)
                    value = json.loads(result_path.read_text())
                    validate_result(value, config, binding, task, key)
                    status["status"] = "DONE"
            except Exception as error:
                status.update(status="FAILED", reason=f"{type(error).__name__}: {error}")
                if result_path.exists():
                    result_path.replace(path/f"rejected-{time.time_ns()}.json")
            with lock:
                states.append(status)
                (out/"status.json").write_text(json.dumps(states, indent=2, ensure_ascii=False))

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in gpus]
        for future in futures:
            future.result()
    states.extend(dict(config=c["id"], task=t, run_key=k, status="PENDING_DEADLINE") for c, t, k in jobs)
    (out/"status.json").write_text(json.dumps(states, indent=2, ensure_ascii=False))
    return states


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", default="configs/catalog.json")
    p.add_argument("--binding", required=True)
    p.add_argument("--runner", required=True, help="JSON file containing argv array")
    p.add_argument("--out", default="results")
    p.add_argument("--gpus", nargs="+", default=["0", "2", "4"])
    p.add_argument("--tasks", nargs="+", default=["GQA", "POPE", "MME", "MMBench"])
    p.add_argument("--hours", type=float, default=6.)
    p.add_argument("--allow-conditional", action="store_true")
    args = p.parse_args()
    load = lambda path: json.loads(Path(path).read_text())
    run(load(args.catalog)["configs"], load(args.binding), load(args.runner), args.out,
        args.gpus, args.tasks, args.hours, args.allow_conditional)

```

## legacy_reference/tests/test_elva.py

```python
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

from elva.budget import Schedule, Trace, feasible_suffixes, choose_suffix
from elva.evidence import (unit, select_stage1, question_visual_mass, rank_drift,
                           average_ranks, compact_sequence)
from elva.counterfactual import Swap, run_swap, answer_nll, ESTIMAND
from elva.toy_adapter import ToyAdapter
from elva.learning import (PairModel, pair_features, split_image, fit_pairs,
                           choose_swap, calibrate_threshold)
from elva.integration import BoundRuntime
from elva.capture import LayerCapture
from make_catalog import build_catalog
from run_queue import digest, validate_result, run_process, conditional_ready, run as run_queue
from elva.benchmark import paired_ttft, summarize
from elva.policy import propose, feature_catalog, decide, collect_catalog


class BudgetTests(unittest.TestCase):
    def test_requested_plateau_and_inverse(self):
        for counts, budget in [((60, 28, 12), 1024), ((90, 42, 18), 1536), ((120, 56, 24), 2048)]:
            p = Schedule(counts, (10, 10, 12))
            self.assertEqual(p.work, budget)
            self.assertEqual(p.events, ((10, counts[1]), (20, counts[2])))
            self.assertEqual(sum(p.trace), budget)
            self.assertEqual((1-p.compression(576))*576*32, budget)
            self.assertEqual(p.trace[9], counts[0])
            self.assertEqual(p.trace[10], counts[1])
            self.assertEqual(p.trace[20], counts[2])

    def test_naive_request_is_over_budget(self):
        self.assertEqual(Schedule((60, 30, 16), (10, 11, 11)).work, 1106)
        self.assertNotEqual(1106, 1024)

    def test_invalid_schedules(self):
        for counts, lengths in [((10, 10), (2, 2)), ((10, 20), (2, 2)), ((10,), (0,)),
                                ((0,), (2,)), ((True,), (2,)), ((10,), (2.5,)), ((), ())]:
            with self.subTest(counts=counts, lengths=lengths), self.assertRaises(ValueError):
                Schedule(counts, lengths)

    def test_square_cost_is_not_work(self):
        a, b = Schedule((60, 28, 12), (10, 10, 12)), Schedule((60, 32, 12), (10, 8, 14))
        self.assertEqual(a.work, b.work)
        self.assertNotEqual(a.proxies(7), b.proxies(7))

    def test_trace_records_before_block(self):
        p = Schedule((5, 2), (2, 3))
        trace = Trace(p.work, p.depth)
        for block, k in enumerate(p.trace, 1):
            trace.before_block(block, k)
        p.validate_trace(trace.finish())
        with self.assertRaises(ValueError):
            trace.before_block(5, 2)

    def test_trace_rejects_resurrection_and_shortfall(self):
        trace = Trace(20, 4)
        trace.before_block(1, 4)
        with self.assertRaises(ValueError):
            trace.before_block(2, 5)
        with self.assertRaises(ValueError):
            trace.finish()

    def test_suffix_solver_against_exhaustive_integer_enumeration(self):
        # Independent brute-force baseline over ALL small catalogs.
        cases = 0
        for depth in range(3, 8):
            for completed in range(1, depth-1):
                boundaries = tuple(range(completed+1, depth))
                for current in range(2, 7):
                    spent = completed*current
                    for budget in range(spent+1, spent+(depth-completed)*current+1):
                        finals = tuple(range(1, current))
                        actual = feasible_suffixes(depth=depth, completed=completed, spent=spent,
                                                  current_k=current, budget=budget,
                                                  boundaries=boundaries, finals=finals)
                        actual_set = {(p.counts, p.lengths) for p in actual}
                        brute = set()
                        for boundary, middle, final in itertools.product(boundaries, range(1, current+1), finals):
                            d1, d2 = boundary-completed, depth-boundary
                            if final < middle and spent+middle*d1+final*d2 == budget:
                                brute.add(((middle, final), (d1, d2)))
                        self.assertEqual(actual_set, brute)
                        cases += 1
        self.assertEqual(cases, 1000)

    def test_suffix_choice_abstains_and_rejects_cost_mismatch(self):
        catalog = feasible_suffixes(depth=32, completed=10, spent=600, current_k=60,
                                    budget=1024, boundaries=(18, 20, 26), finals=(12,))
        self.assertEqual(len(catalog), 3)
        self.assertEqual(choose_suffix(catalog, [0, -.1, .02], threshold=.05), 0)
        self.assertEqual(choose_suffix(catalog, [0, -.1, .2], threshold=.05), 2)
        with self.assertRaises(ValueError):
            choose_suffix([catalog[0], Schedule((5,), (2,))], [0, 1])
        with self.assertRaises(ValueError):
            choose_suffix(catalog, [0, float("nan"), 2])


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        r = np.random.default_rng(3)
        self.pen, self.early = r.normal(size=(8, 5)), r.normal(size=(8, 5))
        self.ids = np.arange(8)*7+100
        self.a, self.b = np.arange(8)/40, np.arange(8)[::-1]/40

    def select(self, **kw):
        return select_stage1(self.pen, self.a, self.ids, 3, early=self.early,
                             early_scores=self.b, early_ids=self.ids, **kw)

    def test_payload_and_projector_width_protected(self):
        for guide in ("pen", "l9", "dual"):
            out = self.select(guide=guide)
            np.testing.assert_array_equal(out["payload"], self.pen[out["keep"]])
            self.assertEqual(out["payload"].shape, (3, 5))

    def test_dual_kernel_is_weighted_cosine(self):
        beta = .3
        out = self.select(guide="dual", beta=beta)
        p = self.pen/np.linalg.norm(self.pen, axis=1, keepdims=True)
        e = self.early/np.linalg.norm(self.early, axis=1, keepdims=True)
        np.testing.assert_allclose(out["metric"]@out["metric"].T,
                                   (1-beta)*(p@p.T)+beta*(e@e.T), atol=1e-12)

    def test_dual_endpoints_match_single_layer(self):
        for beta, single in [(0., "pen"), (1., "l9")]:
            dual, one = self.select(guide="dual", beta=beta), self.select(guide=single)
            np.testing.assert_array_equal(dual["keep"], one["keep"])
            np.testing.assert_array_equal(dual["metric"], one["metric"])

    def test_geometry_and_score_factor_ablation(self):
        geometry_only = self.select(guide="dual", score_guide="pen")
        np.testing.assert_array_equal(geometry_only["scores"], self.a)
        score_only = self.select(guide="pen", score_guide="dual")
        np.testing.assert_array_equal(score_only["metric"], self.pen)
        np.testing.assert_array_equal(score_only["scores"], (self.a+self.b)/2)

    def test_original_cover_required(self):
        with self.assertRaises(RuntimeError):
            self.select(use_cover=True)

    def test_mutating_callback_cannot_change_original_payload(self):
        before = self.pen.copy()
        def malicious(**kwargs):
            kwargs["metric_features"][:] = 0
            kwargs["scores"][:] = 0
            kwargs["patch_ids"][:] = 999
            return np.array([0, 3, 6])
        out = self.select(use_cover=True, cover=malicious)
        np.testing.assert_array_equal(self.pen, before)
        np.testing.assert_array_equal(out["payload"], before[[0, 3, 6]])

    def test_zero_vector_finite_and_frozen_ids(self):
        self.pen[0] = 0
        out = self.select(guide="dual", frozen_keep=[0, 4, 7])
        self.assertTrue(np.isfinite(out["metric"]).all())
        np.testing.assert_array_equal(out["ids"], self.ids[[0, 4, 7]])

    def test_tiny_normalization_matches_documented_epsilon(self):
        result = unit([[0., 0.], [1e-30, 0.]])
        np.testing.assert_allclose(result, [[0, 0], [1e-18, 0]], atol=0)

    def test_alignment_nonfinite_and_duplicate_keep_rejected(self):
        with self.assertRaises(ValueError):
            select_stage1(self.pen, self.a, self.ids, 3, guide="l9", early=self.early,
                          early_scores=self.b, early_ids=self.ids[::-1])
        with self.assertRaises(ValueError):
            self.select(frozen_keep=[0, 0, 2])
        self.pen[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.select()

    def test_causal_normalization_all_visible_keys(self):
        z = np.zeros((2, 4))
        mask = np.array([[1, 1, 0, 0], [1, 1, 1, 1]], bool)
        mass = question_visual_mass(z, mask, [0, 1])
        np.testing.assert_allclose(mass, [.375, .375])
        self.assertLess(mass.sum(), 1)

    def test_attention_mask_padding_extremes(self):
        z = np.array([[10000, -10000, np.nan]])
        np.testing.assert_allclose(question_visual_mass(z, np.array([[1, 1, 0]], bool), [0]), [1])
        with self.assertRaises(ValueError):
            question_visual_mass(z, np.zeros_like(z, dtype=bool), [0])
        with self.assertRaises(ValueError):
            question_visual_mass(z, np.array([[1, 1, 0]]), [0])

    def test_rank_ties_and_history_original_id_alignment(self):
        np.testing.assert_allclose(average_ranks([5, 5, 5]), [.5, .5, .5])
        a = rank_drift([3, 1, 2], [20, 50, 10], [1, 2, 99, 3], [50, 10, 99, 20])
        b = rank_drift([3, 1, 2], [20, 50, 10], [3, 1, 2], [20, 50, 10])
        np.testing.assert_array_equal(a, b)
        with self.assertRaises(ValueError):
            rank_drift([3, 1], [20, 50], [1], [20])

    def test_compaction_keeps_text_and_original_positions(self):
        h = np.arange(24).reshape(6, 4).astype(float)
        compact, p, ids, rows = compact_sequence(h, [2, 5, 7, 9, 12, 20], [-1, 88, 4, 23, -1, -1], [4])
        np.testing.assert_array_equal(rows, [0, 2, 4, 5])
        np.testing.assert_array_equal(p, [2, 7, 12, 20])
        np.testing.assert_array_equal(compact, h[rows])
        with self.assertRaises(ValueError):
            compact_sequence(h, [2, 5, 7, 9, 12, 20], [-1, 88, 4, 23, -1, -1], [999])


class CounterfactualTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ToyAdapter()
        self.prefix = self.adapter.capture(2)
        self.swap = Swap(2, 5, (11, 14, 17), (20, 23), (26, 29), (11, 14))

    def execute(self, mode="native", swap=None):
        return run_swap(self.adapter, self.prefix, swap or self.swap,
                        image_key="fixture-image", question_key="fixture-question", mode=mode)

    def test_actual_causal_decoder_branches_and_work(self):
        out = self.execute()
        self.assertEqual(out["trace"], [12, 12, 5, 5, 5, 2, 2, 2])
        self.assertEqual(out["work"], 45)
        self.assertTrue(math.isfinite(out["delta"]))
        self.assertEqual(out["final_ids"], [11, 14])

    def test_swap_sign_reverses(self):
        reverse = Swap(2, 5, self.swap.common, self.swap.group_b, self.swap.group_a, self.swap.final)
        self.assertAlmostEqual(self.execute()["delta"], -self.execute(swap=reverse)["delta"], places=12)

    def test_support_alignment_changes_cache_estimuand(self):
        native, aligned = self.execute(), self.execute("aligned_support")
        self.assertNotAlmostEqual(native["nll_a"], aligned["nll_a"], places=10)
        self.assertEqual(aligned["cache_mode"], "aligned_support")

    def test_repeated_noop_actual_forwards_are_identical(self):
        values = []
        for _ in range(4):
            state = self.adapter.fork(self.prefix)
            self.adapter.prune(state, self.swap.common+self.swap.group_a)
            self.adapter.advance(state, 5)
            self.adapter.prune(state, self.swap.final)
            values.append(self.adapter.score_suffix(state)["nll"])
        self.assertEqual(max(values)-min(values), 0)

    def test_shared_prefix_immutable(self):
        before = self.adapter.inspect(self.prefix)
        self.execute()
        self.assertEqual(before, self.adapter.inspect(self.prefix))

    def test_aliasing_fork_is_rejected(self):
        self.adapter.fork = lambda state: state
        with self.assertRaises(ValueError):
            self.execute()

    def test_invalid_sets_and_unknown_group(self):
        for args in [(2, 2, (1,), (2,), (3,), (1,)),
                     (2, 5, (1,), (1,), (3,), (1,)),
                     (2, 5, (1,), (2,), (3, 4), (1,)),
                     (2, 5, (1,), (2,), (3,), (7,))]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                Swap(*args)
        unknown = Swap(2, 5, (11, 14), (999,), (20,), (11,))
        with self.assertRaises(ValueError):
            self.execute(swap=unknown)

    def test_answer_nll_shift_not_same_position(self):
        logits = np.log([[.8, .2], [.6, .4], [.1, .9]])
        actual = answer_nll(logits, [0, 1, 0], np.array([0, 1, 1], bool))
        self.assertAlmostEqual(actual, (-math.log(.2)-math.log(.6))/2)
        with self.assertRaises(ValueError):
            answer_nll(logits, [0, 1, 0], np.zeros(3, bool))


class LearningTests(unittest.TestCase):
    def test_shared_policy_catalog_no_resurrection_and_limit(self):
        ids = np.arange(60)*3+19
        scores = np.arange(60)/60
        catalog = propose(scores, ids, np.arange(28), 12)
        self.assertEqual(len(catalog.swaps), 15)
        self.assertTrue(set(catalog.final).issubset(catalog.baseline))
        for common, inserted, removed in catalog.swaps:
            self.assertEqual(len(common)+len(inserted), 28)
            self.assertEqual(len(inserted), len(removed))
            self.assertTrue(set(catalog.final).issubset(common))
        features = np.c_[scores, scores**2]
        matrix = feature_catalog(catalog, features, begin=10, end=20, depth=32)
        model = PairModel(np.ones(8), np.zeros(8), "native", tuple("abcdefgh"), catalog.rule_hash)
        self.assertEqual(decide(catalog, features, model, begin=10, end=20, depth=32, threshold=0)["action"], 0)
        with self.assertRaises(ValueError):
            feature_catalog(catalog, features, begin=10, end=21, depth=32)

    def test_policy_collects_real_fixture_forwards_and_blocks_training(self):
        adapter = ToyAdapter()
        prefix = adapter.capture(2)
        ids = np.array(adapter.inspect(prefix)["active_ids"])
        scores = np.arange(12)/12
        catalog = propose(scores, ids, np.arange(7), 2, group_size=2, limit=4, begin=2, end=5, depth=8)
        x = scores[:, None]
        rows = collect_catalog(adapter, prefix, catalog, ids, x, begin=2, end=5, depth=8,
                               image_key="fixture", question_key="question", feature_names=list("abcd"))
        self.assertEqual(len(rows), 3)
        self.assertEqual({row["work"] for row in rows}, {51})
        self.assertTrue(all(row["fixture"] for row in rows))
        with self.assertRaises(ValueError):
            fit_pairs(rows, feature_names=list("abcd"), catalog_hash=catalog.rule_hash, cache_mode="native")

    def records(self):
        keys = [f"test-generated-{i}" for i in range(100) if split_image(f"test-generated-{i}") == "train"]
        result = []
        for i in range(20):
            x = np.array([i/20, (i % 3)-1.])
            result.append(dict(schema=ESTIMAND, cache_mode="native", catalog_hash="catalog",
                               feature_names=["a", "b"], feature_cutoff=10, begin=10,
                               image_key=keys[i % 5], x=x.tolist(), delta=float(x@np.array([2., -.3]))))
        return result

    def test_pair_features_context_and_antisymmetry(self):
        x = np.arange(30).reshape(10, 3)/10
        a = pair_features(x, [0, 1], [2, 3], [4, 5], begin=10, end=20, depth=32)
        b = pair_features(x, [0, 1], [4, 5], [2, 3], begin=10, end=20, depth=32)
        np.testing.assert_array_equal(a, -b)
        changed = pair_features(x, [6, 7], [2, 3], [4, 5], begin=10, end=20, depth=32)
        self.assertFalse(np.allclose(a, changed))

    def test_grouped_split_stable(self):
        self.assertEqual(split_image("canonical-image"), split_image("canonical-image"))
        self.assertEqual(set(split_image(f"img-{i}") for i in range(100)), {"train", "validation", "test"})

    def test_fit_save_load_and_prediction(self):
        records = self.records()
        model = fit_pairs(records, feature_names=["a", "b"], catalog_hash="catalog", cache_mode="native", ridge=1e-6)
        x = np.array([r["x"] for r in records])
        np.testing.assert_allclose(model.predict(x), [r["delta"] for r in records], atol=1e-6)
        np.testing.assert_allclose(model.predict(-x), -model.predict(x))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"head.json"
            model.save(path)
            loaded = PairModel.load(path, feature_names=["a", "b"], catalog_hash="catalog", cache_mode="native")
            np.testing.assert_array_equal(model.predict(x), loaded.predict(x))
            with self.assertRaises(ValueError):
                PairModel.load(path, feature_names=["a", "b"], catalog_hash="old", cache_mode="native")

    def test_old_schema_future_features_and_fixture_rejected(self):
        for key, value in [("schema", "old_e020"), ("feature_cutoff", 11), ("fixture", True), ("cache_mode", "aligned_support")]:
            records = self.records()
            records[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                fit_pairs(records, feature_names=["a", "b"], catalog_hash="catalog", cache_mode="native")

    def test_validation_images_cannot_enter_fit(self):
        records = self.records()
        records[0]["image_key"] = next(f"held-{i}" for i in range(100) if split_image(f"held-{i}") != "train")
        with self.assertRaises(ValueError):
            fit_pairs(records, feature_names=["a", "b"], catalog_hash="catalog", cache_mode="native")

    def test_candidate_max_calibration_and_all_noop(self):
        p = np.array([[0, .8, .2], [0, .1, .6]])
        good = calibrate_threshold(p, [[0, .4, -.3], [0, -.1, .2]], [0, 0, 0])
        bad = calibrate_threshold(p, [[0, -.4, -.3], [0, -.1, -.2]], [0, 0, 0])
        self.assertEqual(good["status"], "CANDIDATE")
        self.assertEqual(bad["status"], "NO_SUPPORTED_DECISIONS")
        self.assertEqual(bad["accepted"], 0)
        self.assertEqual(choose_swap([0, .1], threshold=.2), 0)


class IntegrationAndCatalogTests(unittest.TestCase):
    def test_queue_runs_resumes_and_recovers_bad_result(self):
        # Mock runner lives only in a temporary directory and labels its
        # metrics/binding as fixtures. No model score is produced.
        source = '''import sys,json,hashlib,pathlib
cfg,task,out,key,binding=sys.argv[1:]
c=json.loads(pathlib.Path(cfg).read_text()); b=json.loads(pathlib.Path(binding).read_text())
m=c['method']; s=m['schedule']; trace=[k for k,d in zip(s['counts'],s['lengths']) for _ in range(d)]
h=lambda v: hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
value=dict(run_key=key,config_hash=c['config_hash'],task=task,backend=b['backend'],cache_mode=m['cache_mode'],samples=2,scorer_sha=b['scorer_sha'],metrics={'fixture_metric':1.},binding_hash=h(b),trace_audit=dict(expected=trace,observed_samples=2,mismatch_count=0))
pathlib.Path(out).write_text(json.dumps(value))
'''
        configs = build_catalog()[:2]
        binding = dict(source_commit="FIXTURE", model_sha="FIXTURE", processor_sha="FIXTURE",
                       scorer_sha="FIXTURE", prompt_sha="FIXTURE", tokenizer_sha="FIXTURE",
                       dataset_sha="FIXTURE", backend="FIXTURE", cache_mode="native",
                       native_selectors={"fixture": "FIXTURE"}, original_tokens=576, depth=32)
        with tempfile.TemporaryDirectory() as d:
            runner, output = Path(d)/"runner.py", Path(d)/"out"
            runner.write_text(source)
            command = [sys.executable, str(runner), "{config}", "{task}", "{out}", "{run_key}", "{binding}"]
            args = (configs, binding, command, output, ["0", "1"], ["FIXTURE_TASK"], .01)
            self.assertEqual([s["status"] for s in run_queue(*args)], ["DONE", "DONE"])
            self.assertEqual([s["status"] for s in run_queue(*args)], ["REUSED_VERIFIED", "REUSED_VERIFIED"])
            path = next(output.glob("*/result.json"))
            value = json.loads(path.read_text())
            value["trace_audit"]["mismatch_count"] = 1
            path.write_text(json.dumps(value))
            self.assertEqual(sorted(s["status"] for s in run_queue(*args)), ["FAILED", "REUSED_VERIFIED"])
            self.assertEqual(sorted(s["status"] for s in run_queue(*args)), ["DONE", "REUSED_VERIFIED"])

    def test_conditional_queue_requires_new_real_evidence(self):
        config = next(c for c in build_catalog() if c["method"]["stage2"]["first_selector"] == "TRANSIENT_PAIR_CURRENT")
        self.assertFalse(conditional_ready(config, {}))
        binding = dict(transient_adapter_ready=True, transient_gate=dict(
            schema=ESTIMAND, status="READY", cache_mode="native", weights_sha="test-only",
            catalog_hash="test-only", real_labels=True, heldout_gain_over_noise=True, generation_checked=True))
        self.assertTrue(conditional_ready(config, binding))
        binding["transient_gate"]["schema"] = "old_e020"
        self.assertFalse(conditional_ready(config, binding))

    def test_first_token_callback_timing_and_missing_token(self):
        rows = paired_ttft(["fixture0", "fixture1"], ("a", "b"), lambda s, m: s+m,
                           lambda: None, rounds=2, warmup=0)
        self.assertEqual(len(rows), 8)
        self.assertEqual(summarize(rows)["a"]["observations"], 4)
        with self.assertRaises(ValueError):
            paired_ttft([0], ("a", "b"), lambda s, m: None, lambda: None, warmup=0)

    def test_capture_completed_block_indices_and_removal(self):
        class Handle:
            def __init__(self, hooks, fn): self.hooks, self.fn = hooks, fn
            def remove(self): self.hooks.remove(self.fn)
        class Block:
            def __init__(self, i): self.i, self.hooks = i, []
            def register_forward_hook(self, fn):
                self.hooks.append(fn)
                return Handle(self.hooks, fn)
            def __call__(self, x):
                y = x+self.i
                for fn in self.hooks: fn(self, (x,), (y,))
                return y
        blocks = [Block(i) for i in range(1, 13)]
        hidden = [np.zeros((1, 5, 3))]
        with LayerCapture(blocks) as capture:
            for block in blocks: hidden.append(block(hidden[-1]))
            early, pen = capture.get()
            np.testing.assert_array_equal(early, hidden[9])
            np.testing.assert_array_equal(pen, hidden[-2])
        self.assertTrue(all(not block.hooks for block in blocks))

    def test_capture_invalid_layer_rejected(self):
        with self.assertRaises(ValueError):
            LayerCapture([None]*8, early_completed=9)

    def test_catalog_distinct_and_all_budgets_exact(self):
        configs = build_catalog()
        self.assertEqual(len(configs), len({c["config_hash"] for c in configs}))
        self.assertEqual(len(configs), 83)
        self.assertEqual(sum(not c["method"]["conditional"] for c in configs), 74)
        for c in configs:
            p = Schedule(**c["method"]["schedule"])
            self.assertEqual(p.work, c["method"]["budget"])
            self.assertEqual(p.depth, 32)

    def test_native_runtime_callbacks_and_positions(self):
        state = dict(ids=[101, 303, 202, 404, 505], protected=[1, 10])
        def inspect(s):
            return dict(active_ids=s["ids"], protected_positions=s["protected"], backend="bound")
        def select(s, *, k, **kwargs):
            return s["ids"][:k]
        def compact(s, *, keep_ids):
            return dict(ids=list(keep_ids), protected=s["protected"].copy())
        p = Schedule((5, 2), (2, 3))
        rt = BoundRuntime(p, select, compact, inspect)
        for block in range(1, 6):
            rt.before_block(block, state)
            state = rt.after_block(block, state, selector="SOURCE_BOUND")
        self.assertEqual(rt.finish(), p.trace)

    def test_runtime_rejects_backend_change(self):
        p = Schedule((2, 1), (1, 1))
        state = dict(ids=[1, 2], backend="flash")
        inspect = lambda s: dict(active_ids=s["ids"], protected_positions=[7], backend=s["backend"])
        rt = BoundRuntime(p, lambda *a, **k: [1], lambda *a, **k: dict(ids=[1], backend="eager"), inspect)
        rt.before_block(1, state)
        with self.assertRaises(ValueError):
            rt.after_block(1, state, selector="native")

    def test_result_reuse_requires_trace_and_provenance(self):
        config = build_catalog()[0]
        binding = dict(backend="fixture", scorer_sha="fixture")
        schedule = Schedule(**config["method"]["schedule"])
        value = dict(run_key="key", config_hash=config["config_hash"], task="GQA", backend="fixture",
                     cache_mode="native", samples=2, scorer_sha="fixture", metrics={"fixture_only": 1.},
                     binding_hash=digest(binding), trace_audit=dict(expected=list(schedule.trace), observed_samples=2, mismatch_count=0))
        validate_result(value, config, binding, "GQA", "key")
        value["trace_audit"]["mismatch_count"] = 1
        with self.assertRaises(ValueError):
            validate_result(value, config, binding, "GQA", "key")
        value["trace_audit"]["mismatch_count"] = 0
        value["metrics"] = {"fixture_only": float("nan")}
        with self.assertRaises(ValueError):
            validate_result(value, config, binding, "GQA", "key")

    def test_timeout_process_group_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            marker = Path(d)/"should-not-exist"
            child = "import time,pathlib; time.sleep(1); pathlib.Path("+repr(str(marker))+").touch()"
            parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',"+repr(child)+"]); time.sleep(9)"
            with (Path(d)/"log").open("w") as log:
                with self.assertRaises(subprocess.TimeoutExpired):
                    run_process([sys.executable, "-c", parent], env=None, log=log, timeout=.15)
            # Parent and child receive group termination; wait through the
            # child's hypothetical write deadline using a short test subprocess.
            subprocess.run([sys.executable, "-c", "import time; time.sleep(1.1)"], check=True)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)

```

## legacy_reference/tests/test_torch_parity.py

```python
"""Run on server before ACC; absence of PyTorch is an explicit skip."""
import importlib.util
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is absent in this workspace")
class TorchParityTests(unittest.TestCase):
    def test_numpy_parity_and_payload_protection(self):
        import torch
        from elva.evidence import select_stage1 as ref, question_visual_mass as mass_ref
        from elva.torch_ops import select_stage1, question_visual_mass
        r = np.random.default_rng(4)
        x, e = r.normal(size=(12, 6)), r.normal(size=(12, 6))
        a, b, ids = r.random(12), r.random(12), np.arange(12)*3
        for guide, beta in [("pen", .5), ("l9", .5), ("dual", 0), ("dual", .5), ("dual", 1)]:
            options = dict(guide=guide, beta=beta)
            reference = ref(x, a, ids, 4, early=e, early_scores=b, early_ids=ids, **options)
            out = select_stage1(torch.tensor(x), torch.tensor(a), torch.tensor(ids), 4,
                                early=torch.tensor(e), early_scores=torch.tensor(b), early_ids=torch.tensor(ids), **options)
            np.testing.assert_array_equal(out["keep"].numpy(), reference["keep"])
            np.testing.assert_array_equal(out["payload"].numpy(), reference["payload"])
            np.testing.assert_allclose(out["metric"].numpy(), reference["metric"], atol=1e-6)
        z = r.normal(size=(2, 3, 15))
        mask = np.ones_like(z, bool)
        cols = np.array([1, 3, 7])
        actual = question_visual_mass(torch.tensor(z), torch.tensor(mask), torch.tensor(cols)).numpy()
        np.testing.assert_allclose(actual, mass_ref(z, mask, cols), atol=1e-6)


if __name__ == "__main__":
    unittest.main()

```

## make_catalog.py

```python
"""Two budgets only. Stage1 first; Stage2 requires an explicit result freeze."""
import argparse
from pathlib import Path
import json
import hashlib
import copy
from elva.budget import Schedule


TASKS = ("GQA", "POPE", "MME", "MMBench")
MAIN = {2048: Schedule((172, 90, 32), (4, 8, 20)),
        1024: Schedule((60, 28, 12), (10, 10, 12))}
SHAPES = {2048: (MAIN[2048], Schedule((172, 61, 32), (4, 16, 12)),
                 Schedule((172, 94, 32), (2, 12, 18))),
          1024: (MAIN[1024], Schedule((60, 28, 12), (8, 16, 8)),
                 Schedule((60, 32, 12), (10, 8, 14)))}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_config(budget, stage, schedule, *, phase, selector="Q_ATTN", label=""):
    if budget not in MAIN or schedule.work != budget or schedule.depth != 32:
        raise ValueError("only exact W2048 / W1024 576x32 schedules are permitted")
    method = dict(original_tokens=576, depth=32, budget=budget,
                  schedule=dict(counts=list(schedule.counts), lengths=list(schedule.lengths)),
                  stage1=copy.deepcopy(stage), stage2=dict(first_selector=selector, later_selector="Q_ATTN"),
                  cache_mode="native", merge=False, learned=False)
    sha = digest(method)
    return dict(id=f"R3_W{budget}_{label}_{sha[:10]}", phase=phase, config_hash=sha, method=method)


def stage(guide, cover_family="none", beta=None, score_guide="same", slots=12):
    value = dict(guide=guide, cover_family=cover_family, cover=cover_family != "none",
                beta=beta if guide == "dual" or score_guide == "dual" else None,
                score_guide=score_guide, payload="pen", saliency="BIND_ORIGINAL_SAME_RULE")
    if cover_family == "protected_coverage":
        from elva.low_budget_cover import POLICY
        value.update(coverage_slots=slots, coverage_policy=POLICY)
    return value


def build_catalog(*, low_cover="protected_coverage", decisions=None, coverage_base=None):
    if low_cover not in ("protected_coverage", "legacy_pool172_then_topk"):
        raise ValueError("explicit K60 COVER strategy required")
    result, seen = [], set()
    def add(config):
        key = (config["phase"], config["config_hash"])
        if key not in seen:
            result.append(config)
            seen.add(key)
    for budget in (2048, 1024):
        family = "legacy_native" if budget == 2048 else "protected_coverage"
        for name, guide, cover, beta in (("V1", "pen", "none", None), ("V2", "l9", "none", None),
                                         ("V3", "dual", "none", .5), ("V4", "pen", family, None),
                                         ("V5", "dual", family, .5)):
            suffix = ("_POOL172" if cover == "legacy_pool172_then_topk" else
                      "_PCOV60_B12" if cover == "protected_coverage" else "")
            add(make_config(budget, stage(guide, cover, beta), MAIN[budget],
                            phase="stage1_core", label=name+suffix))
        for name, guide, beta in (("B025", "dual", .25), ("B075", "dual", .75), ("L9_COVER", "l9", None)):
            add(make_config(budget, stage(guide, family, beta), MAIN[budget],
                            phase="stage1_optional", label=name))
        for name, guide, score in (("GEO_ONLY", "dual", "pen"), ("SCORE_ONLY", "pen", "dual")):
            add(make_config(budget, stage(guide, family, .5, score_guide=score), MAIN[budget],
                            phase="stage1_optional", label=name))
    # Only execute these on the winning guide, if core evidence warrants it.
    # The unchanged PORT60 implementation is a separate source-bound control.
    bases = [stage("pen","protected_coverage"),stage("dual","protected_coverage",.5)]
    if coverage_base is not None:
        if (digest(coverage_base.get("method",{})) != coverage_base.get("config_hash")
                or coverage_base['method'].get('budget') != 1024
                or coverage_base['method']['stage1'].get('cover_family') != 'protected_coverage'
                or coverage_base['method']['stage1'].get('coverage_slots') != 12):
            raise ValueError('coverage refinement requires the actual fingerprinted W1024 PCOV/b12 base config')
        bases = [copy.deepcopy(coverage_base['method']['stage1'])]
    for base in bases:
        guide = base['guide']
        for slots in (6, 18):
            refined=copy.deepcopy(base);refined['coverage_slots']=slots
            add(make_config(1024, refined, MAIN[1024],
                            phase="stage1_optional", label=f"PCOV_{guide}_B{slots}"))
        port=copy.deepcopy(base);port['cover_family']='parameterized_native'
        port.pop('coverage_policy',None);port.pop('coverage_slots',None)
        add(make_config(1024, port, MAIN[1024],
                        phase="stage1_optional", label=f"COVER_PORT60_{guide}_B11"))
    if low_cover == "legacy_pool172_then_topk":
        # Explicit pilot-only diagnostic. It can be identically top-k when
        # legacy COVER changes only the tail of its Top172 pool.
        for name, guide, beta in (("V4", "pen", None), ("V5", "dual", .5)):
            add(make_config(1024, stage(guide, "legacy_pool172_then_topk", beta), MAIN[1024],
                            phase="diagnostic_pool", label="POOL172_CONTROL_"+name))
    if decisions:
        for budget in (2048, 1024):
            item = decisions.get("budgets", {}).get(str(budget))
            if not item:
                continue
            # Preserve a completed refined winner when regenerating Stage2.
            # The workflow revalidates its actual four-task evidence hashes.
            if item.get('winner_config'):
                winner=item['winner_config']
                if digest(winner.get('method',{})) != winner.get('config_hash'):
                    raise ValueError('edited Stage1 winner config')
                add(copy.deepcopy(winner))
            frozen = item["stage1"]
            for shape_index, schedule in enumerate(SHAPES[budget]):
                for selector in ("Q_ATTN", "RDRIFT_NATIVE"):
                    add(make_config(budget, frozen, schedule, phase="stage2",
                                    selector=selector, label=f"S{shape_index}_{selector}"))
            for selector in ("QED_NATIVE", "ATTN_COVER_NATIVE"):
                add(make_config(budget, frozen, MAIN[budget], phase="stage2",
                                selector=selector, label=selector))
    return result


def write_catalog(output, *, low_cover="protected_coverage", decisions=None, coverage_base=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configs = build_catalog(low_cover=low_cover, decisions=decisions, coverage_base=coverage_base)
    prior_path=output/"catalog.json"
    if prior_path.exists():
        previous=json.loads(prior_path.read_text())
        keep={c["id"] for c in configs}
        for old in previous.get("configs", []):
            if old.get("id", "").startswith("R3_") and old["id"] not in keep:
                stale=output/(old["id"]+".json")
                if stale.is_file():
                    if json.loads(stale.read_text()).get("config_hash") != old.get("config_hash"):
                        raise ValueError("edited stale generated config; move it before regenerating")
                    stale.unlink()
    for config in configs:
        (output/(config["id"]+".json")).write_text(json.dumps(config, indent=2)+"\n")
    catalog = dict(schema="elva_r3_two_budget", configs=configs,
                   budgets=[2048, 1024], default_phase="stage1_core", low_cover="protected_coverage",
                   include_pool_diagnostic=low_cover=="legacy_pool172_then_topk")
    (output/"catalog.json").write_text(json.dumps(catalog, indent=2)+"\n")
    return catalog


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="configs")
    parser.add_argument("--include-pool-diagnostic", action="store_true")
    parser.add_argument("--decisions")
    parser.add_argument("--coverage-refine-from", help="actual W1024 PCOV/b12 config after guide/beta choice")
    args = parser.parse_args()
    decisions = json.loads(Path(args.decisions).read_text()) if args.decisions else None
    coverage_base = json.loads(Path(args.coverage_refine_from).read_text()) if args.coverage_refine_from else None
    value = write_catalog(args.out, low_cover="legacy_pool172_then_topk" if args.include_pool_diagnostic else "protected_coverage", decisions=decisions,coverage_base=coverage_base)
    print(json.dumps({"configs": len(value["configs"]), "core": sum(c["phase"]=="stage1_core" for c in value["configs"]), "budgets":value["budgets"]}))

```

## native_result_io.py

```python
"""Call from the real project runner after generation, outside TTFT timing."""
import hashlib
import json
from pathlib import Path
from make_catalog import digest
from elva.workflow import validate_result


def sample_record(*, image_key, question_key, prediction, stage1_ids, stage1_audit,
                  trace, question=None, reference=None):
    if hasattr(stage1_ids, "detach"):
        stage1_ids = stage1_ids.detach().cpu().tolist()
    elif hasattr(stage1_ids, "tolist"):
        stage1_ids = stage1_ids.tolist()
    return dict(image_key=str(image_key), question_key=str(question_key), prediction=prediction,
                question=question, reference=reference, stage1_ids=stage1_ids,
                stage1_audit=stage1_audit, trace=list(trace))


def finish_task(output_path, *, records, official_metrics, config, binding, task, run_key,
                fixture=False):
    """Official scorer supplies metrics; this helper never invents a score."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions = out.parent/"predictions.jsonl"
    with predictions.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
    value = dict(schema="elva_r3_vlm_result", inference_kind="full_vlm_generate", fixture=fixture,
                 config_hash=config["config_hash"], binding_hash=digest(binding), task=task,
                 run_key=run_key, backend=binding["backend"], cache_mode=config["method"]["cache_mode"],
                 scorer_sha=binding["scorer_sha"], samples=len(records), metrics=official_metrics,
                 predictions_file=predictions.name,
                 predictions_sha256=hashlib.sha256(predictions.read_bytes()).hexdigest())
    validate_result(value, config, binding, task, run_key, out.parent, allow_fixture=fixture)
    out.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    return value

```

## partition_manifest.py

```python
"""Apply the global pilot limit BEFORE disjoint sharding."""
import argparse
import hashlib
import json
from pathlib import Path


def partition(rows, *, shards, limit, mode="clip_image"):
    if type(shards) is not int or shards < 1 or type(limit) is not int or limit < 1:
        raise ValueError("positive integer global limit and shard count required")
    if mode not in ("clip_image", "vlm_question"):
        raise ValueError("explicit image-only or question evaluation mode required")
    unique = {}
    for row in rows:
        image = row.get("image_key")
        question = row.get("question_key")
        if not isinstance(image, str) or not image or (mode == "vlm_question" and
                (not isinstance(question, str) or not question)):
            raise ValueError("canonical identities required")
        key = image if mode == "clip_image" else (image, question)
        if key in unique and mode == "vlm_question":
            raise ValueError("duplicate VLM example in the input manifest")
        unique.setdefault(key, row)
    keys = sorted(unique, key=lambda key: hashlib.sha256(repr(key).encode()).hexdigest())[:limit]
    groups = [[unique[key] for i, key in enumerate(keys) if i % shards == shard] for shard in range(shards)]
    assert sum(map(len, groups)) == min(limit, len(unique))
    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSON array of real image/example records")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--limit", type=int, required=True, help="global unique input count, not per-worker count")
    parser.add_argument("--mode", choices=("clip_image", "vlm_question"), default="clip_image")
    args = parser.parse_args()
    groups = partition(json.loads(Path(args.manifest).read_text()), shards=args.shards, limit=args.limit, mode=args.mode)
    out = Path(args.out);out.mkdir(parents=True, exist_ok=True)
    for index, rows in enumerate(groups):
        (out/f"shard_{index:02d}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({"total":sum(map(len,groups)),"per_shard":list(map(len,groups))}))

```

## run_queue.py

```python
"""Explicit phase queue, current GPU checks, source-bound per-sample results."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import math
import tempfile
from make_catalog import digest, TASKS
from elva.workflow import (blocked_reason, result_key, validate_binding, validate_result,
                           validate_decision, collect_complete)
from gpu_resources import inventory, choose, respect_visible, Lease


def run_process(argv, *, env, log, timeout):
    process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
    except BaseException:
        # This process group was created above for this exact queue job.
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except ProcessLookupError:
            pass
        raise
    if code:
        raise subprocess.CalledProcessError(code, argv)


def run(catalog, binding, command, output, *, phase="stage1_core", requested=("auto",),
        workers=3, reserve=1, hours=6., tasks=TASKS, budgets=(2048, 1024),
        decisions=None, gpu_probe=inventory, lease_directory=None, allow_fixture=False, ids=None):
    validate_binding(binding, allow_fixture=allow_fixture)
    if not math.isfinite(hours) or hours <= 0 or phase not in ("stage1_core", "stage1_optional", "stage2"):
        raise ValueError("finite positive deadline and explicit supported phase required")
    if not budgets or set(budgets)-{2048, 1024} or set(tasks)-set(TASKS):
        raise ValueError("only the two declared budgets and bound four tasks are accepted")
    markers = ("{config}", "{task}", "{out}", "{run_key}", "{binding}")
    if (not isinstance(command, list) or not command or any(not isinstance(v, str) for v in command)
            or any(not any(p in token for token in command) for p in markers)):
        raise ValueError("real runner argv must contain all five routing placeholders")
    configs = catalog["configs"]
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    binding_path = out/("binding-"+digest(binding)+".json")
    binding_path.write_text(json.dumps(binding, indent=2)+"\n")
    jobs, states = deque(), []
    for budget in budgets:
        if phase == "stage1_optional":
            collect_complete(configs, binding, out, budget, allow_fixture=allow_fixture)
        if phase == "stage2":
            if decisions is None:
                raise ValueError("Stage2 cannot start before the completed Stage1 result freeze")
            validate_decision(decisions, binding, configs, out, budget, allow_fixture=allow_fixture)
    selected_configs = [c for c in configs if c["phase"] == phase and c["method"]["budget"] in budgets]
    if ids is not None:
        if not ids or set(ids)-{c["id"] for c in selected_configs}:
            raise ValueError("--ids must exactly name configurations in the requested phase/budgets")
        selected_configs = [c for c in selected_configs if c["id"] in ids]
    if not selected_configs:
        raise ValueError("no configs for this phase; generate Stage2 from the actual freeze first")
    for config in selected_configs:
        reason = blocked_reason(config, binding)
        if reason:
            states.append(dict(config=config["id"], status="BLOCKED", reason=reason))
            continue
        for task in tasks:
            jobs.append((config, task, result_key(config, binding, task)))
    selected, unused = choose(respect_visible(gpu_probe(), os.environ.get("CUDA_VISIBLE_DEVICES")), requested, workers, reserve)
    lease_directory = lease_directory or os.environ.get("ELVA_GPU_LOCK_DIR") or str(Path(tempfile.gettempdir())/f"elva-gpu-leases-{os.getuid()}")
    state_lock = threading.Lock()
    deadline = time.monotonic()+hours*3600
    def save_state(status):
        with state_lock:
            states.append(status)
            tmp = out/"status.tmp"
            tmp.write_text(json.dumps(dict(phase=phase, jobs=states,
                           reserved_or_unused=[g.uuid for g in unused]), indent=2)+"\n")
            tmp.replace(out/"status.json")
    def worker(device):
        try:
            with Lease(lease_directory, device.uuid):
                while time.monotonic() < deadline:
                    current = next((g for g in gpu_probe() if g.uuid == device.uuid), None)
                    if current is None or not current.available():
                        save_state(dict(gpu=device.uuid, status="GPU_BECAME_BUSY"))
                        return
                    with state_lock:
                        if not jobs:
                            return
                        config, task, key = jobs.popleft()
                    folder = out/key
                    folder.mkdir(exist_ok=True)
                    config_path, result_path = folder/"config.json", folder/"result.json"
                    config_path.write_text(json.dumps(config, indent=2)+"\n")
                    status = dict(config=config["id"], task=task, run_key=key,
                                  physical_gpu=device.index, gpu_uuid=device.uuid)
                    try:
                        reused = False
                        if result_path.exists():
                            try:
                                value = json.loads(result_path.read_text())
                                validate_result(value, config, binding, task, key, folder, allow_fixture=allow_fixture)
                                reused = True
                            except Exception:
                                result_path.replace(folder/f"rejected-{time.time_ns()}.json")
                        if not reused:
                            replacements = dict(zip(markers, (str(config_path), task, str(result_path), key, str(binding_path))))
                            argv = command.copy()
                            for marker, value in replacements.items():
                                argv = [token.replace(marker, value) for token in argv]
                            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device.uuid,
                                       ELVA_GPU_UUID=device.uuid, ELVA_RUN_KEY=key)
                            with (folder/"runner.log").open("w") as log:
                                run_process(argv, env=env, log=log, timeout=max(.01, deadline-time.monotonic()))
                            value = json.loads(result_path.read_text())
                            validate_result(value, config, binding, task, key, folder, allow_fixture=allow_fixture)
                        status["status"] = "REUSED_VERIFIED" if reused else "DONE"
                    except Exception as error:
                        status.update(status="FAILED", reason=f"{type(error).__name__}: {error}")
                    save_state(status)
        except BlockingIOError:
            save_state(dict(gpu=device.uuid, status="LEASE_BUSY"))
        except Exception as error:
            save_state(dict(gpu=device.uuid, status="GPU_WORKER_FAILED", reason=f"{type(error).__name__}: {error}"))
    if selected and jobs:
        with ThreadPoolExecutor(max_workers=len(selected)) as executor:
            list(executor.map(worker, selected))
    for config, task, key in jobs:
        states.append(dict(config=config["id"], task=task, run_key=key, status="PENDING_RESOURCE_OR_DEADLINE"))
    save_state(dict(status="PHASE_QUEUE_FINISHED", selected_gpu_uuids=[g.uuid for g in selected]))
    return states


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="configs/catalog.json")
    parser.add_argument("--binding", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--out", default="results")
    parser.add_argument("--phase", choices=("stage1_core", "stage1_optional", "stage2"), default="stage1_core")
    parser.add_argument("--gpus", nargs="+", default=["auto"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--reserve", type=int, default=1)
    parser.add_argument("--budgets", nargs="+", type=int, default=[2048, 1024])
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--hours", type=float, default=6.)
    parser.add_argument("--decisions")
    parser.add_argument("--lease-dir")
    parser.add_argument("--ids", nargs="+", help="exact config IDs; use to run a bounded optional ablation")
    args = parser.parse_args()
    load = lambda path: json.loads(Path(path).read_text())
    run(load(args.catalog), load(args.binding), load(args.runner), args.out, phase=args.phase,
        requested=args.gpus, workers=args.workers, reserve=args.reserve, hours=args.hours,
        tasks=args.tasks, budgets=args.budgets, decisions=load(args.decisions) if args.decisions else None,
        lease_directory=args.lease_dir, ids=args.ids)

```

## summarize_results.py

```python
"""List complete four-task results; normalized R4 only with bound dense references."""
import argparse
import json
import math
from pathlib import Path
from make_catalog import TASKS
from elva.workflow import result_key, validate_result


def summarize(configs, binding, results):
    rows = []
    references = binding.get("normalized_metric_reference", {})
    for c in configs:
        scores, missing = {}, []
        normalized = []
        for task in TASKS:
            key = result_key(c, binding, task)
            folder = Path(results)/key
            try:
                value = json.loads((folder/"result.json").read_text())
                validate_result(value, c, binding, task, key, folder)
                scores[task] = value["metrics"]
                ref = references.get(task, {})
                name, dense = ref.get("metric"), ref.get("dense")
                if name in value["metrics"] and type(dense) in (int, float) and math.isfinite(dense) and dense > 0:
                    normalized.append(value["metrics"][name]/dense)
            except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
                missing.append(dict(task=task, reason=str(error)))
        rows.append(dict(config=c["id"], phase=c["phase"], budget=c["method"]["budget"],
                         cover_family=c["method"]["stage1"]["cover_family"], metrics=scores,
                         missing=missing, normalized_R4=100*sum(normalized)/4 if not missing and len(normalized)==4 else None))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="configs/catalog.json")
    parser.add_argument("--binding", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default="results_summary.json")
    args = parser.parse_args()
    load = lambda p: json.loads(Path(p).read_text())
    rows = summarize(load(args.catalog)["configs"], load(args.binding), args.results)
    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False)+"\n")

```

## tail_pool_equivalence.py

```python
"""Conditional design check, not evidence about uninspected native COVER code."""
import itertools
import json


def verify():
    cases=0
    for order in itertools.permutations(range(8)):
        core=list(order[:6]);tail=core[-2:];outside=list(order[6:])
        priority={value:index for index,value in enumerate(order)}
        for count in range(3):
            for removed in itertools.combinations(tail,count):
                for added in itertools.combinations(outside,count):
                    pool=(set(core)-set(removed))|set(added)
                    assert sorted(pool,key=priority.get)[:2]==list(order[:2])
                    cases+=1
    return dict(exhaustive_small_cases=cases,legacy_pool_size=172,tail_capacity=32,
                protected_prefix=172-32,requested=60,requested_inside_protected_prefix=60<=172-32,
                condition="COVER only removes from the bottom b of score TopK and inserts lower-score outside candidates",
                conclusion="same-score top60 after that pool equals plain top60; inspect native code before applying this condition",
                pretrained_vlm_claim=False)


if __name__=='__main__':print(json.dumps(verify(),indent=2))

```

## tests/test_clip_stage1.py

```python
"""Synthetic unfused CLIP fixture only; real CLIP/server parity remains required."""
import unittest
try:
    import torch
except ImportError:
    torch = None
from elva.clip_stage1 import CLIPStage1, encode_projected
from make_catalog import build_catalog


@unittest.skipIf(torch is None,'PyTorch absent locally; execute real forward hook checks on server')
class CLIPHookTests(unittest.TestCase):
    def build(self):
        from torch import nn
        class Attention(nn.Module):
            def __init__(self):
                super().__init__();self.num_heads=2;self.scale=4**-.5
                self.q_proj=nn.Linear(8,8);self.k_proj=nn.Linear(8,8);self.v_proj=nn.Linear(8,8)
            def forward(self,hidden_states,attention_mask=None,causal_attention_mask=None):
                q=self.q_proj(hidden_states).reshape(1,577,2,4).transpose(1,2)
                k=self.k_proj(hidden_states).reshape(1,577,2,4).transpose(1,2)
                v=self.v_proj(hidden_states).reshape(1,577,2,4).transpose(1,2)
                a=((q*self.scale)@k.transpose(-1,-2)).softmax(-1)
                self.last_scores=a[0,:,0,1:].mean(0)
                self.last_scores=self.last_scores/self.last_scores.sum()
                return (a@v).transpose(1,2).reshape(1,577,8)
        class Block(nn.Module):
            def __init__(self):
                super().__init__();self.self_attn=Attention();self.norm=nn.LayerNorm(8)
            def forward(self,x):
                return (x+self.self_attn(self.norm(x)),)
        class Tower(nn.Module):
            def __init__(self):
                super().__init__();self.blocks=nn.ModuleList([Block() for _ in range(10)])
                self.calls=0
            def forward(self,x):
                self.calls+=1
                for i,b in enumerate(self.blocks):
                    x=b(x)[0]
                    if i==8:self.pen=x.detach().clone()
                return self.pen[:,1:]
        torch.manual_seed(3)
        return Tower().eval()

    def test_single_forward_source_scores_raw_payload_and_real_projector(self):
        from torch import nn
        tower=self.build();x=torch.randn(1,577,8);ids=torch.arange(576)
        config=next(c for c in build_catalog() if 'W1024_V3_' in c['id'])
        projector=nn.Linear(8,5).eval()
        projected,selected=encode_projected(vision_tower=tower,projector=projector,images=x,
            blocks=tower.blocks,original_patch_ids=ids,config=config,l9_completed=4)
        self.assertEqual(tower.calls,1)
        self.assertEqual(projected.shape,(1,60,5))
        torch.testing.assert_close(selected['scores'],.5*(tower.blocks[8].self_attn.last_scores+
                                                         tower.blocks[3].self_attn.last_scores),rtol=1e-5,atol=1e-7)
        torch.testing.assert_close(selected['payload'],tower.pen[0,1:][selected['keep']],rtol=0,atol=0)
        torch.testing.assert_close(projected,projector(selected['payload'].unsqueeze(0)),rtol=0,atol=0)
        self.assertFalse(any(len(b._forward_hooks) for b in tower.blocks))

    def test_pen_only_has_no_early_capture_and_bad_payload_is_rejected(self):
        tower=self.build();x=torch.randn(1,577,8);ids=torch.arange(576)
        config=next(c for c in build_catalog() if 'W1024_V1_' in c['id'])
        with CLIPStage1(tower.blocks,config,l9_completed=4) as hooks:
            payload=tower(x)
            self.assertEqual(set(hooks.features),{9})
            with self.assertRaises(ValueError):hooks.select(payload+1,ids)
            selected=hooks.select(payload,ids)
            self.assertFalse(selected['capture_meta']['early_captured'])
            with self.assertRaises(RuntimeError):hooks.select(payload,ids)

    def test_hooks_removed_on_forward_failure_and_training_rejected(self):
        tower=self.build()
        config=next(c for c in build_catalog() if 'W1024_V1_' in c['id'])
        with self.assertRaises(RuntimeError):
            with CLIPStage1(tower.blocks,config,l9_completed=4):
                raise RuntimeError('FIXTURE native forward failure')
        for b in tower.blocks:
            self.assertFalse(b._forward_hooks)
            self.assertFalse(b.self_attn.q_proj._forward_hooks)
        with self.assertRaises(RuntimeError):CLIPStage1(tower.train().blocks,config,l9_completed=4).__enter__()


if __name__=='__main__':unittest.main()

```

## tests/test_coverage_report.py

```python
import copy
import unittest
import numpy as np
from coverage_report import summarize
from elva.low_budget_cover import diagnostics,numpy_keep


def fixture_rows():
    rows=[]
    rng=np.random.default_rng(8)
    for image in ('FIXTURE_A','FIXTURE_B'):
        x=rng.normal(size=(576,4)).astype(np.float32);s=rng.random(576).astype(np.float32)
        for b in (0,6,12,18):
            d=diagnostics(x,s,numpy_keep(x,s,60,b),60,b)
            d.update(schema='elva_r3_cover_audit',fixture=True,image_key=image,config_hash=f'FIXTURE_{b}',
                     source_hash='FIXTURE_SOURCE',guide='pen',beta=None,work=1024)
            rows.append(d)
    return rows


class ReportTests(unittest.TestCase):
    def test_paired_summaries_and_fixture_guard(self):
        rows=fixture_rows()
        with self.assertRaises(ValueError):summarize(rows)
        result=summarize(rows,allow_fixture=True)
        self.assertEqual([r['slots'] for r in result],[0,6,12,18])
        self.assertTrue(all(r['images']==2 for r in result))
        self.assertEqual(result[0]['actual_changed'],0)
        self.assertAlmostEqual(result[0]['saliency_mass_loss_vs_topk'],0)

    def test_mismatched_panel_source_or_guide_and_duplicate_images_rejected(self):
        rows=fixture_rows()
        for altered in (rows[:-1],rows+rows[:1]):
            with self.assertRaises(ValueError):summarize(altered,allow_fixture=True)
        for key,value in (('source_hash','OTHER_FIXTURE'),('guide','dual'),('work',2048),('fixture',False)):
            altered=copy.deepcopy(rows);altered[-1][key]=value
            with self.assertRaises(ValueError):summarize(altered,allow_fixture=True)

    def test_invalid_actual_change_and_nonfinite_values_rejected(self):
        rows=fixture_rows()
        for key,value in (('actual_changed',99),('kept_mass',float('nan')),('protected_count',1)):
            altered=copy.deepcopy(rows);altered[0][key]=value
            with self.assertRaises(ValueError):summarize(altered,allow_fixture=True)


if __name__=='__main__':unittest.main()

```

## tests/test_low_budget_cover.py

```python
import unittest
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from elva.low_budget_cover import numpy_keep, torch_keep, diagnostics, callback, make_spec
from elva.cover_binding import NativeCover, select_checked
from make_catalog import build_catalog


class CoverageTests(unittest.TestCase):
    def test_protected_core_and_actual_additions_reach_output(self):
        # FIXTURE: top 4 are duplicated, a lower-score fifth row covers another direction.
        x = np.array([[1,0],[1,0],[1,0],[1,0],[0,1]], dtype=np.float32)
        s = np.array([5,4,3,2,1], dtype=np.float32)
        keep = numpy_keep(x,s,3,1)
        np.testing.assert_array_equal(keep,[0,1,4])
        audit = diagnostics(x,s,keep,3,1)
        self.assertEqual(audit['actual_changed'],1)
        self.assertEqual(audit['added_local'],[4])
        self.assertGreater(audit['kept_coverage']['weighted'],audit['topk_coverage']['weighted'])
        self.assertLess(audit['kept_mass'],audit['topk_mass'])  # explicit tradeoff

    def test_full_pool_can_select_low_rank_evidence(self):
        x = np.zeros((576,2),dtype=np.float32);x[:,0]=1;x[-1]=[0,1]
        s = np.linspace(1,.01,576,dtype=np.float32)
        keep = numpy_keep(x,s,60,12)
        self.assertIn(575,keep)  # a Top180 candidate restriction would forbid this
        self.assertTrue(set(range(48)).issubset(keep))
        self.assertEqual(len(set(keep)),60)

    def test_zero_affinity_ties_are_stable_topk_and_no_forced_change(self):
        x=np.zeros((80,3),dtype=np.float32)
        for s in (np.zeros(80,dtype=np.float32), np.arange(80,dtype=np.float32)):
            for b in (0,6,12,18,60):
                keep=numpy_keep(x,s,60,b)
                np.testing.assert_array_equal(keep,np.sort(np.argsort(-s,kind='stable')[:60]))

    def test_budget_extremes_and_determinism(self):
        rng=np.random.default_rng(82)
        for n in (1,5,60,80):
            x=rng.normal(size=(n,5)).astype(np.float32);s=rng.random(n).astype(np.float32)
            for k in (1,n):
                for b in (0,k):
                    keep=numpy_keep(x,s,k,b)
                    self.assertEqual(len(keep),k)
                    self.assertEqual(len(np.unique(keep)),k)
                    np.testing.assert_array_equal(keep,numpy_keep(x,s,k,b))
                    self.assertTrue(set(np.argsort(-s,kind='stable')[:k-b]).issubset(keep))

    def test_invalid_inputs_fail(self):
        x=np.eye(3,dtype=np.float32);s=np.ones(3,dtype=np.float32)
        for k,b in ((0,0),(3,4),(3,-1),(True,0)):
            with self.assertRaises(ValueError):numpy_keep(x,s,k,b)
        with self.assertRaises(ValueError):numpy_keep(x,np.array([1,np.nan,1]),2,1)
        with self.assertRaises(ValueError):numpy_keep(x,np.array([-1,1,1]),2,1)

    def test_real_callback_raw_pen_ids_and_not_original_cover_audit(self):
        rng=np.random.default_rng(6)
        x=rng.normal(size=(576,8)).astype(np.float32)
        s=rng.random(576).astype(np.float32)
        ids=np.arange(576,dtype=np.int64)*7+6
        c=next(c for c in build_catalog() if 'W1024_V4_' in c['id'])
        out=select_checked(x,s,ids,c,native_cover=NativeCover(callback,make_spec()))
        np.testing.assert_array_equal(out['payload'],x[out['keep']])
        np.testing.assert_array_equal(out['ids'],ids[out['keep']])
        self.assertEqual(out['stage1_audit']['cover_calls'],1)
        self.assertEqual(out['stage1_audit']['native_calls'],0)
        self.assertEqual(out['stage1_audit']['origin'],'r3_protected_control')
        self.assertEqual(out['stage1_audit']['replacement_slots'],12)
        core=np.argsort(-s,kind='stable')[:48]
        self.assertTrue(set(core).issubset(out['keep']))

    def test_only_one_bounded_coverage_knob_and_explicit_optional_controls(self):
        configs=build_catalog()
        core=[c for c in configs if c['phase']=='stage1_core' and c['method']['budget']==1024]
        covered=[c for c in core if c['method']['stage1']['cover']]
        self.assertEqual(len(covered),2)
        self.assertTrue(all(c['method']['stage1']['coverage_slots']==12 for c in covered))
        extra=[c for c in configs if c['phase']=='stage1_optional' and c['method']['stage1'].get('coverage_slots') in (6,18)]
        self.assertEqual(len(extra),4)
        self.assertEqual(len([c for c in configs if c['method']['stage1']['cover_family']=='parameterized_native']),2)

    def test_refinement_keeps_winning_beta_and_preserves_winner_in_stage2(self):
        base=next(c for c in build_catalog() if c['method']['budget']==1024 and c['method']['stage1'].get('beta')==.25)
        refined=build_catalog(coverage_base=base)
        variants=[c for c in refined if c['method']['stage1'].get('coverage_slots') in (6,18)]
        self.assertEqual(len(variants),2)
        self.assertTrue(all(c['method']['stage1']['beta']==.25 for c in variants))
        winner=variants[0]
        decision={'budgets':{'1024':{'stage1':winner['method']['stage1'],'winner_config':winner}}}
        rebuilt=build_catalog(decisions=decision)
        self.assertTrue(any(c==winner for c in rebuilt))
        self.assertTrue(all(c['method']['stage1']['beta']==.25 for c in rebuilt if c['phase']=='stage2'))


@unittest.skipIf(torch is None,'PyTorch absent locally; run parity on the existing server environment')
class CoverageTorchTests(unittest.TestCase):
    def test_numpy_parity_and_no_payload_change(self):
        rng=np.random.default_rng(33)
        for n,k in ((5,3),(80,60),(576,60)):
            x=rng.normal(size=(n,13)).astype(np.float32);s=rng.random(n).astype(np.float32)
            for b in (0,min(6,k),min(12,k)):
                np.testing.assert_array_equal(torch_keep(torch.from_numpy(x),torch.from_numpy(s),k,b).numpy(),
                                              numpy_keep(x,s,k,b))
        x=torch.zeros(80,3);s=torch.zeros(80)
        torch.testing.assert_close(torch_keep(x,s,60,12),torch.arange(60))


if __name__=='__main__':unittest.main()

```

## tests/test_r3.py

```python
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
try:
    import torch
except ImportError:
    torch=None

from make_catalog import MAIN, SHAPES, TASKS, build_catalog, digest
from elva.cover_binding import NativeCover, select_checked, origin
from elva.workflow import (validate_binding, blocked_reason, result_key, validate_result,
                           collect_complete, freeze, validate_decision, trace_for)
from native_result_io import finish_task
from gpu_resources import GPU, Lease, parse_inventory, choose, respect_visible
from partition_manifest import partition
from run_queue import run


def fixture_cover(*, metric_features, scores, patch_ids, k, context):
    # Fixture callback deliberately uses a different subset from score Top-K.
    return np.arange(k, dtype=np.int64)


def fixture_torch_cover(*, metric_features, scores, patch_ids, k, context):
    return torch.arange(k,device=scores.device,dtype=torch.long)


def fixture_spec(family="legacy_native", supported=(172,)):
    source=Path(__file__).resolve()
    return dict(family=family, supported_k=list(supported), source_files={str(source):hashlib.sha256(source.read_bytes()).hexdigest()},
                replacement_slots={"172":32,"60":11}, legacy172_parity_verified=True)


def fixture_binding():
    from elva.low_budget_cover import make_spec
    protected = make_spec()
    protected['implementation_parity_verified'] = True  # FIXTURE ONLY
    return dict(fixture=True, source_commit="FIXTURE", model_sha="FIXTURE", processor_sha="FIXTURE",
                tokenizer_sha="FIXTURE", prompt_sha="FIXTURE", dataset_sha="FIXTURE", scorer_sha="FIXTURE",
                backend="FIXTURE", cache_mode="native", original_tokens=576, depth=32,
                native_selectors={s:"FIXTURE" for s in ("Q_ATTN","RDRIFT_NATIVE","QED_NATIVE","ATTN_COVER_NATIVE")},
                preflight=dict(off_path_equal=True, clip_layers_checked=True, pen_payload_equal=True,
                               positions_and_kv_checked=True,
                               trace_verified={"2048":True,"1024":True}),
                task_samples={task:1 for task in TASKS},
                native_cover={"legacy_native":fixture_spec(),
                              "protected_coverage":protected,
                              "parameterized_native":fixture_spec("parameterized_native",(60,))})


def fixture_records(c,b):
    m=c["method"]
    family=m["stage1"]["cover_family"]
    k=m["schedule"]["counts"][0]
    if family=="none":
        proof=dict(cover_family=family, requested_k=k,native_calls=0,cover_calls=0,
                   origin=origin(family),replacement_slots=0,fallback=False,
                   payload="pen",cover_source_sha=None,native_k=None)
    else:
        from elva.cover_binding import spec_fingerprint
        spec=b['native_cover'][family]
        native_k=172 if family=="legacy_pool172_then_topk" else k
        slots=m['stage1'].get('coverage_slots') if family=='protected_coverage' else spec['replacement_slots'][str(native_k)]
        proof=dict(cover_family=family,requested_k=k,cover_calls=1,
                   native_calls=int(family!='protected_coverage'),origin=origin(family),
                   replacement_slots=slots,fallback=False,payload="pen",
                   cover_source_sha=spec_fingerprint(b["native_cover"][family]),
                   native_k=native_k)
    return [dict(image_key="FIXTURE_IMAGE",question_key="FIXTURE_QUESTION",prediction="FIXTURE",
                 stage1_ids=list(range(k)),stage1_audit=proof,trace=trace_for(c))]


class BudgetTests(unittest.TestCase):
    def test_only_two_budgets_and_exact_inverse(self):
        from fractions import Fraction
        configs=build_catalog()
        self.assertEqual({c["method"]["budget"] for c in configs},{2048,1024})
        self.assertEqual(sum(c["phase"]=="stage1_core" for c in configs),10)
        for budget, shapes in SHAPES.items():
            for p in shapes:
                self.assertEqual(p.work,budget)
                self.assertEqual(p.depth,32)
                self.assertEqual((1-p.compression(576))*576*32,budget)
                self.assertEqual(p.counts[0],MAIN[budget].counts[0])
                self.assertEqual(p.counts[-1],MAIN[budget].counts[-1])
        self.assertEqual(MAIN[2048].events,((4,90),(12,32)))
        self.assertEqual(MAIN[1024].events,((10,28),(20,12)))

    def test_no_unconditional_stage2_or_learning_and_explicit_pool_identity(self):
        configs=build_catalog()
        self.assertFalse(any(c["phase"]=="stage2" or c["method"]["learned"] for c in configs))
        pooled=build_catalog(low_cover="legacy_pool172_then_topk")
        a=next(c for c in configs if 'W1024_V4_' in c['id'])
        b=next(c for c in pooled if 'W1024_POOL172_CONTROL_V4_' in c['id'])
        self.assertNotEqual(a['config_hash'],b['config_hash'])
        self.assertIn('POOL172',b['id'])
        self.assertEqual(b['phase'],'diagnostic_pool')
        self.assertEqual(sum(c['phase']=='stage1_core' for c in pooled),10)


class CoverTests(unittest.TestCase):
    def setUp(self):
        self.x=np.arange(576*3,dtype=float).reshape(576,3)
        self.s=np.linspace(0,1,576)
        self.ids=np.arange(576)*7+5
        self.config=next(c for c in build_catalog() if 'W2048_V4_' in c['id'])

    def test_missing_cover_is_an_error_not_topk(self):
        with self.assertRaises(RuntimeError):
            select_checked(self.x,self.s,self.ids,self.config)

    def test_native172_calls_source_and_preserves_raw_pen_payload(self):
        cover=NativeCover(fixture_cover,fixture_spec())
        out=select_checked(self.x,self.s,self.ids,self.config,native_cover=cover)
        np.testing.assert_array_equal(out['payload'],self.x[out['keep']])
        np.testing.assert_array_equal(out['keep'],np.arange(172))
        self.assertEqual(out['stage1_audit']['native_calls'],1)
        self.assertEqual(cover.events[0]['replacement_slots'],32)

    def test_fixed172_cannot_claim_direct_k60(self):
        cover=NativeCover(fixture_cover,fixture_spec())
        with self.assertRaises(RuntimeError):
            cover(metric_features=self.x,scores=self.s,patch_ids=self.ids,k=60)
        self.assertEqual(cover.events,[])

    def test_explicit_pool_route_calls172_then_returns60(self):
        cover=NativeCover(fixture_cover,fixture_spec('legacy_pool172_then_topk',(60,)))
        c=next(c for c in build_catalog(low_cover='legacy_pool172_then_topk') if 'W1024_POOL172_CONTROL_V4_' in c['id'])
        out=select_checked(self.x,self.s,self.ids,c,native_cover=cover)
        np.testing.assert_array_equal(out['keep'],np.arange(112,172))
        np.testing.assert_array_equal(out['payload'],self.x[out['keep']])
        self.assertEqual(out['stage1_audit']['native_k'],172)
        self.assertEqual(out['stage1_audit']['requested_k'],60)

    def test_tail_replacement_then_trim_can_be_identically_topk60(self):
        n=576
        scores=np.arange(n,dtype=float)[::-1]
        ids=np.arange(n)
        original_top172=np.arange(172)
        pool=np.r_[original_top172[:140],np.arange(200,232)]
        trimmed=pool[np.argsort(-scores[pool],kind='stable')[:60]]
        np.testing.assert_array_equal(trimmed,np.arange(60))

    def test_changed_native_source_is_blocked(self):
        spec=fixture_spec()
        spec['source_files'][str(Path(__file__).resolve())]='0'*64
        with self.assertRaises(RuntimeError):
            NativeCover(fixture_cover,spec)

    def test_cover_can_legitimately_return_same_ids_as_topk(self):
        # Equal sets alone are not proof that the callback was skipped.
        cover=NativeCover(fixture_cover,fixture_spec())
        out=select_checked(self.x,self.s[::-1].copy(),self.ids,self.config,native_cover=cover)
        self.assertEqual(out['stage1_audit']['native_calls'],1)
        np.testing.assert_array_equal(out['keep'],np.arange(172))


@unittest.skipIf(torch is None,'PyTorch is absent locally; execute this server-side')
class TorchCoverTests(unittest.TestCase):
    def test_native_and_pooled_wrapper_preserve_torch_payload_and_ids(self):
        x=torch.arange(576*3,dtype=torch.float32).reshape(576,3)
        s=torch.linspace(0,1,576)
        ids=torch.arange(576,dtype=torch.long)*7+5
        high=next(c for c in build_catalog() if 'W2048_V5_' in c['id'])
        cover=NativeCover(fixture_torch_cover,fixture_spec())
        out=select_checked(x,s,ids,high,early=x.flip(0),early_scores=s.flip(0),early_ids=ids,
                           native_cover=cover,backend='torch')
        torch.testing.assert_close(out['payload'],x.index_select(0,out['keep']),rtol=0,atol=0)
        torch.testing.assert_close(out['ids'],ids.index_select(0,out['keep']),rtol=0,atol=0)
        low=next(c for c in build_catalog(low_cover='legacy_pool172_then_topk') if 'W1024_POOL172_CONTROL_V4_' in c['id'])
        pool=NativeCover(fixture_torch_cover,fixture_spec('legacy_pool172_then_topk',(60,)))
        out=select_checked(x,s,ids,low,native_cover=pool,backend='torch')
        torch.testing.assert_close(out['keep'],torch.arange(112,172),rtol=0,atol=0)
        self.assertEqual(out['stage1_audit']['native_k'],172)


class GPUAndWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.binding=fixture_binding()
        self.configs=build_catalog()
        self.config=next(c for c in self.configs if 'W2048_V1_' in c['id'])

    def tearDown(self):
        self.temp.cleanup()

    def make_result(self,c,task):
        key=result_key(c,self.binding,task)
        folder=self.root/key
        value=finish_task(folder/'result.json',records=fixture_records(c,self.binding),official_metrics={'FIXTURE':1.},
                          config=c,binding=self.binding,task=task,run_key=key,fixture=True)
        return value,folder,key

    def test_zero_utilization_is_not_an_empty_gpu(self):
        devices=parse_inventory('0, GPU-a, 10, 0\n2, GPU-b, 14632, 0\n4, GPU-c, 5, 0\n', 'GPU-a, 2050863\n')
        self.assertFalse(devices[0].available())
        self.assertFalse(devices[1].available())
        self.assertTrue(devices[2].available())
        acc,unused=choose(devices,['auto'],3,0)
        self.assertEqual([g.index for g in acc],['4'])
        self.assertEqual(choose(devices,['auto'],3,1)[0],[])

    def test_gpu_mapping_unknown_or_na_is_not_assumed_free(self):
        for g,p in [('0, GPU-a, N/A, 0\n',''),('0, GPU-a, 5, 0\n','GPU-b, 1\n')]:
            with self.assertRaises(ValueError): parse_inventory(g,p)
        with self.assertRaises(ValueError): choose([GPU('0','GPU-a',0,0,())],['0','0'],2,0)

    def test_inherited_gpu_allocation_is_respected(self):
        devices=[GPU('0','GPU-a',0,0,()),GPU('7','GPU-b',0,0,())]
        self.assertEqual(respect_visible(devices,'7'),[devices[1]])
        self.assertEqual(respect_visible(devices,'GPU-a'),[devices[0]])
        self.assertEqual(respect_visible(devices,''),[])
        with self.assertRaises(ValueError):respect_visible(devices,'MIG-unknown')

    def test_global_limit_precedes_sharding_and_image_dedup(self):
        rows=[dict(image_key=f'FIXTURE_{i}',question_key=f'Q{q}') for i in range(40) for q in range(2)]
        groups=partition(rows,shards=4,limit=32)
        self.assertEqual(list(map(len,groups)),[8,8,8,8])
        self.assertEqual(len({r['image_key'] for group in groups for r in group}),32)
        with self.assertRaises(ValueError):partition(rows+rows[:1],shards=4,limit=32,mode='vlm_question')

    def test_lease_is_exclusive_and_released(self):
        with Lease(self.root,'GPU-a'):
            with self.assertRaises(BlockingIOError):
                with Lease(self.root,'GPU-a'): pass
        with Lease(self.root,'GPU-a'): pass

    def test_fixture_and_missing_native_capacity_are_blocked(self):
        with self.assertRaises(ValueError):validate_binding(self.binding)
        validate_binding(self.binding,allow_fixture=True)
        c=next(c for c in self.configs if 'W1024_V4_' in c['id'])
        b=copy.deepcopy(self.binding);del b['native_cover']['protected_coverage']
        self.assertIn('BLOCKED_COVER',blocked_reason(c,b))
        self.assertIsNone(blocked_reason(self.config,b))

    def test_per_sample_result_proof_rejects_clip_pilot_and_wrong_trace(self):
        value,folder,key=self.make_result(self.config,'GQA')
        validate_result(value,self.config,self.binding,'GQA',key,folder,allow_fixture=True)
        bad=copy.deepcopy(value);bad['inference_kind']='clip_selection_only'
        with self.assertRaises(ValueError):validate_result(bad,self.config,self.binding,'GQA',key,folder,allow_fixture=True)
        records=fixture_records(self.config,self.binding)
        records[0]['trace'][0]-=1
        with self.assertRaises(ValueError):
            finish_task(folder/'wrong.json',records=records,official_metrics={'FIXTURE':1.},config=self.config,
                        binding=self.binding,task='GQA',run_key=key,fixture=True)

    def test_fake_cover_fallback_is_rejected(self):
        c=next(c for c in self.configs if 'W2048_V4_' in c['id'])
        rows=fixture_records(c,self.binding)
        rows[0]['stage1_audit']['native_calls']=0
        with self.assertRaises(ValueError):
            finish_task(self.root/'bad/result.json',records=rows,official_metrics={'FIXTURE':1.},config=c,
                        binding=self.binding,task='GQA',run_key=result_key(c,self.binding,'GQA'),fixture=True)

    def test_complete_core_freeze_and_changed_results_invalidate(self):
        with self.assertRaises(OSError):collect_complete(self.configs,self.binding,self.root,2048,allow_fixture=True)
        for c in self.configs:
            if c['phase']=='stage1_core' and c['method']['budget']==2048:
                for task in TASKS:self.make_result(c,task)
        decision=freeze(self.configs,self.binding,self.root,2048,self.config['id'],'FIXTURE rationale',allow_fixture=True)
        validate_decision(decision,self.binding,self.configs,self.root,2048,allow_fixture=True)
        generated=build_catalog(decisions=decision)
        self.assertEqual(sum(c['phase']=='stage2' for c in generated),8)
        path=self.root/result_key(self.config,self.binding,'GQA')/'result.json'
        value=json.loads(path.read_text());value['metrics']['FIXTURE']=.5;path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):validate_decision(decision,self.binding,self.configs,self.root,2048,allow_fixture=True)

    def test_completed_optional_beta_can_be_frozen_after_core(self):
        for c in self.configs:
            if c['method']['budget']==2048:
                for task in TASKS:self.make_result(c,task)
        winner=next(c for c in self.configs if c['method']['budget']==2048 and c['method']['stage1']['beta']==.25)
        decision=freeze(self.configs,self.binding,self.root,2048,winner['id'],'FIXTURE optional beta',allow_fixture=True)
        validate_decision(decision,self.binding,self.configs,self.root,2048,allow_fixture=True)
        self.assertEqual(decision['budgets']['2048']['stage1']['beta'],.25)

    def test_queue_runs_resumes_and_rejects_stage2_before_freeze(self):
        script=self.root/'fixture_runner.py'
        script.write_text('''import argparse,json\nfrom pathlib import Path\nfrom native_result_io import finish_task\nfrom elva.workflow import trace_for\np=argparse.ArgumentParser()\nfor name in ('config','binding','task','out','run_key'):p.add_argument('--'+name,required=True)\na=p.parse_args()\nc=json.loads(Path(a.config).read_text());b=json.loads(Path(a.binding).read_text())\nk=c['method']['schedule']['counts'][0]\naudit=dict(cover_family='none',requested_k=k,native_calls=0,cover_calls=0,origin='none',replacement_slots=0,fallback=False,payload='pen',cover_source_sha=None,native_k=None)\nrows=[dict(image_key='FIXTURE_IMAGE',question_key='FIXTURE_Q',prediction='FIXTURE',stage1_ids=list(range(k)),stage1_audit=audit,trace=trace_for(c))]\nfinish_task(a.out,records=rows,official_metrics={'FIXTURE':1.},config=c,binding=b,task=a.task,run_key=a.run_key,fixture=True)\n''')
        command=[sys.executable,str(script),'--config','{config}','--binding','{binding}',
                 '--task','{task}','--out','{out}','--run_key','{run_key}']
        probe=lambda:[GPU('0','GPU-FIXTURE',0,0,())]
        import os
        old=os.environ.get('PYTHONPATH')
        os.environ['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
        try:
            kwargs=dict(phase='stage1_core',requested=['auto'],workers=1,reserve=0,hours=.02,
                        tasks=['GQA'],budgets=[2048],gpu_probe=probe,allow_fixture=True,
                        lease_directory=self.root/'leases')
            states=run({'configs':[self.config]},self.binding,command,self.root/'queue',**kwargs)
            self.assertTrue(any(s['status']=='DONE' for s in states),states)
            states=run({'configs':[self.config]},self.binding,command,self.root/'queue',**kwargs)
            self.assertTrue(any(s['status']=='REUSED_VERIFIED' for s in states))
            kwargs['phase']='stage2'
            with self.assertRaises(ValueError):run({'configs':[self.config]},self.binding,command,self.root/'queue',**kwargs)
        finally:
            if old is None:os.environ.pop('PYTHONPATH',None)
            else:os.environ['PYTHONPATH']=old


if __name__=='__main__':unittest.main()

```

## tests/test_visual_audit.py

```python
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"visual_audit"))
from make_fixture import create
from stage1_visual_audit import (validate_record, load_record, keep_mask, pair_metrics,
                                audit, render_sweep, matched_random_keep)


class VisualAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        create(self.root)
        self.path = self.root/"record_FIXTURE_ONLY.json"
        self.raw = json.loads(self.path.read_text())
        self.raw["image_path"] = str(self.root/"fixture_crop.png")

    def tearDown(self):
        self.temp.cleanup()

    def validate(self, value=None):
        return validate_record(value or self.raw, allow_fixture=True)

    def test_original_ids_are_not_row_indices(self):
        record = self.validate()
        chosen = record["methods"]["pen"]["keep_ids"]
        expected = [i for i, pid in enumerate(record["patch_ids"]) if pid in chosen]
        np.testing.assert_array_equal(np.flatnonzero(keep_mask(record, "pen")), expected)
        self.assertGreater(max(chosen), len(record["patch_ids"]))

    def test_all_and_disjoint_selected_sets(self):
        raw = copy.deepcopy(self.raw)
        raw["methods"] = [dict(name="a", config_hash="a", keep_ids=raw["patch_ids"][:4]),
                          dict(name="b", config_hash="b", keep_ids=raw["patch_ids"][-4:])]
        record = self.validate(raw)
        result = pair_metrics(record, "a", "b")
        self.assertEqual((result["common"], result["union"], result["jaccard"]), (0, 8, 0))
        self.assertEqual(pair_metrics(record, "a", "a")["jaccard"], 1)

    def test_random_supplement_matches_changed_count_and_common_support(self):
        ids = np.array([70, 4, 92, 33, 11, 55])
        baseline, candidate = [70, 4, 92], [4, 11, 55]
        keep, info = matched_random_keep(ids, baseline, candidate, seed=3)
        self.assertEqual(len(keep), len(baseline))
        self.assertEqual(info["changed"], 2)
        self.assertEqual(set(keep)&set(baseline), {4})
        np.testing.assert_array_equal(keep, ids[np.isin(ids, keep)])
        np.testing.assert_array_equal(keep, matched_random_keep(ids, baseline, candidate, seed=3)[0])

    def test_random_supplement_no_change_and_full_keep(self):
        for ids in ([99], [99, 7, 18]):
            keep, info = matched_random_keep(ids, ids, ids, seed=0)
            np.testing.assert_array_equal(keep, ids)
            self.assertEqual(info["changed"], 0)

    def test_random_supplement_cannot_resurrect_ids_or_change_k(self):
        for base, candidate in (([2, 3], [3]), ([2], [100]), ([2, 2], [2, 3])):
            with self.assertRaises(ValueError):
                matched_random_keep([2, 3, 4], base, candidate, seed=1)

    def test_zero_mass_and_uniform_scores_render_without_division(self):
        for score in (0., 1.):
            raw = copy.deepcopy(self.raw)
            for key in ("guide_scores", "pen_scores"):
                raw[key] = [score]*len(raw["patch_ids"])
            p = self.root/f"scores-{score}.json"
            p.write_text(json.dumps(raw))
            summary = audit([p], self.root/f"out-{score}", baseline="pen", candidate="dual050", allow_fixture=True)
            self.assertEqual(summary["shared_raw_vmax"], 1)
            self.assertEqual(summary["score_zero"], score == 0)

    def test_heatmaps_have_actual_scalar_mapped_faces(self):
        from stage1_visual_audit import _boxes
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        fig, ax = plt.subplots()
        faces = _boxes(ax, [[0, 0, 1, 1], [1, 0, 2, 1]], values=[0., 1.], norm=Normalize(0, 1))
        fig.canvas.draw()
        colors = faces.get_facecolors()
        self.assertEqual(colors.shape, (2, 4))
        self.assertFalse(np.allclose(colors[0], colors[1]))
        plt.close(fig)

    def test_image_coordinates_and_cross_crop_overlap_rejected(self):
        for edit in (lambda r: r["boxes_xyxy"][0].__setitem__(0, -1),
                     lambda r: r["boxes_xyxy"].__setitem__(1, r["boxes_xyxy"][0]),
                     lambda r: r["boxes_xyxy"][0].__setitem__(2, 999)):
            raw = copy.deepcopy(self.raw)
            edit(raw)
            with self.assertRaises(ValueError):
                self.validate(raw)

    def test_duplicate_unknown_ids_mismatched_k_and_nan_rejected(self):
        edits = [lambda r: r["patch_ids"].__setitem__(0, r["patch_ids"][1]),
                 lambda r: r["methods"][0]["keep_ids"].__setitem__(0, 99999),
                 lambda r: r["methods"][0]["keep_ids"].pop(),
                 lambda r: r["pen_scores"].__setitem__(0, float("nan"))]
        for edit in edits:
            raw = copy.deepcopy(self.raw)
            edit(raw)
            with self.assertRaises(ValueError):
                self.validate(raw)

    def test_annotated_region_geometry_is_not_attention_coverage(self):
        raw = copy.deepcopy(self.raw)
        raw["methods"] = [dict(name="a", config_hash="a", keep_ids=raw["patch_ids"][:2]),
                          dict(name="b", config_hash="b", keep_ids=raw["patch_ids"][-2:])]
        row = pair_metrics(self.validate(raw), "a", "b")["annotated_region_coverage"][0]
        self.assertEqual(row["baseline_area_fraction"], .5)
        self.assertEqual(row["candidate_area_fraction"], 0.)

    def test_fixture_source_binding_and_repeated_question_guard(self):
        with self.assertRaises(ValueError):
            load_record(self.path)
        with self.assertRaises(ValueError):
            audit([self.path, self.path], self.root/"duplicate", baseline="pen", candidate="dual050", allow_fixture=True)
        raw = copy.deepcopy(self.raw)
        raw["source"]["pen_completed"] = raw["source"]["vision_depth"]
        with self.assertRaises(ValueError):
            self.validate(raw)

    def test_real_layout_shared_scale_and_sweep_gate(self):
        summary = audit([self.path], self.root/"plots", baseline="pen", candidate="dual050", allow_fixture=True)
        self.assertEqual(summary["image_count"], 1)
        self.assertEqual(summary["shared_raw_vmax"], .2)
        matrices = np.asarray(summary["overlap_matrix"])
        np.testing.assert_array_equal(matrices, matrices.T)
        np.testing.assert_array_equal(matrices.diagonal(), 1)
        for p in (self.root/"plots").rglob("*.png"):
            with Image.open(p) as img:
                self.assertGreater(img.width, 100)
                self.assertGreater(img.height, 100)
        ledger = self.root/"sweep_FIXTURE_ONLY.json"
        with self.assertRaises(ValueError):
            render_sweep(ledger, self.root/"sweep")
        render_sweep(ledger, self.root/"sweep", allow_fixture=True)
        raw = json.loads(ledger.read_text())
        raw["rows"][0]["selection_mode"] = "quota"
        ledger.write_text(json.dumps(raw))
        with self.assertRaises(ValueError):
            render_sweep(ledger, self.root/"bad", allow_fixture=True)


if __name__ == "__main__":
    unittest.main()

```

## verify.py

```python
"""CPU checks for the R3 control plane and unchanged R2 numerical modules."""
import compileall
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import re
from make_catalog import write_catalog, MAIN, SHAPES
from tail_pool_equivalence import verify as verify_pool


def main():
    root=Path(__file__).resolve().parent
    out=root/'validation';out.mkdir(exist_ok=True)
    write_catalog(root/'configs')
    assert compileall.compile_dir(str(root/'elva'),quiet=1)
    assert compileall.compile_dir(str(root/'tests'),quiet=1)
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.discover(str(root/'tests')))
    (out/'r3_tests.txt').write_text(stream.getvalue())
    env=dict(os.environ,PYTHONPATH=str(root))
    legacy=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=root/'legacy_reference',
                          env=env,capture_output=True,text=True)
    (out/'r2_regression.txt').write_text(legacy.stdout+legacy.stderr)
    legacy_log=legacy.stdout+legacy.stderr
    legacy_count=int(re.search(r'Ran (\d+) tests?',legacy_log).group(1))
    legacy_skip=re.search(r'OK \(skipped=(\d+)\)',legacy_log)
    legacy_skipped=int(legacy_skip.group(1)) if legacy_skip else 0
    budgets=[dict(budget=w,counts=p.counts,lengths=p.lengths,events=p.events,trace=p.trace,
                  work=p.work,compression_exact=str(p.compression(576))) for w,shapes in SHAPES.items() for p in shapes]
    for row in budgets:
        assert row['budget']==row['work']==sum(row['trace'])
    (out/'budgets.json').write_text(json.dumps(budgets,indent=2)+'\n')
    pool_check=verify_pool()
    (out/'tail_pool_equivalence.json').write_text(json.dumps(pool_check,indent=2)+'\n')
    catalog=write_catalog(root/'configs')
    core=[c for c in catalog['configs'] if c['phase']=='stage1_core']
    summary=dict(r3_tests=result.testsRun,r3_passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
                 r3_failures=len(result.failures),r3_errors=len(result.errors),r3_skipped=len(result.skipped),
                 r2_regression_exit=legacy.returncode,core_configs=len(core),core_four_task_jobs=4*len(core),
                 r2_tests=legacy_count,r2_skipped=legacy_skipped,r2_passed=legacy_count-legacy_skipped if legacy.returncode==0 else None,
                 optional_configs=sum(c['phase']=='stage1_optional' for c in catalog['configs']),budgets=[2048,1024],
                 high_entry=MAIN[2048].counts[0],low_entry=MAIN[1024].counts[0],
                 fixture_jobs_only=True,pretrained_vlm_accuracy_run=False,gpu_ttft_run=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    if not result.wasSuccessful() or legacy.returncode:
        print(stream.getvalue());print(legacy.stderr);raise SystemExit(1)


if __name__=='__main__':main()

```

## visual_audit/make_fixture.py

```python
"""Create clearly labeled synthetic data ONLY for testing the plotting path."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from stage1_visual_audit import SCHEMA


def create(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    height, width, rows, cols = 180, 240, 3, 4
    yy, xx = np.mgrid[:height, :width]
    rgb = np.stack((180+40*xx/width, 190+35*yy/height, 185+15*xx/width), -1).astype(np.uint8)
    Image.fromarray(rgb).save(output/"fixture_crop.png")
    boxes = np.array([[x*width/cols, y*height/rows, (x+1)*width/cols, (y+1)*height/rows]
                      for y in range(rows) for x in range(cols)])
    ids = (np.arange(rows*cols)[::-1]*7+11).tolist()
    guide = np.array([.20, .16, .03, .01, .14, .13, .04, .02, .01, .02, .03, .01])
    pen = guide[::-1].copy()
    methods = []
    for name, beta in (("pen", 0.), ("guide", 1.), ("dual025", .25), ("dual050", .5), ("dual075", .75)):
        scores = (1-beta)*pen+beta*guide
        keep = np.sort(np.argsort(-scores, kind="stable")[:4])
        methods.append(dict(name=name, beta=beta, cover=False, config_hash="FIXTURE_"+name,
                            keep_ids=[ids[i] for i in keep]))
    record = dict(schema=SCHEMA, fixture=True, image_key="FIXTURE_IMAGE", crop_key="FIXTURE_CROP",
                  case_key="FIXTURE_CASE", image_path="fixture_crop.png", patch_ids=ids,
                  boxes_xyxy=boxes.tolist(), guide_scores=guide.tolist(), pen_scores=pen.tolist(),
                  methods=methods, work=1024, category="FIXTURE_ONLY",
                  source=dict(model_sha="FIXTURE", processor_sha="FIXTURE", selector_sha="FIXTURE",
                              saliency_rule="FIXTURE", schedule_hash="FIXTURE", split="development",
                              vision_depth=24, guide_completed=9, pen_completed=23, payload="pen"),
                  regions=[dict(name="FIXTURE_REGION", box_xyxy=[0, 0, 120, 120])])
    (output/"record_FIXTURE_ONLY.json").write_text(json.dumps(record, indent=2)+"\n")
    ledger = dict(schema="elva_stage1_sweep_v1", fixture=True, metric_name="Fixture score (arbitrary)",
                  source_hash="FIXTURE", scorer_sha="FIXTURE", frozen_panel_hash="FIXTURE",
                  timing_protocol_hash="FIXTURE", rows=[dict(beta=b, metric=m, wall_ttft_p50_ms=t,
                  work=1024, cover=False, payload="pen", split="development", selection_mode="weighted_evidence")
                  for b, m, t in [(0., 2., 10.), (.25, 2.2, 10.4), (.5, 2.3, 10.4), (.75, 2.1, 10.4), (1., 1.8, 10.2)]])
    (output/"sweep_FIXTURE_ONLY.json").write_text(json.dumps(ledger, indent=2)+"\n")
    return record


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    create(parser.parse_args().out)

```

## visual_audit/stage1_visual_audit.py

```python
"""Offline Stage1 diagnostics from source-bound scores and original patch IDs.

This module does not run a VLM or infer token utility from attention. Pixel
boxes must refer to the exact processed crop shown, never an assumed reshape.
No image or score is generated when real input is absent. Fixtures are explicit.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, to_rgba
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle


SCHEMA = "elva_stage1_visual_v1"
INK, BASE, NEW, BOTH = "#183e4b", "#c97a43", "#437fac", "#477f6b"


def _ids(value, name):
    a = np.asarray(value)
    if (a.ndim != 1 or not len(a) or a.dtype.kind not in "iu"
            or (a < 0).any() or (a > np.iinfo(np.int64).max).any()
            or len(np.unique(a)) != len(a)):
        raise ValueError(f"{name}: nonempty unique nonnegative integer IDs required")
    return a.astype(np.int64)


def _rect(value, width, height):
    a = np.asarray(value, dtype=float)
    if (a.ndim != 2 or a.shape[1] != 4 or not len(a) or not np.isfinite(a).all()
            or (a[:, :2] < 0).any() or (a[:, 2] > width).any()
            or (a[:, 3] > height).any() or (a[:, 2:] <= a[:, :2]).any()):
        raise ValueError("boxes must be finite positive xyxy inside the processed crop")
    # A crop's patch boxes must not overlap. Otherwise summed region coverage
    # is invalid; separate crops must be separate input records.
    lo = np.maximum(a[:, None, :2], a[None, :, :2])
    hi = np.minimum(a[:, None, 2:], a[None, :, 2:])
    area = np.maximum(0., hi-lo).prod(-1)
    np.fill_diagonal(area, 0.)
    if (area > 1e-7).any():
        raise ValueError("overlapping patch boxes; split image crops into separate records")
    return a


def validate_record(record, *, allow_fixture=False):
    if record.get("schema") != SCHEMA or type(record.get("fixture")) is not bool:
        raise ValueError("explicit schema and fixture flag required")
    if record["fixture"] and not allow_fixture:
        raise ValueError("fixture input requires --allow-fixture")
    for key in ("image_key", "crop_key", "case_key", "image_path"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"nonempty {key} required")
    source = record.get("source", {})
    for key in ("model_sha", "processor_sha", "selector_sha", "saliency_rule",
                "schedule_hash", "split"):
        if not isinstance(source.get(key), str) or not source[key]:
            raise ValueError(f"source.{key} required")
    depth, guide, pen = (source.get(k) for k in
                         ("vision_depth", "guide_completed", "pen_completed"))
    if (any(type(k) is not int for k in (depth, guide, pen))
            or depth < 2 or not 1 <= guide <= depth or pen != depth-1):
        raise ValueError("source must bind completed guide layer and actual penultimate layer")
    if source.get("payload") != "pen":
        raise ValueError("this audit compares selection only; payload must be raw pen")
    if type(record.get("work")) is not int or record["work"] < 1:
        raise ValueError("positive declared token-layer work required")
    image_path = Path(record["image_path"])
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    ids = _ids(record.get("patch_ids"), "patch_ids")
    boxes = _rect(record.get("boxes_xyxy"), rgb.shape[1], rgb.shape[0])
    if len(boxes) != len(ids):
        raise ValueError("one actual pixel box per original patch ID required")
    scores = {}
    for key in ("guide_scores", "pen_scores"):
        a = np.asarray(record.get(key), dtype=float)
        if a.shape != ids.shape or not np.isfinite(a).all() or (a < 0).any():
            raise ValueError(f"{key}: aligned nonnegative finite source scores required")
        scores[key] = a
    methods, all_ids = {}, set(map(int, ids))
    for method in record.get("methods", []):
        name = method.get("name")
        if not isinstance(name, str) or not name or name in methods:
            raise ValueError("unique named methods required")
        if not isinstance(method.get("config_hash"), str) or not method["config_hash"]:
            raise ValueError("each actual keep set must bind a configuration hash")
        keep = _ids(method.get("keep_ids"), name)
        if not set(map(int, keep)).issubset(all_ids):
            raise ValueError("unknown/deleted keep ID; never interpret IDs as row numbers")
        methods[name] = {**method, "keep_ids": keep}
    if len(methods) < 2 or len({len(v["keep_ids"]) for v in methods.values()}) != 1:
        raise ValueError("at least two methods with identical entry K required")
    regions = []
    for region in record.get("regions", []):
        if not isinstance(region.get("name"), str) or not region["name"]:
            raise ValueError("annotated region name required")
        box = _rect([region.get("box_xyxy")], rgb.shape[1], rgb.shape[0])[0]
        regions.append({"name": region["name"], "box": box})
    return {**record, "rgb": rgb, "patch_ids": ids, "boxes_xyxy": boxes,
            **scores, "methods": methods, "regions": regions}


def load_record(path, *, allow_fixture=False):
    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    image = Path(record.get("image_path", ""))
    if not image.is_absolute():
        record["image_path"] = str((path.parent/image).resolve())
    return validate_record(record, allow_fixture=allow_fixture)


def keep_mask(record, method):
    return np.isin(record["patch_ids"], record["methods"][method]["keep_ids"])


def matched_random_keep(patch_ids, baseline_ids, candidate_ids, *, seed):
    """CPU reference for a matched random SUPPLEMENT control.

    Freeze the exact baseline tokens removed by the dual method, replace its
    additions by an equally sized uniform sample outside the baseline. Keep
    original sequence order. This is neither a quota selector nor random
    decoder pruning. A server must bind/record its device-native equivalent.
    """
    ids = _ids(patch_ids, "patch_ids")
    baseline, candidate = _ids(baseline_ids, "baseline"), _ids(candidate_ids, "candidate")
    if (len(baseline) != len(candidate) or not set(baseline).issubset(ids)
            or not set(candidate).issubset(ids) or type(seed) is not int or seed < 0):
        raise ValueError("same-K actual subsets and nonnegative integer seed required")
    common = np.intersect1d(baseline, candidate)
    count = len(baseline)-len(common)
    outside = ids[~np.isin(ids, baseline)]
    added = np.random.default_rng(seed).choice(outside, size=count, replace=False)
    keep = ids[np.isin(ids, np.r_[common, added])]
    return keep, dict(changed=count, seed=seed, removed_ids=baseline[~np.isin(baseline, common)].tolist(),
                      added_ids=added.tolist())


def pair_metrics(record, baseline, candidate):
    a, b = keep_mask(record, baseline), keep_mask(record, candidate)
    common, union = int((a & b).sum()), int((a | b).sum())
    regions = []
    for region in record["regions"]:
        rect = region["box"]
        area = np.maximum(0., np.minimum(record["boxes_xyxy"][:, 2:], rect[2:])
                          - np.maximum(record["boxes_xyxy"][:, :2], rect[:2])).prod(-1)
        denominator = float(np.prod(rect[2:]-rect[:2]))
        regions.append({"name": region["name"],
                        "baseline_area_fraction": float(area[a].sum()/denominator),
                        "candidate_area_fraction": float(area[b].sum()/denominator)})
    return dict(image_key=record["image_key"], crop_key=record["crop_key"],
                case_key=record["case_key"], baseline=baseline, candidate=candidate,
                k=int(a.sum()), work=record["work"], common=common, union=union,
                jaccard=common/union, retention_overlap=common/int(a.sum()),
                baseline_only=list(map(int, record["patch_ids"][a & ~b])),
                candidate_only=list(map(int, record["patch_ids"][b & ~a])),
                annotated_region_coverage=regions or None)


def _axis(ax, rgb, title):
    ax.imshow(rgb)
    ax.set(xlim=(0, rgb.shape[1]), ylim=(rgb.shape[0], 0), title=title)
    ax.set_aspect("equal")
    ax.axis("off")


def _boxes(ax, boxes, *, values=None, cmap="magma", norm=None,
           facecolors="none", edgecolor="none", alpha=1., linewidth=.8):
    collection = PatchCollection([Rectangle((x, y), right-x, bottom-y)
                                  for x, y, right, bottom in boxes],
                                 cmap=cmap, norm=norm,
                                 facecolors=None if values is not None else facecolors,
                                 edgecolors=edgecolor, linewidths=linewidth, alpha=alpha)
    if values is not None:
        collection.set_array(np.asarray(values))
    ax.add_collection(collection)
    return collection


def _save(fig, output, *, fixture):
    if fixture:
        fig.get_layout_engine().set(rect=(0, 0, 1, .92))
        fig.text(.5, .985, "SYNTHETIC FIXTURE — NOT VLM EVIDENCE", ha="center", va="top",
                 fontsize=13, color="#a44131", bbox=dict(facecolor="white", edgecolor="none", pad=3))
    fig.savefig(str(output)+".png", dpi=160, facecolor="white")
    fig.savefig(str(output)+".svg", facecolor="white")
    plt.close(fig)


def render_case(record, output, baseline, candidate, vmax):
    rgb, boxes = record["rgb"], record["boxes_xyxy"]
    guide = record["source"]["guide_completed"]
    pen = record["source"]["pen_completed"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    titles = ["Processed crop", f"CLS score: completed block {guide}",
              f"CLS score: pen / block {pen}", baseline, candidate,
              "Actual selected identities"]
    for ax, title in zip(axes.flat, titles):
        _axis(ax, rgb, title)
    norm = Normalize(0, vmax)
    for ax, key in ((axes[0, 1], "guide_scores"), (axes[0, 2], "pen_scores")):
        _boxes(ax, boxes, values=record[key], norm=norm, alpha=.7)
    scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap="magma")
    fig.colorbar(scalar, ax=list(axes[0, 1:]), label="Raw score; shared scale across this input batch", shrink=.75)
    a, b = keep_mask(record, baseline), keep_mask(record, candidate)
    _boxes(axes[1, 0], boxes[a], edgecolor=BASE, linewidth=1.5)
    _boxes(axes[1, 1], boxes[b], edgecolor=NEW, linewidth=1.5)
    selected = a | b
    colors = [to_rgba(BOTH if aa and bb else BASE if aa else NEW, .65)
              for aa, bb in zip(a[selected], b[selected])]
    _boxes(axes[1, 2], boxes[selected], facecolors=colors, edgecolor="white", linewidth=.4)
    fig.legend(handles=[Patch(color=BOTH, label="Both"),
                        Patch(color=BASE, label="Baseline only"),
                        Patch(color=NEW, label="Candidate only")],
               loc="outside lower center", ncol=3, frameon=False)
    _save(fig, output/"panel", fixture=record["fixture"])
    methods = list(record["methods"])
    rows = math.ceil(len(methods)/3)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 3.9*rows), squeeze=False,
                             layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, name in zip(axes.flat, methods):
        _axis(ax, rgb, f"{name} / K={int(keep_mask(record, name).sum())}")
        _boxes(ax, boxes[keep_mask(record, name)], edgecolor=NEW, linewidth=1.3)
    _save(fig, output/"selection_grid", fixture=record["fixture"])
    # Keep all numeric values, including zero masses. There is no automatic
    # within-visual normalization that could conceal absolute mass changes.
    np.savez_compressed(output/"raw_values.npz", patch_ids=record["patch_ids"],
                        boxes_xyxy=boxes, guide_scores=record["guide_scores"],
                        pen_scores=record["pen_scores"])


def audit(paths, output, *, baseline, candidate, allow_fixture=False):
    records = [load_record(p, allow_fixture=allow_fixture) for p in paths]
    if not records:
        raise ValueError("no actual input records")
    if len({r["fixture"] for r in records}) != 1:
        raise ValueError("do not mix fixture and real records")
    names = list(records[0]["methods"])
    if baseline not in names or candidate not in names or baseline == candidate:
        raise ValueError("two distinct actual method names required")
    signature = lambda r: (r["work"], len(r["methods"][baseline]["keep_ids"]),
                            tuple(r["source"][k] for k in
                                  ("model_sha", "processor_sha", "selector_sha", "saliency_rule",
                                   "schedule_hash", "split", "vision_depth", "guide_completed", "pen_completed")))
    unique = {}
    for r in records:
        if set(r["methods"]) != set(names) or signature(r) != signature(records[0]):
            raise ValueError("one budget, method panel and source binding per audit batch required")
        key = (r["image_key"], r["crop_key"])
        if key in unique:
            raise ValueError("duplicate image/crop: Stage1 panels must not repeat questions as independent images")
        unique[key] = r
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    vmax = max(float(r[k].max()) for r in records for k in ("guide_scores", "pen_scores")) or 1.
    rows, per_image = [], defaultdict(list)
    for r in records:
        case_hash = hashlib.sha256((r["image_key"]+"\0"+r["crop_key"]).encode()).hexdigest()[:16]
        folder = output/case_hash
        folder.mkdir(exist_ok=True)
        render_case(r, folder, baseline, candidate, vmax)
        metrics = pair_metrics(r, baseline, candidate)
        metrics.update(fixture=r["fixture"], source=r["source"], shared_raw_vmax=vmax,
                       category=r.get("category"), predictions=r.get("predictions"),
                       note="Attention, selection and geometric coverage diagnostics; not causal token utility")
        (folder/"metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False)+"\n")
        rows.append(metrics)
        matrix = np.array([[pair_metrics(r, aa, bb)["jaccard"] for bb in names] for aa in names])
        per_image[r["image_key"]].append(matrix)
    # Average crops inside each canonical image before averaging images.
    overlap = np.stack([np.mean(values, axis=0) for values in per_image.values()]).mean(0)
    fig, ax = plt.subplots(figsize=(max(6., len(names)*.9), max(5., len(names)*.8)), layout="constrained")
    im = ax.imshow(overlap, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_title("Selected-set Jaccard; each canonical image weighted equally")
    for i, j in itertools.product(range(len(names)), repeat=2):
        ax.text(j, i, f"{overlap[i, j]:.2f}", ha="center", va="center",
                color="white" if overlap[i, j] < .55 else INK, fontsize=9)
    fig.colorbar(im, ax=ax, label="Jaccard", shrink=.8)
    _save(fig, output/"overlap_matrix", fixture=records[0]["fixture"])
    with (output/"overlap.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["image_key", "crop_key", "case_key", "k", "work", "baseline", "candidate",
                  "common", "union", "jaccard", "retention_overlap"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    result = dict(schema=SCHEMA, fixture=records[0]["fixture"], image_count=len(per_image),
                  crop_count=len(records), methods=names, shared_raw_vmax=vmax,
                  work=records[0]["work"], overlap_matrix=overlap.tolist(),
                  source=records[0]["source"], score_zero=all(not r[k].any() for r in records
                                                           for k in ("guide_scores", "pen_scores")))
    (output/"summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    return result


def render_sweep(path, output, *, allow_fixture=False):
    value = json.loads(Path(path).read_text())
    if value.get("schema") != "elva_stage1_sweep_v1" or type(value.get("fixture")) is not bool:
        raise ValueError("sweep schema and explicit fixture flag required")
    if value["fixture"] and not allow_fixture:
        raise ValueError("fixture sweep requires --allow-fixture")
    for key in ("metric_name", "source_hash", "scorer_sha", "frozen_panel_hash", "timing_protocol_hash"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"bound {key} required")
    groups = defaultdict(list)
    for row in value.get("rows", []):
        if row.get("selection_mode") != "weighted_evidence" or type(row.get("cover")) is not bool:
            raise ValueError("do not mix quota selection, fusion and weighting on one beta curve")
        beta, metric, ttft = (row.get(k) for k in ("beta", "metric", "wall_ttft_p50_ms"))
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (beta, metric, ttft))
                or not 0 <= beta <= 1 or ttft <= 0 or type(row.get("work")) is not int
                or row["work"] <= 0 or row.get("payload") != "pen"
                or row.get("split") not in ("development", "confirmation")):
            raise ValueError("finite beta/official metric/positive wall TTFT and frozen protocol required")
        groups[(row["work"], row["cover"], row["split"])].append(row)
    if not groups:
        raise ValueError("empty sweep; real evaluation records required")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for (work, cover, split), rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["beta"])
        x = [r["beta"] for r in rows]
        if len(set(x)) != len(x):
            plt.close(fig)
            raise ValueError("aggregate true repeats first; duplicate beta rows in one comparison")
        label = f"W={work}, COVER={cover}, {split}"
        axes[0].plot(x, [r["metric"] for r in rows], marker="o", label=label)
        axes[1].plot(x, [r["wall_ttft_p50_ms"] for r in rows], marker="o", label=label)
    for ax in axes:
        ax.set_xlabel("Guide-layer evidence weight beta (not a token quota)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(value["metric_name"])
    axes[1].set_ylabel("End-to-end wall TTFT p50 (ms)")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _save(fig, output/"beta_quality_ttft", fixture=value["fixture"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    image = commands.add_parser("images")
    image.add_argument("records", nargs="+")
    image.add_argument("--baseline", required=True)
    image.add_argument("--candidate", required=True)
    sweep = commands.add_parser("sweep")
    sweep.add_argument("ledger")
    for sub in (image, sweep):
        sub.add_argument("--out", required=True)
        sub.add_argument("--allow-fixture", action="store_true")
    args = parser.parse_args()
    if args.command == "images":
        print(json.dumps(audit(args.records, args.out, baseline=args.baseline,
                               candidate=args.candidate, allow_fixture=args.allow_fixture), indent=2))
    else:
        render_sweep(args.ledger, args.out, allow_fixture=args.allow_fixture)


if __name__ == "__main__":
    main()

```
