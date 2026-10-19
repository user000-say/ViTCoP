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
