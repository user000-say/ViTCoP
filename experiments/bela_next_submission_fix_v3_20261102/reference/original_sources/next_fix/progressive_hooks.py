"""Physical layer-wise Llama pruning using forward hooks (Transformers v4.48-style).

This module is intentionally strict: unknown cache API/missing real Q-attention
must raise rather than silently run Dense. One sample per run, eager attention.
"""
import torch


class InvalidForward(RuntimeError):
    pass


def _trim_dynamic_cache(cache, keep, expected):
    if cache is None or not hasattr(cache,'key_cache') or not hasattr(cache,'value_cache'):
        raise InvalidForward('Expected transformers DynamicCache with key_cache/value_cache')
    if len(cache.key_cache) != len(cache.value_cache):
        raise InvalidForward('Cache key/value layer arrays disagree')
    for index in range(len(cache.key_cache)):
        k,v = cache.key_cache[index],cache.value_cache[index]
        if k is None:
            continue
        if k.ndim!=4 or v.ndim!=4 or k.shape[-2]!=expected or v.shape[-2]!=expected:
            raise InvalidForward(f'KV cache old length mismatch layer={index} expected={expected}')
        cache.key_cache[index] = k.index_select(-2,keep)
        cache.value_cache[index] = v.index_select(-2,keep)
    return cache


class ProgressivePruningHooks:
    """Install only around prefill. Cache physically shrinks at each prune event.

    Stage1 seq masks include text, actual selected native patch tokens, and
    native row-break embeddings. `patch_mask` counts only true image patches.
    """
    def __init__(self, layers, patch_mask, cut12,cut20, *, cut_layers=(12,20),
                 qa_text_mask=None, require_cache=True):
        self.layers = layers
        self.cut_targets = dict(zip(cut_layers,(cut12,cut20)))
        self.first_length=len(patch_mask)
        self.origin_indices=torch.arange(self.first_length,device=patch_mask.device)
        self.base_patch_mask=patch_mask.bool()
        self.base_text_mask=(~patch_mask.bool()) if qa_text_mask is None else qa_text_mask.bool()
        self.require_cache=require_cache
        self.log=[]
        self.handles=[]
        self.cache_ref=None
        if self.base_patch_mask.sum() < max(cut12,cut20):
            raise InvalidForward('Stage1 patch count smaller than stage2 targets')
        if self.first_length==0:
            raise InvalidForward('Empty model input')
        if len(layers)<max(cut_layers):
            raise InvalidForward('Decoder too shallow for requested cuts')

    def __enter__(self):
        for layer_i,layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_pre_hook(self._pre(layer_i+1),with_kwargs=True))
            if layer_i+1 in self.cut_targets:
                self.handles.append(layer.register_forward_hook(self._post(layer_i+1),with_kwargs=True))
        return self

    def __exit__(self,*exc):
        for h in self.handles: h.remove()
        self.handles=[]

    def _pre(self, layer_no):
        def before(module,args,kwargs):
            if not args:
                raise InvalidForward('Decoder must take hidden_states as first argument')
            current_len=len(self.origin_indices)
            if args[0].shape[0]!=1 or args[0].shape[1]!=current_len:
                raise InvalidForward(f'Layer {layer_no} input is not physically pruned: {args[0].shape} != {current_len}')
            kwargs = dict(kwargs)
            mask = kwargs.get('attention_mask')
            if mask is not None:
                if mask.ndim == 4:
                    if mask.shape[-2] == self.first_length:
                        mask=mask.index_select(-2,self.origin_indices).index_select(-1,self.origin_indices)
                    elif mask.shape[-2]!=current_len or mask.shape[-1]!=current_len:
                        raise InvalidForward(f'Layer {layer_no} attention mask inconsistent {mask.shape}')
                elif mask.ndim == 2:
                    if mask.shape[-1]==self.first_length:
                        mask=mask.index_select(-1,self.origin_indices)
                    elif mask.shape[-1]!=current_len:
                        raise InvalidForward('2D attention mask inconsistent')
                else:
                    raise InvalidForward('Unexpected decoder attention mask rank')
                kwargs['attention_mask']=mask
            pos = kwargs.get('position_ids')
            if pos is not None:
                if pos.shape[-1]==self.first_length:
                    kwargs['position_ids']=pos.index_select(-1,self.origin_indices)
                elif pos.shape[-1]!=current_len:
                    raise InvalidForward('Position ID length inconsistent')
            embeddings = kwargs.get('position_embeddings')
            if embeddings is not None:
                if len(embeddings)!=2:
                    raise InvalidForward('Unexpected RoPE position_embeddings')
                fixed=[]
                for x in embeddings:
                    if x.shape[-2]==self.first_length:
                        fixed.append(x.index_select(-2,self.origin_indices))
                    elif x.shape[-2]==current_len:
                        fixed.append(x)
                    else: raise InvalidForward('RoPE length inconsistent')
                kwargs['position_embeddings']=tuple(fixed)
            cp = kwargs.get('cache_position')
            if cp is not None:
                kwargs['cache_position']=torch.arange(current_len,device=cp.device,dtype=cp.dtype)
            self.cache_ref=kwargs.get('past_key_value',self.cache_ref)
            if layer_no in self.cut_targets:
                kwargs['output_attentions']=True
            return args,kwargs
        return before

    def _post(self,layer_no):
        def after(module,args,kwargs,output):
            if not isinstance(output,(tuple,list)) or len(output)<2:
                raise InvalidForward('Q_ATTN missing: decoder did not return attention matrix')
            hidden,attn=output[:2]
            n=len(self.origin_indices)
            if not torch.is_tensor(attn) or attn.ndim!=4 or attn.shape[-2:]!=(n,n):
                raise InvalidForward(f'Q_ATTN missing/invalid at layer {layer_no}: {getattr(attn,"shape",None)}')
            patch = self.base_patch_mask[self.origin_indices]
            text = self.base_text_mask[self.origin_indices] & ~patch
            if not bool(text.any()): raise InvalidForward('No text query tokens for Q_ATTN')
            n_patches=int(patch.sum())
            k=self.cut_targets[layer_no]
            if k>n_patches: raise InvalidForward('Stage2 requested more patches than available')
            patch_at=torch.nonzero(patch,as_tuple=True)[0]
            text_at=torch.nonzero(text,as_tuple=True)[0]
            # selected question/text queries -> selected visual keys, averaged over heads/queries
            a=attn[0].index_select(-2,text_at).index_select(-1,patch_at)
            scores=a.float().mean(dim=(0,1))
            if not bool(torch.isfinite(scores).all()):
                raise InvalidForward('Non-finite Q_ATTN scores')
            top=torch.argsort(scores,descending=True,stable=True)[:k]
            keep_mask=(~patch)
            keep_mask[patch_at[top]]=True
            local_keep=torch.nonzero(keep_mask,as_tuple=True)[0]
            if self.require_cache:
                _trim_dynamic_cache(self.cache_ref,local_keep,n)
            new_hidden=hidden.index_select(1,local_keep)
            old=self.origin_indices
            self.origin_indices=old.index_select(0,local_keep)
            self.log.append({'layer':layer_no,'before_patches':n_patches,
                'after_patches':k,'before_sequence':n,'after_sequence':len(local_keep),
                'original_stage1_indices':self.origin_indices[self.base_patch_mask[self.origin_indices]].tolist()})
            return (new_hidden,*output[1:]) if isinstance(output,tuple) else [new_hidden,*output[1:]]
        return after

    def verify(self):
        if len(self.log)!=len(self.cut_targets):
            raise InvalidForward(f'PRUNING_NOT_ACTIVE: got only {len(self.log)} events')
        for rec in self.log:
            if rec['after_patches']!=self.cut_targets[rec['layer']]:
                raise InvalidForward('Wrong actual per-layer Patch count')
        if self.require_cache:
            expected=len(self.origin_indices)
            cache=self.cache_ref
            if cache is None or len(cache.key_cache)<len(self.layers):
                raise InvalidForward('KV cache missing layers after pruning')
            for i,k in enumerate(cache.key_cache):
                if k.shape[-2]!=expected:
                    raise InvalidForward(f'Final KV cache layer {i} length {k.shape[-2]} != {expected}')
        return True
