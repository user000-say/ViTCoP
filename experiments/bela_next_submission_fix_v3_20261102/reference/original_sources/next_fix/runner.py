"""Reference HF LLaVA-NeXT v4.48-style inference adapter.

A real implementation (not pseudocode) for pre-expanded image placeholders and
HF Llama eager attention + DynamicCache. UNKNOWN model versions fail closed.
V100: fp16, attn_implementation=eager, batch_size=1, greedy decoding only.
It is NOT validated against user's remote V100 checkpoint/source.
"""
from dataclasses import dataclass
import torch
from .native_layout import map_native_packing,pack_dual_depth,structural_keep_mask, _normalize_packer_result
from .legacy_bridge import run_original_bela,BelaConfig
from .progressive_hooks import ProgressivePruningHooks,InvalidForward


@dataclass
class Prepared:
    embeds: torch.Tensor
    position_ids: torch.Tensor
    patch_mask: torch.Tensor
    text_mask: torch.Tensor
    full_dense_length: int
    native_patch_count: int
    native_newlines: int
    stage1_newlines: int
    qa_log: dict
    stage1_visual_id_origin: torch.Tensor


def _llm_and_layers(model):
    llm=getattr(model,'language_model',None)
    if llm is None or not hasattr(llm,'model') or not hasattr(llm.model,'layers'):
        raise InvalidForward('Only HF v4.48-style LlavaNextForConditionalGeneration + Llama supported')
    layers=llm.model.layers
    if len(layers)!=32:
        raise InvalidForward(f'Expected NeXT-7B 32 layers, found {len(layers)}')
    if getattr(llm.config,'_attn_implementation',None)!='eager':
        raise InvalidForward('Require attn_implementation=eager (real Q_ATTN extraction)')
    return llm,layers


@torch.inference_mode()
def prepare_native_bela_prefill(model, model_inputs, cfg:BelaConfig, dual_fn,qa_fn,
                                *, question_token_positions=None, enabled=True):
    """Run the checkpoint's vision tower and native packer exactly once.

    Never force [[672,672]] or simple_crop_concat. Use the SAME processor
    outputs that the Dense benchmark uses, including image_sizes/grid pinpoints.
    """
    cfg.check()
    llm,layers=_llm_and_layers(model)
    ids=model_inputs['input_ids']
    pixels=model_inputs['pixel_values']
    image_sizes=model_inputs['image_sizes']
    attention_mask=model_inputs.get('attention_mask',torch.ones_like(ids))
    if ids.ndim!=2 or ids.shape[0]!=1:
        raise InvalidForward('Batch size 1 required; avoid mixing image lengths')
    if not bool(attention_mask.eq(1).all()):
        raise InvalidForward('Padded batches unsupported: preserve original nonpadding positions first')
    if hasattr(model.config,'image_token_index') is False:
        raise InvalidForward('Not a NeXT image_token_index checkpoint')
    vision_mask=ids[0].eq(model.config.image_token_index)
    if not bool(vision_mask.any()):
        raise InvalidForward('No expanded image placeholders in input')
    if question_token_positions is None:
        question_token_positions=torch.nonzero(~vision_mask,as_tuple=True)[0]
    else:
        question_token_positions=torch.as_tensor(question_token_positions,device=ids.device,dtype=torch.long)
        if question_token_positions.numel()==0 or bool(vision_mask[question_token_positions].any()):
            raise InvalidForward('question_token_positions invalid or include image placeholders')
    hidden_trace=[]
    def _capture(tower,args,outputs):
        hs=getattr(outputs,'hidden_states',None)
        if hs is None:
            raise InvalidForward('Vision tower did not emit hidden states')
        hidden_trace.append(hs)
    handle=model.vision_tower.register_forward_hook(_capture)
    try:
        feature_layer=model.config.vision_feature_layer
        strategy=model.config.vision_feature_select_strategy
        per_image=model.get_image_features(pixels,image_sizes,
                            vision_feature_layer=feature_layer,
                            vision_feature_select_strategy=strategy)
    finally:
        handle.remove()
    if len(hidden_trace)!=1 or not isinstance(per_image,(list,tuple)) or not per_image or per_image[0].ndim!=3:
        raise InvalidForward('Unknown HF NeXT get_image_features API; do not fabricate dual features')
    native_packed, native_lens=_normalize_packer_result(model.pack_image_features(
                per_image,image_sizes,vision_feature_select_strategy=strategy,image_newline=model.image_newline))
    layout=map_native_packing(model.pack_image_features,per_image,image_sizes,strategy)
    if not torch.equal(layout.image_lens,native_lens):
        raise InvalidForward('Real packed features and native metadata disagree')
    if int(vision_mask.sum())!=len(native_packed):
        raise InvalidForward(f'Native packed {len(native_packed)} != processor image placeholders {int(vision_mask.sum())}; '
                             'DO NOT fallback to simple_crop_concat')
    if len(ids[0])<2 or ids[0,-1]==model.config.image_token_index:
        raise InvalidForward('Last input token must be a real text/query token')
    full_embeds=model.get_input_embeddings()(ids)
    if full_embeds.device!=native_packed.device:
        raise InvalidForward('Model/device sharding not supported by this reference adapter')
    full_embeds=full_embeds.clone()
    full_embeds[0,vision_mask] = native_packed.to(dtype=full_embeds.dtype)
    dense_positions=torch.arange(ids.shape[1],device=ids.device,dtype=torch.long)
    image_positions=torch.nonzero(vision_mask,as_tuple=True)[0]
    patch_positions=image_positions[layout.patch_mask]
    full_patch_mask=torch.zeros(len(ids[0]),device=ids.device,dtype=torch.bool)
    full_patch_mask[patch_positions]=True
    full_text_mask=~vision_mask
    qa_log={'swaps':0,'enabled':enabled,'legacy_dual_called':False,'legacy_qa_called':False}
    if enabled:
        mid,pen=pack_dual_depth(model.pack_image_features,hidden_trace[0],per_image,image_sizes,
                    layout, cfg.mid_layer,cfg.pen_layer,strategy)
        question=full_embeds[0,question_token_positions]
        qids=ids[0,question_token_positions]
        keep_ids,qa_log=run_original_bela(mid,pen,question,cfg,dual_fn,qa_fn,question_ids=qids,
              extra_context={'model':model,'pixel_values':pixels,'image_sizes':image_sizes,
                             'native_layout':layout,'processor_inputs':model_inputs})
        qa_log.update({'enabled':True,'legacy_dual_called':True,'legacy_qa_called':cfg.qa_swaps>0})
        local_keep=structural_keep_mask(layout,keep_ids)
        full_keep=torch.ones(len(ids[0]),device=ids.device,dtype=torch.bool)
        full_keep[image_positions]=local_keep
        chosen_patch_ids=keep_ids
    else:
        full_keep=torch.ones(len(ids[0]),device=ids.device,dtype=torch.bool)
        chosen_patch_ids=torch.arange(layout.patch_count,device=ids.device)
    seq_indices=torch.nonzero(full_keep,as_tuple=True)[0]
    packed_patch_mask=full_patch_mask[seq_indices]
    packed_text_mask=full_text_mask[seq_indices]
    if int(packed_patch_mask.sum())!=len(chosen_patch_ids):
        raise InvalidForward('Stage1 packed visual count wrong')
    stage1_nl=int((vision_mask[seq_indices] & ~packed_patch_mask).sum())
    return Prepared(full_embeds.index_select(1,seq_indices),dense_positions[seq_indices][None,:],
            packed_patch_mask,packed_text_mask,len(ids[0]),layout.patch_count,
            int(layout.newline_mask.sum()),stage1_nl,qa_log,chosen_patch_ids)


@torch.inference_mode()
def greedy_native_bela(model, model_inputs, cfg,dual_fn=None,qa_fn=None, *,
             enabled=True,progressive=True,max_new_tokens=32,eos_token_id=None,
             question_token_positions=None):
    """Compare against unmodified Dense .generate before enabling full eval.

    Does NOT alter answer punctuation or stop on newline. EOS only. Produces
    generated token IDs (no prompt echo), plus a machine-verifiable trace.
    """
    llm,layers=_llm_and_layers(model)
    if not 1<=max_new_tokens<=1024:
        raise ValueError('max_new_tokens must be 1..1024')
    prep=prepare_native_bela_prefill(model,model_inputs,cfg,dual_fn,qa_fn,
                     enabled=enabled,question_token_positions=question_token_positions)
    if progressive and not enabled:
        raise InvalidForward('Cannot enable progressive pruning with Stage1 disabled')
    if not progressive and enabled:
        raise InvalidForward('For validation, disable both Stage1 and Stage2 together')
    if eos_token_id is None:
        eos_token_id=llm.config.eos_token_id
    end_tokens={int(x) for x in (eos_token_id if isinstance(eos_token_id,(tuple,list,set)) else [eos_token_id]) if x is not None}
    hooks=ProgressivePruningHooks(layers,prep.patch_mask,cfg.cut12,cfg.cut20,
                   qa_text_mask=prep.text_mask) if progressive else None
    if hooks is not None:
        with hooks:
            out=llm(inputs_embeds=prep.embeds,attention_mask=torch.ones(prep.embeds.shape[:2],
                    device=prep.embeds.device,dtype=torch.long),position_ids=prep.position_ids,
                    use_cache=True,output_attentions=False,return_dict=True)
        hooks.verify()
        prefill_len=len(hooks.origin_indices)
    else:
        out=llm(inputs_embeds=prep.embeds,attention_mask=torch.ones(prep.embeds.shape[:2],
                device=prep.embeds.device,dtype=torch.long),position_ids=prep.position_ids,
                use_cache=True,output_attentions=False,return_dict=True)
        prefill_len=prep.embeds.shape[1]
    cache=out.past_key_values
    if cache is None or cache.get_seq_length()!=prefill_len:
        raise InvalidForward('Model KV cache length differs from physical prefill length')
    first=out.logits[:,-1,:].argmax(dim=-1,keepdim=True)
    generated=[first]
    for step in range(1,max_new_tokens):
        if int(generated[-1].item()) in end_tokens:
            break
        phys_len=cache.get_seq_length()
        next_out=llm(input_ids=generated[-1],past_key_values=cache,use_cache=True,return_dict=True,
          attention_mask=torch.ones((1,phys_len+1),device=first.device,dtype=torch.long),
          cache_position=torch.tensor([phys_len],device=first.device,dtype=torch.long),
          position_ids=torch.tensor([[prep.full_dense_length+step-1]],device=first.device,dtype=torch.long))
        cache=next_out.past_key_values
        if cache.get_seq_length()!=phys_len+1:
            raise InvalidForward('Decode KV cache did not extend exactly once')
        generated.append(next_out.logits[:,-1,:].argmax(-1,keepdim=True))
    answer=torch.cat(generated,dim=-1)
    expected=[cfg.stage1,cfg.cut12,cfg.cut20] if progressive else [prep.native_patch_count]
    actual=[int(prep.patch_mask.sum())]+[x['after_patches'] for x in hooks.log] if progressive else [prep.native_patch_count]
    if actual!=expected:
        raise InvalidForward(f'PRUNING_NOT_ACTIVE expected {expected}, got {actual}')
    trace={'visual_patch_trajectory':actual,
           'native_visual_patches':prep.native_patch_count,
           'native_newlines':prep.native_newlines,'stage1_newlines':prep.stage1_newlines,
           'prefill_physical_cache_tokens':prefill_len,
           'original_dense_prompt_tokens':prep.full_dense_length,
           'qa_log':prep.qa_log,'prune_log':hooks.log if hooks is not None else [],
           'position_preserved_after_pruning':bool(
               int(prep.position_ids[0,-1])==prep.full_dense_length-1
               and prep.position_ids.unique().numel()==prep.position_ids.numel()),
           'no_decode_answer_postprocessing':True}
    return answer,trace
