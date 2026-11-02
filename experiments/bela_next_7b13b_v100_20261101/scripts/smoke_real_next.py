#!/usr/bin/env python3
"""Real-GPU NeXT-7B smoke: native Dense agreement then A0..A4.
Requires the original BELA wrappers. No scoring / benchmark data changes.
Usage: CUDA_VISIBLE_DEVICES=4 python smoke_real_next.py --checkpoint PATH \
  --image PATH --dual your_project_bela_compat:original_dual --qa your_project_bela_compat:original_qa
"""
import argparse
import json
import os
from pathlib import Path
import torch
from bela_next_fix.legacy_bridge import ARMS,import_fn
from bela_next_fix.runner import greedy_native_bela


def cli():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',default='llava-hf/llava-v1.6-vicuna-7b-hf')
    parser.add_argument('--image',required=True,help='Absolute path to one existing real image')
    parser.add_argument('--prompt',default='USER: <image>\nWhat is shown in this picture? ASSISTANT:')
    parser.add_argument('--dual',help='module:function wrapper using REAL historic Dual COVER')
    parser.add_argument('--qa',help='module:function wrapper using REAL historic QA-COVER')
    parser.add_argument('--max-new',type=int,default=8)
    parser.add_argument('--arms',nargs='+',default=['A0','A1','A2','A3','A4'])
    parser.add_argument('--json-out',default='next_bela_smoke.json')
    parser.add_argument('--dense-only',action='store_true')
    return parser.parse_args()


def main():
    a=cli()
    import transformers
    from transformers import AutoProcessor,LlavaNextForConditionalGeneration
    from PIL import Image
    print('transformers:',transformers.__version__)
    print('CUDA_VISIBLE_DEVICES:',os.environ.get('CUDA_VISIBLE_DEVICES'))
    if not torch.cuda.is_available():
        raise RuntimeError('This smoke must run on the selected free V100; no silent CPU fallback')
    if not Path(a.image).is_file(): raise FileNotFoundError(a.image)
    # No device_map=auto! The reference hook needs ONE fully resident Llama.
    model=LlavaNextForConditionalGeneration.from_pretrained(
        a.checkpoint,torch_dtype=torch.float16,attn_implementation='eager',
        low_cpu_mem_usage=True).to('cuda:0').eval()
    proc=AutoProcessor.from_pretrained(a.checkpoint)
    image=Image.open(a.image).convert('RGB')
    encoded=proc(images=image,text=a.prompt,return_tensors='pt')
    inp={k:(v.to(device='cuda:0',dtype=torch.float16) if k=='pixel_values' else v.to('cuda:0'))
         for k,v in encoded.items() if torch.is_tensor(v)}
    print('Native image_grid_pinpoints:',model.config.image_grid_pinpoints)
    print('Processor image_sizes:',inp['image_sizes'].tolist())
    print('Native expanded placeholders:',int((inp['input_ids']==model.config.image_token_index).sum()))
    print('Prompt head:',inp['input_ids'].shape)
    if (inp['input_ids']==model.config.image_token_index).sum()<320:
        raise RuntimeError('Image has fewer than A0 Stage1 patches; use a representative NeXT image')
    cfg=ARMS['A0']
    with torch.inference_mode():
        native=model.generate(**inp,max_new_tokens=a.max_new,do_sample=False,use_cache=True)
        if native.shape[1] <= inp['input_ids'].shape[1]:
            raise AssertionError('Native generate did not return output tokens')
        native_new=native[:,inp['input_ids'].shape[1]:]
        dense,trace_dense=greedy_native_bela(model,inp,cfg,enabled=False,progressive=False,
                                             max_new_tokens=a.max_new)
        if not torch.equal(dense,native_new):
            raise AssertionError('DENSE_EQUIVALENCE_FAILED: original checkpoint generate differs from sidecar Dense. '
              'Do not run compressed benchmarking before examining position/EOS/processor differences. '
              f'native={native_new.tolist()} sidecar={dense.tolist()}')
        print('DENSE_EQUIVALENCE_PASS:',dense.tolist())
        print('DENSE packed geometry:',{k:trace_dense[k] for k in ('native_visual_patches','native_newlines')})
        results={'checkpoint':a.checkpoint,'transformers':transformers.__version__,
                 'image_sizes':inp['image_sizes'].tolist(), 'dense':trace_dense,'arms':{}}
        if a.dense_only:
            Path(a.json_out).write_text(json.dumps(results,indent=2,ensure_ascii=False))
            return
        if a.dual is None or a.qa is None:
            raise RuntimeError('REAL_BELA_ALGORITHMS_MISSING: provide --dual and --qa wrappers from 1.5-7B code')
        dual=import_fn(a.dual)
        qa=import_fn(a.qa)
        for name in a.arms:
            if name not in ARMS: raise ValueError(name)
            tokens,trace=greedy_native_bela(model,inp,ARMS[name],dual,qa,
                                      max_new_tokens=a.max_new)
            assert trace['visual_patch_trajectory']==[ARMS[name].stage1,
                                     ARMS[name].cut12,ARMS[name].cut20]
            assert trace['qa_log']['legacy_qa_called']
            print(name,'TOKENS',trace['visual_patch_trajectory'],
                  'QA swaps',trace['qa_log']['swaps'],
                  'row_tokens',trace['stage1_newlines'],
                  'answer IDs',tokens.tolist())
            results['arms'][name]={'trace':trace,'generated_token_ids':tokens.tolist()}
            Path(a.json_out).write_text(json.dumps(results,indent=2,ensure_ascii=False))
    print('ALL_REAL_MODEL_SMOKES_PASSED')

if __name__=='__main__': main()
