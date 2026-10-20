"""Server-only actual-feature CUDA parity. Deliberately fails without CUDA."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from current_recycle import recycle_current as reference
from current_torch import recycle_current as device_impl


def main():
    p=argparse.ArgumentParser()
    p.add_argument('actual_npz')
    p.add_argument('output_json')
    args=p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; validation not run')
    z=np.load(args.actual_npz, allow_pickle=False)
    needed=('hidden','keep','metric','coords','groups','positions')
    if not all(k in z for k in needed): raise ValueError('missing actual capture fields')
    outputs=[]
    for dtype in (torch.float32,torch.float16,torch.bfloat16):
        tensors={k:torch.tensor(z[k],device='cuda',dtype=(dtype if k in ('hidden','metric') else torch.float32 if k=='coords' else torch.long)) for k in needed}
        source={k:v.float().cpu().numpy() if k in ('hidden','metric') else v.cpu().numpy() for k,v in tensors.items()}
        for alpha in (0.,.25,.5,1.):
            for prefix in (False,True):
                kwargs=dict(alpha=alpha,prefix_only=prefix)
                got,a=device_impl(**tensors,**kwargs,debug=True)
                expected,b=reference(**source,**kwargs)
                target=torch.tensor(expected,device='cuda').to(dtype).float().cpu().numpy()
                actual=got.float().cpu().numpy()
                np.testing.assert_allclose(actual,target,rtol=.03 if dtype!=torch.float32 else 2e-5,
                    atol=.03 if dtype!=torch.float32 else 2e-5)
                if not a['off']:
                    np.testing.assert_array_equal(a['receiver'].cpu().numpy(),b['receiver'])
                outputs.append(dict(dtype=str(dtype),alpha=alpha,prefix=prefix,max_error=float(np.abs(actual-target).max())))
    Path(args.output_json).write_text(json.dumps(dict(actual_capture=str(args.actual_npz),checks=outputs),indent=2)+'\n')


if __name__=='__main__': main()
