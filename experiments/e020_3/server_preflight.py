"""Real-feature CUDA parity check. Requires a server-captured NPZ fixture.

Required arrays: hidden[N,D], keep[K], affinity[N,N], coords[N,2].
Optional: groups[N], weights[N]. This is not a VLM accuracy benchmark.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("fixture")
    p.add_argument("report")
    p.add_argument("--device",default="cuda:0")
    args=p.parse_args()
    import torch
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("real CUDA is required; this check cannot silently skip")
    if __package__:
        from .recycle import recycle as reference
        from .torch_backend import recycle as actual
    else:
        from recycle import recycle as reference
        from torch_backend import recycle as actual
    with np.load(args.fixture,allow_pickle=False) as d:
        data={k:d[k] for k in d.files}
    for key in ("hidden","keep","affinity","coords"):
        if key not in data:
            raise ValueError("missing actual feature field: "+key)
    # Explicit float32 arithmetic parity. Actual model dtype/logit/generation
    # parity and timing are separate integration tasks.
    x=np.asarray(data["hidden"],np.float32)
    keep=np.asarray(data["keep"],np.int64)
    device=torch.device(args.device)
    tx=torch.tensor(x,device=device)
    tk=torch.tensor(keep,device=device)
    kw={k:torch.tensor(data[k],device=device) for k in ("affinity","coords","groups","weights") if k in data}
    raw={k:data[k] for k in ("affinity","coords","groups","weights") if k in data}
    rows=[]
    for alpha in (0.,.25,.5,1.):
        for preserve_norm in (False,True):
            expected=reference(x,keep,alpha=alpha,preserve_norm=preserve_norm,**raw)
            got,_=actual(tx,tk,alpha=alpha,preserve_norm=preserve_norm,debug=True,**kw)
            arr=got.cpu().numpy()
            np.testing.assert_allclose(arr,expected.hidden,rtol=1e-4,atol=1e-5)
            if alpha==0:
                assert arr.tobytes()==x[keep].tobytes()
            rows.append(dict(alpha=alpha,preserve_norm=preserve_norm,
                             max_abs_error=float(np.abs(arr-expected.hidden).max())))
    report=dict(schema="e020_3_real_feature_parity_v1",passed=True,
                fixture=str(Path(args.fixture).resolve()),device=str(device),
                GPU=torch.cuda.get_device_name(device),dtype="float32",cases=rows,
                note="Does not establish VLM ACC or end-to-end TTFT")
    Path(args.report).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"passed":True,"cases":len(rows),"report":args.report}))


if __name__=="__main__": main()
