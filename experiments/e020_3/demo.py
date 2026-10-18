"""Synthetic compaction/identity smoke only; no reported VLM accuracy."""
import json
import numpy as np
from runtime import RecyclingState
from make_matrix import operator
from costs import PROFILES,work_report


def main():
    rng=np.random.default_rng(20261018)
    n=172
    f=rng.normal(size=(n,16))
    xy=np.column_stack((np.arange(n)//24,np.arange(n)%24))
    h=rng.normal(size=(n,128)).astype(np.float32)
    patch_ids=np.arange(n)*2
    state=RecyclingState("synthetic:entryhash",f,xy,patch_ids,
                         operator(("first","final"),alpha=.5,variant="loose_gate"))
    first,ids1=state.compress("first",h,np.arange(116))
    final,ids2=state.compress("final",first.hidden,np.arange(32))
    assert len(ids1)==116 and len(ids2)==32
    assert np.isfinite(final.hidden).all()
    off=RecyclingState("synthetic:entryhash",None,None,patch_ids,operator(()))
    original,_=off.compress("first",h,np.arange(116))
    assert original.hidden.tobytes()==h[:116].tobytes()
    print(json.dumps({"synthetic_only":True,"shapes":[list(h.shape),list(first.hidden.shape),list(final.hidden.shape)],
                      "first_audit":first.audit,"final_audit":final.audit,
                      "off_is_bit_exact":True,
                      "work":{k:work_report(v)["visual_token_layers"] for k,v in PROFILES.items()}},indent=2))


if __name__=="__main__": main()
