"""Actual official/external anchors; never reconstruct S0 from a chat score.

Export the SUCCESSFUL project config with native_method and verified per-block
visual counts. Its bridge must dispatch that original method, bypassing E020.
Reported W=2034 is context, not an invented schedule or verified model trace.
"""
import copy
from core_matrix import identify


def external_anchors(runtime):
    anchors=[]
    for raw in runtime.get('external_anchors',[]):
        c=copy.deepcopy(raw)
        if not c.get('native_method') or c.get('application')!='original_project_path':
            raise ValueError('external anchor must identify the original project method path')
        if c.get('selector',{}).get('kind')!='reference':
            raise ValueError('external anchor cannot load an E020 learned selector')
        counts=c.get('budget',{}).get('per_block_visual',[])
        if len(counts)!=runtime['model_depth'] or any(type(v) is not int or v<1 for v in counts):
            raise ValueError('external anchor needs an actual per-block visual trace')
        if sum(counts)!=c['budget'].get('visual_token_layers'):
            raise ValueError('external anchor W disagrees with its actual trace')
        c['budget']['visual_square_sum']=sum(v*v for v in counts)
        anchors.append(identify(c))
    return anchors
