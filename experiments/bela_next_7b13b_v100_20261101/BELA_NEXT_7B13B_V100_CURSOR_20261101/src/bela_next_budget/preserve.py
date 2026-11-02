"""Keep nonpatch tokens unchanged when planning physical decoder pruning.

Patch positions include ONLY real visual patches (exclude anyres image_newline,
text, separators and padding). Output is a sorted full-sequence keep index.
Does NOT update hidden states/KV cache/rotary positions; Cursor must integrate
into the already-proven safe gather implementation and validate generation.
"""
import torch

def plan_physical_keep(sequence_length: int, patch_positions: torch.Tensor,
                       selected_patch_relative: torch.Tensor, *,
                       original_patch_ids=None, crop_ids=None, validate=True):
    if patch_positions.ndim!=1 or selected_patch_relative.ndim!=1:
        raise ValueError("Patch positions and selected indices must be flat")
    if patch_positions.dtype!=torch.long or selected_patch_relative.dtype!=torch.long:
        raise ValueError("Index dtype must be long")
    if patch_positions.device!=selected_patch_relative.device:
        raise ValueError("Device mismatch")
    if sequence_length<=0: raise ValueError("Need nonempty sequence")
    if validate:
        if not bool(((patch_positions>=0)&(patch_positions<sequence_length)).all()):
            raise ValueError("Invalid absolute patch sequence position")
        if patch_positions.unique().numel()!=patch_positions.numel():
            raise ValueError("Duplicate patch seq positions")
        if not bool(((selected_patch_relative>=0)&
                     (selected_patch_relative<patch_positions.numel())).all()):
            raise ValueError("Invalid patch-relative index")
        if selected_patch_relative.unique().numel()!=selected_patch_relative.numel():
            raise ValueError("Duplicate selected patch indexes")
    flag=torch.ones(sequence_length,dtype=torch.bool,device=patch_positions.device)
    flag[patch_positions]=False
    flag[patch_positions[selected_patch_relative]]=True
    full_keep=flag.nonzero(as_tuple=False).flatten()
    result={"full_keep_indices":full_keep,
            "selected_patch_abs_positions":patch_positions[selected_patch_relative],
            "selected_patch_relative":selected_patch_relative,
            "kept_patch_count":int(selected_patch_relative.numel()),
            "kept_nonpatch_count":int(sequence_length-patch_positions.numel())}
    if original_patch_ids is not None:
        if original_patch_ids.shape != patch_positions.shape:
            raise ValueError("original_patch_ids shape")
        result["retained_original_patch_ids"]=original_patch_ids[selected_patch_relative]
    if crop_ids is not None:
        if crop_ids.shape != patch_positions.shape:
            raise ValueError("crop_ids shape")
        result["retained_crop_ids"]=crop_ids[selected_patch_relative]
    return result
