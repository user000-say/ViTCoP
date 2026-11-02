"""Native AnyRes packing provenance for HF LLaVA-NeXT.

Never reconstruct spatial_unpad by concatenating crops. Ask the model's own
packer to pack synthetic per-crop IDs through the IDENTICAL geometry path.
"""
from dataclasses import dataclass
import torch


class PackingMismatch(RuntimeError):
    pass


@dataclass
class NativeLayout:
    patch_ids: torch.Tensor  # [packed_len], -1 means image_newline
    image_lens: torch.Tensor
    origin_crop: torch.Tensor  # same shape, -1 for newline
    origin_local_patch: torch.Tensor
    base_image_patch_counts: tuple
    crop_counts: tuple

    @property
    def patch_mask(self):
        return self.patch_ids.ge(0)

    @property
    def newline_mask(self):
        return self.patch_ids.eq(-1)

    @property
    def patch_count(self):
        return int(self.patch_mask.sum().item())


def _normalize_packer_result(result):
    if isinstance(result, tuple) and len(result) == 2:
        feats, lens = result
    else:
        raise PackingMismatch('Native packer must return (packed_features, feature_lens)')
    if isinstance(feats, list):
        feats = torch.cat(feats, dim=0)
    return feats, torch.as_tensor(lens, device=feats.device, dtype=torch.long)


def map_native_packing(pack_fn, per_image_features, image_sizes, strategy='default'):
    """Produce exact packed row/newline map via model.pack_image_features.

    per_image_features: list of [n_crops, n_patch_per_crop, hidden_dim].
    Keep ID scalar in fp32 to avoid fp16 ID collisions; tested up to 16M.
    """
    if not per_image_features:
        raise PackingMismatch('No image features')
    device = per_image_features[0].device
    src, crop, local = [], [], []
    offset, crop_offset = 0, 0
    base_image_patch_counts=[]
    crop_counts=[]
    for features in per_image_features:
        if features.ndim != 3:
            raise PackingMismatch('Expected [crops, patches, channels]')
        n_crop, n_patch = features.shape[:2]
        base_image_patch_counts.append(int(n_patch))
        crop_counts.append(int(n_crop))
        item_ids = torch.arange(offset, offset + n_crop*n_patch, device=device, dtype=torch.float32)
        src.append(item_ids.view(n_crop, n_patch, 1))
        crop.append(torch.arange(crop_offset, crop_offset+n_crop, device=device).repeat_interleave(n_patch))
        local.append(torch.arange(n_patch,device=device).repeat(n_crop))
        offset += n_crop*n_patch
        crop_offset += n_crop
    if offset >= 16777216:
        raise PackingMismatch('Too many patches for unambiguous fp32 index tagging')
    raw, lens = _normalize_packer_result(pack_fn(src, image_sizes,
                    vision_feature_select_strategy=strategy,
                    image_newline=torch.tensor([-1.], dtype=torch.float32, device=device)))
    if raw.ndim != 2 or raw.shape[1] != 1:
        raise PackingMismatch('Unexpected native packer output')
    packed = raw[:,0]
    ids = packed.to(torch.long)
    if not torch.equal(ids.to(dtype=packed.dtype), packed):
        raise PackingMismatch('Native packer changed sentinel token IDs')
    if bool(((ids < -1) | (ids >= offset)).any()):
        raise PackingMismatch('Out-of-range native provenance ID')
    flat_crop = torch.cat(crop)
    flat_local = torch.cat(local)
    ix = ids.clamp(min=0)
    source_crop = flat_crop[ix].where(ids.ge(0), torch.full_like(ids,-1))
    source_local = flat_local[ix].where(ids.ge(0), torch.full_like(ids,-1))
    if ids[ids.ge(0)].unique().numel() != ids.ge(0).sum():
        raise PackingMismatch('Native packing duplicated a visual patch')
    if int(lens.sum()) != len(ids):
        raise PackingMismatch('Native feature lengths are inconsistent')
    return NativeLayout(ids, lens, source_crop, source_local,
                        tuple(base_image_patch_counts),tuple(crop_counts))


def pack_dual_depth(pack_fn, tower_hidden_states, per_image_features, image_sizes,
                    expected_layout: NativeLayout, mid_index=9, pen_index=-2,
                    strategy='default'):
    """Pack actual ViT mid & penultimate *unprojected* features with native geometry.

    Caller supplies the hidden_states emitted in the SAME vision-tower invocation
    that yielded the projected features. Zero dummy newline excludes structure
    markers from patch similarity (they are addressed via provenance mapping).
    """
    if tower_hidden_states is None or len(tower_hidden_states) <= mid_index:
        raise PackingMismatch('ViT hidden states missing; cannot do true dual-depth COVER')
    n_crops = sum(int(x.shape[0]) for x in per_image_features)
    all_patches = sum(int(x.shape[0]*x.shape[1]) for x in per_image_features)
    output = []
    for idx in (mid_index, pen_index):
        arr = tower_hidden_states[idx]
        if arr.shape[0] != n_crops:
            raise PackingMismatch('ViT crop count != projector crop count')
        if strategy == 'default':
            arr = arr[:,1:,:]
        if sum(int(x.shape[0]*x.shape[1]) for x in per_image_features) != arr.shape[0]*arr.shape[1]:
            raise PackingMismatch('ViT feature tokens do not match projected crop tokens')
        splits = torch.split(arr, [int(x.shape[0]) for x in per_image_features], dim=0)
        packed, lens = _normalize_packer_result(pack_fn(
            splits, image_sizes, vision_feature_select_strategy=strategy,
            image_newline=torch.zeros(arr.shape[-1], dtype=arr.dtype,device=arr.device)))
        if packed.shape[0] != len(expected_layout.patch_ids) or not torch.equal(lens,expected_layout.image_lens):
            raise PackingMismatch('Dual-depth feature packing differs from projected packing')
        output.append(packed[expected_layout.patch_mask])
    return tuple(output)


def structural_keep_mask(layout: NativeLayout, selected_patch_flat_ids):
    """Keep only newlines attached to retained high-resolution image rows.

    A newline belongs to the immediately preceding row (since the previous
    newline, but ignoring earlier base crop). Empty rows lose their newline.
    """
    selected_patch_flat_ids = torch.as_tensor(selected_patch_flat_ids,
                                  device=layout.patch_ids.device, dtype=torch.long)
    if selected_patch_flat_ids.numel() != selected_patch_flat_ids.unique().numel():
        raise PackingMismatch('Selection has duplicate patch IDs')
    if selected_patch_flat_ids.numel() and (bool(selected_patch_flat_ids.min() < 0) or
                                            bool(selected_patch_flat_ids.max() >= layout.patch_count)):
        raise PackingMismatch('Selection is outside native packed patch namespace')
    selected_mask = torch.zeros(layout.patch_count, device=layout.patch_ids.device,dtype=torch.bool)
    selected_mask[selected_patch_flat_ids] = True
    keep = torch.zeros(len(layout.patch_ids), dtype=torch.bool,device=layout.patch_ids.device)
    patch_rows = torch.nonzero(layout.patch_mask,as_tuple=True)[0]
    keep[patch_rows] = selected_mask
    # Retain newline iff at least one patch survived in its immediately preceding HR row.
    # A base-only image can have a final newline: preserve it iff any patch survived.
    off = 0
    for img_index,image_len in enumerate(layout.image_lens.tolist()):
        end = off + image_len
        region = torch.arange(off,end,device=keep.device)
        newline_indices = region[layout.newline_mask[off:end]]
        prev = off if layout.crop_counts[img_index] == 1 else off + layout.base_image_patch_counts[img_index]
        for nl in newline_indices.tolist():
            row = keep[prev:nl]
            if bool(row.any()):
                keep[nl] = True
            prev = nl + 1
        off = end
    return keep
