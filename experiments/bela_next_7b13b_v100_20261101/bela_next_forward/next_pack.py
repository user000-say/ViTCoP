"""NeXT spatial_unpad + image_newline packing with optional patch keep-sets.

Mirrors HF Transformers 5.20 `LlavaNextModel.pack_image_features` / `unpad_image`
exactly. Selection only drops *patch* tokens; every image_newline that stock
NeXT would emit is always kept. Token-type metadata is returned so Stage2 can
prune patches only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from transformers.models.llava_next.modeling_llava_next import (
    get_anyres_image_grid_shape,
    unpad_image,
)


@dataclass
class PackedVisual:
    """One image's packed visual stream (patches + newlines)."""

    features: torch.Tensor  # [L, D]
    is_patch: torch.Tensor  # [L] bool
    crop_id: torch.Tensor  # [L] long; -1 for newline
    local_patch_id: torch.Tensor  # [L] long; -1 for newline
    n_patches: int
    n_newlines: int
    grid_nh: int
    grid_nw: int
    unpad_hw: Tuple[int, int]
    base_patches: int
    tile_patches: int


def _as_hw(image_size) -> Tuple[int, int]:
    if isinstance(image_size, torch.Tensor):
        image_size = image_size.tolist()
    if isinstance(image_size, (list, tuple)):
        if len(image_size) == 2:
            return int(image_size[0]), int(image_size[1])
        if len(image_size) == 1 and isinstance(image_size[0], (list, tuple)):
            return int(image_size[0][0]), int(image_size[0][1])
    raise TypeError(f"bad image_size: {type(image_size)} {image_size}")


def _tile_maps(nh: int, nw: int, height: int, width: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Maps matching pack_image_features permute/flatten → [nh*h, nw*w]."""
    crop = (
        torch.arange(nh * nw, device=device, dtype=torch.long)
        .view(nh, nw, 1, 1)
        .expand(nh, nw, height, width)
    )
    yy = (
        torch.arange(height, device=device, dtype=torch.long)
        .view(1, 1, height, 1)
        .expand(nh, nw, height, width)
    )
    xx = (
        torch.arange(width, device=device, dtype=torch.long)
        .view(1, 1, 1, width)
        .expand(nh, nw, height, width)
    )
    local = yy * width + xx
    # features: view(nh,nw,h,w,D).permute(4,0,2,1,3).flatten(1,2).flatten(2,3)
    # equivalent spatial: permute(0,2,1,3).flatten(0,1).flatten(1,2)
    crop_map = crop.permute(0, 2, 1, 3).contiguous().flatten(0, 1).flatten(1, 2)
    local_map = local.permute(0, 2, 1, 3).contiguous().flatten(0, 1).flatten(1, 2)
    return crop_map, local_map


def valid_tile_local_ids(
    image_size_hw,
    *,
    image_grid_pinpoints,
    vision_image_size: int,
    patch_size: int,
    device=None,
) -> List[torch.Tensor]:
    """Per-tile-crop sorted local patch ids that survive unpad (base excluded)."""
    device = device or torch.device("cpu")
    orig_h, orig_w = _as_hw(image_size_hw)
    height = width = vision_image_size // patch_size
    nh, nw = get_anyres_image_grid_shape(image_size_hw, image_grid_pinpoints, vision_image_size)
    crop_map, local_map = _tile_maps(nh, nw, height, width, device)
    # unpad on float maps
    crop_u = unpad_image(crop_map.float().unsqueeze(0), (orig_h, orig_w))[0].long()
    local_u = unpad_image(local_map.float().unsqueeze(0), (orig_h, orig_w))[0].long()
    per: List[List[int]] = [[] for _ in range(nh * nw)]
    flat_c = crop_u.reshape(-1)
    flat_l = local_u.reshape(-1)
    for c, lid in zip(flat_c.tolist(), flat_l.tolist()):
        per[int(c)].append(int(lid))
    out = []
    for ids in per:
        t = torch.tensor(sorted(set(ids)), dtype=torch.long, device=device)
        out.append(t)
    return out


@torch.no_grad()
def pack_spatial_unpad_selected(
    projected_crops: torch.Tensor,
    image_size_hw,
    keep_per_crop: Sequence[torch.Tensor],
    image_newline: torch.Tensor,
    *,
    image_grid_pinpoints,
    vision_image_size: int,
    patch_size: int,
) -> PackedVisual:
    """Pack like stock NeXT, emitting only selected patches (+ all newlines).

    Args:
        projected_crops: [num_crops, 576, D] (base + tiles)
        keep_per_crop: length num_crops; each is 1D long local ids in [0,576)
        image_newline: [D]
    """
    if projected_crops.ndim != 3:
        raise ValueError(f"projected_crops must be [C,576,D], got {tuple(projected_crops.shape)}")
    num_crops, n_patches, dim = projected_crops.shape
    if n_patches != (vision_image_size // patch_size) ** 2:
        raise ValueError(f"expected {(vision_image_size // patch_size) ** 2} patches, got {n_patches}")
    if len(keep_per_crop) != num_crops:
        raise ValueError(f"keep_per_crop len {len(keep_per_crop)} != num_crops {num_crops}")

    device = projected_crops.device
    dtype = projected_crops.dtype
    nl = image_newline.to(device=device, dtype=dtype)
    keep_sets = []
    for ci, k in enumerate(keep_per_crop):
        kk = k.to(device=device, dtype=torch.long).unique()
        if kk.numel() and (int(kk.min()) < 0 or int(kk.max()) >= n_patches):
            raise RuntimeError(f"crop{ci} keep id out of range")
        keep_sets.append(set(int(x) for x in kk.tolist()))

    orig_h, orig_w = _as_hw(image_size_hw)
    height = width = vision_image_size // patch_size

    feats: List[torch.Tensor] = []
    is_patch: List[bool] = []
    crop_ids: List[int] = []
    local_ids: List[int] = []

    if num_crops == 1:
        # single-crop path (stock else-branch)
        base = projected_crops[0]
        for lid in sorted(keep_sets[0]):
            feats.append(base[lid])
            is_patch.append(True)
            crop_ids.append(0)
            local_ids.append(lid)
        feats.append(nl)
        is_patch.append(False)
        crop_ids.append(-1)
        local_ids.append(-1)
        packed = torch.stack(feats, dim=0)
        return PackedVisual(
            features=packed,
            is_patch=torch.tensor(is_patch, dtype=torch.bool, device=device),
            crop_id=torch.tensor(crop_ids, dtype=torch.long, device=device),
            local_patch_id=torch.tensor(local_ids, dtype=torch.long, device=device),
            n_patches=sum(is_patch),
            n_newlines=1,
            grid_nh=1,
            grid_nw=1,
            unpad_hw=(height, width),
            base_patches=sum(is_patch),
            tile_patches=0,
        )

    # multi-crop AnyRes: base + spatial_unpad tiles
    base = projected_crops[0]
    for lid in sorted(keep_sets[0]):
        feats.append(base[lid])
        is_patch.append(True)
        crop_ids.append(0)
        local_ids.append(lid)
    base_n = sum(is_patch)

    nh, nw = get_anyres_image_grid_shape(image_size_hw, image_grid_pinpoints, vision_image_size)
    if 1 + nh * nw != num_crops:
        raise RuntimeError(
            f"crop layout mismatch: num_crops={num_crops} vs 1+{nh}*{nw}={1 + nh * nw} "
            f"for image_size={(orig_h, orig_w)} pinpoints={image_grid_pinpoints}"
        )

    tiles = projected_crops[1:]  # [nh*nw, 576, D]
    image_feature = tiles.view(nh, nw, height, width, dim)
    image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
    image_feature = image_feature.flatten(1, 2).flatten(2, 3)  # [D, nh*h, nw*w]
    image_feature = unpad_image(image_feature, (orig_h, orig_w))  # [D, H', W']

    crop_map, local_map = _tile_maps(nh, nw, height, width, device)
    crop_u = unpad_image(crop_map.float().unsqueeze(0), (orig_h, orig_w))[0].long()
    local_u = unpad_image(local_map.float().unsqueeze(0), (orig_h, orig_w))[0].long()
    if crop_u.shape != image_feature.shape[1:]:
        raise RuntimeError(f"unpad map shape {tuple(crop_u.shape)} != feat {tuple(image_feature.shape[1:])}")

    H_u, W_u = int(image_feature.shape[1]), int(image_feature.shape[2])
    # append newline column like stock
    image_feature = torch.cat(
        (
            image_feature,
            nl[:, None, None].expand(dim, H_u, 1),
        ),
        dim=-1,
    )  # [D, H', W'+1]

    tile_n = 0
    nl_n = 0
    for y in range(H_u):
        for x in range(W_u):
            tile_crop = int(crop_u[y, x].item())  # 0..nh*nw-1 among tiles
            lid = int(local_u[y, x].item())
            global_crop = tile_crop + 1  # +1 for base
            if lid in keep_sets[global_crop]:
                # use projected value (same as image_feature[:,y,x] numerically)
                feats.append(projected_crops[global_crop, lid])
                is_patch.append(True)
                crop_ids.append(global_crop)
                local_ids.append(lid)
                tile_n += 1
        # newline at end of row (stock always)
        feats.append(nl)
        is_patch.append(False)
        crop_ids.append(-1)
        local_ids.append(-1)
        nl_n += 1

    packed = torch.stack(feats, dim=0)
    return PackedVisual(
        features=packed,
        is_patch=torch.tensor(is_patch, dtype=torch.bool, device=device),
        crop_id=torch.tensor(crop_ids, dtype=torch.long, device=device),
        local_patch_id=torch.tensor(local_ids, dtype=torch.long, device=device),
        n_patches=int(sum(is_patch)),
        n_newlines=nl_n,
        grid_nh=nh,
        grid_nw=nw,
        unpad_hw=(H_u, W_u),
        base_patches=base_n,
        tile_patches=tile_n,
    )


@torch.no_grad()
def stock_pack_from_projected(
    projected_crops: torch.Tensor,
    image_size_hw,
    image_newline: torch.Tensor,
    *,
    image_grid_pinpoints,
    vision_image_size: int,
    patch_size: int,
    pack_fn,
    vision_feature_select_strategy: str = "default",
) -> torch.Tensor:
    """Call the model's stock pack_image_features for parity checks."""
    # stock expects list of [num_crops, 576, D] per image
    feats, _lens = pack_fn(
        [projected_crops],
        torch.tensor([list(_as_hw(image_size_hw))], device=projected_crops.device),
        vision_feature_select_strategy,
        image_newline=image_newline,
    )
    return feats[0]
