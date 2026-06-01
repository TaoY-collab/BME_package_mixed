# -*- coding: utf-8 -*-
"""
src/distance_utils.py

用于 Hybrid-Swin-SDF-CoreNet 项目的 3D mask 辅助监督目标生成工具。

主要功能：
1. 将输入 mask 统一为 [D, H, W]；
2. 根据 3D binary mask 计算 signed distance field，简称 SDF；
3. 根据 SDF 生成边界权重图；
4. 生成稀疏 core mask；
5. 提取 3D 边界 mask；
6. 一次性构建 sdf、boundary、core 三种辅助监督目标。

约定：
- 输入 mask 可以是 [D, H, W] 或 [1, D, H, W]；
- 输出 sdf、boundary、core 均为 np.float32；
- build_auxiliary_targets 输出统一为 [1, D, H, W]；
- mask 内部 SDF 为负值，mask 外部 SDF 为正值；
- 空 mask 不会导致程序崩溃。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
    maximum_filter,
)

try:
    from skimage.morphology import skeletonize_3d
except Exception:
    skeletonize_3d = None


def ensure_3d(mask: np.ndarray) -> np.ndarray:
    """
    将输入 mask 统一转换为 [D, H, W] 形式。

    参数
    ----
    mask:
        输入 3D mask，支持以下形状：
        - [D, H, W]
        - [1, D, H, W]

    返回
    ----
    mask_3d:
        形状为 [D, H, W] 的 numpy 数组。

    说明
    ----
    这里不强制转换为 binary mask，只负责维度统一。
    后续函数会根据需要自行进行二值化。
    """
    mask = np.asarray(mask)

    if mask.ndim == 3:
        return mask

    if mask.ndim == 4:
        if mask.shape[0] != 1:
            raise ValueError(
                f"输入 mask 为 4D 时，要求形状为 [1, D, H, W]，"
                f"但当前 shape = {mask.shape}"
            )
        return mask[0]

    raise ValueError(
        f"输入 mask 只支持 [D, H, W] 或 [1, D, H, W]，"
        f"但当前 shape = {mask.shape}"
    )


def _to_binary_mask(mask: np.ndarray) -> np.ndarray:
    """
    将输入 mask 转换为 bool 类型的 3D binary mask。

    参数
    ----
    mask:
        输入 mask，支持 [D, H, W] 或 [1, D, H, W]。

    返回
    ----
    mask_bool:
        bool 类型的 [D, H, W] mask。
    """
    mask_3d = ensure_3d(mask)
    mask_bool = mask_3d > 0
    return mask_bool


def compute_sdf(
    mask: np.ndarray,
    normalize: bool = True,
    clip_value: float = 20.0,
) -> np.ndarray:
    """
    计算 3D signed distance field，简称 SDF。

    参数
    ----
    mask:
        输入 3D binary mask，支持 [D, H, W] 或 [1, D, H, W]。

    normalize:
        是否将 SDF 裁剪并归一化到 [-1, 1]。

    clip_value:
        SDF 裁剪范围。
        当 normalize=True 时，会先裁剪到 [-clip_value, clip_value]，
        再除以 clip_value，得到 [-1, 1] 范围。

    返回
    ----
    sdf:
        np.float32 类型的 [D, H, W] 数组。
        - mask 内部为负值；
        - mask 外部为正值；
        - 边界附近接近 0。

    说明
    ----
    使用公式：
        sdf = distance_outside - distance_inside

    其中：
    - distance_outside 表示背景区域到前景区域的距离；
    - distance_inside 表示前景区域到背景区域的距离。
    """
    if clip_value <= 0:
        raise ValueError(f"clip_value 必须大于 0，但当前 clip_value = {clip_value}")

    mask_bool = _to_binary_mask(mask)
    sdf_shape = mask_bool.shape

    # 空 mask：整个 patch 都是背景。
    # 此时没有真实边界，为了保持“外部为正”的语义，直接返回正距离。
    if not np.any(mask_bool):
        sdf = np.full(sdf_shape, fill_value=clip_value, dtype=np.float32)
        if normalize:
            sdf = sdf / float(clip_value)
        return sdf.astype(np.float32)

    # 全 mask：整个 patch 都是前景。
    # 此时没有真实背景，为了保持“内部为负”的语义，直接返回负距离。
    if np.all(mask_bool):
        sdf = np.full(sdf_shape, fill_value=-clip_value, dtype=np.float32)
        if normalize:
            sdf = sdf / float(clip_value)
        return sdf.astype(np.float32)

    # 背景区域到前景区域的距离。
    # 对于背景 voxel，值为到最近前景 voxel 的距离；
    # 对于前景 voxel，值为 0。
    distance_outside = distance_transform_edt(~mask_bool)

    # 前景区域到背景区域的距离。
    # 对于前景 voxel，值为到最近背景 voxel 的距离；
    # 对于背景 voxel，值为 0。
    distance_inside = distance_transform_edt(mask_bool)

    # 外部为正，内部为负。
    sdf = distance_outside - distance_inside
    sdf = sdf.astype(np.float32)

    if normalize:
        sdf = np.clip(sdf, -clip_value, clip_value)
        sdf = sdf / float(clip_value)

    return sdf.astype(np.float32)


def compute_boundary_weight(
    mask: np.ndarray,
    sigma: float = 3.0,
    w0: float = 5.0,
) -> np.ndarray:
    """
    根据 SDF 生成边界权重图。

    参数
    ----
    mask:
        输入 3D binary mask，支持 [D, H, W] 或 [1, D, H, W]。

    sigma:
        控制边界权重衰减速度。
        sigma 越大，边界附近高权重区域越宽。

    w0:
        边界区域的额外权重强度。
        理论上最大权重接近 1 + w0。

    返回
    ----
    weight:
        np.float32 类型的 [D, H, W] 数组。
        权重范围大致为 [1, 1 + w0]。

    公式
    ----
    weight = 1 + w0 * exp(-abs(sdf_raw) / sigma)
    """
    if sigma <= 0:
        raise ValueError(f"sigma 必须大于 0，但当前 sigma = {sigma}")

    mask_bool = _to_binary_mask(mask)

    # 空 mask 或全 mask 没有明确边界，直接返回全 1 权重。
    if not np.any(mask_bool) or np.all(mask_bool):
        return np.ones(mask_bool.shape, dtype=np.float32)

    sdf_raw = compute_sdf(mask_bool, normalize=False)
    weight = 1.0 + float(w0) * np.exp(-np.abs(sdf_raw) / float(sigma))

    return weight.astype(np.float32)


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    """
    提取 3D binary mask 的边界区域。

    参数
    ----
    mask:
        输入 3D binary mask，支持 [D, H, W] 或 [1, D, H, W]。

    返回
    ----
    boundary:
        np.float32 类型的 [D, H, W] 3D 边界 mask。
        边界处为 1，其余位置为 0。

    说明
    ----
    通过 binary erosion 得到内部腐蚀区域：
        boundary = mask - eroded_mask

    对 3D 肺结节而言，这可以得到前景 mask 的表面边界。
    """
    mask_bool = _to_binary_mask(mask)

    if not np.any(mask_bool):
        return np.zeros(mask_bool.shape, dtype=np.float32)

    structure = generate_binary_structure(rank=3, connectivity=1)
    eroded = binary_erosion(mask_bool, structure=structure, border_value=0)

    boundary = mask_bool & (~eroded)

    return boundary.astype(np.float32)


def _fallback_core_from_distance(mask_bool: np.ndarray) -> np.ndarray:
    """
    当 skeletonize_3d 不可用时，基于距离变换生成稀疏 core 近似。

    参数
    ----
    mask_bool:
        bool 类型的 [D, H, W] binary mask。

    返回
    ----
    core:
        bool 类型的 [D, H, W] 稀疏 core mask。

    设计思路
    ----
    1. 对前景 mask 计算 distance transform；
    2. 在前景内部寻找距离图的局部极大值；
    3. 这些局部极大值通常位于结构中心区域，可作为 skeleton/core 的近似；
    4. 若局部极大值为空，则使用最大距离点作为兜底。
    """
    core = np.zeros_like(mask_bool, dtype=bool)

    if not np.any(mask_bool):
        return core

    dist = distance_transform_edt(mask_bool)

    max_dist = float(dist.max())
    if max_dist <= 0:
        return core

    # 寻找 3D 局部极大值。
    local_max = dist == maximum_filter(dist, size=3)
    local_max = local_max & mask_bool

    # 为了避免边缘噪声点过多，只保留距离较大的中心区域。
    # 阈值不宜过高，否则细长结构可能只剩极少数点。
    threshold = max(1.0, 0.3 * max_dist)
    core = local_max & (dist >= threshold)

    # 如果仍然没有 core，则至少保留一个最大距离点。
    if not np.any(core):
        max_index = np.unravel_index(np.argmax(dist), dist.shape)
        core[max_index] = True

    return core


def compute_core_mask(mask: np.ndarray) -> np.ndarray:
    """
    生成稀疏 3D core mask。

    参数
    ----
    mask:
        输入 3D binary mask，支持 [D, H, W] 或 [1, D, H, W]。

    返回
    ----
    core:
        np.float32 类型的 [D, H, W] 稀疏 3D mask。

    说明
    ----
    优先使用 skimage.morphology.skeletonize_3d。
    如果当前环境中的 skimage 不支持 skeletonize_3d，
    则使用基于 distance transform 的局部极大值方法进行近似。
    """
    mask_bool = _to_binary_mask(mask)

    if not np.any(mask_bool):
        return np.zeros(mask_bool.shape, dtype=np.float32)

    core_bool = None

    if skeletonize_3d is not None:
        try:
            skeleton = skeletonize_3d(mask_bool.astype(np.uint8))
            core_bool = skeleton > 0
        except Exception:
            core_bool = None

    # 如果 skeletonize_3d 不可用，或者运行后结果为空，则使用 fallback。
    if core_bool is None or not np.any(core_bool):
        core_bool = _fallback_core_from_distance(mask_bool)

    return core_bool.astype(np.float32)


def build_auxiliary_targets(mask: np.ndarray) -> Dict[str, np.ndarray]:
    """
    根据输入 3D mask 一次性构建辅助监督目标。

    参数
    ----
    mask:
        输入 3D binary mask，支持 [D, H, W] 或 [1, D, H, W]。

    返回
    ----
    targets:
        字典，包含：
        {
            "sdf":      [1, D, H, W] np.float32,
            "boundary": [1, D, H, W] np.float32,
            "core":     [1, D, H, W] np.float32
        }

    说明
    ----
    这里的输出形状与项目中 image、mask、sdf、boundary、core 的统一约定一致：
        [C, D, H, W]，其中 C = 1。
    """
    mask_3d = ensure_3d(mask)

    sdf = compute_sdf(mask_3d, normalize=True)
    boundary = extract_boundary(mask_3d)
    core = compute_core_mask(mask_3d)

    targets = {
        "sdf": sdf[np.newaxis, ...].astype(np.float32),
        "boundary": boundary[np.newaxis, ...].astype(np.float32),
        "core": core[np.newaxis, ...].astype(np.float32),
    }

    return targets


if __name__ == "__main__":
    # 简单自检：构造一个 64×64×64 的球形 3D mask。
    D, H, W = 64, 64, 64
    z, y, x = np.ogrid[:D, :H, :W]

    center_z, center_y, center_x = 32, 32, 32
    radius = 10

    demo_mask = (
        (z - center_z) ** 2
        + (y - center_y) ** 2
        + (x - center_x) ** 2
        <= radius ** 2
    ).astype(np.float32)

    aux_targets = build_auxiliary_targets(demo_mask)

    print("sdf shape:", aux_targets["sdf"].shape, aux_targets["sdf"].dtype)
    print("boundary shape:", aux_targets["boundary"].shape, aux_targets["boundary"].dtype)
    print("core shape:", aux_targets["core"].shape, aux_targets["core"].dtype)

    print("sdf min/max:", aux_targets["sdf"].min(), aux_targets["sdf"].max())
    print("boundary sum:", aux_targets["boundary"].sum())
    print("core sum:", aux_targets["core"].sum())

    empty_mask = np.zeros((64, 64, 64), dtype=np.float32)
    empty_targets = build_auxiliary_targets(empty_mask)

    print("empty sdf shape:", empty_targets["sdf"].shape)
    print("empty boundary sum:", empty_targets["boundary"].sum())
    print("empty core sum:", empty_targets["core"].sum())