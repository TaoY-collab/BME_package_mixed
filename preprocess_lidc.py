#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hybrid-Swin-SDF-CoreNet: LIDC-IDRI 预处理脚本

功能：
1. 使用 pylidc 读取 LIDC-IDRI 的 DICOM 与 XML 标注信息；
2. 读取每个 scan 的 CT volume；
3. 将 CT 值转换为 HU，并截断到 [-1000, 400]；
4. 归一化到 [0, 1]；
5. 尽量重采样到 1×1×1 mm；
6. 对每个结节读取多个医生 annotation；
7. 只保留 annotation 数量 >= min_annotations 的结节；
8. 基于多医生标注生成共识 mask；
9. 只保留直径 >= min_diameter_mm 的结节；
10. 以结节中心裁剪 64×64×64 patch；
11. 调用 distance_utils.py 生成 sdf、boundary、core；
12. 保存为 npz：image, mask, sdf, boundary, core；
13. 按 patient_id 划分 train / val / test，避免同一病人泄漏到多个集合。

输出目录结构：
output_root/
    patches/
        LIDC-IDRI-0001_scan0001_nodule0001.npz
        ...
    train.csv
    val.csv
    test.csv
    failed_cases.txt
    preprocess_summary.csv

注意：
- pylidc 依赖本地配置文件定位 DICOM 根目录。
- 本脚本会在提供 --dicom-root 后，自动写入或更新 pylidc 配置文件。
- --xml-root 主要用于路径校验和记录；pylidc 实际读取 XML 标注依赖其自身数据库。
"""

from __future__ import annotations

import argparse
import configparser
import csv
import importlib
import multiprocessing
import os
import random
import shutil
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool
from scipy import ndimage


HU_MIN = -1000.0
HU_MAX = 400.0
TARGET_SPACING_HWD = (1.0, 1.0, 1.0)
DEFAULT_WORKERS = max(1, min(2, os.cpu_count() or 1))

_WORKER_PL: Any = None
_WORKER_CONSENSUS_FN: Any = None


@dataclass
class PatchRecord:
    """单个结节 patch 的记录，用于最终写 CSV。"""

    npz_path: str
    patient_id: str
    scan_id: str
    nodule_id: str
    num_annotations: int
    diameter_mm: float
    patch_type: str = "center"
    sample_label: int = 1
    source_nodule_id: str = ""
    nodule_size_class: str = ""
    split: str = ""


@dataclass
class VolumeRecord:
    """Whole-scan image/mask record for full CT validation or inference."""

    image: str
    label: str
    patient_id: str
    scan_id: str
    split: str = ""


def parse_patch_size(value: str) -> Tuple[int, int, int]:
    """
    解析 patch size。

    支持：
    - "64"
    - "64,64,64"
    - "64x64x64"
    """
    value = str(value).strip().lower().replace("x", ",")
    parts = [p.strip() for p in value.split(",") if p.strip()]

    if len(parts) == 1:
        size = int(parts[0])
        return size, size, size

    if len(parts) == 3:
        return int(parts[0]), int(parts[1]), int(parts[2])

    raise argparse.ArgumentTypeError(
        "--patch-size 必须是单个整数，例如 64，或者三元组，例如 64,64,64"
    )


def get_pylidc_config_path() -> Path:
    """
    获取 pylidc 配置文件路径。

    Linux / macOS:
        ~/.pylidcrc

    Windows:
        ~/pylidc.conf
    """
    home = Path.home()
    if os.name == "nt":
        return home / "pylidc.conf"
    return home / ".pylidcrc"


def prepare_pylidc_config(dicom_root: Optional[Path]) -> Optional[Path]:
    """
    根据 --dicom-root 自动写入 pylidc 配置文件。

    pylidc 默认通过用户目录下的配置文件寻找 DICOM 根目录。
    如果用户传入 --dicom-root，本函数会更新配置文件中的 [dicom] path。

    为了安全，如果原配置文件存在，会自动创建一个 .bak 备份。
    """
    if dicom_root is None:
        return None

    dicom_root = dicom_root.expanduser().resolve()
    if not dicom_root.exists():
        raise FileNotFoundError(f"--dicom-root 不存在：{dicom_root}")

    config_path = get_pylidc_config_path()
    config = configparser.ConfigParser()

    if config_path.exists():
        try:
            config.read(config_path, encoding="utf-8")
        except Exception:
            config.read(config_path)

        backup_path = config_path.with_name(config_path.name + ".bak")
        if not backup_path.exists():
            shutil.copy2(config_path, backup_path)

    if not config.has_section("dicom"):
        config.add_section("dicom")

    config.set("dicom", "path", str(dicom_root))
    config.set("dicom", "warn", "True")

    with config_path.open("w", encoding="utf-8") as f:
        config.write(f)

    return config_path


def import_pylidc():
    """
    导入 pylidc。

    如果 pylidc 不可用，给出清晰错误提示。
    """
    try:
        import pylidc as pl
        from pylidc.utils import consensus
    except ImportError as exc:
        raise ImportError(
            "\n未检测到 pylidc，无法读取 LIDC-IDRI 的 DICOM/XML 标注。\n"
            "请先安装：\n"
            "    pip install pylidc\n\n"
            "并确保已经完成 pylidc 所需的数据配置。\n"
            "如果你已经下载了 LIDC-IDRI DICOM，请运行本脚本时传入：\n"
            "    --dicom-root /path/to/LIDC-IDRI\n"
        ) from exc

    return pl, consensus


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[float, float, float]:
    """检查并归一化数据集划分比例。"""
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=np.float64)

    if np.any(ratios < 0):
        raise ValueError("train_ratio / val_ratio / test_ratio 不能为负数。")

    total = float(ratios.sum())
    if total <= 0:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须大于 0。")

    ratios = ratios / total
    return float(ratios[0]), float(ratios[1]), float(ratios[2])


def safe_float(value: Any, default: float = 0.0) -> float:
    """安全转换 float。"""
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def get_scan_identifier(scan: Any) -> str:
    """获取 scan 的稳定标识。"""
    sid = getattr(scan, "id", None)
    if sid is not None:
        return f"scan{int(sid):04d}"

    series_uid = getattr(scan, "series_instance_uid", None)
    if series_uid:
        return str(series_uid).replace(".", "_")

    return "scan_unknown"


def get_patient_id(scan: Any) -> str:
    """获取 patient_id。"""
    patient_id = getattr(scan, "patient_id", None)
    if patient_id is None:
        return "UNKNOWN_PATIENT"
    return str(patient_id)


def dicom_slice_to_hu(ds: Any) -> np.ndarray:
    """
    将单张 DICOM slice 转换为 HU。

    HU = pixel_array * RescaleSlope + RescaleIntercept
    """
    arr = ds.pixel_array.astype(np.float32)

    slope = safe_float(getattr(ds, "RescaleSlope", 1.0), 1.0)
    intercept = safe_float(getattr(ds, "RescaleIntercept", 0.0), 0.0)

    arr = arr * slope + intercept
    return arr.astype(np.float32)


def load_scan_volume_hu_hwd(scan: Any) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    读取一个 scan 的 CT volume，并转换为 HU。

    返回：
        volume_hwd: [H, W, D]
        spacing_hwd: [row_spacing, col_spacing, slice_spacing]
    """
    images = scan.load_all_dicom_images(verbose=False)
    if len(images) == 0:
        raise RuntimeError("scan.load_all_dicom_images() 返回空列表。")

    slices = [dicom_slice_to_hu(ds) for ds in images]
    volume_hwd = np.stack(slices, axis=-1).astype(np.float32)

    first = images[0]

    # 平面内 spacing 优先读取 DICOM PixelSpacing。
    pixel_spacing = getattr(first, "PixelSpacing", None)
    if pixel_spacing is not None and len(pixel_spacing) >= 2:
        row_spacing = safe_float(pixel_spacing[0], safe_float(getattr(scan, "pixel_spacing", 1.0), 1.0))
        col_spacing = safe_float(pixel_spacing[1], safe_float(getattr(scan, "pixel_spacing", 1.0), 1.0))
    else:
        ps = safe_float(getattr(scan, "pixel_spacing", 1.0), 1.0)
        row_spacing = ps
        col_spacing = ps

    # z 方向 spacing 优先使用 pylidc 的 slice_spacing。
    slice_spacing = safe_float(getattr(scan, "slice_spacing", None), 0.0)
    if slice_spacing <= 0:
        slice_spacing = safe_float(getattr(scan, "slice_thickness", 1.0), 1.0)

    if slice_spacing <= 0:
        slice_spacing = 1.0

    spacing_hwd = (float(row_spacing), float(col_spacing), float(slice_spacing))
    return volume_hwd, spacing_hwd


def clip_and_normalize_hu(volume_hu: np.ndarray) -> np.ndarray:
    """
    HU 截断到 [-1000, 400]，再归一化到 [0, 1]。
    """
    volume = np.clip(volume_hu.astype(np.float32), HU_MIN, HU_MAX)
    volume = (volume - HU_MIN) / (HU_MAX - HU_MIN)
    volume = np.clip(volume, 0.0, 1.0)
    return volume.astype(np.float32)


def bbox_to_valid_slices(
    bbox: Sequence[slice],
    full_shape: Sequence[int],
    local_shape: Sequence[int],
) -> Tuple[Tuple[slice, slice, slice], Tuple[slice, slice, slice]]:
    """
    将 pylidc 返回的 bbox 修正到 full volume 范围内。

    返回：
        full_slices: 放入 full volume 的切片
        local_slices: 从 local mask 中截取的切片
    """
    full_slices: List[slice] = []
    local_slices: List[slice] = []

    for axis, s in enumerate(bbox):
        start = 0 if s.start is None else int(s.start)
        stop = int(full_shape[axis]) if s.stop is None else int(s.stop)

        full_start = max(0, start)
        full_stop = min(int(full_shape[axis]), stop)

        if full_stop <= full_start:
            raise ValueError("bbox 与 full volume 没有有效交集。")

        local_start = max(0, -start)
        local_stop = local_start + (full_stop - full_start)

        local_stop = min(local_stop, int(local_shape[axis]))
        valid_len = local_stop - local_start

        if valid_len <= 0:
            raise ValueError("local mask 与 bbox 没有有效交集。")

        full_stop = full_start + valid_len

        full_slices.append(slice(full_start, full_stop))
        local_slices.append(slice(local_start, local_stop))

    return tuple(full_slices), tuple(local_slices)  # type: ignore[return-value]


def make_consensus_mask_hwd(
    cluster: Sequence[Any],
    volume_shape_hwd: Sequence[int],
    min_annotations: int,
    consensus_fn: Any,
) -> np.ndarray:
    """
    基于一个结节的多个医生 annotation 生成共识 mask。

    共识规则：
        多个 annotation mask 相加；
        count >= min_annotations 的体素置为 1。

    返回：
        full_mask_hwd: [H, W, D]
    """
    if len(cluster) < min_annotations:
        raise ValueError(f"annotation 数量不足：{len(cluster)} < {min_annotations}")

    # clevel 只用于让 pylidc 生成一个 common bbox；
    # 真正的 count >= min_annotations 由下面的 mask_stack.sum 实现。
    clevel = min(1.0, max(1.0 / max(len(cluster), 1), min_annotations / max(len(cluster), 1)))

    _, bbox, masks = consensus_fn(list(cluster), clevel=clevel, pad=0)

    if masks is None or len(masks) == 0:
        raise RuntimeError("pylidc.utils.consensus 未返回有效的医生 mask。")

    mask_stack = np.stack([np.asarray(m, dtype=np.uint8) for m in masks], axis=0)
    count_map = mask_stack.sum(axis=0)
    local_consensus = count_map >= int(min_annotations)

    if local_consensus.sum() == 0:
        raise RuntimeError("共识 mask 为空。")

    full_mask = np.zeros(tuple(volume_shape_hwd), dtype=np.uint8)
    full_slices, local_slices = bbox_to_valid_slices(
        bbox=bbox,
        full_shape=volume_shape_hwd,
        local_shape=local_consensus.shape,
    )
    full_mask[full_slices] = local_consensus[local_slices].astype(np.uint8)

    if full_mask.sum() == 0:
        raise RuntimeError("放回 full volume 后共识 mask 为空。")

    return full_mask


def estimate_diameter_mm_from_mask(
    mask_hwd: np.ndarray,
    spacing_hwd: Sequence[float],
) -> float:
    """
    根据共识 mask 的外接框估计结节最大直径，单位 mm。

    这里取 H/W/D 三个方向物理跨度的最大值。
    """
    coords = np.argwhere(mask_hwd > 0)
    if coords.size == 0:
        return 0.0

    min_coord = coords.min(axis=0)
    max_coord = coords.max(axis=0)
    extent_vox = max_coord - min_coord + 1

    spacing = np.asarray(spacing_hwd, dtype=np.float32)
    extent_mm = extent_vox.astype(np.float32) * spacing

    return float(np.max(extent_mm))


def estimate_diameter_mm(
    cluster: Sequence[Any],
    mask_hwd: np.ndarray,
    spacing_hwd: Sequence[float],
) -> float:
    """
    综合 annotation.diameter 和 mask 外接框估计结节直径。

    优先尽量保守地取二者最大值，避免误删接近 3mm 的小结节。
    """
    mask_diameter = estimate_diameter_mm_from_mask(mask_hwd, spacing_hwd)

    ann_diameters: List[float] = []
    for ann in cluster:
        diameter = safe_float(getattr(ann, "diameter", None), 0.0)
        if diameter > 0:
            ann_diameters.append(diameter)

    if len(ann_diameters) == 0:
        return mask_diameter

    return float(max(mask_diameter, max(ann_diameters)))


def resample_hwd(
    arr_hwd: np.ndarray,
    spacing_hwd: Sequence[float],
    target_spacing_hwd: Sequence[float] = TARGET_SPACING_HWD,
    order: int = 1,
) -> np.ndarray:
    """
    将 [H, W, D] 数组重采样到目标 spacing。

    对 image 使用 order=1；
    对 mask 使用 order=0。
    """
    spacing = np.asarray(spacing_hwd, dtype=np.float32)
    target = np.asarray(target_spacing_hwd, dtype=np.float32)

    if np.any(spacing <= 0):
        raise ValueError(f"非法 spacing：{spacing_hwd}")

    zoom_factors = spacing / target

    # scipy.ndimage.zoom 要求每个轴的缩放因子，对应 H/W/D。
    out = ndimage.zoom(arr_hwd, zoom=zoom_factors, order=order)

    return out


def hwd_to_dhw(arr_hwd: np.ndarray) -> np.ndarray:
    """[H, W, D] -> [D, H, W]。"""
    return np.transpose(arr_hwd, (2, 0, 1))


def get_mask_center_dhw(mask_dhw: np.ndarray) -> Tuple[int, int, int]:
    """
    根据 mask 的体素坐标计算中心点。

    如果 mask 为空，则报错。
    """
    coords = np.argwhere(mask_dhw > 0)
    if coords.size == 0:
        raise ValueError("mask 为空，无法计算结节中心。")

    centroid = coords.mean(axis=0)
    nearest_index = int(np.argmin(np.sum((coords - centroid) ** 2, axis=1)))
    center = coords[nearest_index].astype(int)
    return int(center[0]), int(center[1]), int(center[2])


def crop_or_pad_center_dhw(
    arr_dhw: np.ndarray,
    center_dhw: Sequence[int],
    patch_size_dhw: Sequence[int],
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    以 center_dhw 为中心裁剪 [D, H, W] patch。

    如果靠近边界，则自动 padding。
    """
    arr = np.asarray(arr_dhw)
    patch_size = np.asarray(patch_size_dhw, dtype=int)
    center = np.asarray(center_dhw, dtype=int)

    if arr.ndim != 3:
        raise ValueError(f"crop_or_pad_center_dhw 只支持 3D 数组，当前 shape={arr.shape}")

    out = np.full(tuple(patch_size.tolist()), pad_value, dtype=arr.dtype)

    start = center - patch_size // 2
    end = start + patch_size

    src_start = np.maximum(start, 0)
    src_end = np.minimum(end, np.asarray(arr.shape, dtype=int))

    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)

    if np.any(src_end <= src_start):
        raise ValueError(
            f"裁剪区域无效：arr_shape={arr.shape}, center={center_dhw}, patch_size={patch_size_dhw}"
        )

    src_slices = tuple(slice(int(src_start[i]), int(src_end[i])) for i in range(3))
    dst_slices = tuple(slice(int(dst_start[i]), int(dst_end[i])) for i in range(3))

    out[dst_slices] = arr[src_slices]
    return out


def classify_nodule_size(diameter_mm: float) -> str:
    """Classify nodules by diameter for sampling policy."""
    if diameter_mm < 6.0:
        return "small"
    if diameter_mm < 10.0:
        return "medium"
    return "large"


def sampling_plan_for_diameter(diameter_mm: float) -> Dict[str, int]:
    """
    Return per-nodule patch counts.

    The medium plan is the recommended 6 patches:
    1 center + 3 random positive + 1 boundary + 1 hard negative.
    Small nodules get a little more sampling; large nodules get slightly less.
    """
    size_class = classify_nodule_size(float(diameter_mm))
    if size_class == "small":
        return {"random_positive": 4, "boundary": 2, "hard_negative": 1}
    if size_class == "large":
        return {"random_positive": 2, "boundary": 1, "hard_negative": 1}
    return {"random_positive": 3, "boundary": 1, "hard_negative": 1}


def offset_radius_for_diameter(diameter_mm: float) -> int:
    """Choose a random positive offset radius in voxels."""
    size_class = classify_nodule_size(float(diameter_mm))
    if size_class == "small":
        return 8
    if size_class == "large":
        return 20
    return 12


def clip_center_to_volume(center_dhw: Sequence[int], shape_dhw: Sequence[int]) -> Tuple[int, int, int]:
    """Clamp a center coordinate to the valid volume extent."""
    center = np.asarray(center_dhw, dtype=int)
    shape = np.asarray(shape_dhw, dtype=int)
    center = np.minimum(np.maximum(center, 0), np.maximum(shape - 1, 0))
    return int(center[0]), int(center[1]), int(center[2])


def sample_positive_center(
    positive_coords_dhw: np.ndarray,
    patch_size_dhw: Sequence[int],
    shape_dhw: Sequence[int],
    rng: random.Random,
) -> Tuple[int, int, int]:
    """
    Place a random positive voxel at a random interior location in the patch.

    This guarantees intersection with the target nodule while avoiding the
    center-biased sampling produced by small offsets around the centroid.
    """
    coords = np.asarray(positive_coords_dhw, dtype=int)
    if coords.ndim != 2 or coords.shape[0] == 0 or coords.shape[1] != 3:
        raise ValueError("positive_coords_dhw must contain [D, H, W] coordinates.")

    patch_size = np.asarray(patch_size_dhw, dtype=int)
    anchor = coords[rng.randrange(len(coords))]
    margin = np.maximum(1, patch_size // 8)
    target_position = np.asarray(
        [
            rng.randint(int(margin[axis]), int(patch_size[axis] - margin[axis] - 1))
            for axis in range(3)
        ],
        dtype=int,
    )
    center = anchor + patch_size // 2 - target_position
    return clip_center_to_volume(center, shape_dhw)


def boundary_voxels_dhw(mask_dhw: np.ndarray) -> np.ndarray:
    """Return boundary voxel coordinates in [D, H, W]."""
    mask = np.asarray(mask_dhw) > 0
    if not np.any(mask):
        return np.empty((0, 3), dtype=int)

    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    boundary = mask & (~eroded)
    coords = np.argwhere(boundary)
    return coords.astype(int, copy=False)


def random_boundary_center(
    boundary_coords_dhw: np.ndarray,
    shape_dhw: Sequence[int],
    rng: random.Random,
    jitter_vox: int = 4,
) -> Optional[Tuple[int, int, int]]:
    """Pick a patch center near a random nodule boundary voxel."""
    coords = np.asarray(boundary_coords_dhw, dtype=int)
    if coords.size == 0:
        return None

    coord = coords[rng.randrange(len(coords))].astype(int)
    jitter = np.asarray([rng.randint(-jitter_vox, jitter_vox) for _ in range(3)], dtype=int)
    return clip_center_to_volume(coord + jitter, shape_dhw)


def patch_has_tissue(image_patch: np.ndarray) -> bool:
    """Reject trivial all-air/all-padding negatives."""
    image = np.asarray(image_patch, dtype=np.float32)
    non_air_fraction = float(np.mean(image > 0.03))
    return non_air_fraction >= 0.05 and float(image.mean()) > 0.02


def random_unit_vector(rng: random.Random) -> np.ndarray:
    """Sample a random 3D unit vector."""
    vec = np.asarray([rng.uniform(-1.0, 1.0) for _ in range(3)], dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    return vec / norm


def sample_hard_negative_center(
    image_dhw: np.ndarray,
    full_mask_dhw: np.ndarray,
    nodule_mask_dhw: np.ndarray,
    center_dhw: Sequence[int],
    patch_size_dhw: Sequence[int],
    rng: random.Random,
    max_attempts: int = 80,
) -> Optional[Tuple[int, int, int]]:
    """
    Sample a nearby background patch that contains no nodule mask.

    Candidates are drawn from an annulus around the current nodule first, then
    fall back to whole-volume tissue-like locations.
    """
    shape = np.asarray(image_dhw.shape, dtype=int)
    patch_size = np.asarray(patch_size_dhw, dtype=np.float32)
    center = np.asarray(center_dhw, dtype=np.float32)

    coords = np.argwhere(nodule_mask_dhw > 0)
    if coords.size == 0:
        return None

    radius = float(np.max(coords.max(axis=0) - coords.min(axis=0) + 1)) / 2.0
    base_distance = float(np.max(patch_size) / 2.0 + radius + 2.0)

    for attempt in range(max_attempts):
        if attempt < int(max_attempts * 0.75):
            direction = random_unit_vector(rng)
            distance = base_distance + rng.uniform(0.0, max(4.0, float(np.max(patch_size)) * 0.35))
            candidate = np.rint(center + direction * distance).astype(int)
        else:
            candidate = np.asarray([rng.randrange(max(int(v), 1)) for v in shape], dtype=int)

        candidate = clip_center_to_volume(candidate, shape)
        mask_patch = crop_or_pad_center_dhw(
            full_mask_dhw,
            center_dhw=candidate,
            patch_size_dhw=patch_size_dhw,
            pad_value=0,
        )
        if np.any(mask_patch > 0):
            continue

        image_patch = crop_or_pad_center_dhw(
            image_dhw,
            center_dhw=candidate,
            patch_size_dhw=patch_size_dhw,
            pad_value=0.0,
        )
        if not patch_has_tissue(image_patch):
            continue

        return candidate

    return None


def fallback_distance_targets(mask_dhw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    当 distance_utils.py 不存在或函数名不匹配时，使用内置后备方法生成 sdf/boundary/core。

    SDF 约定：
        mask 内部为负值；
        mask 外部为正值；
        最后裁剪并归一化到 [-1, 1]。

    boundary：
        使用 dilation 与 erosion 的差得到一个窄边界区域。

    core：
        使用 mask 内部距离图提取高置信核心区域。
    """
    mask = (mask_dhw > 0).astype(np.uint8)

    if mask.sum() == 0:
        sdf = np.ones_like(mask, dtype=np.float32)
        boundary = np.zeros_like(mask, dtype=np.float32)
        core = np.zeros_like(mask, dtype=np.float32)
        return sdf, boundary, core

    outside_dist = ndimage.distance_transform_edt(mask == 0)
    inside_dist = ndimage.distance_transform_edt(mask > 0)

    sdf = outside_dist - inside_dist
    sdf = np.clip(sdf, -32.0, 32.0) / 32.0
    sdf = sdf.astype(np.float32)

    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    dilated = ndimage.binary_dilation(mask > 0, structure=structure, iterations=1)
    eroded = ndimage.binary_erosion(mask > 0, structure=structure, iterations=1)
    boundary = np.logical_xor(dilated, eroded).astype(np.float32)

    positive_inside = inside_dist[mask > 0]
    if positive_inside.size > 0:
        threshold = max(1.0, float(np.percentile(positive_inside, 50)))
        core = (inside_dist >= threshold).astype(np.float32)
    else:
        core = np.zeros_like(mask, dtype=np.float32)

    if core.sum() == 0:
        core = eroded.astype(np.float32)

    if core.sum() == 0:
        core = mask.astype(np.float32)

    return sdf, boundary, core


def parse_distance_result(result: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    解析 distance_utils.py 的输出。

    支持两类返回：
    1. dict:
        {"sdf": ..., "boundary": ..., "core": ...}
    2. tuple/list:
        (sdf, boundary, core)
    """
    if isinstance(result, dict):
        if not all(k in result for k in ["sdf", "boundary", "core"]):
            raise KeyError("distance_utils 返回 dict 时必须包含 sdf、boundary、core。")
        return result["sdf"], result["boundary"], result["core"]

    if isinstance(result, (tuple, list)) and len(result) >= 3:
        return result[0], result[1], result[2]

    raise TypeError("无法解析 distance_utils.py 的返回结果。")


def compute_auxiliary_targets(mask_dhw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    调用 distance_utils.py 生成 sdf、boundary、core。

    为了增强模块联动性，这里兼容多种常见函数命名：
    - generate_sdf_boundary_core(mask)
    - create_sdf_boundary_core(mask)
    - make_sdf_boundary_core(mask)
    - build_distance_targets(mask)
    - compute_sdf / compute_boundary / compute_core 组合

    如果 distance_utils.py 不存在或函数名不匹配，则使用 fallback_distance_targets。
    """
    mask = (mask_dhw > 0).astype(np.uint8)

    try:
        distance_utils = importlib.import_module("distance_utils")
    except Exception:
        return fallback_distance_targets(mask)

    combined_function_names = [
        "generate_sdf_boundary_core",
        "create_sdf_boundary_core",
        "make_sdf_boundary_core",
        "build_distance_targets",
        "generate_distance_targets",
    ]

    for name in combined_function_names:
        fn = getattr(distance_utils, name, None)
        if callable(fn):
            try:
                sdf, boundary, core = parse_distance_result(fn(mask))
                return (
                    np.asarray(sdf, dtype=np.float32),
                    np.asarray(boundary, dtype=np.float32),
                    np.asarray(core, dtype=np.float32),
                )
            except Exception as exc:
                warnings.warn(
                    f"调用 distance_utils.{name} 失败，将尝试其他函数或使用后备实现。错误：{exc}"
                )

    sdf_fn = None
    boundary_fn = None
    core_fn = None

    for name in ["compute_sdf", "mask_to_sdf", "generate_sdf"]:
        candidate = getattr(distance_utils, name, None)
        if callable(candidate):
            sdf_fn = candidate
            break

    for name in ["compute_boundary", "mask_to_boundary", "generate_boundary"]:
        candidate = getattr(distance_utils, name, None)
        if callable(candidate):
            boundary_fn = candidate
            break

    for name in ["compute_core", "mask_to_core", "generate_core"]:
        candidate = getattr(distance_utils, name, None)
        if callable(candidate):
            core_fn = candidate
            break

    if sdf_fn is not None and boundary_fn is not None and core_fn is not None:
        try:
            sdf = sdf_fn(mask)
            boundary = boundary_fn(mask)
            core = core_fn(mask)
            return (
                np.asarray(sdf, dtype=np.float32),
                np.asarray(boundary, dtype=np.float32),
                np.asarray(core, dtype=np.float32),
            )
        except Exception as exc:
            warnings.warn(f"调用 distance_utils 单独函数失败，将使用后备实现。错误：{exc}")

    return fallback_distance_targets(mask)


def ensure_channel_first_4d(arr_dhw: np.ndarray, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    将 [D, H, W] 转为 [1, D, H, W]。
    """
    arr = np.asarray(arr_dhw, dtype=dtype)
    if arr.ndim != 3:
        raise ValueError(f"期望输入为 [D, H, W]，实际 shape={arr.shape}")
    return arr[None, ...]


def append_failed_case(
    failed_file: Path,
    patient_id: str,
    scan_id: str,
    stage: str,
    error: BaseException,
) -> None:
    """
    将失败病例追加写入 failed_cases.txt。

    不让单个病例导致整个预处理崩掉。
    """
    failed_file.parent.mkdir(parents=True, exist_ok=True)

    entry = (
        "=" * 100
        + "\n"
        + f"patient_id: {patient_id}\n"
        + f"scan_id: {scan_id}\n"
        + f"stage: {stage}\n"
        + f"error_type: {type(error).__name__}\n"
        + f"error_message: {str(error)}\n"
        + "traceback:\n"
        + traceback.format_exc()
        + "\n"
    )
    fd = os.open(str(failed_file), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, entry.encode("utf-8", errors="replace"))
    finally:
        os.close(fd)


def save_npz(path: Path, compression: str, **arrays: Any) -> None:
    """Save a patch using fast ZIP storage or CPU-heavy compression."""
    if compression == "compressed":
        np.savez_compressed(path, **arrays)
    elif compression == "fast":
        np.savez(path, **arrays)
    else:
        raise ValueError(f"Unsupported NPZ compression mode: {compression}")


def save_sampled_patch(
    volume_img_iso_dhw: np.ndarray,
    full_mask_iso_dhw: np.ndarray,
    accept_mask_iso_dhw: np.ndarray,
    center_dhw: Sequence[int],
    patch_size_dhw: Tuple[int, int, int],
    output_root: Path,
    patient_id: str,
    scan_id: str,
    nodule_id: str,
    patch_type: str,
    patch_index: int,
    num_annotations: int,
    diameter_mm: float,
    nodule_size_class: str,
    require_positive: bool,
    npz_compression: str = "compressed",
    min_positive_fraction: float = 0.0,
    positive_reference_voxels: Optional[int] = None,
) -> Optional[PatchRecord]:
    """Crop, validate, save one sampled patch, and return its CSV record."""
    image_patch = crop_or_pad_center_dhw(
        volume_img_iso_dhw,
        center_dhw=center_dhw,
        patch_size_dhw=patch_size_dhw,
        pad_value=0.0,
    ).astype(np.float32)

    mask_patch = crop_or_pad_center_dhw(
        full_mask_iso_dhw,
        center_dhw=center_dhw,
        patch_size_dhw=patch_size_dhw,
        pad_value=0,
    )
    mask_patch = (mask_patch > 0).astype(np.float32)

    accept_patch = crop_or_pad_center_dhw(
        accept_mask_iso_dhw,
        center_dhw=center_dhw,
        patch_size_dhw=patch_size_dhw,
        pad_value=0,
    )
    accept_sum = float(np.asarray(accept_patch > 0).sum())
    mask_sum = float(mask_patch.sum())

    if require_positive:
        reference_voxels = (
            int(positive_reference_voxels)
            if positive_reference_voxels is not None
            else int(np.count_nonzero(accept_mask_iso_dhw))
        )
        required_voxels = max(
            1,
            int(np.ceil(max(0.0, float(min_positive_fraction)) * reference_voxels)),
        )
        if accept_sum < required_voxels:
            return None
    if not require_positive and mask_sum > 0:
        return None

    sdf_patch, boundary_patch, core_patch = compute_auxiliary_targets(mask_patch)

    sdf_patch = np.asarray(sdf_patch, dtype=np.float32)
    boundary_patch = np.asarray(boundary_patch, dtype=np.float32)
    core_patch = np.asarray(core_patch, dtype=np.float32)

    expected_shape = tuple(patch_size_dhw)
    for name, arr in [
        ("image", image_patch),
        ("mask", mask_patch),
        ("sdf", sdf_patch),
        ("boundary", boundary_patch),
        ("core", core_patch),
    ]:
        if tuple(arr.shape) != expected_shape:
            raise RuntimeError(
                f"{name} shape 涓嶇鍚堣姹傦細鏈熸湜 {expected_shape}锛屽疄闄?{arr.shape}"
            )

    image_4d = ensure_channel_first_4d(image_patch, dtype=np.float32)
    mask_4d = ensure_channel_first_4d(mask_patch, dtype=np.uint8)
    sdf_4d = ensure_channel_first_4d(sdf_patch, dtype=np.float32)
    boundary_4d = ensure_channel_first_4d(boundary_patch, dtype=np.uint8)
    core_4d = ensure_channel_first_4d(core_patch, dtype=np.uint8)

    patches_dir = output_root / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    safe_patch_type = str(patch_type).replace("/", "_").replace("\\", "_").replace(":", "_")
    file_name = f"{patient_id}_{scan_id}_{nodule_id}_{safe_patch_type}{int(patch_index):02d}.npz"
    file_name = file_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    save_path = patches_dir / file_name

    sample_label = 1 if require_positive else 0

    save_npz(
        save_path,
        npz_compression,
        image=image_4d,
        mask=mask_4d,
        sdf=sdf_4d,
        boundary=boundary_4d,
        core=core_4d,
        patient_id=patient_id,
        scan_id=scan_id,
        nodule_id=nodule_id,
        source_nodule_id=nodule_id,
        patch_type=patch_type,
        sample_label=np.int64(sample_label),
        nodule_size_class=nodule_size_class,
        patch_center_dhw=np.asarray(center_dhw, dtype=np.int32),
        positive_voxels=np.int64(round(accept_sum)),
        foreground_voxels=np.int64(round(mask_sum)),
        num_annotations=int(num_annotations),
        diameter_mm=float(diameter_mm),
        spacing_mm=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
    )

    rel_path = save_path.relative_to(output_root).as_posix()
    return PatchRecord(
        npz_path=rel_path,
        patient_id=patient_id,
        scan_id=scan_id,
        nodule_id=f"{nodule_id}_{safe_patch_type}{int(patch_index):02d}",
        num_annotations=int(num_annotations),
        diameter_mm=float(diameter_mm),
        patch_type=patch_type,
        sample_label=sample_label,
        source_nodule_id=nodule_id,
        nodule_size_class=nodule_size_class,
    )


def split_by_patient(
    records: Sequence[PatchRecord],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> List[PatchRecord]:
    """
    按 patient_id 划分数据集，避免同一病人出现在多个集合。
    """
    train_ratio, val_ratio, test_ratio = validate_ratios(train_ratio, val_ratio, test_ratio)

    patients = sorted({r.patient_id for r in records})
    rng = random.Random(seed)
    rng.shuffle(patients)

    n = len(patients)
    if n == 0:
        return []

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    # 修正边界，确保总数不超过 n。
    n_train = min(max(n_train, 0), n)
    n_val = min(max(n_val, 0), n - n_train)

    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train:n_train + n_val])
    test_patients = set(patients[n_train + n_val:])

    # 如果由于 round 导致 test 为空，但整体病人数足够，则从 train 中挪一个到 test。
    if n >= 3 and len(test_patients) == 0 and len(train_patients) > 1:
        moved = sorted(train_patients)[-1]
        train_patients.remove(moved)
        test_patients.add(moved)

    output: List[PatchRecord] = []
    for r in records:
        if r.patient_id in train_patients:
            r.split = "train"
        elif r.patient_id in val_patients:
            r.split = "val"
        else:
            r.split = "test"
        output.append(r)

    return output


def write_split_csvs(records: Sequence[PatchRecord], output_root: Path) -> None:
    """
    写 train.csv、val.csv、test.csv。
    """
    fieldnames = [
        "case_id",
        "npz_path",
        "patient_id",
        "scan_id",
        "nodule_id",
        "source_nodule_id",
        "patch_type",
        "sample_label",
        "nodule_size_class",
        "num_annotations",
        "diameter_mm",
        "split",
    ]

    def write_rows(csv_path: Path, selected_records: Sequence[PatchRecord]) -> None:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in selected_records:
                case_id = Path(r.npz_path).stem
                writer.writerow(
                    {
                        "case_id": case_id,
                        "npz_path": r.npz_path,
                        "patient_id": r.patient_id,
                        "scan_id": r.scan_id,
                        "nodule_id": r.nodule_id,
                        "source_nodule_id": r.source_nodule_id or r.nodule_id,
                        "patch_type": r.patch_type,
                        "sample_label": int(r.sample_label),
                        "nodule_size_class": r.nodule_size_class,
                        "num_annotations": r.num_annotations,
                        "diameter_mm": f"{r.diameter_mm:.4f}",
                        "split": r.split,
                    }
                )

    for split in ["train", "val", "test"]:
        csv_path = output_root / f"{split}.csv"
        split_records = [r for r in records if r.split == split]
        if split in {"val", "test"}:
            split_records = [r for r in split_records if r.patch_type == "center"]
        write_rows(csv_path, split_records)

    write_rows(output_root / "all_patches.csv", records)
    val_records = [r for r in records if r.split == "val"]
    test_records = [r for r in records if r.split == "test"]
    write_rows(output_root / "val_all.csv", val_records)
    write_rows(output_root / "val_center.csv", [r for r in val_records if r.patch_type == "center"])
    write_rows(
        output_root / "val_offset.csv",
        [r for r in val_records if r.patch_type in {"random_positive", "boundary"}],
    )
    write_rows(output_root / "test_all.csv", test_records)
    write_rows(output_root / "test_center.csv", [r for r in test_records if r.patch_type == "center"])


def write_volume_split_csvs(records: Sequence[VolumeRecord], output_root: Path) -> None:
    """Write train_volume.csv / val_volume.csv / test_volume.csv for whole CT data."""
    fieldnames = [
        "image",
        "label",
        "patient_id",
        "scan_id",
        "case_id",
        "split",
    ]

    for split in ["train", "val", "test"]:
        csv_path = output_root / f"{split}_volume.csv"
        split_records = [r for r in records if r.split == split]

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in split_records:
                writer.writerow(
                    {
                        "image": r.image,
                        "label": r.label,
                        "patient_id": r.patient_id,
                        "scan_id": r.scan_id,
                        "case_id": f"{r.patient_id}_{r.scan_id}",
                        "split": r.split,
                    }
                )


def write_summary_csv(records: Sequence[PatchRecord], output_root: Path) -> None:
    """
    写 preprocess_summary.csv，便于快速查看各集合数量。
    """
    summary_path = output_root / "preprocess_summary.csv"

    split_names = ["train", "val", "test"]
    rows: List[Dict[str, Any]] = []

    for split in split_names:
        split_records = [r for r in records if r.split == split]
        patients = sorted({r.patient_id for r in split_records})
        rows.append(
            {
                "split": split,
                "num_patients": len(patients),
                "num_patches": len(split_records),
            }
        )

    rows.append(
        {
            "split": "all",
            "num_patients": len({r.patient_id for r in records}),
            "num_patches": len(records),
        }
    )

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "num_patients", "num_patches"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    type_summary_path = output_root / "patch_type_summary.csv"
    type_rows: List[Dict[str, Any]] = []
    for split in [*split_names, "all"]:
        split_records = records if split == "all" else [r for r in records if r.split == split]
        for patch_type in ["center", "random_positive", "boundary", "hard_negative"]:
            selected = [r for r in split_records if r.patch_type == patch_type]
            type_rows.append(
                {
                    "split": split,
                    "patch_type": patch_type,
                    "num_patients": len({r.patient_id for r in selected}),
                    "num_patches": len(selected),
                }
            )

    with type_summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "patch_type", "num_patients", "num_patches"],
        )
        writer.writeheader()
        writer.writerows(type_rows)


def process_one_scan(
    scan: Any,
    consensus_fn: Any,
    output_root: Path,
    patch_size_dhw: Tuple[int, int, int],
    min_annotations: int,
    min_diameter_mm: float,
    failed_file: Path,
    sampling_seed: int = 42,
    volume_records: Optional[List[VolumeRecord]] = None,
    npz_compression: str = "compressed",
) -> List[PatchRecord]:
    """
    处理单个 LIDC scan，返回该 scan 中成功保存的 patch 记录。
    """
    records: List[PatchRecord] = []

    patient_id = get_patient_id(scan)
    scan_id = get_scan_identifier(scan)

    try:
        volume_hu_hwd, spacing_hwd = load_scan_volume_hu_hwd(scan)
        volume_img_hwd = clip_and_normalize_hu(volume_hu_hwd)

        # 对 image 先整图重采样到 1mm isotropic。
        volume_img_iso_hwd = resample_hwd(
            volume_img_hwd,
            spacing_hwd=spacing_hwd,
            target_spacing_hwd=TARGET_SPACING_HWD,
            order=1,
        ).astype(np.float32)

        volume_img_iso_dhw = hwd_to_dhw(volume_img_iso_hwd)
    except Exception as exc:
        append_failed_case(failed_file, patient_id, scan_id, "load_or_resample_scan", exc)
        return records

    volume_mask_iso_dhw = np.zeros_like(volume_img_iso_dhw, dtype=np.uint8)

    try:
        clusters = scan.cluster_annotations(verbose=False)
    except TypeError:
        try:
            clusters = scan.cluster_annotations()
        except Exception as exc:
            append_failed_case(failed_file, patient_id, scan_id, "cluster_annotations", exc)
            return records
    except Exception as exc:
        append_failed_case(failed_file, patient_id, scan_id, "cluster_annotations", exc)
        return records

    nodule_items: List[Dict[str, Any]] = []

    for nodule_index, cluster in enumerate(clusters, start=1):
        nodule_id = f"nodule{nodule_index:04d}"

        try:
            if len(cluster) < min_annotations:
                continue

            mask_hwd = make_consensus_mask_hwd(
                cluster=cluster,
                volume_shape_hwd=volume_img_hwd.shape,
                min_annotations=min_annotations,
                consensus_fn=consensus_fn,
            )

            diameter_mm = estimate_diameter_mm(
                cluster=cluster,
                mask_hwd=mask_hwd,
                spacing_hwd=spacing_hwd,
            )

            if diameter_mm < float(min_diameter_mm):
                continue

            # mask 使用 nearest neighbor 重采样，保持标签离散。
            mask_iso_hwd = resample_hwd(
                mask_hwd.astype(np.uint8),
                spacing_hwd=spacing_hwd,
                target_spacing_hwd=TARGET_SPACING_HWD,
                order=0,
            )
            mask_iso_hwd = (mask_iso_hwd > 0).astype(np.uint8)
            mask_iso_dhw = hwd_to_dhw(mask_iso_hwd)

            if mask_iso_dhw.sum() == 0:
                raise RuntimeError("重采样后 mask 为空。")

            volume_mask_iso_dhw = np.logical_or(
                volume_mask_iso_dhw > 0,
                mask_iso_dhw > 0,
            ).astype(np.uint8)

            center_dhw = get_mask_center_dhw(mask_iso_dhw)
            nodule_items.append(
                {
                    "nodule_id": nodule_id,
                    "mask": mask_iso_dhw,
                    "center": center_dhw,
                    "num_annotations": int(len(cluster)),
                    "diameter_mm": float(diameter_mm),
                    "size_class": classify_nodule_size(float(diameter_mm)),
                }
            )

            image_patch = crop_or_pad_center_dhw(
                volume_img_iso_dhw,
                center_dhw=center_dhw,
                patch_size_dhw=patch_size_dhw,
                pad_value=0.0,
            ).astype(np.float32)

            mask_patch = crop_or_pad_center_dhw(
                mask_iso_dhw,
                center_dhw=center_dhw,
                patch_size_dhw=patch_size_dhw,
                pad_value=0,
            )
            mask_patch = (mask_patch > 0).astype(np.float32)

            if mask_patch.sum() == 0:
                raise RuntimeError("裁剪后 mask patch 为空。")

            sdf_patch, boundary_patch, core_patch = compute_auxiliary_targets(mask_patch)

            # 统一为 float32，并保证 shape 完全一致。
            sdf_patch = np.asarray(sdf_patch, dtype=np.float32)
            boundary_patch = np.asarray(boundary_patch, dtype=np.float32)
            core_patch = np.asarray(core_patch, dtype=np.float32)

            expected_shape = tuple(patch_size_dhw)
            for name, arr in [
                ("image", image_patch),
                ("mask", mask_patch),
                ("sdf", sdf_patch),
                ("boundary", boundary_patch),
                ("core", core_patch),
            ]:
                if tuple(arr.shape) != expected_shape:
                    raise RuntimeError(
                        f"{name} shape 不符合要求：期望 {expected_shape}，实际 {arr.shape}"
                    )

            image_4d = ensure_channel_first_4d(image_patch, dtype=np.float32)
            mask_4d = ensure_channel_first_4d(mask_patch, dtype=np.uint8)
            sdf_4d = ensure_channel_first_4d(sdf_patch, dtype=np.float32)
            boundary_4d = ensure_channel_first_4d(boundary_patch, dtype=np.uint8)
            core_4d = ensure_channel_first_4d(core_patch, dtype=np.uint8)

            patches_dir = output_root / "patches"
            patches_dir.mkdir(parents=True, exist_ok=True)

            file_name = f"{patient_id}_{scan_id}_{nodule_id}.npz"
            file_name = file_name.replace("/", "_").replace("\\", "_").replace(":", "_")
            save_path = patches_dir / file_name

            save_npz(
                save_path,
                npz_compression,
                image=image_4d,
                mask=mask_4d,
                sdf=sdf_4d,
                boundary=boundary_4d,
                core=core_4d,
                patient_id=patient_id,
                scan_id=scan_id,
                nodule_id=nodule_id,
                num_annotations=int(len(cluster)),
                diameter_mm=float(diameter_mm),
                patch_type="center",
                sample_label=1,
                source_nodule_id=nodule_id,
                nodule_size_class=classify_nodule_size(float(diameter_mm)),
                spacing_mm=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            )

            rel_path = save_path.relative_to(output_root).as_posix()

            records.append(
                PatchRecord(
                    npz_path=rel_path,
                    patient_id=patient_id,
                    scan_id=scan_id,
                    nodule_id=nodule_id,
                    num_annotations=int(len(cluster)),
                    diameter_mm=float(diameter_mm),
                    patch_type="center",
                    sample_label=1,
                    source_nodule_id=nodule_id,
                    nodule_size_class=classify_nodule_size(float(diameter_mm)),
                )
            )

        except Exception as exc:
            append_failed_case(
                failed_file,
                patient_id,
                scan_id,
                f"process_{nodule_id}",
                exc,
            )
            continue

    seed_text = f"{int(sampling_seed)}|{patient_id}|{scan_id}|{tuple(patch_size_dhw)}"
    rng_seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(seed_text)) & 0xFFFFFFFF
    rng = random.Random(rng_seed)

    for item in nodule_items:
        nodule_id = str(item["nodule_id"])
        nodule_mask = np.asarray(item["mask"], dtype=np.uint8)
        center_dhw = item["center"]
        num_annotations = int(item["num_annotations"])
        diameter_mm = float(item["diameter_mm"])
        size_class = str(item["size_class"])
        plan = sampling_plan_for_diameter(diameter_mm)
        positive_coords = np.argwhere(nodule_mask > 0)
        boundary_coords = boundary_voxels_dhw(nodule_mask)
        positive_reference_voxels = int(len(positive_coords))
        used_centers = {tuple(int(v) for v in center_dhw)}

        try:
            random_saved = 0
            random_attempts = max(16, int(plan["random_positive"]) * 12)
            for _ in range(random_attempts):
                if random_saved >= int(plan["random_positive"]):
                    break
                shifted_center = sample_positive_center(
                    positive_coords_dhw=positive_coords,
                    patch_size_dhw=patch_size_dhw,
                    shape_dhw=volume_img_iso_dhw.shape,
                    rng=rng,
                )
                if shifted_center in used_centers:
                    continue
                record = save_sampled_patch(
                    volume_img_iso_dhw=volume_img_iso_dhw,
                    full_mask_iso_dhw=volume_mask_iso_dhw,
                    accept_mask_iso_dhw=nodule_mask,
                    center_dhw=shifted_center,
                    patch_size_dhw=patch_size_dhw,
                    output_root=output_root,
                    patient_id=patient_id,
                    scan_id=scan_id,
                    nodule_id=nodule_id,
                    patch_type="random_positive",
                    patch_index=random_saved + 1,
                    num_annotations=num_annotations,
                    diameter_mm=diameter_mm,
                    nodule_size_class=size_class,
                    require_positive=True,
                    npz_compression=npz_compression,
                    min_positive_fraction=0.10,
                    positive_reference_voxels=positive_reference_voxels,
                )
                if record is not None:
                    records.append(record)
                    random_saved += 1
                    used_centers.add(shifted_center)

            boundary_saved = 0
            boundary_attempts = max(16, int(plan["boundary"]) * 12)
            for _ in range(boundary_attempts):
                if boundary_saved >= int(plan["boundary"]):
                    break
                boundary_center = random_boundary_center(
                    boundary_coords_dhw=boundary_coords,
                    shape_dhw=volume_img_iso_dhw.shape,
                    rng=rng,
                    jitter_vox=max(2, offset_radius_for_diameter(diameter_mm) // 3),
                )
                if boundary_center is None:
                    break
                if boundary_center in used_centers:
                    continue
                record = save_sampled_patch(
                    volume_img_iso_dhw=volume_img_iso_dhw,
                    full_mask_iso_dhw=volume_mask_iso_dhw,
                    accept_mask_iso_dhw=nodule_mask,
                    center_dhw=boundary_center,
                    patch_size_dhw=patch_size_dhw,
                    output_root=output_root,
                    patient_id=patient_id,
                    scan_id=scan_id,
                    nodule_id=nodule_id,
                    patch_type="boundary",
                    patch_index=boundary_saved + 1,
                    num_annotations=num_annotations,
                    diameter_mm=diameter_mm,
                    nodule_size_class=size_class,
                    require_positive=True,
                    npz_compression=npz_compression,
                    min_positive_fraction=0.05,
                    positive_reference_voxels=positive_reference_voxels,
                )
                if record is not None:
                    records.append(record)
                    boundary_saved += 1
                    used_centers.add(boundary_center)

            negative_saved = 0
            negative_attempts = max(4, int(plan["hard_negative"]) * 4)
            for _ in range(negative_attempts):
                if negative_saved >= int(plan["hard_negative"]):
                    break
                negative_center = sample_hard_negative_center(
                    image_dhw=volume_img_iso_dhw,
                    full_mask_dhw=volume_mask_iso_dhw,
                    nodule_mask_dhw=nodule_mask,
                    center_dhw=center_dhw,
                    patch_size_dhw=patch_size_dhw,
                    rng=rng,
                )
                if negative_center is None:
                    continue
                if negative_center in used_centers:
                    continue
                record = save_sampled_patch(
                    volume_img_iso_dhw=volume_img_iso_dhw,
                    full_mask_iso_dhw=volume_mask_iso_dhw,
                    accept_mask_iso_dhw=volume_mask_iso_dhw,
                    center_dhw=negative_center,
                    patch_size_dhw=patch_size_dhw,
                    output_root=output_root,
                    patient_id=patient_id,
                    scan_id=scan_id,
                    nodule_id=nodule_id,
                    patch_type="hard_negative",
                    patch_index=negative_saved + 1,
                    num_annotations=num_annotations,
                    diameter_mm=diameter_mm,
                    nodule_size_class=size_class,
                    require_positive=False,
                    npz_compression=npz_compression,
                )
                if record is not None:
                    records.append(record)
                    negative_saved += 1
                    used_centers.add(negative_center)

            saved_counts = {
                "random_positive": random_saved,
                "boundary": boundary_saved,
                "hard_negative": negative_saved,
            }
            missing = {
                name: int(plan[name]) - saved_counts[name]
                for name in saved_counts
                if saved_counts[name] < int(plan[name])
            }
            if missing:
                warnings.warn(
                    f"{patient_id}/{scan_id}/{nodule_id} patch sampling incomplete: "
                    f"missing={missing}, planned={plan}, saved={saved_counts}"
                )

        except Exception as exc:
            append_failed_case(
                failed_file,
                patient_id,
                scan_id,
                f"sample_patches_{nodule_id}",
                exc,
            )
            continue

    if volume_records is not None and records:
        volumes_image_dir = output_root / "volumes" / "images"
        volumes_label_dir = output_root / "volumes" / "labels"
        volumes_image_dir.mkdir(parents=True, exist_ok=True)
        volumes_label_dir.mkdir(parents=True, exist_ok=True)

        safe_scan_name = f"{patient_id}_{scan_id}".replace("/", "_").replace("\\", "_").replace(":", "_")
        image_path = volumes_image_dir / f"{safe_scan_name}_image.npy"
        label_path = volumes_label_dir / f"{safe_scan_name}_label.npy"

        np.save(image_path, volume_img_iso_dhw.astype(np.float32))
        np.save(label_path, volume_mask_iso_dhw.astype(np.uint8))

        volume_records.append(
            VolumeRecord(
                image=image_path.relative_to(output_root).as_posix(),
                label=label_path.relative_to(output_root).as_posix(),
                patient_id=patient_id,
                scan_id=scan_id,
            )
        )

    return records


def get_all_scans(pl: Any) -> List[Any]:
    """
    从 pylidc 数据库中读取全部 scan。
    """
    try:
        scans = list(pl.query(pl.Scan).all())
    except Exception as exc:
        raise RuntimeError(
            "\n无法从 pylidc 查询 LIDC-IDRI scans。\n"
            "可能原因：\n"
            "1. pylidc 未正确安装；\n"
            "2. pylidc 的 XML annotation 数据库未正确初始化；\n"
            "3. DICOM 根目录配置错误；\n"
            "4. 当前 Python 环境无法访问 LIDC-IDRI 数据。\n\n"
            "建议检查：\n"
            "    pip install pylidc\n"
            "    python -c \"import pylidc as pl; print(pl.query(pl.Scan).count())\"\n"
        ) from exc

    return scans


def init_preprocess_worker() -> None:
    """Initialize pylidc once in each worker process."""
    global _WORKER_PL, _WORKER_CONSENSUS_FN
    _WORKER_PL, _WORKER_CONSENSUS_FN = import_pylidc()


def process_scan_by_database_id(
    scan_database_id: int,
    output_root: Path,
    patch_size_dhw: Tuple[int, int, int],
    min_annotations: int,
    min_diameter_mm: float,
    failed_file: Path,
    sampling_seed: int,
    npz_compression: str,
) -> Tuple[List[PatchRecord], List[VolumeRecord]]:
    """Load and process one scan without sharing SQLAlchemy objects."""
    if _WORKER_PL is None or _WORKER_CONSENSUS_FN is None:
        init_preprocess_worker()

    scan = _WORKER_PL.query(_WORKER_PL.Scan).filter(
        _WORKER_PL.Scan.id == int(scan_database_id)
    ).first()
    if scan is None:
        raise RuntimeError(f"pylidc scan id not found: {scan_database_id}")

    volume_records: List[VolumeRecord] = []
    records = process_one_scan(
        scan=scan,
        consensus_fn=_WORKER_CONSENSUS_FN,
        output_root=output_root,
        patch_size_dhw=patch_size_dhw,
        min_annotations=min_annotations,
        min_diameter_mm=min_diameter_mm,
        failed_file=failed_file,
        sampling_seed=sampling_seed,
        volume_records=volume_records,
        npz_compression=npz_compression,
    )
    return records, volume_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess LIDC-IDRI into 64×64×64 nodule patches for Hybrid-Swin-SDF-CoreNet."
    )

    parser.add_argument(
        "--dicom-root",
        type=str,
        default="/home/lembert/Desktop/BME-4/BME-WYZ/data",
        help="LIDC-IDRI DICOM 根目录，例如 /data/LIDC-IDRI。",
    )
    parser.add_argument(
        "--xml-root",
        type=str,
        default=None,
        help=(
            "LIDC-IDRI XML 标注根目录。注意：pylidc 通常通过自身数据库读取 XML，"
            "此参数主要用于路径校验和记录。"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/home/lembert/Desktop/BME-4/BME-WYZ/data_processed/LIDC-IDRI",
        help="预处理输出目录。",
    )
    parser.add_argument(
        "--patch-size",
        type=parse_patch_size,
        default=(64, 64, 64),
        help="patch 大小，支持 64 或 64,64,64。默认 64。",
    )
    parser.add_argument(
        "--min-annotations",
        type=int,
        default=3,
        help="共识结节最少医生 annotation 数。默认 3。",
    )
    parser.add_argument(
        "--min-diameter-mm",
        type=float,
        default=3.0,
        help="最小结节直径，单位 mm。默认 3.0。",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="训练集病人比例。默认 0.7。",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="验证集病人比例。默认 0.1。",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="测试集病人比例。默认 0.2。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="按 patient_id 划分数据集的随机种子。默认 42。",
    )
    parser.add_argument(
        "--max-scans",
        type=int,
        default=None,
        help="调试用：最多处理多少个 scan。默认处理全部。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="并行处理 scan 的进程数。默认最多 2 个；设为 1 可串行处理。",
    )
    parser.add_argument(
        "--npz-compression",
        choices=["fast", "compressed"],
        default="fast",
        help="fast 写入更快但占用更多磁盘；compressed 更省空间。默认 fast。",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dicom_root = Path(args.dicom_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    failed_file = output_root / "failed_cases.txt"
    if failed_file.exists():
        failed_file.unlink()

    xml_root: Optional[Path] = None
    if args.xml_root is not None:
        xml_root = Path(args.xml_root).expanduser().resolve()
        if not xml_root.exists():
            warnings.warn(
                f"--xml-root 不存在：{xml_root}。"
                "pylidc 可能仍可通过其数据库运行，但请确认 XML 标注已正确导入。"
            )

    if args.min_annotations <= 0:
        raise ValueError("--min-annotations 必须大于 0。")

    if args.min_diameter_mm < 0:
        raise ValueError("--min-diameter-mm 不能为负数。")

    if args.workers <= 0:
        raise ValueError("--workers 必须大于 0。")

    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    config_path = prepare_pylidc_config(dicom_root)
    if config_path is not None:
        print(f"[INFO] 已配置 pylidc DICOM 路径：{dicom_root}")
        print(f"[INFO] pylidc 配置文件：{config_path}")

    if xml_root is not None:
        print(f"[INFO] XML root：{xml_root}")
        print("[INFO] 注意：pylidc 读取 XML 标注依赖其自身数据库，--xml-root 主要用于记录和校验。")

    pl, consensus_fn = import_pylidc()

    scans = get_all_scans(pl)
    if args.max_scans is not None:
        scans = scans[: int(args.max_scans)]

    print(f"[INFO] 待处理 scan 数量：{len(scans)}")
    print(f"[INFO] 输出目录：{output_root}")
    print(f"[INFO] patch_size[D,H,W]：{args.patch_size}")
    print(f"[INFO] min_annotations：{args.min_annotations}")
    print(f"[INFO] min_diameter_mm：{args.min_diameter_mm}")
    print(f"[INFO] workers：{args.workers}")
    print(f"[INFO] npz_compression：{args.npz_compression}")

    all_records: List[PatchRecord] = []
    all_volume_records: List[VolumeRecord] = []

    if args.workers == 1 or len(scans) <= 1:
        for idx, scan in enumerate(scans, start=1):
            patient_id = get_patient_id(scan)
            scan_id = get_scan_identifier(scan)
            print(f"[{idx:04d}/{len(scans):04d}] 处理 {patient_id} / {scan_id} ...")

            try:
                records = process_one_scan(
                    scan=scan,
                    consensus_fn=consensus_fn,
                    output_root=output_root,
                    patch_size_dhw=args.patch_size,
                    min_annotations=args.min_annotations,
                    min_diameter_mm=args.min_diameter_mm,
                    failed_file=failed_file,
                    sampling_seed=args.seed,
                    volume_records=all_volume_records,
                    npz_compression=args.npz_compression,
                )
                all_records.extend(records)
                print(f"    成功保存 patch 数：{len(records)}")
            except Exception as exc:
                append_failed_case(failed_file, patient_id, scan_id, "process_one_scan_unhandled", exc)
                print(f"    [WARNING] scan 处理失败，已记录到 failed_cases.txt：{exc}")
    else:
        scan_jobs = [
            (int(scan.id), get_patient_id(scan), get_scan_identifier(scan))
            for scan in scans
        ]
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(int(args.workers), len(scan_jobs)),
            mp_context=context,
            initializer=init_preprocess_worker,
        ) as executor:
            future_to_scan = {
                executor.submit(
                    process_scan_by_database_id,
                    scan_database_id,
                    output_root,
                    args.patch_size,
                    args.min_annotations,
                    args.min_diameter_mm,
                    failed_file,
                    args.seed,
                    args.npz_compression,
                ): (patient_id, scan_id)
                for scan_database_id, patient_id, scan_id in scan_jobs
            }
            completed = 0
            for future in as_completed(future_to_scan):
                completed += 1
                patient_id, scan_id = future_to_scan[future]
                try:
                    records, volume_records = future.result()
                    all_records.extend(records)
                    all_volume_records.extend(volume_records)
                    print(
                        f"[{completed:04d}/{len(scan_jobs):04d}] "
                        f"{patient_id} / {scan_id}: {len(records)} patches"
                    )
                except Exception as exc:
                    append_failed_case(
                        failed_file,
                        patient_id,
                        scan_id,
                        "process_one_scan_worker",
                        exc,
                    )
                    print(
                        f"[{completed:04d}/{len(scan_jobs):04d}] "
                        f"[WARNING] {patient_id} / {scan_id}: {exc}"
                    )

    if len(all_records) == 0:
        print(
            "\n[WARNING] 没有生成任何 patch。\n"
            "请检查：\n"
            "1. pylidc 是否能查询到 scan；\n"
            "2. DICOM root 是否正确；\n"
            "3. min_annotations 是否过高；\n"
            "4. min_diameter_mm 是否过高；\n"
            "5. failed_cases.txt 中的具体错误。\n"
        )
        return

    all_records.sort(
        key=lambda record: (
            record.patient_id,
            record.scan_id,
            record.source_nodule_id or record.nodule_id,
            record.patch_type,
            record.nodule_id,
        )
    )
    all_volume_records.sort(key=lambda record: (record.patient_id, record.scan_id))

    all_records = split_by_patient(
        records=all_records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    patient_to_split = {r.patient_id: r.split for r in all_records}
    for r in all_volume_records:
        r.split = patient_to_split.get(r.patient_id, "test")

    write_split_csvs(all_records, output_root)
    write_volume_split_csvs(all_volume_records, output_root)
    write_summary_csv(all_records, output_root)

    num_train = sum(1 for r in all_records if r.split == "train")
    num_val = sum(1 for r in all_records if r.split == "val")
    num_test = sum(1 for r in all_records if r.split == "test")
    num_train_volume = sum(1 for r in all_volume_records if r.split == "train")
    num_val_volume = sum(1 for r in all_volume_records if r.split == "val")
    num_test_volume = sum(1 for r in all_volume_records if r.split == "test")
    num_patients = len({r.patient_id for r in all_records})

    print("\n[INFO] 预处理完成。")
    print(f"[INFO] 总病人数：{num_patients}")
    print(f"[INFO] 总 patch 数：{len(all_records)}")
    print(f"[INFO] train patch 数：{num_train}")
    print(f"[INFO] val patch 数：{num_val}")
    print(f"[INFO] test patch 数：{num_test}")
    print(f"[INFO] train.csv：{output_root / 'train.csv'}")
    print(f"[INFO] val.csv：{output_root / 'val.csv'}")
    print(f"[INFO] test.csv：{output_root / 'test.csv'}")
    print(f"[INFO] failed_cases.txt：{failed_file}")


if __name__ == "__main__":
    main()
