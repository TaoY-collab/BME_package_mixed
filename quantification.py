# -*- coding: utf-8 -*-
"""
src/quantification.py

Hybrid-Swin-SDF-CoreNet 项目的 3D 肺结节量化分析模块。

目标：
根据预测得到的肺结节 pred_mask 计算三维量化指标，用于竞赛报告、实验分析和结果展示。

支持指标：
1. volume_mm3:
   结节体积，单位 mm^3。

2. equivalent_diameter_mm:
   等效球直径，即与当前结节体积相同的球体直径。

3. max_diameter_mm:
   最大三维直径，近似为结节表面点之间的最大欧氏距离。

4. surface_area_mm2:
   表面积，单位 mm^2。
   优先使用 skimage.measure.marching_cubes + mesh_surface_area。

5. compactness:
   紧致度。
   使用公式：
       compactness = 36 * pi * volume^2 / surface_area^3
   对理想球体，该值接近 1；形状越不规则，通常越小。

6. max_axial_area_mm2:
   轴向切片上的最大截面积，默认 D 轴为 axial 方向。

7. num_components:
   连通域数量。

核心函数：
- keep_largest_component(mask)
- quantify_nodule(mask, spacing)
- batch_quantify_prediction_folder(prediction_dir, output_csv)

输入 mask 支持：
- [D, H, W]
- [1, D, H, W]
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from scipy.ndimage import binary_erosion, generate_binary_structure, label
from scipy.spatial.distance import pdist

try:
    from train import DEFAULT_CONFIG_PATH, DEFAULT_OUTPUT_DIR, resolve_existing_path, resolve_runtime_path
except Exception:
    DEFAULT_CONFIG_PATH = "configs/task_adaptive.yaml"
    DEFAULT_OUTPUT_DIR = "outputs"

    def resolve_existing_path(value):
        return Path(value).expanduser().resolve()

    def resolve_runtime_path(value):
        return Path(value).expanduser().resolve()

try:
    from skimage.measure import marching_cubes, mesh_surface_area
except Exception:
    marching_cubes = None
    mesh_surface_area = None


QUANTIFICATION_KEYS = [
    "volume_mm3",
    "equivalent_diameter_mm",
    "max_diameter_mm",
    "surface_area_mm2",
    "compactness",
    "max_axial_area_mm2",
    "num_components",
]


def ensure_3d(mask: np.ndarray) -> np.ndarray:
    """
    将输入 mask 统一为 [D, H, W]。

    参数
    ----
    mask:
        输入 mask，支持：
        - [D, H, W]
        - [1, D, H, W]

    返回
    ----
    mask_3d:
        [D, H, W] 形状的 numpy 数组。
    """
    mask = np.asarray(mask)

    if mask.ndim == 3:
        return mask

    if mask.ndim == 4:
        if mask.shape[0] != 1:
            raise ValueError(
                f"4D mask 必须是 [1, D, H, W]，但当前 shape = {mask.shape}"
            )
        return mask[0]

    raise ValueError(
        f"mask 只支持 [D, H, W] 或 [1, D, H, W]，但当前 shape = {mask.shape}"
    )


def to_binary_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    将输入 mask 转为 bool 类型的 3D binary mask。

    参数
    ----
    mask:
        输入 mask，可以是 binary mask，也可以是 probability map。

    threshold:
        二值化阈值。

    返回
    ----
    mask_bool:
        bool 类型的 [D, H, W] mask。
    """
    mask_3d = ensure_3d(mask)
    mask_bool = mask_3d >= float(threshold)
    return mask_bool.astype(bool)


def count_components(mask: np.ndarray, connectivity: int = 3) -> int:
    """
    计算 3D mask 中的连通域数量。

    参数
    ----
    mask:
        输入 binary mask，shape 为 [D, H, W] 或 [1, D, H, W]。

    connectivity:
        连通性。
        - 1：6 邻域；
        - 2：18 邻域；
        - 3：26 邻域。
        默认使用 26 邻域，更适合 3D 医学体数据。

    返回
    ----
    num_components:
        连通域数量。
    """
    mask_bool = to_binary_mask(mask)

    if not np.any(mask_bool):
        return 0

    structure = generate_binary_structure(rank=3, connectivity=connectivity)
    _, num_components = label(mask_bool, structure=structure)

    return int(num_components)


def keep_largest_component(mask: np.ndarray, connectivity: int = 3) -> np.ndarray:
    """
    只保留 3D mask 中最大的连通域。

    参数
    ----
    mask:
        输入 binary mask，shape 为 [D, H, W] 或 [1, D, H, W]。

    connectivity:
        连通性，默认 26 邻域。

    返回
    ----
    largest:
        bool 类型的 [D, H, W] mask。
        若输入为空 mask，则返回全 False mask。

    说明
    ----
    在肺结节分割中，模型预测可能出现少量孤立假阳性区域。
    保留最大连通域可以作为一种简单后处理方式。
    """
    mask_bool = to_binary_mask(mask)

    if not np.any(mask_bool):
        return np.zeros_like(mask_bool, dtype=bool)

    structure = generate_binary_structure(rank=3, connectivity=connectivity)
    labeled, num_components = label(mask_bool, structure=structure)

    if num_components == 0:
        return np.zeros_like(mask_bool, dtype=bool)

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0

    largest_label = int(np.argmax(component_sizes))
    largest = labeled == largest_label

    return largest.astype(bool)


def extract_surface(mask: np.ndarray) -> np.ndarray:
    """
    提取 3D mask 的表面 voxel。

    参数
    ----
    mask:
        bool 类型或可转 bool 的 [D, H, W] mask。

    返回
    ----
    surface:
        bool 类型的 [D, H, W] 表面 mask。
    """
    mask_bool = np.asarray(mask).astype(bool)

    if not np.any(mask_bool):
        return np.zeros_like(mask_bool, dtype=bool)

    structure = generate_binary_structure(rank=3, connectivity=1)
    eroded = binary_erosion(mask_bool, structure=structure, border_value=0)

    surface = mask_bool & (~eroded)

    return surface.astype(bool)


def compute_volume_mm3(
    mask: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> float:
    """
    计算结节体积。

    参数
    ----
    mask:
        bool 类型 3D mask。

    spacing:
        体素间距，顺序为 (D_spacing, H_spacing, W_spacing)。

    返回
    ----
    volume_mm3:
        体积，单位 mm^3。
    """
    mask_bool = np.asarray(mask).astype(bool)
    voxel_volume = float(np.prod(np.asarray(spacing, dtype=np.float64)))

    volume_mm3 = float(mask_bool.sum()) * voxel_volume

    return float(volume_mm3)


def compute_equivalent_diameter_mm(volume_mm3: float) -> float:
    """
    根据体积计算等效球直径。

    公式：
        V = 4/3 * pi * r^3
        d = 2 * r
        d = (6V / pi)^(1/3)

    参数
    ----
    volume_mm3:
        体积，单位 mm^3。

    返回
    ----
    equivalent_diameter_mm:
        等效球直径，单位 mm。
    """
    if volume_mm3 <= 0:
        return 0.0

    diameter = (6.0 * float(volume_mm3) / math.pi) ** (1.0 / 3.0)

    return float(diameter)


def compute_max_axial_area_mm2(
    mask: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> float:
    """
    计算轴向最大截面积。

    参数
    ----
    mask:
        bool 类型 [D, H, W] mask。

    spacing:
        体素间距，顺序为 (D_spacing, H_spacing, W_spacing)。

    返回
    ----
    max_axial_area_mm2:
        最大 axial 切片面积，单位 mm^2。

    说明
    ----
    这里默认 D 轴是 axial 切片方向。
    因此单个 axial 切片上的像素面积为：
        H_spacing * W_spacing
    """
    mask_bool = np.asarray(mask).astype(bool)

    if not np.any(mask_bool):
        return 0.0

    spacing = tuple(float(v) for v in spacing)
    pixel_area = spacing[1] * spacing[2]

    slice_areas = mask_bool.sum(axis=(1, 2)).astype(np.float64) * pixel_area
    max_area = float(np.max(slice_areas))

    return max_area


def compute_surface_area_mm2(
    mask: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> float:
    """
    使用 marching cubes 计算 3D mask 表面积。

    参数
    ----
    mask:
        bool 类型 [D, H, W] mask。

    spacing:
        体素间距，顺序为 (D_spacing, H_spacing, W_spacing)。

    返回
    ----
    surface_area_mm2:
        表面积，单位 mm^2。

    说明
    ----
    为了处理结节贴近 patch 边界的情况，这里会先对 mask 做一圈 padding，
    再执行 marching cubes。padding 只会平移坐标，不会影响表面积。
    """
    mask_bool = np.asarray(mask).astype(bool)

    if not np.any(mask_bool):
        return 0.0

    if marching_cubes is None or mesh_surface_area is None:
        return float("nan")

    spacing = tuple(float(v) for v in spacing)

    try:
        padded = np.pad(mask_bool.astype(np.float32), pad_width=1, mode="constant")
        verts, faces, _, _ = marching_cubes(
            volume=padded,
            level=0.5,
            spacing=spacing,
        )
        area = mesh_surface_area(verts, faces)
        return float(area)
    except Exception:
        return float("nan")


def _get_mesh_vertices_for_diameter(
    mask: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> Optional[np.ndarray]:
    """
    使用 marching cubes 获取表面 mesh 顶点，用于计算最大三维直径。

    返回
    ----
    verts:
        [N, 3] mesh 顶点坐标，单位 mm。
        如果 marching cubes 不可用或失败，则返回 None。
    """
    mask_bool = np.asarray(mask).astype(bool)

    if not np.any(mask_bool):
        return None

    if marching_cubes is None:
        return None

    spacing = tuple(float(v) for v in spacing)

    try:
        padded = np.pad(mask_bool.astype(np.float32), pad_width=1, mode="constant")
        verts, _, _, _ = marching_cubes(
            volume=padded,
            level=0.5,
            spacing=spacing,
        )
        return verts.astype(np.float64, copy=False)
    except Exception:
        return None


def _get_surface_voxel_points(
    mask: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    获取表面 voxel 中心点坐标，用于 marching cubes 不可用时的 fallback。

    返回
    ----
    points:
        [N, 3] 坐标点，单位 mm。
    """
    mask_bool = np.asarray(mask).astype(bool)
    surface = extract_surface(mask_bool)

    if not np.any(surface):
        return np.zeros((0, 3), dtype=np.float64)

    spacing_arr = np.asarray(spacing, dtype=np.float64)
    coords = np.argwhere(surface).astype(np.float64)
    points = coords * spacing_arr[np.newaxis, :]

    return points


def _max_pairwise_distance(
    points: np.ndarray,
    max_exact_points: int = 6000,
) -> float:
    """
    计算点集最大两两欧氏距离。

    参数
    ----
    points:
        [N, 3] 点坐标，单位 mm。

    max_exact_points:
        精确计算 pairwise distance 的最大点数。
        若点太多，会进行确定性下采样，避免内存爆炸。

    返回
    ----
    max_distance:
        最大两点距离。
    """
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points 必须是 [N, 3]，但当前 shape = {points.shape}")

    num_points = points.shape[0]

    if num_points <= 1:
        return 0.0

    if num_points > max_exact_points:
        # 确定性均匀采样，避免随机性影响复现实验。
        indices = np.linspace(0, num_points - 1, num=max_exact_points).astype(np.int64)
        points = points[indices]

    distances = pdist(points, metric="euclidean")

    if distances.size == 0:
        return 0.0

    return float(np.max(distances))


def compute_max_diameter_mm(
    mask: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> float:
    """
    计算最大 3D 直径。

    参数
    ----
    mask:
        bool 类型 [D, H, W] mask。

    spacing:
        体素间距，顺序为 (D_spacing, H_spacing, W_spacing)。

    返回
    ----
    max_diameter_mm:
        最大 3D 直径，单位 mm。

    说明
    ----
    优先使用 marching cubes 得到的表面 mesh 顶点计算最大距离。
    如果 marching cubes 不可用，则退化为使用表面 voxel 中心点。
    """
    mask_bool = np.asarray(mask).astype(bool)

    if not np.any(mask_bool):
        return 0.0

    verts = _get_mesh_vertices_for_diameter(mask_bool, spacing=spacing)

    if verts is not None and verts.shape[0] > 1:
        return _max_pairwise_distance(verts)

    points = _get_surface_voxel_points(mask_bool, spacing=spacing)

    return _max_pairwise_distance(points)


def compute_compactness(
    volume_mm3: float,
    surface_area_mm2: float,
) -> float:
    """
    计算紧致度 compactness。

    公式：
        compactness = 36 * pi * V^2 / A^3

    参数
    ----
    volume_mm3:
        体积，单位 mm^3。

    surface_area_mm2:
        表面积，单位 mm^2。

    返回
    ----
    compactness:
        紧致度。
        对球体接近 1。
        如果体积或表面积无效，则返回 0。
    """
    if volume_mm3 <= 0:
        return 0.0

    if surface_area_mm2 is None:
        return 0.0

    if np.isnan(surface_area_mm2) or surface_area_mm2 <= 0:
        return 0.0

    compactness = (
        36.0
        * math.pi
        * float(volume_mm3) ** 2
        / (float(surface_area_mm2) ** 3)
    )

    return float(compactness)


def quantify_nodule(
    mask: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    keep_largest: bool = True,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    对单个肺结节 mask 进行三维量化分析。

    参数
    ----
    mask:
        输入 pred_mask，shape 为 [D, H, W] 或 [1, D, H, W]。

    spacing:
        体素间距，顺序为 (D_spacing, H_spacing, W_spacing)。

    keep_largest:
        是否只对最大连通域进行量化。
        默认 True，适合模型预测结果中可能存在小型假阳性的情况。

    threshold:
        二值化阈值。

    返回
    ----
    result:
        {
            "volume_mm3": ...,
            "equivalent_diameter_mm": ...,
            "max_diameter_mm": ...,
            "surface_area_mm2": ...,
            "compactness": ...,
            "max_axial_area_mm2": ...,
            "num_components": ...
        }
    """
    spacing = tuple(float(v) for v in spacing)

    if len(spacing) != 3:
        raise ValueError(f"spacing 必须是长度为 3 的序列，但当前为 {spacing}")

    mask_bool = to_binary_mask(mask, threshold=threshold)

    num_components = count_components(mask_bool)

    if not np.any(mask_bool):
        return {
            "volume_mm3": 0.0,
            "equivalent_diameter_mm": 0.0,
            "max_diameter_mm": 0.0,
            "surface_area_mm2": 0.0,
            "compactness": 0.0,
            "max_axial_area_mm2": 0.0,
            "num_components": 0,
        }

    if keep_largest:
        quant_mask = keep_largest_component(mask_bool)
    else:
        quant_mask = mask_bool

    volume_mm3 = compute_volume_mm3(quant_mask, spacing=spacing)
    equivalent_diameter_mm = compute_equivalent_diameter_mm(volume_mm3)
    max_diameter_mm = compute_max_diameter_mm(quant_mask, spacing=spacing)
    surface_area_mm2 = compute_surface_area_mm2(quant_mask, spacing=spacing)
    compactness = compute_compactness(
        volume_mm3=volume_mm3,
        surface_area_mm2=surface_area_mm2,
    )
    max_axial_area_mm2 = compute_max_axial_area_mm2(quant_mask, spacing=spacing)

    result = {
        "volume_mm3": float(volume_mm3),
        "equivalent_diameter_mm": float(equivalent_diameter_mm),
        "max_diameter_mm": float(max_diameter_mm),
        "surface_area_mm2": float(surface_area_mm2),
        "compactness": float(compactness),
        "max_axial_area_mm2": float(max_axial_area_mm2),
        "num_components": int(num_components),
    }

    return result


def _format_float(value: Any) -> str:
    """
    格式化数值，方便写入 CSV。
    """
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return "nan"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.8f}"


def _load_prediction_mask_from_npz(
    npz_path: Path,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    从预测 npz 文件中读取 mask。

    优先级：
    1. pred_mask
    2. pred_prob，经 threshold 二值化
    3. mask

    返回
    ----
    mask:
        [D, H, W] bool mask。
    """
    with np.load(npz_path) as data:
        keys = set(data.files)

        if "pred_mask" in keys:
            return to_binary_mask(data["pred_mask"], threshold=threshold)

        if "pred_prob" in keys:
            return to_binary_mask(data["pred_prob"], threshold=threshold)

        if "mask" in keys:
            return to_binary_mask(data["mask"], threshold=threshold)

        raise KeyError(
            f"预测文件中没有 pred_mask、pred_prob 或 mask 字段：{npz_path}，"
            f"当前字段为 {sorted(list(keys))}"
        )


def _load_spacing_from_npz(
    npz_path: Path,
    default_spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> Tuple[float, float, float]:
    """
    如果 npz 中包含 spacing 字段，则读取 spacing，否则使用默认 spacing。
    """
    with np.load(npz_path) as data:
        if "spacing" not in data.files:
            return tuple(float(v) for v in default_spacing)

        spacing = np.asarray(data["spacing"]).reshape(-1)

        if spacing.size != 3:
            return tuple(float(v) for v in default_spacing)

        return tuple(float(v) for v in spacing.tolist())


def batch_quantify_prediction_folder(
    prediction_dir: str | Path,
    output_csv: str | Path,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    threshold: float = 0.5,
    keep_largest: bool = True,
) -> None:
    """
    批量量化 prediction_dir 中的所有预测 npz 文件。

    参数
    ----
    prediction_dir:
        预测结果目录。
        默认可对应 evaluate.py 或 predict.py 生成的 outputs/predictions。

    output_csv:
        输出 CSV 文件路径。

    spacing:
        默认体素间距。
        如果某个 npz 文件中包含 spacing 字段，则优先使用 npz 内的 spacing。

    threshold:
        如果文件中只有 pred_prob，则使用该阈值生成 pred_mask。

    keep_largest:
        是否只量化最大连通域。

    输出 CSV 字段
    ----
    - file_name
    - volume_mm3
    - equivalent_diameter_mm
    - max_diameter_mm
    - surface_area_mm2
    - compactness
    - max_axial_area_mm2
    - num_components
    """
    prediction_dir = Path(prediction_dir).expanduser().resolve()
    output_csv = Path(output_csv).expanduser().resolve()

    if not prediction_dir.exists():
        raise FileNotFoundError(f"找不到预测目录：{prediction_dir}")

    npz_files = sorted(prediction_dir.glob("*.npz"))

    if len(npz_files) == 0:
        raise RuntimeError(f"预测目录中没有 npz 文件：{prediction_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["file_name"] + QUANTIFICATION_KEYS

    rows: List[Dict[str, Any]] = []

    for npz_path in npz_files:
        pred_mask = _load_prediction_mask_from_npz(
            npz_path=npz_path,
            threshold=threshold,
        )

        file_spacing = _load_spacing_from_npz(
            npz_path=npz_path,
            default_spacing=spacing,
        )

        result = quantify_nodule(
            mask=pred_mask,
            spacing=file_spacing,
            keep_largest=keep_largest,
            threshold=threshold,
        )

        row: Dict[str, Any] = {"file_name": npz_path.name}

        for key in QUANTIFICATION_KEYS:
            row[key] = result[key]

        rows.append(row)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            formatted_row = {"file_name": row["file_name"]}

            for key in QUANTIFICATION_KEYS:
                formatted_row[key] = _format_float(row[key])

            writer.writerow(formatted_row)

    print(f"批量量化完成：{output_csv}")
    print(f"共处理 {len(rows)} 个 npz 文件。")


def parse_args() -> argparse.Namespace:
    """
    命令行参数。

    示例：
    python src/quantification.py \
        --prediction-dir outputs/predictions \
        --output-csv outputs/metrics/quantification.csv \
        --spacing 1.0 1.0 1.0 \
        --threshold 0.5
    """
    parser = argparse.ArgumentParser(
        description="Quantify 3D nodule morphology from prediction npz files."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"训练配置文件路径，默认 {DEFAULT_CONFIG_PATH}",
    )

    parser.add_argument(
        "--prediction-dir",
        type=str,
        default=None,
        help="预测 npz 文件目录，例如 outputs/predictions",
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="量化结果 CSV 输出路径，例如 outputs/metrics/quantification.csv",
    )

    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        help="体素间距，顺序为 D H W，默认 1.0 1.0 1.0",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="pred_prob 二值化阈值，默认 0.5",
    )

    parser.add_argument(
        "--no-keep-largest",
        action="store_true",
        help="不保留最大连通域，直接量化完整 mask",
    )

    return parser.parse_args()


def _demo_single_case() -> None:
    """
    简单自检：构造一个球形 3D mask 并计算量化指标。
    """
    d, h, w = 64, 64, 64
    z, y, x = np.ogrid[:d, :h, :w]

    center = (32, 32, 32)
    radius = 10

    demo_mask = (
        (z - center[0]) ** 2
        + (y - center[1]) ** 2
        + (x - center[2]) ** 2
        <= radius ** 2
    ).astype(np.float32)

    result = quantify_nodule(
        mask=demo_mask,
        spacing=(1.0, 1.0, 1.0),
        keep_largest=True,
        threshold=0.5,
    )

    print("单样本量化自检：")
    for key, value in result.items():
        print(f"{key}: {value}")


def infer_output_dir(config_path: str) -> Path:
    path = resolve_existing_path(config_path)
    if not path.exists():
        return resolve_runtime_path(DEFAULT_OUTPUT_DIR)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        return resolve_runtime_path(DEFAULT_OUTPUT_DIR)
    return resolve_runtime_path(cfg.get("output", {}).get("output_dir", DEFAULT_OUTPUT_DIR))


def main() -> None:
    """
    主入口。

    如果提供 --prediction-dir 和 --output-csv，则执行批量量化；
    否则执行一个简单自检。
    """
    args = parse_args()

    output_dir = infer_output_dir(args.config)
    prediction_dir = args.prediction_dir
    output_csv = args.output_csv

    if prediction_dir is None and output_csv is None:
        default_prediction_dir = output_dir / "predictions"
        if default_prediction_dir.exists():
            prediction_dir = str(default_prediction_dir)
            output_csv = str(output_dir / "metrics" / "quantification.csv")
    elif prediction_dir is None:
        prediction_dir = str(output_dir / "predictions")
    elif output_csv is None:
        output_csv = str(output_dir / "metrics" / "quantification.csv")

    if prediction_dir is not None and output_csv is not None:
        batch_quantify_prediction_folder(
            prediction_dir=prediction_dir,
            output_csv=output_csv,
            spacing=tuple(args.spacing),
            threshold=float(args.threshold),
            keep_largest=not bool(args.no_keep_largest),
        )
    else:
        _demo_single_case()


if __name__ == "__main__":
    main()
