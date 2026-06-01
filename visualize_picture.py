#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
visualize_whole_ct_checkpoint.py

根据 train.py 的保存路径，随机选择验证集整例 CT，
加载 outputs/checkpoints/best_dice.pt 或 latest.pt，使用滑窗推理整例体积并可视化。

建议放置位置：
    与 train.py 放在同一目录。

常用运行：
    python visualize_random_checkpoint.py --config configs/task_adaptive.yaml --num 3 --save-only

显示窗口：
    python visualize_random_checkpoint.py --config configs/task_adaptive.yaml --num 1

指定 checkpoint：
    python visualize_random_checkpoint.py --config configs/task_adaptive.yaml --ckpt outputs/checkpoints/latest.pt --num 3 --save-only

输出：
    1. 可视化图片：
       outputs/visualized_checkpoint/*.png

    2. 可供 quantification.py 继续量化的预测 npz：
       outputs/predictions_from_checkpoint/*.npz
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from monai.data import PersistentDataset
from monai.inferers import sliding_window_inference


DEFAULT_VOLUME_ROOT = "/home/lembert/Desktop/BME-4/BME_package_mixed"


# =========================
# 1. 导入训练脚本中的构建逻辑
# =========================

def add_project_to_path() -> Path:
    """
    让脚本既可以放在项目根目录，也可以放在 src/ 目录下运行。
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parent,
        here.parent.parent,
    ]
    for p in candidates:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return here.parent


add_project_to_path()

try:
    from train import (  # type: ignore
        DEFAULT_CONFIG_PATH,
        DEFAULT_OUTPUT_DIR,
        IMAGE_KEY,
        LABEL_KEY,
        MASK_ALIAS_KEY,
        build_transforms,
        build_model,
        ensure_3d_label_np,
        fast_postprocess_prediction,
        infer_input_format,
        load_checkpoint,
        load_yaml,
        normalize_records_for_image_label,
        parse_class_dict_or_scalar,
        read_csv_records,
        resolve_path,
        resolve_split_csv_paths,
        resolve_existing_path,
        resolve_runtime_path,
    )
except Exception as exc:
    raise ImportError(
        "无法导入 train.py。\n"
        "请确认本脚本与 train.py 在同一目录，"
        "或者在项目根目录运行。\n"
        f"原始错误：{repr(exc)}"
    ) from exc


# =========================
# 2. 参数
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random visualization from trained checkpoint."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="训练时使用的 YAML 配置文件。",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="auto",
        help=(
            "checkpoint 路径。默认 auto：优先 output_dir/checkpoints/best_dice.pt，"
            "不存在则使用 output_dir/checkpoints/latest.pt。"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="覆盖配置文件中的 output.output_dir。",
    )

    parser.add_argument(
        "--num",
        type=int,
        default=1,
        help="随机可视化几个验证集样本。",
    )

    parser.add_argument(
        "--val-csv",
        type=str,
        default=None,
        help=(
            "整例验证 CSV。默认使用 config.data.val_csv。"
            "CSV 必须包含 image,label 或 image,mask；npz_path patch CSV 不支持整例 CT 可视化。"
        ),
    )

    parser.add_argument(
        "--volume-root",
        type=str,
        default=DEFAULT_VOLUME_ROOT,
        help=(
            "整例 CT 所在目录。若 val.csv 是 npz_path patch 列表，脚本会从 val.csv 提取病例 ID，"
            "再到该目录寻找 {case_id}_img.nii.gz 和 {case_id}_mask.nii.gz。"
        ),
    )

    parser.add_argument(
        "--sw-batch-size",
        type=int,
        default=1,
        help="滑窗推理时每次送入模型的窗口数。",
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="滑窗推理窗口重叠比例，范围 [0, 1)。",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子；设置后每次抽样一致。",
    )

    parser.add_argument(
        "--axis",
        type=str,
        default="axial",
        choices=["axial", "coronal", "sagittal"],
        help="显示方向。",
    )

    parser.add_argument(
        "--slice",
        type=int,
        default=None,
        help="指定切片编号；不指定则自动选择 pred/gt 面积最大的切片。",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="预测二值化阈值。默认读取 config.eval.threshold。",
    )

    parser.add_argument(
        "--save-only",
        action="store_true",
        help="只保存图片，不弹窗。服务器或 ToDesk 不稳定时建议使用。",
    )

    parser.add_argument(
        "--no-save-npz",
        action="store_true",
        help="不保存 predictions_from_checkpoint/*.npz。",
    )

    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="图像显示窗下限。不指定则使用 1% 分位数。",
    )

    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="图像显示窗上限。不指定则使用 99% 分位数。",
    )

    return parser.parse_args()


# =========================
# 3. 配置与路径
# =========================

def apply_visual_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg.setdefault("output", {})
    cfg.setdefault("data", {})
    cfg.setdefault("eval", {})

    if args.output_dir is not None:
        cfg["output"]["output_dir"] = args.output_dir

    if args.val_csv is not None:
        cfg["data"]["val_csv"] = args.val_csv

    if args.volume_root is not None:
        cfg["data"]["volume_root"] = args.volume_root

    # 可视化时不需要多进程，避免 PyCharm / Windows / 服务器上卡住或报 pickle 问题
    cfg["data"]["num_workers"] = 0

    return cfg


def get_output_dir(cfg: Dict[str, Any]) -> Path:
    return resolve_runtime_path(cfg.get("output", {}).get("output_dir", DEFAULT_OUTPUT_DIR))


def resolve_checkpoint_path(cfg: Dict[str, Any], ckpt_arg: str) -> Path:
    output_dir = get_output_dir(cfg)
    ckpt_dir = output_dir / "checkpoints"

    if ckpt_arg.lower() != "auto":
        path = Path(ckpt_arg).expanduser()
        path = path.resolve() if path.is_absolute() else resolve_existing_path(path, base_dirs=[ckpt_dir, output_dir])
        if not path.exists():
            raise FileNotFoundError(f"找不到指定 checkpoint：{path}")
        return path

    latest_path = ckpt_dir / "latest.pt"
    best_path = ckpt_dir / "best_dice.pt"

    if latest_path.exists():
        return latest_path.resolve()

    if best_path.exists():
        return best_path.resolve()

    raise FileNotFoundError(
        "没有找到可用 checkpoint。\n"
        f"尝试过：\n"
        f"1. {latest_path}\n"
        f"2. {best_path}\n"
        "请先训练，或使用 --ckpt 手动指定权重文件。"
    )


def extract_case_id(record: Dict[str, Any]) -> str:
    for key in ("patient_id", "case_id"):
        value = str(record.get(key, "")).strip()
        if value:
            return value

    for key in ("npz_path", "image", "label", "mask"):
        value = str(record.get(key, "")).strip()
        if not value:
            continue
        match = re.search(r"LIDC-IDRI-\d+", Path(value).name)
        if match:
            return match.group(0)

    raise ValueError(f"无法从记录中提取病例 ID：{record}")


def find_volume_pair(volume_root: Path, case_id: str) -> Tuple[Path, Path]:
    image_patterns = [
        f"{case_id}_img.nii.gz",
        f"{case_id}_*_img.nii.gz",
        f"{case_id}_image.nii.gz",
        f"{case_id}_*_image.nii.gz",
        f"{case_id}_ct.nii.gz",
        f"{case_id}_*_ct.nii.gz",
        f"{case_id}*img*.nii.gz",
        f"{case_id}*image*.nii.gz",
        f"{case_id}*ct*.nii.gz",
    ]
    label_patterns = [
        f"{case_id}_mask.nii.gz",
        f"{case_id}_*_mask.nii.gz",
        f"{case_id}_label.nii.gz",
        f"{case_id}_*_label.nii.gz",
        f"{case_id}_seg.nii.gz",
        f"{case_id}_*_seg.nii.gz",
        f"{case_id}*mask*.nii.gz",
        f"{case_id}*label*.nii.gz",
        f"{case_id}*seg*.nii.gz",
    ]

    def first_match(patterns: Sequence[str], forbidden: Sequence[str] = ()) -> Optional[Path]:
        for pattern in patterns:
            candidates = sorted(
                p for p in volume_root.rglob(pattern)
                if p.is_file() and not any(token in p.name.lower() for token in forbidden)
            )
            if candidates:
                return candidates[0]
        return None

    image = first_match(image_patterns, forbidden=("mask", "label", "seg", "dist"))
    label = first_match(label_patterns, forbidden=("dist",))

    if image is None or label is None:
        raise FileNotFoundError(
            f"找不到 {case_id} 的整例 CT 或 mask。\n"
            f"搜索目录：{volume_root}\n"
            f"期望类似：{case_id}_img.nii.gz / {case_id}_mask.nii.gz，"
            f"或带序号的 {case_id}_247_img.nii.gz / {case_id}_247_mask.nii.gz"
        )

    return image.resolve(), label.resolve()


def records_from_patch_csv(records: Sequence[Dict[str, Any]], volume_root: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()

    for record in records:
        case_id = extract_case_id(record)
        if case_id in seen:
            continue
        seen.add(case_id)

        image, label = find_volume_pair(volume_root, case_id)
        out.append(
            {
                "image": str(image),
                "label": str(label),
                "case_id": case_id,
                "patient_id": case_id,
            }
        )

    return out


def autodetect_volume_root(val_csv: Path) -> Optional[Path]:
    here = Path(__file__).resolve()
    candidates: List[Path] = [
        Path(DEFAULT_VOLUME_ROOT),
    ]

    for base in [Path.cwd(), here.parent, here.parent.parent, *val_csv.parents]:
        candidates.append(base / "BME_package_mixed")
        candidates.append(base / "old1" / "processed_data")
        candidates.append(base / "processed_data")

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.is_dir():
            continue
        has_ready = any(candidate.glob("*.bt_ready"))
        has_images = any(candidate.glob("*_img.nii.gz"))
        if has_ready or has_images:
            return candidate

    return None


def build_whole_volume_val_dataset(cfg: Dict[str, Any]) -> Tuple[PersistentDataset, str, Path]:
    data_cfg = cfg.get("data", {})

    data_root = resolve_runtime_path(data_cfg.get("data_root", "data"))
    val_csv_value = data_cfg.get("val_csv", "val.csv")
    use_default_split_paths = (
        str(data_cfg.get("data_root", "data")) == "data"
        and str(data_cfg.get("train_csv", "train.csv")) == "train.csv"
        and str(val_csv_value) == "val.csv"
    )
    if use_default_split_paths:
        data_root, _, val_csv = resolve_split_csv_paths(data_cfg)
    else:
        val_csv = resolve_path(val_csv_value, data_root)

    records = read_csv_records(val_csv, val_csv.parent)

    input_format = infer_input_format(records, "auto")
    if input_format == "npz_patch":
        volume_root_value = data_cfg.get("volume_root", None)
        if volume_root_value is None:
            detected_volume_root = autodetect_volume_root(val_csv)
            if detected_volume_root is not None:
                volume_root = detected_volume_root
                print(f"[AUTO VOLUME ROOT] {volume_root}")
            else:
                raise ValueError(
                    "val.csv 是 npz_path patch 列表。若要按验证集病例可视化整例 CT，"
                    "请传入 --volume-root 指向整例 CT 目录，例如 old1/processed_data。"
                )
        else:
            volume_root = Path(str(volume_root_value)).expanduser()
            if not volume_root.is_absolute():
                volume_root = resolve_runtime_path(volume_root)
            volume_root = volume_root.resolve()
        if not volume_root.is_dir():
            raise ValueError(
                f"整例 CT 目录不存在：{volume_root}\n"
                "请检查 --volume-root 是否指向包含 *_img.nii.gz / *_mask.nii.gz 的目录。"
            )
        records = records_from_patch_csv(records, volume_root)
        input_format = "image_label"
    elif input_format != "image_label":
        raise ValueError(
            "整例 CT 可视化需要验证 CSV 包含 image,label 或 image,mask 列。\n"
            f"当前识别到 input_format={input_format}，通常表示 val.csv 仍是 npz patch 列表。\n"
            "请提供整例验证 CSV，或在 patch val.csv 情况下传入 --volume-root。"
        )

    records = normalize_records_for_image_label(records)

    roi_size = data_cfg.get("roi_size", data_cfg.get("patch_size", [96, 96, 96]))
    target_spacing = data_cfg.get("target_spacing", [1.0, 1.0, 1.0])
    intensity_mode = data_cfg.get("intensity_mode", "ct")
    ct_window = data_cfg.get("ct_window", [-1000.0, 400.0])
    profile_ratios = data_cfg.get("profile_ratios", [1, 1, 1, 0.5, 0.5])
    small_cc_voxels = int(data_cfg.get("small_cc_voxels", 128))
    large_cc_voxels = int(data_cfg.get("large_cc_voxels", 4096))
    overwrite_aux = bool(data_cfg.get("overwrite_aux", False))
    use_d2 = bool(data_cfg.get("use_d2", False))
    d2_percentile = float(data_cfg.get("d2_percentile", 99.0))

    val_tfms = build_transforms(
        input_format="image_label",
        roi_size=roi_size,
        target_spacing=target_spacing,
        intensity_mode=intensity_mode,
        ct_window=ct_window,
        profile_ratios=profile_ratios,
        samples_per_volume=1,
        small_cc_voxels=small_cc_voxels,
        large_cc_voxels=large_cc_voxels,
        overwrite_aux=overwrite_aux,
        is_train=False,
        use_d2=use_d2,
        d2_percentile=d2_percentile,
    )

    output_dir = get_output_dir(cfg)
    cache_root = (
        resolve_runtime_path(data_cfg["cache_dir"])
        if data_cfg.get("cache_dir", None) is not None
        else output_dir / "persistent_cache"
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    val_ds = PersistentDataset(
        data=records,
        transform=val_tfms,
        cache_dir=cache_root / ("val_whole_volume_d2" if use_d2 else "val_whole_volume"),
    )

    return val_ds, input_format, val_csv


# =========================
# 4. 数组处理
# =========================

def tensor_to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def image_to_3d_first_channel(image: Any) -> np.ndarray:
    """
    输入通常是 [C,D,H,W]，取第 0 通道得到 [D,H,W]。
    """
    arr = tensor_to_numpy(image)
    arr = np.asarray(arr)

    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 4:
        return arr[0].astype(np.float32)

    if arr.ndim == 3:
        return arr.astype(np.float32)

    arr = np.squeeze(arr)
    if arr.ndim == 3:
        return arr.astype(np.float32)

    raise ValueError(f"无法把 image 转为 [D,H,W]，当前 shape={arr.shape}")


def label_to_3d(label: Any) -> np.ndarray:
    arr = tensor_to_numpy(label)
    return ensure_3d_label_np(arr).astype(np.int64)


def normalize_for_show(
    image: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)

    if not np.any(finite):
        return np.zeros_like(image, dtype=np.float32)

    valid = image[finite]

    if vmin is None:
        vmin = float(np.percentile(valid, 1))
    if vmax is None:
        vmax = float(np.percentile(valid, 99))

    if vmax <= vmin:
        vmax = vmin + 1.0

    image = np.clip(image, vmin, vmax)
    image = (image - vmin) / (vmax - vmin + 1e-8)

    return image.astype(np.float32)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax_np(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def get_case_id(item: Dict[str, Any], index: int) -> str:
    for key in ["case_id", "nodule_id", "npz_path", "image"]:
        if key in item:
            value = item[key]
            if isinstance(value, (list, tuple)) and len(value) > 0:
                value = value[0]
            value = str(value)
            if value:
                if key in {"npz_path", "image"}:
                    return Path(value).stem
                return value.replace("/", "_").replace("\\", "_")
    return f"case_{index:04d}"


# =========================
# 5. 切片选择与绘图
# =========================

def choose_slice_index(
    pred: np.ndarray,
    gt: Optional[np.ndarray],
    axis: str,
    fixed_slice: Optional[int] = None,
) -> int:
    axis_to_dim = {
        "sagittal": 0,
        "coronal": 1,
        "axial": 2,
    }

    dim = axis_to_dim[axis]
    max_index = pred.shape[dim] - 1

    if fixed_slice is not None:
        return int(np.clip(fixed_slice, 0, max_index))

    if axis == "axial":
        sum_axes = (0, 1)
    elif axis == "coronal":
        sum_axes = (0, 2)
    else:
        sum_axes = (1, 2)

    if gt is not None:
        gt_areas = (gt > 0).astype(np.float32).sum(axis=sum_axes)
        if np.max(gt_areas) > 0:
            return int(np.argmax(gt_areas))

    pred_areas = (pred > 0).astype(np.float32).sum(axis=sum_axes)
    if np.max(pred_areas) > 0:
        return int(np.argmax(pred_areas))

    return pred.shape[dim] // 2


def get_2d_slice(arr: np.ndarray, axis: str, index: int) -> np.ndarray:
    if axis == "axial":
        slc = arr[:, :, index]
    elif axis == "coronal":
        slc = arr[:, index, :]
    elif axis == "sagittal":
        slc = arr[index, :, :]
    else:
        raise ValueError(f"未知 axis：{axis}")

    return np.rot90(slc)


def draw_contour(ax: Any, mask_2d: np.ndarray, color: str, linewidth: float = 1.2) -> None:
    if np.any(mask_2d > 0):
        ax.contour(
            (mask_2d > 0).astype(np.float32),
            levels=[0.5],
            colors=color,
            linewidths=linewidth,
        )


def compute_fg_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bin = pred > 0
    gt_bin = gt > 0
    inter = float(np.logical_and(pred_bin, gt_bin).sum())
    denom = float(pred_bin.sum() + gt_bin.sum())
    if denom <= 0:
        return 1.0
    return float(2.0 * inter / denom)


def visualize_case(
    *,
    image_3d: np.ndarray,
    gt_3d: Optional[np.ndarray],
    pred_3d: np.ndarray,
    case_id: str,
    axis: str,
    fixed_slice: Optional[int],
    vmin: Optional[float],
    vmax: Optional[float],
    out_png: Path,
    save_only: bool,
    plt: Any,
) -> None:
    image_show = normalize_for_show(image_3d, vmin=vmin, vmax=vmax)

    slice_index = choose_slice_index(
        pred=pred_3d,
        gt=gt_3d,
        axis=axis,
        fixed_slice=fixed_slice,
    )

    img_2d = get_2d_slice(image_show, axis, slice_index)
    pred_2d = get_2d_slice(pred_3d, axis, slice_index)
    gt_2d = None if gt_3d is None else get_2d_slice(gt_3d, axis, slice_index)

    has_gt = gt_2d is not None
    ncols = 4 if has_gt else 3

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), dpi=120)

    axes[0].imshow(img_2d, cmap="gray")
    axes[0].set_title("Image")
    axes[0].axis("off")

    if has_gt:
        axes[1].imshow(img_2d, cmap="gray")
        draw_contour(axes[1], gt_2d, color="lime")
        axes[1].set_title("GT contour")
        axes[1].axis("off")

        axes[2].imshow(img_2d, cmap="gray")
        draw_contour(axes[2], pred_2d, color="red")
        axes[2].set_title("Pred contour")
        axes[2].axis("off")

        overlay_ax = axes[3]
    else:
        axes[1].imshow((pred_2d > 0).astype(np.float32), cmap="gray")
        axes[1].set_title("Pred mask")
        axes[1].axis("off")
        overlay_ax = axes[2]

    overlay_ax.imshow(img_2d, cmap="gray")
    if has_gt:
        draw_contour(overlay_ax, gt_2d, color="lime")
    draw_contour(overlay_ax, pred_2d, color="red")
    overlay_ax.set_title("Overlay: pred=red, gt=green")
    overlay_ax.axis("off")

    dice_text = ""
    if gt_3d is not None:
        dice_text = f" | fg_dice={compute_fg_dice(pred_3d, gt_3d):.4f}"

    fig.suptitle(
        f"{case_id} | axis={axis} | slice={slice_index}{dice_text}",
        fontsize=10,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", dpi=160)
    print(f"[SAVE PNG] {out_png}")

    if save_only:
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


# =========================
# 6. 推理
# =========================

@torch.no_grad()
def predict_one(
    model: torch.nn.Module,
    item: Dict[str, Any],
    device: torch.device,
    cfg: Dict[str, Any],
    threshold: float,
    sw_batch_size: int,
    overlap: float,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    返回：
        image_3d, pred_mask_3d, gt_3d, pred_prob
    """
    image = item[IMAGE_KEY]
    image_tensor = torch.as_tensor(tensor_to_numpy(image)).float()

    if image_tensor.ndim == 4:
        image_tensor = image_tensor.unsqueeze(0)  # [1,C,D,H,W]
    elif image_tensor.ndim == 5:
        pass
    else:
        raise ValueError(f"image 必须是 [C,D,H,W] 或 [B,C,D,H,W]，当前 shape={tuple(image_tensor.shape)}")

    image_tensor = image_tensor.to(device, non_blocking=True)

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("eval", {})

    num_classes = int(model_cfg.get("num_classes", 1))
    roi_size = tuple(int(v) for v in data_cfg.get("roi_size", data_cfg.get("patch_size", model_cfg.get("img_size", [96, 96, 96]))))
    spacing = eval_cfg.get("spacing", data_cfg.get("target_spacing", [1.0, 1.0, 1.0]))
    min_voxels = parse_class_dict_or_scalar(eval_cfg.get("min_voxels", 16))
    min_volume_mm3 = parse_class_dict_or_scalar(eval_cfg.get("min_volume_mm3", None))
    keep_largest = parse_class_dict_or_scalar(eval_cfg.get("keep_largest", False))

    def predictor(window_tensor: torch.Tensor) -> torch.Tensor:
        outputs = model(window_tensor)
        return outputs["mask_logits"] if isinstance(outputs, dict) else outputs

    logits = sliding_window_inference(
        inputs=image_tensor,
        roi_size=roi_size,
        sw_batch_size=max(1, int(sw_batch_size)),
        predictor=predictor,
        overlap=float(overlap),
    )
    logits_np = logits.detach().float().cpu().numpy()[0]  # [C,D,H,W]

    pred_mask = fast_postprocess_prediction(
        pred=logits_np,
        num_classes=num_classes,
        threshold=threshold,
        spacing=spacing,
        min_voxels=min_voxels,
        min_volume_mm3=min_volume_mm3,
        keep_largest=keep_largest,
        connectivity=3,
        input_is_logits=True,
    ).astype(np.uint8)

    if num_classes <= 1:
        pred_prob = sigmoid_np(logits_np[0]).astype(np.float32)
    else:
        pred_prob = softmax_np(logits_np, axis=0).astype(np.float32)

    image_3d = image_to_3d_first_channel(item[IMAGE_KEY])

    label = item.get(LABEL_KEY, item.get(MASK_ALIAS_KEY, None))
    gt_3d = None
    if label is not None:
        gt_3d = label_to_3d(label).astype(np.uint8)

    return image_3d, pred_mask, gt_3d, pred_prob


# =========================
# 7. 主入口
# =========================

def main() -> None:
    args = parse_args()

    import matplotlib
    if args.save_only:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    cfg = load_yaml(args.config)
    cfg = apply_visual_overrides(cfg, args)

    output_dir = get_output_dir(cfg)
    ckpt_path = resolve_checkpoint_path(cfg, args.ckpt)

    vis_dir = output_dir / "visualized_checkpoint"
    pred_dir = output_dir / "predictions_from_checkpoint"
    vis_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(cfg.get("eval", {}).get("threshold", 0.5))
    )
    if not 0.0 <= float(args.overlap) < 1.0:
        raise ValueError(f"--overlap 必须在 [0, 1) 范围内，当前为 {args.overlap}")

    print("=" * 80)
    print(f"[CONFIG] {resolve_existing_path(args.config)}")
    print(f"[OUTPUT DIR] {output_dir}")
    print(f"[CHECKPOINT] {ckpt_path}")
    print(f"[DEVICE] {device}")
    print(f"[THRESHOLD] {threshold}")
    print("=" * 80)

    # Build whole-volume validation data instead of the patch validation loader.
    val_ds, input_format, val_csv = build_whole_volume_val_dataset(cfg)

    if len(val_ds) == 0:
        raise RuntimeError("验证集为空，无法随机可视化。")

    rng = random.Random(args.seed)
    num = max(1, min(int(args.num), len(val_ds)))
    indices = rng.sample(range(len(val_ds)), k=num)

    model = build_model(cfg).to(device)
    loaded_epoch, best_score = load_checkpoint(
        ckpt_path,
        model=model,
        optimizer=None,
        scheduler=None,
        device=device,
    )
    model.eval()

    print(f"[LOADED] epoch={loaded_epoch}, best_score={best_score:.6f}")
    print(f"[INPUT FORMAT] {input_format}")
    print(f"[VAL CSV] {val_csv}")
    print(f"[SLIDING WINDOW] sw_batch_size={args.sw_batch_size}, overlap={args.overlap}")
    print(f"[SELECTED INDICES] {indices}")

    summary_rows = []

    for index in indices:
        item = val_ds[index]
        case_id = get_case_id(item, index)

        print("-" * 80)
        print(f"[CASE] index={index}, case_id={case_id}")

        image_3d, pred_mask, gt_3d, pred_prob = predict_one(
            model=model,
            item=item,
            device=device,
            cfg=cfg,
            threshold=threshold,
            sw_batch_size=args.sw_batch_size,
            overlap=args.overlap,
        )

        safe_case_id = case_id.replace(" ", "_").replace("/", "_").replace("\\", "_")
        out_png = vis_dir / f"{safe_case_id}_{args.axis}.png"

        visualize_case(
            image_3d=image_3d,
            gt_3d=gt_3d,
            pred_3d=pred_mask,
            case_id=case_id,
            axis=args.axis,
            fixed_slice=args.slice,
            vmin=args.vmin,
            vmax=args.vmax,
            out_png=out_png,
            save_only=bool(args.save_only),
            plt=plt,
        )

        dice = None
        if gt_3d is not None:
            dice = compute_fg_dice(pred_mask, gt_3d)

        pred_npz_path = None
        if not args.no_save_npz:
            pred_npz_path = pred_dir / f"{safe_case_id}.npz"
            np.savez_compressed(
                pred_npz_path,
                image=image_3d.astype(np.float32),
                pred_mask=pred_mask.astype(np.uint8),
                pred_prob=pred_prob.astype(np.float32),
                gt_mask=np.zeros_like(pred_mask, dtype=np.uint8) if gt_3d is None else gt_3d.astype(np.uint8),
                spacing=np.asarray(
                    cfg.get("eval", {}).get(
                        "spacing",
                        cfg.get("data", {}).get("target_spacing", [1.0, 1.0, 1.0]),
                    ),
                    dtype=np.float32,
                ),
            )
            print(f"[SAVE NPZ] {pred_npz_path}")

        summary_rows.append({
            "index": index,
            "case_id": case_id,
            "png": str(out_png),
            "npz": "" if pred_npz_path is None else str(pred_npz_path),
            "fg_dice": None if dice is None else round(float(dice), 6),
            "pred_voxels": int((pred_mask > 0).sum()),
            "gt_voxels": None if gt_3d is None else int((gt_3d > 0).sum()),
        })

    summary_path = vis_dir / "visualization_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"[SUMMARY] {summary_path}")
    print("完成。")


if __name__ == "__main__":
    main()
