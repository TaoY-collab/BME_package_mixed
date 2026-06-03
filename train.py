#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_task_adaptive.py

Task-Adaptive 3D Medical Segmentation Training Script

适配目标：
1. 新任务 image/label 全体积训练，例如 NIfTI / NRRD / MHA：CSV 需要 image,label 两列；
2. 旧项目 npz patch 训练，例如 LIDC-IDRI patch：CSV 需要 npz_path 列，npz 内建议包含 image/mask/sdf/boundary/core；
3. MONAI PersistentDataset 硬盘缓存；
4. Profiled 智能切块采样：背景、普通前景、边界/难例、小连通域、大连通域按比例采样；
5. 动态对抗损失：Warmup Dice+CE/BCE，中后期降低 Dice/CE 并拉高 Tversky、Boundary、SDF；
6. Stage2 小连通域反向加权；
7. AMP：优先 bfloat16，回退 float16 + GradScaler；
8. 梯度累加、梯度裁剪、checkpoint/resume；
9. 验证阶段 NumPy 向量化连通域后处理。

推荐运行：
python train_task_adaptive.py --config configs/task_adaptive.yaml

最小配置示例见文件底部注释。
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
    label as cc_label,
)
from torch.optim import AdamW
from tqdm import tqdm

try:
    from monai.data import DataLoader, Dataset, PersistentDataset
    from monai.transforms import (
        Compose,
        CropForegroundd,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        MapTransform,
        Orientationd,
        RandCropByLabelClassesd,
        RandFlipd,
        RandGaussianNoised,
        RandRotate90d,
        RandScaleIntensityd,
        RandShiftIntensityd,
        ScaleIntensityRanged,
        ScaleIntensityRangePercentilesd,
        Spacingd,
    )
    from monai.networks.nets import SwinUNETR
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "本训练脚本依赖 MONAI，请先安装：pip install monai\n"
        f"原始错误：{repr(exc)}"
    ) from exc


# 尽量兼容你的旧项目结构；如果这些文件不存在，脚本会使用内置 fallback。
try:
    from model_hybrid_swin_sdf_core import HybridSwinSDFCoreNet  # type: ignore
except Exception:
    HybridSwinSDFCoreNet = None  # type: ignore


# =========================
# 全局字段
# =========================

LOG_COLUMNS = [
    "epoch",
    "lr",
    "train_total",
    "train_dice",
    "train_ce",
    "train_tversky",
    "train_boundary",
    "train_sdf",
    "train_core",
    "train_center",
    "w_dice",
    "w_ce",
    "w_tversky",
    "w_boundary",
    "w_sdf",
    "w_core",
    "w_center",
    "stage2_active",
    "val_dice",
    "val_iou",
    "val_soft_dice",
    "val_post_dice",
    "val_post_iou",
    "val_pred_voxels",
    "val_post_pred_voxels",
    "val_gt_voxels",
    "val_score",
    "epoch_time_sec",
]

IMAGE_KEY = "image"
LABEL_KEY = "label"
MASK_ALIAS_KEY = "mask"
PATCH_CENTER_KEY = "patch_center_dhw"
SPACING_KEY = "spacing_mm"
POSITION_COORDS_KEY = "position_coords"

DEFAULT_CONFIG_PATH = "configs/task_adaptive.yaml"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_DATA_ROOT = "data"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "deterministic": False,
    "data": {
        "data_root": "/home/lembert/Desktop/BME-4/BME-WYZ/data_processed/LIDC-IDRI",
        "val_data_root": "/home/lembert/Desktop/BME-4/BME-WYZ/data_processed/LIDC-IDRI/volumes",
        "train_csv": "train.csv",
        "val_csv": "val.csv",
        "test_csv": "test.csv",
        "val_volume_csv": "val_volume.csv",
        "auto_split": True,
        "split_ratios": [0.8, 0.1, 0.1],
        "split_dir": "outputs/splits",
        "sample_pos_neg_ratio": 3.0,
        "input_format": "npz_patch",
        "val_input_format": "npy_volume",
        "cache_dir": "outputs/persistent_cache",
        "use_persistent_cache": False,
        "train_num_workers": 2,
        "val_num_workers": 0,
        "prefetch_factor": 1,
        "persistent_workers": False,
        "pin_memory": False,
        "roi_size": [96, 96, 96],
        "target_spacing": [1.0, 1.0, 1.0],
        "intensity_mode": "ct",
        "ct_window": [-1000, 400],
        "profile_ratios": [1, 1, 3, 1, 1],
        "samples_per_volume": 8,
        "small_cc_voxels": 128,
        "large_cc_voxels": 4096,
        "overwrite_aux": False,
        "num_workers": 2,
    },
    "model": {
        "name": "hybrid_swin_sdf_core",
        "num_classes": 1,
        "img_size": [96, 96, 96],
        "in_channels": 1,
        "ct_in_channels": 1,
        "swin_feature_channels": 12,
        "two_d_feature_channels": 12,
        "fusion_channels": 24,
        "feature_size": 12,
        "use_checkpoint": True,
        "use_absolute_position_encoding": True,
        "absolute_position_scale_mm": 128.0,
    },
    "loss": {
        "warmup_epochs": 15,
        "stage2_start_epoch": 80,
        "dice_weight_start": 1.0,
        "dice_weight_end": 0.35,
        "ce_weight_start": 1.0,
        "ce_weight_end": 0.25,
        "tversky_weight_start": 0.0,
        "tversky_weight_end": 1.2,
        "boundary_weight_start": 0.0,
        "boundary_weight_end": 0.8,
        "sdf_weight_start": 0.0,
        "sdf_weight_end": 0.3,
        "core_weight_start": 0.0,
        "core_weight_end": 0.3,
        "center_weight_start": 0.0,
        "center_weight_end": 0.01,
        "center_source": "core",
        "center_use_smooth_l1": True,
        "tversky_alpha": 0.3,
        "tversky_beta": 0.7,
        "boundary_voxel_boost": 5.0,
        "dist_sigma": 0.2,
        "stage2_weight_power": 1.0,
        "max_stage2_weight": 8.0,
    },
    "train": {
        "epochs": 300,
        "batch_size": 1,
        "lr": 0.0001,
        "min_lr": 0.000001,
        "weight_decay": 0.00001,
        "scheduler": "cosine",
        "amp": True,
        "enforce_24gb_safety": True,
        "max_gpu_memory_gb": 24,
        "grad_accum_steps": 4,
        "grad_clip_norm": 12.0,
        "drop_last": False,
    },
    "eval": {
        "threshold": 0.5,
        "val_interval": 10,
        "roi_size": [96, 96, 96],
        "sw_batch_size": 1,
        "overlap": 0.0,
        "max_cases_per_epoch": 8,
        "validate_on_start": False,
        "validate_on_final": True,
        "compute_soft_dice": False,
        "empty_cache": True,
        "spacing": [1.0, 1.0, 1.0],
        "min_voxels": 16,
        "min_volume_mm3": None,
        "keep_largest": False,
    },
    "output": {
        "output_dir": DEFAULT_OUTPUT_DIR,
    },
}


# =========================
# 基础工具函数
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train task-adaptive 3D segmentation model.")

    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--amp", type=str, default=None, choices=["true", "false"])
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one synthetic train step to check CUDA memory, then exit.",
    )

    return parser.parse_args()


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        key = str(resolved)
        if key not in seen:
            out.append(resolved)
            seen.add(key)
    return out


def is_within_project(path: Union[str, Path]) -> bool:
    try:
        Path(path).expanduser().resolve().relative_to(PROJECT_ROOT)
        return True
    except ValueError:
        return False


def require_project_path(path: Union[str, Path], label: str = "path") -> Path:
    resolved = Path(path).expanduser().resolve()
    if not is_within_project(resolved):
        raise ValueError(f"{label} 必须位于项目目录内：{PROJECT_ROOT}，当前为：{resolved}")
    return resolved


def _candidate_base_dirs(
    base_dirs: Optional[Sequence[Union[str, Path]]] = None,
) -> List[Path]:
    bases: List[Path] = []
    if base_dirs is not None:
        bases.extend(Path(str(p)).expanduser() for p in base_dirs)
    bases.extend([PROJECT_ROOT, SCRIPT_DIR])
    cwd = Path.cwd()
    if is_within_project(cwd):
        bases.append(cwd)
    return _unique_paths(bases)


def path_candidates(
    value: Union[str, Path],
    base_dirs: Optional[Sequence[Union[str, Path]]] = None,
) -> List[Path]:
    raw = Path(str(value)).expanduser()
    if raw.is_absolute():
        return [require_project_path(raw)]
    return [require_project_path(base / raw) for base in _candidate_base_dirs(base_dirs)]


def resolve_existing_path(
    value: Union[str, Path],
    base_dirs: Optional[Sequence[Union[str, Path]]] = None,
) -> Path:
    """
    Resolve a possibly relative path without assuming the process cwd matches the
    project root. This is important when train.py is launched from src/.
    """
    candidates = path_candidates(value, base_dirs=base_dirs)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_runtime_path(
    value: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Resolve output/cache/data roots. Relative defaults are anchored at the
    project root, not at src/, so IDE working-directory changes do not move them.
    """
    raw = Path(str(value)).expanduser()
    if raw.is_absolute():
        return require_project_path(raw)
    base = Path(str(base_dir)).expanduser() if base_dir is not None else PROJECT_ROOT
    return require_project_path(base / raw)


def resolve_data_path(value: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> Path:
    raw = Path(str(value)).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    base = Path(str(base_dir)).expanduser() if base_dir is not None else PROJECT_ROOT
    return (base / raw).resolve()


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    requested_path = path
    resolved_path = resolve_existing_path(requested_path)
    if not resolved_path.exists():
        if str(requested_path) == DEFAULT_CONFIG_PATH:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            with resolved_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(DEFAULT_CONFIG, f, allow_unicode=True, sort_keys=False)
            print(f"未找到默认配置文件，已在项目目录内创建：{resolved_path}")
            return yaml.safe_load(yaml.safe_dump(DEFAULT_CONFIG))
        tried = "\n".join(f"  - {p}" for p in path_candidates(requested_path))
        raise FileNotFoundError(f"找不到配置文件：{resolved_path}\n已在项目目录内尝试：\n{tried}")
    with resolved_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件顶层必须是 dict，但当前是 {type(cfg)}")
    return cfg


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg.setdefault("data", {})
    cfg.setdefault("train", {})
    cfg.setdefault("output", {})

    if args.data_root is not None:
        cfg["data"]["data_root"] = args.data_root
    if args.output_dir is not None:
        cfg["output"]["output_dir"] = args.output_dir
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        cfg["train"]["lr"] = float(args.lr)
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = int(args.num_workers)
    if args.amp is not None:
        cfg["train"]["amp"] = args.amp.lower() == "true"
    if args.resume is not None:
        cfg["train"]["resume"] = args.resume

    return cfg


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def resolve_path(value: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return p.resolve()
    if base_dir is not None:
        return (base_dir / p).resolve()
    return p.resolve()


def read_csv_records(csv_path: Path, data_root: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV：{csv_path}")

    records: List[Dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 没有表头：{csv_path}")

        for row in reader:
            item: Dict[str, Any] = {k: v for k, v in row.items() if k is not None}

            # 旧项目：npz_path
            if "npz_path" in item and str(item["npz_path"]).strip():
                item["npz_path"] = str(resolve_path(item["npz_path"], data_root))

            # 新任务：image,label
            if "image" in item and str(item["image"]).strip():
                item["image"] = str(resolve_path(item["image"], data_root))
            if "label" in item and str(item["label"]).strip():
                item["label"] = str(resolve_path(item["label"], data_root))
            if "mask" in item and str(item["mask"]).strip():
                item["mask"] = str(resolve_path(item["mask"], data_root))

            records.append(item)

    if len(records) == 0:
        raise RuntimeError(f"CSV 中没有样本：{csv_path}")

    return records


def read_raw_csv_rows(csv_path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not csv_path.exists():
        return [], []

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        rows = [{k: v for k, v in row.items() if k is not None} for row in reader]
        return rows, list(reader.fieldnames)


def write_raw_csv_rows(csv_path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(csv_path.parent)

    ordered_fields: List[str] = []
    seen = set()
    for name in fieldnames:
        if name not in seen:
            ordered_fields.append(name)
            seen.add(name)
    for row in rows:
        for name in row.keys():
            if name not in seen:
                ordered_fields.append(name)
                seen.add(name)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in ordered_fields})


def _volume_pair_key(path: Path) -> str:
    stem = path.stem
    for suffix in ("_image", "-image", "_img", "-img", "_label", "-label", "_mask", "-mask"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def discover_npy_volume_rows(data_root: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    images_dir = data_root / "images"
    labels_dir = data_root / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return [], []

    image_paths = sorted(images_dir.glob("*.npy"))
    label_paths = sorted(labels_dir.glob("*.npy"))
    labels_by_key = {_volume_pair_key(path): path for path in label_paths}

    rows: List[Dict[str, Any]] = []
    for image_path in image_paths:
        key = _volume_pair_key(image_path)
        label_path = labels_by_key.get(key)
        if label_path is None:
            continue
        rows.append(
            {
                "image": image_path.relative_to(data_root).as_posix(),
                "label": label_path.relative_to(data_root).as_posix(),
                "case_id": key,
            }
        )

    return rows, ["image", "label", "case_id"]


def parse_split_ratios(data_cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    raw = data_cfg.get("split_ratios", [0.8, 0.1, 0.1])
    if isinstance(raw, dict):
        values = [
            float(raw.get("train", 0.8)),
            float(raw.get("val", 0.1)),
            float(raw.get("test", 0.1)),
        ]
    else:
        values = [float(v) for v in raw]

    if len(values) != 3:
        raise ValueError("data.split_ratios must contain train/val/test ratios.")
    if any(v < 0 for v in values):
        raise ValueError("data.split_ratios cannot contain negative values.")

    total = sum(values)
    if total <= 0:
        raise ValueError("data.split_ratios sum must be greater than 0.")

    return values[0] / total, values[1] / total, values[2] / total


def split_group_key(row: Dict[str, Any]) -> str:
    for key in ("patient_id", "case_id", "scan_id"):
        value = str(row.get(key, "")).strip()
        if value:
            return f"{key}:{value}"

    for key in ("npz_path", "image", "label", "mask"):
        value = str(row.get(key, "")).strip()
        if not value:
            continue
        match = re.search(r"LIDC-IDRI-\d+", value)
        if match:
            return f"case:{match.group(0)}"
        return f"{key}:{Path(value).stem}"

    return "row:" + json.dumps(row, sort_keys=True, ensure_ascii=False)


def row_identity(row: Dict[str, Any]) -> Tuple[str, ...]:
    keys = (
        "npz_path",
        "image",
        "label",
        "mask",
        "patient_id",
        "case_id",
        "scan_id",
        "nodule_id",
        "source_nodule_id",
    )
    return tuple(str(row.get(k, "")).strip() for k in keys)


def split_counts(num_groups: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    if num_groups <= 0:
        return 0, 0, 0

    exact = [num_groups * r for r in ratios]
    counts = [int(math.floor(v)) for v in exact]
    remainder = num_groups - sum(counts)
    order = sorted(range(3), key=lambda i: exact[i] - counts[i], reverse=True)
    for i in order[:remainder]:
        counts[i] += 1

    for i, ratio in enumerate(ratios):
        if num_groups >= 3 and ratio > 0 and counts[i] == 0:
            donor = max(range(3), key=lambda j: counts[j])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[i] += 1

    return counts[0], counts[1], counts[2]


def split_rows_by_group(
    rows: Sequence[Dict[str, Any]],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(split_group_key(row), []).append(row)

    group_keys = sorted(grouped.keys())
    rng = random.Random(int(seed))
    rng.shuffle(group_keys)

    n_train, n_val, _ = split_counts(len(group_keys), ratios)
    train_keys = set(group_keys[:n_train])
    val_keys = set(group_keys[n_train:n_train + n_val])
    test_keys = set(group_keys[n_train + n_val:])

    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    for key in group_keys:
        if key in train_keys:
            train_rows.extend(grouped[key])
        elif key in val_keys:
            val_rows.extend(grouped[key])
        elif key in test_keys:
            test_rows.extend(grouped[key])

    return train_rows, val_rows, test_rows


def build_auto_split_csvs(
    data_cfg: Dict[str, Any],
    data_root: Path,
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
) -> Tuple[Path, Path, Path]:
    split_dir = resolve_runtime_path(data_cfg.get("split_dir", "outputs/splits"))
    out_train = split_dir / "train.csv"
    out_val = split_dir / "val.csv"
    out_test = split_dir / "test.csv"

    all_csv_value = data_cfg.get("all_csv", None)
    source_paths: List[Path] = []
    if all_csv_value is not None and str(all_csv_value).strip():
        source_paths.append(resolve_path(all_csv_value, data_root))
    else:
        source_paths.extend([p for p in (train_csv, val_csv, test_csv) if p.exists()])

    all_rows: List[Dict[str, Any]] = []
    fieldnames: List[str] = []
    seen_rows = set()
    for source_path in source_paths:
        rows, source_fields = read_raw_csv_rows(source_path)
        for name in source_fields:
            if name not in fieldnames:
                fieldnames.append(name)
        for row in rows:
            identity = row_identity(row)
            if identity in seen_rows:
                continue
            seen_rows.add(identity)
            all_rows.append(row)

    if not all_rows:
        all_rows, fieldnames = discover_npy_volume_rows(data_root)

    if not all_rows:
        tried = "\n".join(f"  - {p}" for p in [train_csv, val_csv, test_csv])
        raise FileNotFoundError(
            "No source CSV rows or paired npy volumes found for auto split:\n"
            f"{tried}\n"
            f"Expected npy layout: {data_root / 'images'} and {data_root / 'labels'}"
        )

    ratios = parse_split_ratios(data_cfg)
    seed = int(data_cfg.get("split_seed", data_cfg.get("seed", DEFAULT_CONFIG.get("seed", 42))))
    train_rows, val_rows, test_rows = split_rows_by_group(all_rows, ratios, seed)

    write_raw_csv_rows(out_train, train_rows, fieldnames)
    write_raw_csv_rows(out_val, val_rows, fieldnames)
    write_raw_csv_rows(out_test, test_rows, fieldnames)

    total = len(train_rows) + len(val_rows) + len(test_rows)
    print(
        "Auto split CSVs: "
        f"train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}, total={total}, "
        f"ratios={ratios[0]:.2f}/{ratios[1]:.2f}/{ratios[2]:.2f}, dir={split_dir}"
    )

    return out_train, out_val, out_test


def is_positive_record(row: Dict[str, Any]) -> Optional[bool]:
    value = row.get("sample_label", None)
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value) > 0
    except ValueError:
        return str(value).strip().lower() in {"pos", "positive", "true", "foreground"}


def resample_positive_negative_records(
    records: Sequence[Dict[str, Any]],
    pos_neg_ratio: float,
    seed: int,
) -> List[Dict[str, Any]]:
    if pos_neg_ratio <= 0:
        return list(records)

    positives: List[Dict[str, Any]] = []
    negatives: List[Dict[str, Any]] = []
    unknowns: List[Dict[str, Any]] = []
    for row in records:
        is_pos = is_positive_record(row)
        if is_pos is None:
            unknowns.append(row)
        elif is_pos:
            positives.append(row)
        else:
            negatives.append(row)

    if unknowns or not positives or not negatives:
        return list(records)

    rng = random.Random(int(seed))
    target_pos = max(1, int(round(len(negatives) * float(pos_neg_ratio))))

    strata: Dict[str, List[Dict[str, Any]]] = {}
    for row in positives:
        patch_type = str(row.get("patch_type", "positive") or "positive").strip() or "positive"
        size_class = str(row.get("nodule_size_class", "unknown") or "unknown").strip() or "unknown"
        strata.setdefault(f"{patch_type}:{size_class}", []).append(row)

    for rows in strata.values():
        rng.shuffle(rows)

    sampled_pos: List[Dict[str, Any]] = []
    if strata:
        stratum_keys = sorted(strata.keys())
        cursors = {key: 0 for key in stratum_keys}
        while len(sampled_pos) < target_pos:
            made_progress = False
            rng.shuffle(stratum_keys)
            for key in stratum_keys:
                rows = strata[key]
                if not rows:
                    continue
                cursor = cursors[key]
                sampled_pos.append(rows[cursor % len(rows)])
                cursors[key] = cursor + 1
                made_progress = True
                if len(sampled_pos) >= target_pos:
                    break
            if not made_progress:
                break
    elif len(positives) >= target_pos:
        sampled_pos = rng.sample(positives, target_pos)
    else:
        sampled_pos = list(positives)
        sampled_pos.extend(rng.choice(positives) for _ in range(target_pos - len(positives)))

    sampled = sampled_pos + list(negatives)
    rng.shuffle(sampled)
    print(
        "Train positive/negative sampling: "
        f"positive={len(sampled_pos)}, negative={len(negatives)}, ratio={pos_neg_ratio:.2f}:1, "
        f"positive_strata={len(strata)}"
    )
    return sampled


def find_split_data_root(
    train_csv_name: Union[str, Path] = "train.csv",
    val_csv_name: Union[str, Path] = "val.csv",
) -> Optional[Path]:
    train_rel = Path(str(train_csv_name))
    val_rel = Path(str(val_csv_name))
    if train_rel.is_absolute() or val_rel.is_absolute():
        return None

    candidates: List[Path] = []
    for train_csv in PROJECT_ROOT.rglob(train_rel.name):
        if any(part in {"__pycache__", ".git"} for part in train_csv.parts):
            continue
        if train_csv.name != train_rel.name:
            continue
        root = train_csv.parent
        if (root / val_rel).exists():
            candidates.append(root)

    if not candidates:
        return None

    candidates = _unique_paths(candidates)
    candidates.sort(key=lambda p: (len(p.relative_to(PROJECT_ROOT).parts), str(p)))
    return candidates[0]


def resolve_split_csv_paths(data_cfg: Dict[str, Any]) -> Tuple[Path, Path, Path]:
    data_root_value = data_cfg.get("data_root", DEFAULT_DATA_ROOT)
    train_csv_value = data_cfg.get("train_csv", "train.csv")
    val_csv_value = data_cfg.get("val_csv", "val.csv")
    test_csv_value = data_cfg.get("test_csv", "test.csv")

    data_root = resolve_data_path(data_root_value)
    train_csv = resolve_path(train_csv_value, data_root)
    val_csv = resolve_path(val_csv_value, data_root)
    test_csv = resolve_path(test_csv_value, data_root)

    if bool(data_cfg.get("auto_split", False)):
        split_train, split_val, split_test = build_auto_split_csvs(data_cfg, data_root, train_csv, val_csv, test_csv)
        data_cfg["train_csv"] = str(split_train)
        data_cfg["val_csv"] = str(split_val)
        data_cfg["test_csv"] = str(split_test)
        return data_root, split_train, split_val

    if train_csv.exists() and val_csv.exists():
        return data_root, train_csv, val_csv

    using_default_split_paths = (
        str(data_root_value) == DEFAULT_DATA_ROOT
        and str(train_csv_value) == "train.csv"
        and str(val_csv_value) == "val.csv"
    )
    detected_root = find_split_data_root(train_csv_value, val_csv_value) if using_default_split_paths else None
    if detected_root is not None:
        detected_train = resolve_path(train_csv_value, detected_root)
        detected_val = resolve_path(val_csv_value, detected_root)
        print(f"默认 data/train.csv 未找到，已在项目内自动使用数据目录：{detected_root}")
        return detected_root, detected_train, detected_val

    tried = "\n".join(f"  - {p}" for p in [train_csv, val_csv])
    hint = (
        f"项目目录：{PROJECT_ROOT}\n"
        "请把 train.csv 和 val.csv 放在项目内同一数据目录中，"
        "或修改 configs/task_adaptive.yaml 的 data.data_root / data.train_csv / data.val_csv。"
    )
    raise FileNotFoundError(f"找不到训练/验证 CSV：\n{tried}\n{hint}")


def infer_input_format(records: Sequence[Dict[str, Any]], configured: str = "auto") -> str:
    configured = str(configured).lower().strip()
    if configured in {"npz", "npz_patch", "patch"}:
        return "npz_patch"
    if configured in {"npy", "npy_volume", "volume_npy"}:
        return "npy_volume"
    if configured in {"image_label", "nifti", "volume"}:
        return "image_label"
    if configured != "auto":
        raise ValueError(f"未知 input_format：{configured}")

    first = records[0]
    if "npz_path" in first and str(first.get("npz_path", "")).strip():
        return "npz_patch"
    if "image" in first and ("label" in first or "mask" in first):
        image_value = str(first.get("image", "")).lower()
        label_value = str(first.get("label", first.get("mask", ""))).lower()
        if image_value.endswith(".npy") and label_value.endswith(".npy"):
            return "npy_volume"
        return "image_label"

    raise ValueError(
        "无法自动判断数据格式。CSV 需要：\n"
        "1. npz patch 格式：npz_path 列；或\n"
        "2. 全体积格式：image,label 两列。"
    )


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(v, device) for v in batch)
    return batch


def model_position_kwargs(batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    kwargs: Dict[str, torch.Tensor] = {}
    position_coords = batch.get(POSITION_COORDS_KEY)
    patch_center = batch.get(PATCH_CENTER_KEY)
    spacing = batch.get(SPACING_KEY)
    if torch.is_tensor(position_coords):
        kwargs["position_coords"] = position_coords.float()
    if torch.is_tensor(patch_center):
        kwargs["patch_center_dhw"] = patch_center.float()
    if torch.is_tensor(spacing):
        kwargs["spacing_mm"] = spacing.float()
    return kwargs


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def format_float(x: Any) -> str:
    if x is None:
        return ""
    try:
        x = float(x)
    except Exception:
        return str(x)
    if math.isnan(x):
        return "nan"
    return f"{x:.8f}"


def init_csv_log(path: Path, resume: bool = False) -> None:
    if resume and path.exists():
        return
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writeheader()


def append_csv_log(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writerow({k: row.get(k, "") for k in LOG_COLUMNS})


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(str(k).startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {str(k)[7:] if str(k).startswith("module.") else str(k): v for k, v in state_dict.items()}


# =========================
# 自定义 Transform：npz 加载与辅助目标构建
# =========================

def _ensure_channel_first_np(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 3:
        x = x[None]
    elif x.ndim == 4:
        # 默认认为已经是 [C,D,H,W]
        pass
    elif x.ndim == 5 and x.shape[0] == 1:
        x = x[0]
    else:
        raise ValueError(f"{name} 只支持 [D,H,W]、[C,D,H,W] 或 [1,C,D,H,W]，当前 shape={x.shape}")
    return x


class LoadNPZPatchd(MapTransform):
    """
    读取旧项目 npz patch。

    输入 data 至少包含：
        npz_path

    npz 内支持：
        image
        mask 或 label
        sdf / dist_map
        boundary
        core
    """

    def __init__(self, npz_key: str = "npz_path") -> None:
        super().__init__(keys=[npz_key], allow_missing_keys=False)
        self.npz_key = npz_key

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        npz_path = Path(str(d[self.npz_key])).expanduser().resolve()
        if not npz_path.exists():
            raise FileNotFoundError(f"找不到 npz patch：{npz_path}")

        with np.load(npz_path) as npz:
            keys = set(npz.files)
            if IMAGE_KEY not in keys:
                raise KeyError(f"npz 缺少 image：{npz_path}")
            if LABEL_KEY in keys:
                label_arr = npz[LABEL_KEY]
            elif MASK_ALIAS_KEY in keys:
                label_arr = npz[MASK_ALIAS_KEY]
            else:
                raise KeyError(f"npz 缺少 label/mask：{npz_path}")

            d[IMAGE_KEY] = _ensure_channel_first_np(npz[IMAGE_KEY], IMAGE_KEY).astype(np.float32)
            d[LABEL_KEY] = _ensure_channel_first_np(label_arr, LABEL_KEY).astype(np.int64)
            d[MASK_ALIAS_KEY] = d[LABEL_KEY]

            if "sdf" in keys:
                d["sdf"] = _ensure_channel_first_np(npz["sdf"], "sdf").astype(np.float32)
            if "dist_map" in keys:
                d["dist_map"] = _ensure_channel_first_np(npz["dist_map"], "dist_map").astype(np.float32)
            if "boundary" in keys:
                d["boundary"] = _ensure_channel_first_np(npz["boundary"], "boundary").astype(np.float32)
            if "core" in keys:
                d["core"] = _ensure_channel_first_np(npz["core"], "core").astype(np.float32)
            if PATCH_CENTER_KEY in keys:
                d[PATCH_CENTER_KEY] = np.asarray(npz[PATCH_CENTER_KEY], dtype=np.float32)
            if SPACING_KEY in keys:
                d[SPACING_KEY] = np.asarray(npz[SPACING_KEY], dtype=np.float32)
            if POSITION_COORDS_KEY in keys and PATCH_CENTER_KEY not in keys:
                d[POSITION_COORDS_KEY] = np.asarray(npz[POSITION_COORDS_KEY], dtype=np.float32)

        d.setdefault("case_id", npz_path.stem)
        d.setdefault("nodule_id", "")
        return d


def build_absolute_coord_grid(
    shape_dhw: Sequence[int],
    spacing_mm: Sequence[float] = (1.0, 1.0, 1.0),
    dtype: Any = np.float32,
) -> np.ndarray:
    depth, height, width = (int(v) for v in shape_dhw)
    spacing = np.asarray(spacing_mm, dtype=np.float32)
    z = np.arange(depth, dtype=np.float32) * float(spacing[0])
    y = np.arange(height, dtype=np.float32) * float(spacing[1])
    x = np.arange(width, dtype=np.float32) * float(spacing[2])
    zz = np.broadcast_to(z[:, None, None], (depth, height, width))
    yy = np.broadcast_to(y[None, :, None], (depth, height, width))
    xx = np.broadcast_to(x[None, None, :], (depth, height, width))
    return np.stack([zz, yy, xx], axis=0).astype(dtype, copy=False)


class LoadNumpyVolumed(MapTransform):
    """Load whole-case npy image/label volumes from images/ and labels/."""

    def __init__(
        self,
        image_key: str = IMAGE_KEY,
        label_key: str = LABEL_KEY,
        spacing_mm: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__(keys=[image_key, label_key], allow_missing_keys=False)
        self.image_key = image_key
        self.label_key = label_key
        self.spacing_mm = tuple(float(v) for v in spacing_mm)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        image_path = Path(str(d[self.image_key])).expanduser()
        label_path = Path(str(d[self.label_key])).expanduser()
        if not image_path.exists():
            raise FileNotFoundError(f"找不到 npy image：{image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"找不到 npy label：{label_path}")

        image = _ensure_channel_first_np(np.load(image_path), self.image_key).astype(np.float32)
        label = _ensure_channel_first_np(np.load(label_path), self.label_key).astype(np.int64)
        if image.shape[1:] != label.shape[1:]:
            raise ValueError(
                f"image/label 空间尺寸不一致：image={image.shape}, label={label.shape}, "
                f"image_path={image_path}, label_path={label_path}"
            )

        d[self.image_key] = image
        d[self.label_key] = label
        d[MASK_ALIAS_KEY] = label
        d[SPACING_KEY] = np.asarray(self.spacing_mm, dtype=np.float32)
        d.setdefault("case_id", image_path.stem)
        d.setdefault("nodule_id", "")
        return d


class AddPositionCoordsd(MapTransform):
    """Build absolute coordinate maps for existing patch arrays when metadata is available."""

    def __init__(
        self,
        image_key: str = IMAGE_KEY,
        center_key: str = PATCH_CENTER_KEY,
        spacing_key: str = SPACING_KEY,
        out_key: str = POSITION_COORDS_KEY,
    ) -> None:
        super().__init__(keys=[image_key], allow_missing_keys=False)
        self.image_key = image_key
        self.center_key = center_key
        self.spacing_key = spacing_key
        self.out_key = out_key

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if self.center_key not in d:
            return d

        image = np.asarray(d[self.image_key])
        if image.ndim == 3:
            shape_dhw = image.shape
        elif image.ndim == 4:
            shape_dhw = image.shape[1:]
        else:
            raise ValueError(f"{self.image_key} must be [D,H,W] or [C,D,H,W], got {image.shape}")

        center = np.asarray(d[self.center_key], dtype=np.float32).reshape(-1)[:3]
        spacing = np.asarray(d.get(self.spacing_key, np.ones(3, dtype=np.float32)), dtype=np.float32).reshape(-1)[:3]
        patch_shape = np.asarray(shape_dhw, dtype=np.float32)
        start = center - np.floor(patch_shape / 2.0)

        coords = build_absolute_coord_grid(shape_dhw, spacing_mm=spacing, dtype=np.float32)
        coords += (start * spacing).reshape(3, 1, 1, 1).astype(np.float32)
        d[self.out_key] = coords
        return d


class EnsureMaskAliasd(MapTransform):
    """保证 batch 中同时存在 label 和 mask，兼容旧 loss / 新 loss。"""

    def __init__(self, label_key: str = LABEL_KEY) -> None:
        super().__init__(keys=[label_key], allow_missing_keys=False)
        self.label_key = label_key

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        if self.label_key != LABEL_KEY:
            d[LABEL_KEY] = d[self.label_key]
        d[MASK_ALIAS_KEY] = d[LABEL_KEY]
        return d


class BuildAuxTargetsd(MapTransform):
    """
    由 label 构建 boundary / dist_map / component_weight / sample_label。

    sample_label 编码：
        0 = 背景
        1 = 普通前景
        2 = 边界 / hard 区域
        3 = 小连通域
        4 = 大连通域
    """

    def __init__(
        self,
        label_key: str = LABEL_KEY,
        overwrite_existing: bool = False,
        small_cc_voxels: int = 128,
        large_cc_voxels: int = 4096,
        sdf_clip: float = 20.0,
        inv_weight_alpha: float = 0.5,
        max_inv_weight: float = 8.0,
    ) -> None:
        super().__init__(keys=[label_key], allow_missing_keys=False)
        self.label_key = label_key
        self.overwrite_existing = bool(overwrite_existing)
        self.small_cc_voxels = int(small_cc_voxels)
        self.large_cc_voxels = int(large_cc_voxels)
        self.sdf_clip = float(sdf_clip)
        self.inv_weight_alpha = float(inv_weight_alpha)
        self.max_inv_weight = float(max_inv_weight)

    @staticmethod
    def _to_3d(x: Any) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 4 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 3:
            raise ValueError(f"label 必须是 [D,H,W] 或 [1,D,H,W]，当前 shape={x.shape}")
        return x

    @staticmethod
    def _add_ch(x: np.ndarray, dtype: Any) -> np.ndarray:
        return x[None].astype(dtype, copy=False)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        label_np = self._to_3d(d[self.label_key])
        label_int = label_np.astype(np.int64, copy=False)
        fg = label_int > 0

        structure_26 = generate_binary_structure(rank=3, connectivity=3)
        structure_6 = generate_binary_structure(rank=3, connectivity=1)

        # ---------- boundary ----------
        need_boundary = self.overwrite_existing or "boundary" not in d
        if need_boundary:
            if np.any(fg):
                dilated = binary_dilation(fg, structure=structure_6)
                eroded = binary_erosion(fg, structure=structure_6, border_value=0)
                boundary = (dilated ^ eroded).astype(np.float32)
            else:
                boundary = np.zeros_like(fg, dtype=np.float32)
            d["boundary"] = self._add_ch(boundary, np.float32)
        else:
            boundary = self._to_3d(d["boundary"]).astype(np.float32)

        # ---------- signed distance map ----------
        need_sdf = self.overwrite_existing or ("sdf" not in d and "dist_map" not in d)
        if need_sdf:
            if np.any(fg) and np.any(~fg):
                dist_out = distance_transform_edt(~fg)
                dist_in = distance_transform_edt(fg)
                sdf = dist_out - dist_in
                sdf = np.clip(sdf, -self.sdf_clip, self.sdf_clip) / max(self.sdf_clip, 1e-6)
                sdf = sdf.astype(np.float32)
            else:
                sdf = np.zeros_like(label_np, dtype=np.float32)
            d["sdf"] = self._add_ch(sdf, np.float32)
            d["dist_map"] = self._add_ch(sdf, np.float32)
        else:
            if "dist_map" not in d and "sdf" in d:
                d["dist_map"] = d["sdf"]
            if "sdf" not in d and "dist_map" in d:
                d["sdf"] = d["dist_map"]

        # ---------- connected component inverse weight ----------
        component_weight = np.ones_like(label_np, dtype=np.float32)
        sample_label = np.zeros_like(label_int, dtype=np.uint8)

        if np.any(fg):
            cc, num_cc = cc_label(fg, structure=structure_26)
            cc_sizes = np.bincount(cc.reshape(-1))
            cc_sizes[0] = 0

            valid_sizes = cc_sizes[cc_sizes > 0]
            ref_size = float(np.median(valid_sizes)) if valid_sizes.size > 0 else 1.0

            lut = np.ones_like(cc_sizes, dtype=np.float32)
            for cc_id in range(1, num_cc + 1):
                size = max(float(cc_sizes[cc_id]), 1.0)
                w = (ref_size / size) ** self.inv_weight_alpha
                lut[cc_id] = np.clip(w, 1.0, self.max_inv_weight)

            component_weight = lut[cc].astype(np.float32)
            sample_label[fg] = 1

            small_lut = cc_sizes <= self.small_cc_voxels
            large_lut = cc_sizes >= self.large_cc_voxels
            small_lut[0] = False
            large_lut[0] = False

            sample_label[small_lut[cc]] = 3
            sample_label[large_lut[cc]] = 4

        # 边界难例优先级最高
        sample_label[np.asarray(boundary) > 0] = 2

        d["component_weight"] = self._add_ch(component_weight, np.float32)
        d["sample_label"] = self._add_ch(sample_label, np.int64)

        if "core" not in d:
            # 没有 core 标签时给一个全 0，loss 会根据 outputs 是否有 core_logits 决定是否使用
            d["core"] = self._add_ch(np.zeros_like(label_np, dtype=np.float32), np.float32)

        d[MASK_ALIAS_KEY] = d[LABEL_KEY]
        return d


# =========================
# Transform 构建
# =========================

def build_transforms(
    input_format: str,
    roi_size: Sequence[int],
    target_spacing: Sequence[float],
    intensity_mode: str,
    ct_window: Sequence[float],
    profile_ratios: Sequence[float],
    samples_per_volume: int,
    small_cc_voxels: int,
    large_cc_voxels: int,
    overwrite_aux: bool,
    is_train: bool,
) -> Compose:
    roi_size = tuple(int(v) for v in roi_size)
    target_spacing = tuple(float(v) for v in target_spacing)
    profile_ratios = list(float(v) for v in profile_ratios)
    rotate_spatial_axes = (0, 1)

    aux_builder = BuildAuxTargetsd(
        label_key=LABEL_KEY,
        overwrite_existing=overwrite_aux,
        small_cc_voxels=small_cc_voxels,
        large_cc_voxels=large_cc_voxels,
    )

    if input_format == "npz_patch":
        keys = [IMAGE_KEY, LABEL_KEY, MASK_ALIAS_KEY, "sdf", "dist_map", "boundary", "core", "component_weight", "sample_label"]
        spatial_keys = list(keys) + [POSITION_COORDS_KEY]
        typed_keys = spatial_keys + [PATCH_CENTER_KEY, SPACING_KEY]
        transforms: List[Any] = [
            LoadNPZPatchd(npz_key="npz_path"),
            AddPositionCoordsd(),
        ]
        transforms += [
            EnsureMaskAliasd(label_key=LABEL_KEY),
            aux_builder,
            EnsureTyped(keys=typed_keys, track_meta=False, allow_missing_keys=True),
        ]

        # npz patch 通常已经完成 spacing/window/ROI 对齐，不再做全体积切块。
        # 如确实需要在大 npz volume 上切块，可把 data.input_format 改为 image_label 或自行打开 enable_patch_crop。
        if is_train:
            transforms += [
                RandFlipd(keys=spatial_keys, spatial_axis=0, prob=0.5, allow_missing_keys=True),
                RandFlipd(keys=spatial_keys, spatial_axis=1, prob=0.5, allow_missing_keys=True),
                RandFlipd(keys=spatial_keys, spatial_axis=2, prob=0.5, allow_missing_keys=True),
                RandRotate90d(keys=spatial_keys, prob=0.2, max_k=3, spatial_axes=rotate_spatial_axes, allow_missing_keys=True),
                RandGaussianNoised(keys=[IMAGE_KEY], prob=0.15, mean=0.0, std=0.01),
                RandScaleIntensityd(keys=[IMAGE_KEY], factors=0.10, prob=0.30),
                RandShiftIntensityd(keys=[IMAGE_KEY], offsets=0.10, prob=0.30),
            ]
        return Compose(transforms)

    if input_format == "npy_volume":
        base_keys = [IMAGE_KEY, LABEL_KEY, MASK_ALIAS_KEY]
        crop_keys = [
            IMAGE_KEY,
            LABEL_KEY,
            MASK_ALIAS_KEY,
            POSITION_COORDS_KEY,
            "sdf",
            "dist_map",
            "boundary",
            "core",
            "component_weight",
            "sample_label",
        ]
        deterministic: List[Any] = [
            LoadNumpyVolumed(spacing_mm=target_spacing),
            EnsureMaskAliasd(label_key=LABEL_KEY),
        ]
        if intensity_mode.lower() == "ct":
            deterministic.append(
                ScaleIntensityRanged(
                    keys=[IMAGE_KEY],
                    a_min=float(ct_window[0]),
                    a_max=float(ct_window[1]),
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                )
            )
        else:
            deterministic.append(
                ScaleIntensityRangePercentilesd(
                    keys=[IMAGE_KEY],
                    lower=1.0,
                    upper=99.0,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                )
            )

        if is_train:
            deterministic.append(
                CropForegroundd(
                    keys=base_keys,
                    source_key=IMAGE_KEY,
                    margin=8,
                    allow_smaller=True,
                )
            )

        if is_train:
            deterministic.append(aux_builder)
            typed_keys = crop_keys + [SPACING_KEY]
        else:
            typed_keys = base_keys + [SPACING_KEY]

        deterministic.append(
            EnsureTyped(keys=typed_keys, track_meta=False, allow_missing_keys=True)
        )

        if not is_train:
            return Compose(deterministic)

        random_part = [
            RandCropByLabelClassesd(
                keys=crop_keys,
                label_key="sample_label",
                spatial_size=roi_size,
                ratios=profile_ratios,
                num_classes=5,
                num_samples=int(samples_per_volume),
                allow_smaller=False,
                allow_missing_keys=True,
            ),
            RandFlipd(keys=crop_keys, spatial_axis=0, prob=0.5, allow_missing_keys=True),
            RandFlipd(keys=crop_keys, spatial_axis=1, prob=0.5, allow_missing_keys=True),
            RandFlipd(keys=crop_keys, spatial_axis=2, prob=0.5, allow_missing_keys=True),
            RandRotate90d(keys=crop_keys, prob=0.2, max_k=3, spatial_axes=rotate_spatial_axes, allow_missing_keys=True),
            RandGaussianNoised(keys=[IMAGE_KEY], prob=0.15, mean=0.0, std=0.01),
            RandScaleIntensityd(keys=[IMAGE_KEY], factors=0.10, prob=0.30),
            RandShiftIntensityd(keys=[IMAGE_KEY], offsets=0.10, prob=0.30),
        ]
        return Compose(deterministic + random_part)

    if input_format != "image_label":
        raise ValueError(f"未知 input_format：{input_format}")

    label_source_key = LABEL_KEY
    # 如果 CSV 里是 mask 列，需要在 records 里统一映射，后面 build_loaders 处理。

    base_keys = [IMAGE_KEY, label_source_key]
    crop_keys = [IMAGE_KEY, LABEL_KEY, MASK_ALIAS_KEY, "sdf", "dist_map", "boundary", "core", "component_weight", "sample_label"]

    deterministic: List[Any] = [
        LoadImaged(keys=base_keys),
        EnsureChannelFirstd(keys=base_keys),
        Orientationd(keys=base_keys, axcodes="RAS"),
        Spacingd(
            keys=base_keys,
            pixdim=target_spacing,
            mode=("bilinear", "nearest"),
        ),
    ]

    if intensity_mode.lower() == "ct":
        deterministic.append(
            ScaleIntensityRanged(
                keys=[IMAGE_KEY],
                a_min=float(ct_window[0]),
                a_max=float(ct_window[1]),
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        )
    else:
        deterministic.append(
            ScaleIntensityRangePercentilesd(
                keys=[IMAGE_KEY],
                lower=1.0,
                upper=99.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        )

    deterministic += [
        EnsureMaskAliasd(label_key=LABEL_KEY),
        CropForegroundd(
            keys=[IMAGE_KEY, LABEL_KEY, MASK_ALIAS_KEY],
            source_key=IMAGE_KEY,
            margin=8,
            allow_smaller=True,
        ),
        aux_builder,
        EnsureTyped(keys=crop_keys, track_meta=False, allow_missing_keys=True),
    ]

    if not is_train:
        return Compose(deterministic)

    random_part = [
        RandCropByLabelClassesd(
            keys=crop_keys,
            label_key="sample_label",
            spatial_size=roi_size,
            ratios=profile_ratios,
            num_classes=5,
            num_samples=int(samples_per_volume),
            allow_smaller=False,
            allow_missing_keys=True,
        ),
        RandFlipd(keys=crop_keys, spatial_axis=0, prob=0.5, allow_missing_keys=True),
        RandFlipd(keys=crop_keys, spatial_axis=1, prob=0.5, allow_missing_keys=True),
        RandFlipd(keys=crop_keys, spatial_axis=2, prob=0.5, allow_missing_keys=True),
        RandRotate90d(keys=crop_keys, prob=0.2, max_k=3, spatial_axes=rotate_spatial_axes, allow_missing_keys=True),
        RandGaussianNoised(keys=[IMAGE_KEY], prob=0.15, mean=0.0, std=0.01),
        RandScaleIntensityd(keys=[IMAGE_KEY], factors=0.10, prob=0.30),
        RandShiftIntensityd(keys=[IMAGE_KEY], offsets=0.10, prob=0.30),
    ]

    return Compose(deterministic + random_part)


def normalize_records_for_image_label(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把 mask 列映射为 label，方便 MONAI LoadImaged 使用统一 key。"""
    out: List[Dict[str, Any]] = []
    for r in records:
        d = dict(r)
        if LABEL_KEY not in d and MASK_ALIAS_KEY in d:
            d[LABEL_KEY] = d[MASK_ALIAS_KEY]
        if IMAGE_KEY not in d or LABEL_KEY not in d:
            raise KeyError("image_label 格式的 CSV 必须包含 image,label 两列，或 image,mask 两列。")
        out.append(d)
    return out


def _has_image_label_columns(rows: Sequence[Dict[str, Any]]) -> bool:
    if not rows:
        return False
    return all(
        str(row.get(IMAGE_KEY, "")).strip()
        and (str(row.get(LABEL_KEY, "")).strip() or str(row.get(MASK_ALIAS_KEY, "")).strip())
        for row in rows
    )


def _volume_csv_candidates(data_root: Path, csv_value: Any) -> List[Path]:
    raw = Path(str(csv_value)).expanduser()
    if raw.is_absolute():
        return [raw.resolve()]
    return _unique_paths([data_root / raw, data_root.parent / raw])


def load_npy_volume_records(data_root: Path, csv_value: Any = "val_volume.csv") -> List[Dict[str, Any]]:
    for csv_path in _volume_csv_candidates(data_root, csv_value):
        if not csv_path.exists():
            continue

        rows = read_csv_records(csv_path, csv_path.parent)
        if _has_image_label_columns(rows):
            return normalize_records_for_image_label(rows)

        print(
            f"[WARNING] 整例验证 CSV 不是 image/label 格式，已忽略并尝试扫描 volumes: {csv_path}"
        )
        break

    rows, _ = discover_npy_volume_rows(data_root)
    if not rows:
        raise FileNotFoundError(
            f"找不到整例验证 CSV：{csv_path}，也没有发现 {data_root / 'images'} 和 {data_root / 'labels'} 下的 npy 配对。"
        )

    resolved: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item[IMAGE_KEY] = str(resolve_path(item[IMAGE_KEY], data_root))
        item[LABEL_KEY] = str(resolve_path(item[LABEL_KEY], data_root))
        resolved.append(item)
    return resolved


def build_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, str]:
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    data_cfg.setdefault("seed", cfg.get("seed", DEFAULT_CONFIG.get("seed", 42)))

    data_root, train_csv, val_csv = resolve_split_csv_paths(data_cfg)

    train_records = read_csv_records(train_csv, data_root)
    default_val_records = read_csv_records(val_csv, data_root)

    train_input_format = infer_input_format(
        train_records,
        data_cfg.get("train_input_format", data_cfg.get("input_format", "auto")),
    )
    val_input_config = data_cfg.get("val_input_format", data_cfg.get("validation_input_format", train_input_format))
    use_separate_val_root = any(
        key in data_cfg for key in ("val_data_root", "volume_data_root", "val_volume_csv")
    ) or str(val_input_config).lower() in {"npy", "npy_volume", "volume_npy"}
    if use_separate_val_root:
        val_data_root = resolve_data_path(
            data_cfg.get("val_data_root", data_cfg.get("volume_data_root", data_cfg.get("data_root", DEFAULT_DATA_ROOT)))
        )
        val_records = load_npy_volume_records(
            val_data_root,
            csv_value=data_cfg.get("val_volume_csv", "val_volume.csv"),
        )
    else:
        val_records = default_val_records
    val_input_format = infer_input_format(val_records, val_input_config)

    train_records = resample_positive_negative_records(
        train_records,
        pos_neg_ratio=float(data_cfg.get("sample_pos_neg_ratio", 3.0)),
        seed=int(data_cfg.get("seed", DEFAULT_CONFIG.get("seed", 42))),
    )
    if train_input_format == "image_label":
        train_records = normalize_records_for_image_label(train_records)
    if val_input_format == "image_label":
        val_records = normalize_records_for_image_label(val_records)

    roi_size = data_cfg.get("roi_size", data_cfg.get("patch_size", [96, 96, 96]))
    target_spacing = data_cfg.get("target_spacing", [1.0, 1.0, 1.0])
    intensity_mode = data_cfg.get("intensity_mode", "ct")
    ct_window = data_cfg.get("ct_window", [-1000.0, 400.0])
    profile_ratios = data_cfg.get("profile_ratios", [1, 1, 1, 0.5, 0.5])
    samples_per_volume = int(data_cfg.get("samples_per_volume", 8))
    small_cc_voxels = int(data_cfg.get("small_cc_voxels", 128))
    large_cc_voxels = int(data_cfg.get("large_cc_voxels", 4096))
    overwrite_aux = bool(data_cfg.get("overwrite_aux", False))

    train_tfms = build_transforms(
        input_format=train_input_format,
        roi_size=roi_size,
        target_spacing=target_spacing,
        intensity_mode=intensity_mode,
        ct_window=ct_window,
        profile_ratios=profile_ratios,
        samples_per_volume=samples_per_volume,
        small_cc_voxels=small_cc_voxels,
        large_cc_voxels=large_cc_voxels,
        overwrite_aux=overwrite_aux,
        is_train=True,
    )
    val_tfms = build_transforms(
        input_format=val_input_format,
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
    )

    output_dir = resolve_runtime_path(cfg.get("output", {}).get("output_dir", DEFAULT_OUTPUT_DIR))
    use_persistent_cache = bool(data_cfg.get("use_persistent_cache", False))
    cache_root = (
        resolve_runtime_path(data_cfg["cache_dir"])
        if data_cfg.get("cache_dir", None) is not None
        else output_dir / "persistent_cache"
    )
    if use_persistent_cache:
        ensure_dir(cache_root)

    train_cache_tag = f"{train_input_format}_poscoords"
    val_cache_tag = f"{val_input_format}_poscoords"

    if use_persistent_cache:
        train_ds = PersistentDataset(
            data=train_records,
            transform=train_tfms,
            cache_dir=cache_root / f"train_{train_cache_tag}",
        )
    else:
        train_ds = Dataset(data=train_records, transform=train_tfms)

    if val_input_format == "npy_volume":
        val_ds = Dataset(data=val_records, transform=val_tfms)
    elif use_persistent_cache:
        val_ds = PersistentDataset(
            data=val_records,
            transform=val_tfms,
            cache_dir=cache_root / f"val_{val_cache_tag}",
        )
    else:
        val_ds = Dataset(data=val_records, transform=val_tfms)

    batch_size = int(train_cfg.get("batch_size", 1) or 1)
    if batch_size <= 0:
        raise ValueError(f"train.batch_size 必须大于 0，当前为 {batch_size}")
    default_num_workers = int(data_cfg.get("num_workers", 2) or 0)
    train_num_workers = int(data_cfg.get("train_num_workers", default_num_workers) or 0)
    default_val_workers = 0 if val_input_format == "npy_volume" else default_num_workers
    val_num_workers = int(data_cfg.get("val_num_workers", default_val_workers) or 0)
    prefetch_factor = max(1, int(data_cfg.get("prefetch_factor", 1) or 1))
    train_persistent_workers = (
        bool(data_cfg.get("train_persistent_workers", data_cfg.get("persistent_workers", False)))
        and train_num_workers > 0
    )
    val_persistent_workers = (
        bool(data_cfg.get("val_persistent_workers", False))
        and val_num_workers > 0
    )
    pin_memory_default = bool(data_cfg.get("pin_memory", False))
    train_pin_memory = bool(data_cfg.get("train_pin_memory", pin_memory_default))
    val_pin_memory = bool(data_cfg.get("val_pin_memory", False if val_input_format == "npy_volume" else pin_memory_default))

    train_loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": train_num_workers,
        "pin_memory": train_pin_memory,
        "drop_last": bool(train_cfg.get("drop_last", False)),
    }
    if train_num_workers > 0:
        train_loader_kwargs["persistent_workers"] = train_persistent_workers
        train_loader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(train_ds, **train_loader_kwargs)

    val_loader_kwargs: Dict[str, Any] = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": val_num_workers,
        "pin_memory": val_pin_memory,
        "drop_last": False,
    }
    if val_num_workers > 0:
        val_loader_kwargs["persistent_workers"] = val_persistent_workers
        val_loader_kwargs["prefetch_factor"] = prefetch_factor
    val_loader = DataLoader(val_ds, **val_loader_kwargs)

    print(f"数据格式 train_input_format = {train_input_format}, val_input_format = {val_input_format}")
    print(f"训练样本数：{len(train_ds)}，验证样本数：{len(val_ds)}")
    print(
        "DataLoader safety: "
        f"train_workers={train_num_workers}, val_workers={val_num_workers}, "
        f"prefetch_factor={prefetch_factor}, train_pin_memory={train_pin_memory}, val_pin_memory={val_pin_memory}, "
        f"persistent_cache={use_persistent_cache}"
    )
    if not use_persistent_cache:
        print("PersistentDataset disabled by default to avoid duplicating large position_coords caches.")
    elif val_input_format == "npy_volume":
        print(f"PersistentDataset cache_dir = {cache_root} (train cached; whole-volume validation uncached)")
    else:
        print(f"PersistentDataset cache_dir = {cache_root}")

    return train_loader, val_loader, train_input_format


# =========================
# 模型构建
# =========================

class GenericSwinUNETRDict(nn.Module):
    """当没有使用旧 Hybrid 模型时，提供一个通用 SwinUNETR 字典输出包装。"""

    def __init__(
        self,
        img_size: Sequence[int] = (96, 96, 96),
        in_channels: int = 1,
        num_classes: int = 1,
        feature_size: int = 12,
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        out_channels = 1 if self.num_classes <= 1 else self.num_classes

        common_kwargs = dict(
            spatial_dims=3,
            in_channels=int(in_channels),
            out_channels=out_channels,
            feature_size=int(feature_size),
            use_checkpoint=bool(use_checkpoint),
            norm_name="instance",
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
        )

        # 兼容 MONAI 版本差异：新版本可能不需要 img_size，旧版本可能需要。
        try:
            self.net = SwinUNETR(**common_kwargs)
        except TypeError:
            self.net = SwinUNETR(img_size=tuple(int(v) for v in img_size), **common_kwargs)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.net(x)
        return {"mask_logits": logits}


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    name = str(model_cfg.get("name", "hybrid_swin_sdf_core")).lower()
    num_classes = int(model_cfg.get("num_classes", 1))
    img_size = model_cfg.get("img_size", data_cfg.get("roi_size", data_cfg.get("patch_size", [96, 96, 96])))
    use_absolute_position_encoding = bool(
        model_cfg.get(
            "use_absolute_position_encoding",
            model_cfg.get("use_global_position_encoding", True),
        )
    )

    if name in {"hybrid", "hybrid_swin_sdf_core", "hybrid_swin_sdf_corenet"}:
        if HybridSwinSDFCoreNet is None:
            print("警告：未找到 model_hybrid_swin_sdf_core.py，将使用 GenericSwinUNETRDict。")
        else:
            if num_classes != 1:
                print("警告：旧 HybridSwinSDFCoreNet 默认是二分类输出。num_classes != 1 时建议改 model.py 输出头。")
            return HybridSwinSDFCoreNet(
                img_size=tuple(int(v) for v in img_size),
                in_channels=int(model_cfg.get("in_channels", 1)),
                ct_in_channels=model_cfg.get("ct_in_channels", None),
                swin_feature_channels=int(model_cfg.get("swin_feature_channels", 16)),
                two_d_feature_channels=int(model_cfg.get("two_d_feature_channels", 16)),
                fusion_channels=int(model_cfg.get("fusion_channels", 32)),
                feature_size=int(model_cfg.get("feature_size", 24)),
                use_checkpoint=bool(model_cfg.get("use_checkpoint", True)),
                use_absolute_position_encoding=use_absolute_position_encoding,
                absolute_position_scale_mm=float(model_cfg.get("absolute_position_scale_mm", 128.0)),
            )

    if name in {"swinunetr", "generic_swinunetr", "swin_unetr"} or HybridSwinSDFCoreNet is None:
        return GenericSwinUNETRDict(
            img_size=img_size,
            in_channels=int(model_cfg.get("in_channels", 1)),
            num_classes=num_classes,
            feature_size=int(model_cfg.get("feature_size", 12)),
            use_checkpoint=bool(model_cfg.get("use_checkpoint", True)),
        )

    raise ValueError(f"未知 model.name：{name}")


# =========================
# 动态对抗 Loss
# =========================

from losses import AdaptiveDynamicSegLoss


def build_criterion(cfg: Dict[str, Any]) -> AdaptiveDynamicSegLoss:
    """
    构建三阶段动态复合损失。

    YAML 推荐配置：
    loss:
      warmup_epochs: 15
      stage2_start_epoch: 80
      dice_weight_start: 1.0
      dice_weight_end: 0.35
      ce_weight_start: 1.0
      ce_weight_end: 0.25
      tversky_weight_start: 0.0
      tversky_weight_end: 1.2
      boundary_weight_start: 0.0
      boundary_weight_end: 0.8
      sdf_weight_start: 0.0
      sdf_weight_end: 0.3
      core_weight_start: 0.0
      core_weight_end: 0.3
      tversky_alpha: 0.3
      tversky_beta: 0.7
    """
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    loss_cfg = cfg.get("loss", {})

    return AdaptiveDynamicSegLoss(
        num_classes=int(model_cfg.get("num_classes", 1)),
        max_epochs=int(train_cfg.get("epochs", 300)),
        warmup_epochs=int(loss_cfg.get("warmup_epochs", 15)),
        stage2_start_epoch=int(loss_cfg.get("stage2_start_epoch", 80)),
        include_background=bool(loss_cfg.get("include_background", False)),
        dice_weight_start=float(loss_cfg.get("dice_weight_start", 1.0)),
        dice_weight_end=float(loss_cfg.get("dice_weight_end", 0.35)),
        ce_weight_start=float(loss_cfg.get("ce_weight_start", loss_cfg.get("ce_weight", 1.0))),
        ce_weight_end=float(loss_cfg.get("ce_weight_end", 0.25)),
        tversky_weight_start=float(loss_cfg.get("tversky_weight_start", 0.0)),
        tversky_weight_end=float(loss_cfg.get("tversky_weight_end", 1.2)),
        boundary_weight_start=float(loss_cfg.get("boundary_weight_start", 0.0)),
        boundary_weight_end=float(loss_cfg.get("boundary_weight_end", 0.8)),
        sdf_weight_start=float(loss_cfg.get("sdf_weight_start", 0.0)),
        sdf_weight_end=float(loss_cfg.get("sdf_weight_end", 0.3)),
        core_weight_start=float(loss_cfg.get("core_weight_start", 0.0)),
        core_weight_end=float(loss_cfg.get("core_weight_end", 0.3)),
        center_weight_start=float(loss_cfg.get("center_weight_start", 0.0)),
        center_weight_end=float(loss_cfg.get("center_weight_end", 0.01)),
        center_source=str(loss_cfg.get("center_source", "core")),
        center_use_smooth_l1=bool(loss_cfg.get("center_use_smooth_l1", True)),
        tversky_alpha=float(loss_cfg.get("tversky_alpha", 0.3)),
        tversky_beta=float(loss_cfg.get("tversky_beta", 0.7)),
        boundary_voxel_boost=float(loss_cfg.get("boundary_voxel_boost", 5.0)),
        dist_sigma=float(loss_cfg.get("dist_sigma", 0.2)),
        stage2_weight_power=float(loss_cfg.get("stage2_weight_power", 1.0)),
        max_stage2_weight=float(loss_cfg.get("max_stage2_weight", 8.0)),
    )


# =========================
# AMP 策略
# =========================

@dataclass
class AMPPolicy:
    enabled: bool
    dtype: torch.dtype
    scaler: Optional[torch.cuda.amp.GradScaler]


def build_amp_policy(device: torch.device, requested: bool = True) -> AMPPolicy:
    if not requested or device.type != "cuda":
        return AMPPolicy(enabled=False, dtype=torch.float32, scaler=None)
    if torch.cuda.is_bf16_supported():
        print("AMP: 使用 bfloat16，不需要 GradScaler。")
        return AMPPolicy(enabled=True, dtype=torch.bfloat16, scaler=None)
    print("AMP: 使用 float16 + GradScaler。")
    return AMPPolicy(enabled=True, dtype=torch.float16, scaler=torch.cuda.amp.GradScaler(enabled=True))


def autocast_context(policy: AMPPolicy):
    if not policy.enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=policy.dtype, enabled=True)


def report_gpu_memory_safety(cfg: Dict[str, Any], device: torch.device) -> None:
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("eval", {})

    enforce = bool(train_cfg.get("enforce_24gb_safety", False))
    max_gpu_gb = float(train_cfg.get("max_gpu_memory_gb", 24.0))

    if device.type != "cuda":
        print("GPU safety: CUDA 不可用，无法在本机确认 24GB 显存占用；将在 CPU/当前设备继续。")
        return

    props = torch.cuda.get_device_properties(device)
    total_gb = float(props.total_memory) / (1024.0 ** 3)
    name = getattr(props, "name", "cuda")
    print(f"GPU safety: {name}, total_memory={total_gb:.2f} GB")

    roi = _to_3tuple_int(data_cfg.get("roi_size", model_cfg.get("img_size", [96, 96, 96])))
    eval_roi = _to_3tuple_int(eval_cfg.get("roi_size", roi))
    feature_size = int(model_cfg.get("feature_size", 12))
    swin_feature_channels = int(model_cfg.get("swin_feature_channels", 12))
    two_d_feature_channels = int(model_cfg.get("two_d_feature_channels", 12))
    fusion_channels = int(model_cfg.get("fusion_channels", 24))
    batch_size = int(train_cfg.get("batch_size", 1) or 1)
    val_sw_batch_size = int(eval_cfg.get("sw_batch_size", 1) or 1)
    val_overlap = float(eval_cfg.get("overlap", 0.0))
    train_input_format = str(data_cfg.get("input_format", "")).lower()
    val_input_format = str(data_cfg.get("val_input_format", "")).lower()
    val_num_workers = int(data_cfg.get("val_num_workers", data_cfg.get("num_workers", 0)) or 0)

    issues: List[str] = []
    if int(np.prod(roi)) > 96 ** 3:
        issues.append(f"data.roi_size={list(roi)} 超过 96^3，24GB 上风险很高")
    if int(np.prod(eval_roi)) > 96 ** 3:
        issues.append(f"eval.roi_size={list(eval_roi)} 超过 96^3，整例滑窗风险很高")
    if batch_size > 1:
        issues.append(f"train.batch_size={batch_size}，24GB 建议固定为 1")
    if not bool(train_cfg.get("amp", True)):
        issues.append("train.amp=false，24GB 必须开启 AMP")
    if not bool(model_cfg.get("use_checkpoint", True)):
        issues.append("model.use_checkpoint=false，24GB 建议必须开启 gradient checkpointing")
    if feature_size > 12:
        issues.append(f"model.feature_size={feature_size}，24GB 安全档建议 <= 12")
    if swin_feature_channels > 12:
        issues.append(f"model.swin_feature_channels={swin_feature_channels}，24GB 安全档建议 <= 12")
    if two_d_feature_channels > 12:
        issues.append(f"model.two_d_feature_channels={two_d_feature_channels}，24GB 安全档建议 <= 12")
    if fusion_channels > 24:
        issues.append(f"model.fusion_channels={fusion_channels}，24GB 安全档建议 <= 24")
    if val_sw_batch_size > 1:
        issues.append(f"eval.sw_batch_size={val_sw_batch_size}，整例验证建议固定为 1")
    if val_overlap > 0.0:
        issues.append(f"eval.overlap={val_overlap}，24GB/远程服务器安全档建议为 0.0")
    if bool(data_cfg.get("use_persistent_cache", False)):
        issues.append("data.use_persistent_cache=true，会复制大 patch/position_coords 到缓存，磁盘和 I/O 风险高")
    if val_input_format in {"npy", "npy_volume", "volume_npy"} and val_num_workers > 0:
        issues.append(f"整例验证 val_input_format={val_input_format} 但 val_num_workers={val_num_workers}，建议为 0")
    if train_input_format in {"npy", "npy_volume", "volume_npy"} and int(data_cfg.get("samples_per_volume", 1)) > 1:
        issues.append("整例训练会在 worker 内随机裁多个样本；当前推荐使用 npz_patch 训练")

    if issues:
        message = "24GB GPU safety check found risky settings:\n" + "\n".join(f"  - {item}" for item in issues)
        if enforce and total_gb <= max_gpu_gb + 0.75:
            raise RuntimeError(message)
        print("[WARNING] " + message)
    else:
        print("GPU safety: 当前配置符合 24GB 安全档。")


def run_cuda_smoke_test(
    model: nn.Module,
    criterion: nn.Module,
    cfg: Dict[str, Any],
    device: torch.device,
    amp_policy: AMPPolicy,
) -> None:
    if device.type != "cuda":
        print("Smoke test: 当前不是 CUDA 设备，只做 CPU 语法路径检查。")

    data_cfg = cfg.get("data", {})
    loss_cfg = cfg.get("loss", {})
    roi = _to_3tuple_int(data_cfg.get("roi_size", cfg.get("model", {}).get("img_size", [96, 96, 96])))
    if int(np.prod(roi)) > 96 ** 3:
        raise RuntimeError(f"Smoke test refusing roi_size={list(roi)}; 24GB safety limit is 96^3.")

    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    try:
        image = torch.randn((1, 1, *roi), device=device, dtype=torch.float32)
        label = torch.zeros((1, 1, *roi), device=device, dtype=torch.long)
        d0 = max(roi[0] // 2 - 4, 0)
        h0 = max(roi[1] // 2 - 4, 0)
        w0 = max(roi[2] // 2 - 4, 0)
        label[..., d0:d0 + 8, h0:h0 + 8, w0:w0 + 8] = 1
        spacing = torch.ones(3, device=device, dtype=torch.float32)
        position_coords = _build_window_position_coords(
            start_dhw=(0, 0, 0),
            roi_size=roi,
            spacing=spacing,
            device=device,
            dtype=torch.float32,
        )
        batch = {
            IMAGE_KEY: image,
            LABEL_KEY: label,
            MASK_ALIAS_KEY: label,
            POSITION_COORDS_KEY: position_coords,
            SPACING_KEY: spacing.view(1, 3),
            "boundary": (label > 0).float(),
            "sdf": torch.zeros_like(image),
            "dist_map": torch.zeros_like(image),
            "core": torch.zeros_like(image),
            "component_weight": torch.ones_like(image),
        }

        epoch = max(int(loss_cfg.get("warmup_epochs", 15)) + 1, 1)
        with autocast_context(amp_policy):
            outputs = model(image, position_coords=position_coords)
            loss, loss_dict = criterion(outputs, batch, epoch=epoch)

        if amp_policy.scaler is not None:
            amp_policy.scaler.scale(loss).backward()
        else:
            loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_gb = float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 3)
            reserved_gb = float(torch.cuda.max_memory_reserved(device)) / (1024.0 ** 3)
            print(
                "Smoke test passed: "
                f"loss={float(loss.detach().cpu()):.5f}, "
                f"peak_allocated={peak_gb:.2f} GB, peak_reserved={reserved_gb:.2f} GB"
            )
        else:
            print(f"Smoke test passed on CPU: loss={float(loss.detach().cpu()):.5f}")

        print(
            "Smoke loss parts: "
            + ", ".join(
                f"{key}={float(value.detach().cpu()):.4f}"
                for key, value in loss_dict.items()
                if torch.is_tensor(value) and value.ndim == 0
            )
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                "Smoke test hit CUDA OOM. Keep roi_size at [96,96,96], "
                "feature_size at 12, batch_size at 1, amp/checkpoint on; "
                "if it still fails, reduce roi_size to [64,64,64] and regenerate patches."
            ) from exc
        raise
    finally:
        model.zero_grad(set_to_none=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


# =========================
# 后处理与指标
# =========================

def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit_threshold(threshold: float) -> float:
    threshold = float(threshold)
    if threshold <= 0.0:
        return float("-inf")
    if threshold >= 1.0:
        return float("inf")
    return float(math.log(threshold / (1.0 - threshold)))


def _softmax_np(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def remove_small_components_3d_vectorized(mask: np.ndarray, min_voxels: int = 16, keep_largest: bool = False, connectivity: int = 3) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    structure = generate_binary_structure(rank=3, connectivity=connectivity)
    labeled, num_cc = cc_label(mask, structure=structure)
    if num_cc == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labeled.reshape(-1))
    counts[0] = 0
    if keep_largest:
        keep_lut = np.zeros_like(counts, dtype=bool)
        keep_lut[int(np.argmax(counts))] = True
    else:
        keep_lut = counts >= int(min_voxels)
        keep_lut[0] = False
    return keep_lut[labeled].astype(bool)


def fast_postprocess_prediction(
    pred: np.ndarray,
    num_classes: int,
    threshold: float = 0.5,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    min_voxels: Union[int, Dict[int, int]] = 16,
    min_volume_mm3: Optional[Union[float, Dict[int, float]]] = None,
    keep_largest: Union[bool, Dict[int, bool]] = False,
    connectivity: int = 3,
    input_is_logits: bool = True,
) -> np.ndarray:
    pred = np.asarray(pred)
    voxel_volume = float(np.prod(np.asarray(spacing, dtype=np.float64)))

    def resolve_min_voxels(cls_id: int) -> int:
        if min_volume_mm3 is not None:
            vol = float(min_volume_mm3.get(cls_id, 0.0)) if isinstance(min_volume_mm3, dict) else float(min_volume_mm3)
            if vol > 0:
                return int(np.ceil(vol / max(voxel_volume, 1e-8)))
        if isinstance(min_voxels, dict):
            return int(min_voxels.get(cls_id, 0))
        return int(min_voxels)

    def resolve_keep_largest(cls_id: int) -> bool:
        if isinstance(keep_largest, dict):
            return bool(keep_largest.get(cls_id, False))
        return bool(keep_largest)

    if num_classes <= 1:
        if pred.ndim == 4:
            pred = pred[0]
        raw = pred >= _logit_threshold(threshold) if input_is_logits else pred >= float(threshold)
        return remove_small_components_3d_vectorized(
            raw,
            min_voxels=resolve_min_voxels(1),
            keep_largest=resolve_keep_largest(1),
            connectivity=connectivity,
        )

    if pred.ndim != 4:
        raise ValueError(f"多分类 pred 必须是 [C,D,H,W]，当前 shape={pred.shape}")
    prob = _softmax_np(pred, axis=0) if input_is_logits else pred
    raw_cls = np.argmax(prob, axis=0).astype(np.int64)
    final = np.zeros_like(raw_cls, dtype=np.int64)
    for cls_id in range(1, int(num_classes)):
        cls_mask = raw_cls == cls_id
        cleaned = remove_small_components_3d_vectorized(
            cls_mask,
            min_voxels=resolve_min_voxels(cls_id),
            keep_largest=resolve_keep_largest(cls_id),
            connectivity=connectivity,
        )
        final[cleaned] = cls_id
    return final


def ensure_3d_label_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 5 and x.shape[0] == 1 and x.shape[1] == 1:
        x = x[0, 0]
    elif x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    elif x.ndim == 4 and x.shape[1] == 1:
        x = x[0, 0]
    elif x.ndim == 3:
        pass
    else:
        x = np.squeeze(x)
        if x.ndim != 3:
            raise ValueError(f"无法转换为 3D label，shape={x.shape}")
    return x


def compute_simple_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int) -> Dict[str, float]:
    pred = ensure_3d_label_np(pred_mask)
    gt = ensure_3d_label_np(gt_mask)
    eps = 1e-6

    if num_classes <= 1:
        pred_bin = pred > 0
        gt_bin = gt > 0
        inter = float(np.logical_and(pred_bin, gt_bin).sum())
        pred_sum = float(pred_bin.sum())
        gt_sum = float(gt_bin.sum())
        union = float(np.logical_or(pred_bin, gt_bin).sum())
        dice = (2.0 * inter + eps) / (pred_sum + gt_sum + eps)
        iou = (inter + eps) / (union + eps)
        return {"dice": float(dice), "iou": float(iou), "score": float(dice)}

    dices: List[float] = []
    ious: List[float] = []
    for cls_id in range(1, int(num_classes)):
        pred_c = pred == cls_id
        gt_c = gt == cls_id
        if not np.any(gt_c) and not np.any(pred_c):
            continue
        inter = float(np.logical_and(pred_c, gt_c).sum())
        pred_sum = float(pred_c.sum())
        gt_sum = float(gt_c.sum())
        union = float(np.logical_or(pred_c, gt_c).sum())
        dices.append((2.0 * inter + eps) / (pred_sum + gt_sum + eps))
        ious.append((inter + eps) / (union + eps))

    dice = float(np.mean(dices)) if dices else 1.0
    iou = float(np.mean(ious)) if ious else 1.0
    return {"dice": dice, "iou": iou, "score": dice}


def compute_soft_binary_dice(prob: np.ndarray, gt_mask: np.ndarray) -> float:
    prob_3d = ensure_3d_label_np(prob).astype(np.float64, copy=False)
    gt_3d = (ensure_3d_label_np(gt_mask) > 0).astype(np.float64, copy=False)
    eps = 1e-6
    inter = float(np.sum(prob_3d * gt_3d))
    denom = float(np.sum(prob_3d) + np.sum(gt_3d))
    return float((2.0 * inter + eps) / (denom + eps))


def parse_class_dict_or_scalar(value: Any) -> Any:
    """支持 YAML 里的标量或 {'1': 10, '2': 50}。"""
    if isinstance(value, dict):
        out: Dict[int, Any] = {}
        for k, v in value.items():
            out[int(k)] = v
        return out
    return value


# =========================
# 训练 / 验证 / checkpoint
# =========================

def average_loss_dict(sum_dict: Dict[str, float], count: int) -> Dict[str, float]:
    if count <= 0:
        return {k: 0.0 for k in sum_dict}
    return {k: float(v) / float(count) for k, v in sum_dict.items()}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: AdaptiveDynamicSegLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_policy: AMPPolicy,
    epoch: int,
    accum_steps: int = 1,
    grad_clip_norm: Optional[float] = None,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accum_steps = max(int(accum_steps), 1)

    loss_sums = {
        "total": 0.0,
        "dice": 0.0,
        "ce": 0.0,
        "tversky": 0.0,
        "boundary": 0.0,
        "sdf": 0.0,
        "core": 0.0,
        "center": 0.0,
        "w_dice": 0.0,
        "w_ce": 0.0,
        "w_tversky": 0.0,
        "w_boundary": 0.0,
        "w_sdf": 0.0,
        "w_core": 0.0,
        "w_center": 0.0,
        "stage2_active": 0.0,
    }
    count = 0

    progress = tqdm(loader, desc=f"Epoch {epoch} [train]", dynamic_ncols=True, leave=False)

    for step, batch in enumerate(progress):
        batch = move_batch_to_device(batch, device)
        images = batch[IMAGE_KEY].float()
        position_kwargs = model_position_kwargs(batch)

        with autocast_context(amp_policy):
            outputs = model(images, **position_kwargs)
            loss, loss_dict = criterion(outputs, batch, epoch=epoch)
            loss_for_backward = loss / accum_steps

        if amp_policy.scaler is not None:
            amp_policy.scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        should_update = ((step + 1) % accum_steps == 0) or ((step + 1) == len(loader))
        if should_update:
            if grad_clip_norm is not None and float(grad_clip_norm) > 0:
                if amp_policy.scaler is not None:
                    amp_policy.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))

            if amp_policy.scaler is not None:
                amp_policy.scaler.step(optimizer)
                amp_policy.scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        bs = int(images.shape[0])
        count += bs
        for k in loss_sums.keys():
            loss_sums[k] += float(loss_dict.get(k, torch.tensor(0.0)).detach().cpu()) * bs

        progress.set_postfix({
            "loss": f"{float(loss.detach().cpu()):.4f}",
            "dice_l": f"{float(loss_dict['dice'].detach().cpu()):.4f}",
            "ctr": f"{float(loss_dict.get('center', torch.tensor(0.0)).detach().cpu()):.4f}",
            "w_ce": f"{float(loss_dict.get('w_ce', torch.tensor(0.0)).detach().cpu()):.3f}",
            "w_bnd": f"{float(loss_dict.get('w_boundary', torch.tensor(0.0)).detach().cpu()):.3f}",
        })

    return average_loss_dict(loss_sums, count)


def _to_3tuple_int(value: Sequence[int] | int) -> Tuple[int, int, int]:
    if isinstance(value, int):
        return int(value), int(value), int(value)
    if len(value) != 3:
        raise ValueError(f"roi_size must be int or length-3 sequence, got {value}")
    return tuple(int(v) for v in value)


def _window_starts(size: int, roi: int, overlap: float) -> List[int]:
    if size <= roi:
        return [0]
    step = max(1, int(round(float(roi) * (1.0 - float(overlap)))))
    starts = list(range(0, max(size - roi, 0) + 1, step))
    last = size - roi
    if starts[-1] != last:
        starts.append(last)
    return starts


def _select_position_kwargs(
    position_kwargs: Dict[str, torch.Tensor],
    index: int,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    selected: Dict[str, torch.Tensor] = {}
    for key, value in position_kwargs.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
            selected[key] = value[index:index + 1]
        else:
            selected[key] = value
    return selected


def _spacing_from_kwargs(
    position_kwargs: Dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    spacing = position_kwargs.get(SPACING_KEY)
    if torch.is_tensor(spacing):
        spacing = spacing.to(device=device, dtype=dtype).reshape(-1)
        if spacing.numel() >= 3:
            return spacing[:3]
    return torch.ones(3, device=device, dtype=dtype)


def _build_window_position_coords(
    start_dhw: Tuple[int, int, int],
    roi_size: Tuple[int, int, int],
    spacing: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    d0, h0, w0 = start_dhw
    rd, rh, rw = roi_size
    z = (torch.arange(rd, device=device, dtype=dtype) + float(d0)) * spacing[0]
    y = (torch.arange(rh, device=device, dtype=dtype) + float(h0)) * spacing[1]
    x = (torch.arange(rw, device=device, dtype=dtype) + float(w0)) * spacing[2]
    zz = z.view(1, 1, rd, 1, 1).expand(1, 1, rd, rh, rw)
    yy = y.view(1, 1, 1, rh, 1).expand(1, 1, rd, rh, rw)
    xx = x.view(1, 1, 1, 1, rw).expand(1, 1, rd, rh, rw)
    return torch.cat([zz, yy, xx], dim=1)


def sliding_window_logits(
    model: nn.Module,
    images: torch.Tensor,
    position_kwargs: Dict[str, torch.Tensor],
    amp_policy: AMPPolicy,
    roi_size: Sequence[int] | int,
    sw_batch_size: int = 1,
    overlap: float = 0.25,
) -> torch.Tensor:
    """
    Sliding-window inference for validation.

    Each window is [roi_size] so SwinUNETR sees dimensions divisible by 32.
    position_coords are cropped per window and passed separately from image.
    """
    if images.ndim != 5:
        raise ValueError(f"images must be [B,C,D,H,W], got {tuple(images.shape)}")

    roi = _to_3tuple_int(roi_size)
    sw_batch_size = max(1, int(sw_batch_size))
    overlap = min(max(float(overlap), 0.0), 0.95)

    batch_size, _, depth, height, width = images.shape
    outputs_per_sample: List[torch.Tensor] = []

    for sample_idx in range(batch_size):
        image = images[sample_idx:sample_idx + 1]
        sample_kwargs = _select_position_kwargs(position_kwargs, sample_idx, batch_size)

        pad_d = max(roi[0] - depth, 0)
        pad_h = max(roi[1] - height, 0)
        pad_w = max(roi[2] - width, 0)
        pad = (0, pad_w, 0, pad_h, 0, pad_d)
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            image = F.pad(image, pad, mode="constant", value=0.0)
            if POSITION_COORDS_KEY in sample_kwargs and torch.is_tensor(sample_kwargs[POSITION_COORDS_KEY]):
                sample_kwargs[POSITION_COORDS_KEY] = F.pad(
                    sample_kwargs[POSITION_COORDS_KEY],
                    pad,
                    mode="replicate",
                )

        _, _, padded_d, padded_h, padded_w = image.shape
        d_starts = _window_starts(padded_d, roi[0], overlap)
        h_starts = _window_starts(padded_h, roi[1], overlap)
        w_starts = _window_starts(padded_w, roi[2], overlap)
        starts = [(d, h, w) for d in d_starts for h in h_starts for w in w_starts]
        full_position_coords = sample_kwargs.get(POSITION_COORDS_KEY)
        has_full_position_coords = torch.is_tensor(full_position_coords)
        spacing = _spacing_from_kwargs(sample_kwargs, image.device, image.dtype)

        accum: Optional[torch.Tensor] = None
        count_map = torch.zeros((1, 1, padded_d, padded_h, padded_w), dtype=torch.int32, device="cpu")

        for start_idx in range(0, len(starts), sw_batch_size):
            chunk_starts = starts[start_idx:start_idx + sw_batch_size]
            patch_images: List[torch.Tensor] = []
            patch_position_coords: List[torch.Tensor] = []
            for d0, h0, w0 in chunk_starts:
                rd, rh, rw = roi
                patch_images.append(image[..., d0:d0 + rd, h0:h0 + rh, w0:w0 + rw])
                if has_full_position_coords:
                    patch_position_coords.append(
                        full_position_coords[..., d0:d0 + rd, h0:h0 + rh, w0:w0 + rw]
                    )
                else:
                    patch_position_coords.append(
                        _build_window_position_coords(
                            start_dhw=(d0, h0, w0),
                            roi_size=roi,
                            spacing=spacing,
                            device=image.device,
                            dtype=image.dtype,
                        )
                    )

            patch_batch = torch.cat(patch_images, dim=0)
            patch_kwargs = {
                key: value
                for key, value in sample_kwargs.items()
                if key != POSITION_COORDS_KEY
            }
            if patch_position_coords:
                patch_kwargs[POSITION_COORDS_KEY] = torch.cat(patch_position_coords, dim=0)

            with autocast_context(amp_policy):
                patch_outputs = model(patch_batch, **patch_kwargs)
            patch_logits = patch_outputs["mask_logits"].detach().float().cpu()

            if accum is None:
                out_channels = int(patch_logits.shape[1])
                accum = torch.zeros(
                    (1, out_channels, padded_d, padded_h, padded_w),
                    dtype=torch.float32,
                    device="cpu",
                )

            for local_idx, (d0, h0, w0) in enumerate(chunk_starts):
                rd, rh, rw = roi
                accum[..., d0:d0 + rd, h0:h0 + rh, w0:w0 + rw] += patch_logits[local_idx:local_idx + 1]
                count_map[..., d0:d0 + rd, h0:h0 + rh, w0:w0 + rw] += 1

        if accum is None:
            raise RuntimeError("sliding_window_logits produced no windows.")

        logits = accum / count_map.clamp_min(1).to(dtype=accum.dtype)
        outputs_per_sample.append(logits[..., :depth, :height, :width])

    return torch.cat(outputs_per_sample, dim=0)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_policy: AMPPolicy,
    num_classes: int,
    threshold: float,
    spacing: Sequence[float],
    min_voxels: Any,
    min_volume_mm3: Any,
    keep_largest: Any,
    roi_size: Sequence[int] | int,
    sw_batch_size: int = 1,
    overlap: float = 0.25,
    max_cases: Optional[int] = None,
    compute_soft_dice: bool = False,
    empty_cache: bool = True,
) -> Dict[str, float]:
    model.eval()
    rows: List[Dict[str, float]] = []
    processed_cases = 0

    progress = tqdm(loader, desc="Validation", dynamic_ncols=True, leave=False)
    for batch in progress:
        if max_cases is not None and processed_cases >= int(max_cases):
            break

        batch = move_batch_to_device(batch, device)
        images = batch[IMAGE_KEY].float()
        position_kwargs = model_position_kwargs(batch)
        labels = batch.get(LABEL_KEY, batch.get(MASK_ALIAS_KEY))
        if labels is None:
            raise KeyError(f"验证 batch 缺少 {LABEL_KEY}/{MASK_ALIAS_KEY} 标签字段。")

        logits_tensor = sliding_window_logits(
            model=model,
            images=images,
            position_kwargs=position_kwargs,
            amp_policy=amp_policy,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
        )

        logits = logits_tensor.numpy()
        gt = labels.detach().cpu().numpy()

        b = int(logits.shape[0])
        for i in range(b):
            if max_cases is not None and processed_cases >= int(max_cases):
                break

            gt_i = ensure_3d_label_np(gt[i])
            if num_classes <= 1:
                logit_i = ensure_3d_label_np(logits[i])
                raw_pred_i = logit_i >= _logit_threshold(threshold)
                if compute_soft_dice:
                    prob_i = _sigmoid_np(logit_i.astype(np.float32, copy=False))
                    soft_dice = compute_soft_binary_dice(prob_i, gt_i)
                else:
                    soft_dice = float("nan")
            else:
                prob_i = _softmax_np(logits[i], axis=0)
                raw_pred_i = np.argmax(prob_i, axis=0).astype(np.int64)
                soft_dice = float("nan")

            raw_metrics = compute_simple_metrics(raw_pred_i, gt_i, num_classes=num_classes)
            post_pred_i = fast_postprocess_prediction(
                pred=logits[i],
                num_classes=num_classes,
                threshold=threshold,
                spacing=spacing,
                min_voxels=min_voxels,
                min_volume_mm3=min_volume_mm3,
                keep_largest=keep_largest,
                connectivity=3,
                input_is_logits=True,
            )
            post_metrics = compute_simple_metrics(post_pred_i, gt_i, num_classes=num_classes)
            metrics = {
                "dice": raw_metrics["dice"],
                "iou": raw_metrics["iou"],
                "score": raw_metrics["score"],
                "soft_dice": soft_dice,
                "post_dice": post_metrics["dice"],
                "post_iou": post_metrics["iou"],
                "pred_voxels": float(np.sum(np.asarray(raw_pred_i) > 0)),
                "post_pred_voxels": float(np.sum(np.asarray(post_pred_i) > 0)),
                "gt_voxels": float(np.sum(np.asarray(gt_i) > 0)),
            }
            rows.append(metrics)
            processed_cases += 1

        if rows:
            progress.set_postfix({
                "raw_dice": f"{np.mean([r['dice'] for r in rows]):.4f}",
                "post_dice": f"{np.mean([r['post_dice'] for r in rows]):.4f}",
                "cases": processed_cases,
            })

        del logits_tensor, logits, gt, images, labels, position_kwargs
        gc.collect()
        if empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        return {"dice": 0.0, "iou": 0.0, "score": 0.0}

    soft_values = np.asarray([r["soft_dice"] for r in rows], dtype=np.float64)
    soft_dice_mean = float(np.nanmean(soft_values)) if not np.all(np.isnan(soft_values)) else float("nan")

    return {
        "dice": float(np.mean([r["dice"] for r in rows])),
        "iou": float(np.mean([r["iou"] for r in rows])),
        "soft_dice": soft_dice_mean,
        "post_dice": float(np.mean([r["post_dice"] for r in rows])),
        "post_iou": float(np.mean([r["post_iou"] for r in rows])),
        "pred_voxels": float(np.mean([r["pred_voxels"] for r in rows])),
        "post_pred_voxels": float(np.mean([r["post_pred_voxels"] for r in rows])),
        "gt_voxels": float(np.mean([r["gt_voxels"] for r in rows])),
        "score": float(np.mean([r["score"] for r in rows])),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    best_score: float,
    cfg: Dict[str, Any],
) -> None:
    ensure_dir(path.parent)
    payload = {
        "epoch": int(epoch),
        "best_score": float(best_score),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, float]:
    if not path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{path}")
    checkpoint = torch.load(path, map_location=device or "cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = strip_module_prefix(checkpoint["model_state_dict"])
        model.load_state_dict(state, strict=False)
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return int(checkpoint.get("epoch", 0)), float(checkpoint.get("best_score", -1.0))

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(strip_module_prefix(checkpoint["state_dict"]), strict=False)
        return 0, -1.0

    if isinstance(checkpoint, dict):
        model.load_state_dict(strip_module_prefix(checkpoint), strict=False)
        return 0, -1.0

    raise RuntimeError(f"无法识别 checkpoint 格式：{path}")


def build_optimizer_and_scheduler(cfg: Dict[str, Any], model: nn.Module) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
    train_cfg = cfg.get("train", {})
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler_name = str(train_cfg.get("scheduler", "cosine")).lower()
    epochs = int(train_cfg.get("epochs", 300))
    if scheduler_name in {"none", "null", "off"}:
        return optimizer, None
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1),
            eta_min=float(train_cfg.get("min_lr", 1e-6)),
        )
        return optimizer, scheduler
    raise ValueError(f"未知 scheduler：{scheduler_name}")


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = apply_cli_overrides(cfg, args)

    seed = int(cfg.get("seed", 42))
    deterministic = bool(cfg.get("deterministic", False))
    set_seed(seed, deterministic=deterministic)

    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})
    model_cfg = cfg.get("model", {})
    output_cfg = cfg.get("output", {})
    data_cfg = cfg.get("data", {})

    output_dir = ensure_dir(resolve_runtime_path(output_cfg.get("output_dir", DEFAULT_OUTPUT_DIR)))
    log_dir = ensure_dir(output_dir / "logs")
    ckpt_dir = ensure_dir(output_dir / "checkpoints")

    log_path = log_dir / "train_log.csv"
    latest_path = ckpt_dir / "latest.pt"
    best_path = ckpt_dir / "best_dice.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    report_gpu_memory_safety(cfg, device)

    if args.smoke_test:
        model = build_model(cfg).to(device)
        criterion = build_criterion(cfg).to(device)
        amp_policy = build_amp_policy(device, requested=bool(train_cfg.get("amp", True)))
        run_cuda_smoke_test(
            model=model,
            criterion=criterion,
            cfg=cfg,
            device=device,
            amp_policy=amp_policy,
        )
        return

    train_loader, val_loader, input_format = build_loaders(cfg)

    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model)

    amp_policy = build_amp_policy(device, requested=bool(train_cfg.get("amp", True)))

    epochs = int(train_cfg.get("epochs", 300))
    accum_steps = max(int(train_cfg.get("grad_accum_steps", train_cfg.get("accum_steps", 1))), 1)
    grad_clip_norm = train_cfg.get("grad_clip_norm", None)
    if grad_clip_norm is not None:
        grad_clip_norm = float(grad_clip_norm)

    num_classes = int(model_cfg.get("num_classes", 1))
    threshold = float(eval_cfg.get("threshold", 0.5))
    val_interval = max(1, int(eval_cfg.get("val_interval", train_cfg.get("val_interval", 5))))
    val_roi_size = eval_cfg.get("roi_size", data_cfg.get("roi_size", data_cfg.get("patch_size", [96, 96, 96])))
    val_sw_batch_size = max(1, int(eval_cfg.get("sw_batch_size", 1)))
    val_overlap = float(eval_cfg.get("overlap", 0.25))
    raw_val_max_cases = eval_cfg.get("max_cases_per_epoch", None)
    val_max_cases = None if raw_val_max_cases is None else max(1, int(raw_val_max_cases))
    validate_on_start = bool(eval_cfg.get("validate_on_start", False))
    validate_on_final = bool(eval_cfg.get("validate_on_final", True))
    val_compute_soft_dice = bool(eval_cfg.get("compute_soft_dice", False))
    val_empty_cache = bool(eval_cfg.get("empty_cache", True))
    spacing = eval_cfg.get("spacing", data_cfg.get("target_spacing", [1.0, 1.0, 1.0]))
    min_voxels = parse_class_dict_or_scalar(eval_cfg.get("min_voxels", 16))
    min_volume_mm3 = parse_class_dict_or_scalar(eval_cfg.get("min_volume_mm3", None))
    keep_largest = parse_class_dict_or_scalar(eval_cfg.get("keep_largest", False))

    start_epoch = 1
    best_score = -1.0

    resume_path = args.resume or train_cfg.get("resume", None)
    resume_flag = False
    if resume_path is not None and str(resume_path).strip():
        loaded_epoch, loaded_best = load_checkpoint(
            resolve_existing_path(resume_path, base_dirs=[ckpt_dir, output_dir]),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        start_epoch = loaded_epoch + 1
        best_score = loaded_best
        resume_flag = True
        print(f"已恢复 checkpoint：epoch={loaded_epoch}, best_score={best_score:.6f}")

    init_csv_log(log_path, resume=resume_flag)

    print("训练配置摘要：")
    print(json.dumps({
        "input_format": input_format,
        "epochs": epochs,
        "batch_size": int(train_cfg.get("batch_size", 1) or 1),
        "grad_accum_steps": accum_steps,
        "num_classes": num_classes,
        "amp": bool(train_cfg.get("amp", True)),
        "val_interval": val_interval,
        "val_roi_size": val_roi_size,
        "val_sw_batch_size": val_sw_batch_size,
        "val_overlap": val_overlap,
        "val_max_cases": val_max_cases,
        "validate_on_start": validate_on_start,
        "validate_on_final": validate_on_final,
        "val_compute_soft_dice": val_compute_soft_dice,
        "threshold": threshold,
        "min_voxels": min_voxels,
        "min_volume_mm3": min_volume_mm3,
        "keep_largest": keep_largest,
    }, ensure_ascii=False, indent=2))

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_losses = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            amp_policy=amp_policy,
            epoch=epoch,
            accum_steps=accum_steps,
            grad_clip_norm=grad_clip_norm,
        )

        should_validate = (
            (validate_on_start and epoch == start_epoch)
            or (validate_on_final and epoch == epochs)
            or (epoch % val_interval == 0)
        )
        val_metrics: Dict[str, Any] = {}
        if should_validate:
            val_metrics = validate(
                model=model,
                loader=val_loader,
                device=device,
                amp_policy=amp_policy,
                num_classes=num_classes,
                threshold=threshold,
                spacing=spacing,
                min_voxels=min_voxels,
                min_volume_mm3=min_volume_mm3,
                keep_largest=keep_largest,
                roi_size=val_roi_size,
                sw_batch_size=val_sw_batch_size,
                overlap=val_overlap,
                max_cases=val_max_cases,
                compute_soft_dice=val_compute_soft_dice,
                empty_cache=val_empty_cache,
            )

        if scheduler is not None:
            scheduler.step()

        current_lr = get_current_lr(optimizer)
        epoch_time = time.time() - t0
        val_score = float(val_metrics.get("score", val_metrics.get("dice", best_score)))

        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_score=max(best_score, val_score) if should_validate else best_score,
            cfg=cfg,
        )

        improved = should_validate and val_score > best_score
        if improved:
            best_score = val_score
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_score=best_score,
                cfg=cfg,
            )

        row = {
            "epoch": epoch,
            "lr": format_float(current_lr),
            "train_total": format_float(train_losses["total"]),
            "train_dice": format_float(train_losses["dice"]),
            "train_ce": format_float(train_losses["ce"]),
            "train_tversky": format_float(train_losses["tversky"]),
            "train_boundary": format_float(train_losses["boundary"]),
            "train_sdf": format_float(train_losses["sdf"]),
            "train_core": format_float(train_losses["core"]),
            "train_center": format_float(train_losses["center"]),
            "w_dice": format_float(train_losses.get("w_dice", 0.0)),
            "w_ce": format_float(train_losses.get("w_ce", 0.0)),
            "w_tversky": format_float(train_losses.get("w_tversky", 0.0)),
            "w_boundary": format_float(train_losses.get("w_boundary", 0.0)),
            "w_sdf": format_float(train_losses.get("w_sdf", 0.0)),
            "w_core": format_float(train_losses.get("w_core", 0.0)),
            "w_center": format_float(train_losses.get("w_center", 0.0)),
            "stage2_active": format_float(train_losses.get("stage2_active", 0.0)),
            "val_dice": format_float(val_metrics.get("dice", None)),
            "val_iou": format_float(val_metrics.get("iou", None)),
            "val_soft_dice": format_float(val_metrics.get("soft_dice", None)),
            "val_post_dice": format_float(val_metrics.get("post_dice", None)),
            "val_post_iou": format_float(val_metrics.get("post_iou", None)),
            "val_pred_voxels": format_float(val_metrics.get("pred_voxels", None)),
            "val_post_pred_voxels": format_float(val_metrics.get("post_pred_voxels", None)),
            "val_gt_voxels": format_float(val_metrics.get("gt_voxels", None)),
            "val_score": format_float(val_score if should_validate else None),
            "epoch_time_sec": format_float(epoch_time),
        }
        append_csv_log(log_path, row)

        flag = " *best*" if improved else ""
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"loss={train_losses['total']:.5f} | "
            f"center={train_losses.get('center', 0.0):.4f} | "
            f"w_ce={train_losses.get('w_ce', 0.0):.2f} | "
            f"w_bnd={train_losses.get('w_boundary', 0.0):.2f} | "
            f"val_dice={val_metrics.get('dice', float('nan')):.5f} | "
            f"val_post={val_metrics.get('post_dice', float('nan')):.5f} | "
            f"val_iou={val_metrics.get('iou', float('nan')):.5f} | "
            f"lr={current_lr:.3e} | "
            f"time={epoch_time:.1f}s"
            f"{' | val=skipped' if not should_validate else ''}{flag}"
        )

    print("训练完成。")
    print(f"latest checkpoint: {latest_path}")
    print(f"best checkpoint:   {best_path}")
    print(f"train log:         {log_path}")


if __name__ == "__main__":
    main()


"""
最小 YAML 配置示例：

seed: 42
deterministic: false

data:
  data_root: /home/lembert/Desktop/BME-4/BME-WYZ/data_processed/LIDC-IDRI/volumes
  train_csv: train.csv
  val_csv: val.csv
  test_csv: test.csv
  auto_split: true
  split_ratios: [0.8, 0.1, 0.1]
  split_dir: outputs/splits
  input_format: npy_volume  # auto / image_label / npz_patch / npy_volume
  cache_dir: outputs/persistent_cache
  roi_size: [96, 96, 96]
  target_spacing: [1.0, 1.0, 1.0]
  intensity_mode: ct        # ct / percentile
  ct_window: [-1000, 400]
  profile_ratios: [1, 1, 1, 0.5, 0.5]  # class0 negative : classes1-4 positive = 1:3
  samples_per_volume: 8
  small_cc_voxels: 128
  large_cc_voxels: 4096
  overwrite_aux: false
  num_workers: 4

model:
  name: hybrid_swin_sdf_core   # hybrid_swin_sdf_core / generic_swinunetr
  num_classes: 1
  img_size: [96, 96, 96]
  in_channels: 1
  swin_feature_channels: 16
  two_d_feature_channels: 16
  fusion_channels: 32
  feature_size: 24
  use_checkpoint: true
  use_absolute_position_encoding: true
  absolute_position_scale_mm: 128.0

loss:
  warmup_epochs: 15
  stage2_start_epoch: 80
  dice_weight_start: 1.0
  dice_weight_end: 0.35
  ce_weight_start: 1.0
  ce_weight_end: 0.25
  tversky_weight_start: 0.0
  tversky_weight_end: 1.2
  boundary_weight_start: 0.0
  boundary_weight_end: 0.8
  sdf_weight_start: 0.0
  sdf_weight_end: 0.3
  core_weight_start: 0.0
  core_weight_end: 0.3
  center_weight_start: 0.0
  center_weight_end: 0.01
  center_source: core
  center_use_smooth_l1: true
  tversky_alpha: 0.3
  tversky_beta: 0.7
  boundary_voxel_boost: 5.0
  dist_sigma: 0.2
  stage2_weight_power: 1.0
  max_stage2_weight: 8.0

train:
  epochs: 300
  batch_size: 1
  lr: 0.0001
  min_lr: 0.000001
  weight_decay: 0.00001
  scheduler: cosine
  amp: true
  grad_accum_steps: 4
  grad_clip_norm: 12.0
  drop_last: false

eval:
  threshold: 0.5
  spacing: [1.0, 1.0, 1.0]
  min_voxels: 16
  min_volume_mm3: null
  keep_largest: false

output:
  output_dir: outputs
"""
