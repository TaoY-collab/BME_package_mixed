#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_task_adaptive.py

Task-Adaptive 3D Medical Segmentation Training Script

适配目标：
1. 新任务 image/label 全体积训练，例如 NIfTI / NRRD / MHA：CSV 需要 image,label 两列；
2. 旧项目 npz patch 训练，例如 LIDC-IDRI patch：CSV 需要 npz_path 列，npz 内建议包含 image/mask/boundary/dist_map；
3. MONAI PersistentDataset 硬盘缓存；
4. Profiled 智能切块采样：背景、普通前景、边界/难例、小连通域、大连通域按比例采样；
5. 动态复合损失：Warmup Dice+CE/BCE，中后期降低 Dice/CE 并拉高 Tversky、Boundary；
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
import inspect
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

# Reduce allocator fragmentation for large, varying 3D activation tensors.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    from monai.data import DataLoader, PersistentDataset
    from monai.inferers import sliding_window_inference
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
    "train_tversky",
    "train_boundary",
    "w_dice",
    "w_tversky",
    "w_boundary",
    "val_dice",
    "val_iou",
    "val_soft_dice",
    "val_post_dice",
    "val_post_iou",
    "val_pred_voxels",
    "val_post_pred_voxels",
    "val_gt_voxels",
    "val_pred_gt_ratio",
    "val_post_pred_gt_ratio",
    "val_threshold",
    "val_min_voxels",
    "val_score",
    "epoch_time_sec",
]

IMAGE_KEY = "image"
LABEL_KEY = "label"
MASK_ALIAS_KEY = "mask"

DEFAULT_CONFIG_PATH = "configs/task_adaptive.yaml"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_DATA_ROOT = "data"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "deterministic": False,
    "data": {
        "data_root": DEFAULT_DATA_ROOT,
        "train_csv": "train.csv",
        "val_csv": "val.csv",
        "test_csv": "test.csv",
        "auto_split": True,
        "split_ratios": [0.8, 0.1, 0.1],
        "split_dir": "outputs/splits",
        "sample_pos_neg_ratio": 3.0,
        "input_format": "image_label",
        "cache_dir": "outputs/persistent_cache",
        "roi_size": [96, 96, 96],
        "target_spacing": [1.0, 1.0, 1.0],
        "intensity_mode": "ct",
        "ct_window": [-1000, 400],
        "profile_ratios": [1, 1, 1, 0.5, 0.5],
        "samples_per_volume": 1,
        "train_repeat_factor": 1,
        "small_cc_voxels": 128,
        "large_cc_voxels": 4096,
        "overwrite_aux": False,
        "num_workers": 2,
        "pin_memory": False,
        "prefetch_factor": 1,
    },
    "model": {
        "name": "hybrid_swin_sdf_core",
        "num_classes": 1,
        "img_size": [96, 96, 96],
        "in_channels": 1,
        "ct_in_channels": 1,
        "swin_feature_channels": 16,
        "two_d_feature_channels": 16,
        "two_d_mode": "z_axis_adjacent_triplet",
        "neighbor_radius": 1,
        "two_d_slice_chunk_size": 8,
        "fusion_channels": 32,
        "feature_size": 48,
        "use_checkpoint": True,
        "use_global_position_encoding": True,
    },
    "loss": {
        "dice_weight": 1.0,
        "tversky_weight": 0.4,
        "boundary_start_epoch": 40,
        "boundary_end_epoch": 120,
        "boundary_max_weight": 0.02,
        "tversky_alpha": 0.6,
        "tversky_beta": 0.4,
        "boundary_voxel_boost": 5.0,
        "dist_sigma": 0.2,
    },
    "train": {
        "epochs": 300,
        "batch_size": 1,
        "lr": 0.0001,
        "min_lr": 0.000001,
        "weight_decay": 0.00001,
        "scheduler": "cosine",
        "amp": True,
        "cuda_memory_limit_gb": 20.0,
        "host_memory_limit_gb": 48.0,
        "grad_accum_steps": 4,
        "grad_clip_norm": 12.0,
        "drop_last": False,
    },
    "eval": {
        "threshold": 0.5,
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

class RepeatDataset(Dataset):
    """Repeat a dataset without duplicating MONAI PersistentDataset cache entries."""

    def __init__(self, dataset: Dataset, repeat_factor: int) -> None:
        self.dataset = dataset
        self.repeat_factor = max(int(repeat_factor), 1)

    def __len__(self) -> int:
        return len(self.dataset) * self.repeat_factor

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index % len(self.dataset)]


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
    parser.add_argument("--cuda-memory-limit-gb", type=float, default=None)

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
    del label
    return Path(os.path.expandvars(str(path))).expanduser().resolve()


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
    raw = Path(os.path.expandvars(str(value))).expanduser()
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
    raw = Path(os.path.expandvars(str(value))).expanduser()
    if raw.is_absolute():
        return require_project_path(raw)
    base = Path(str(base_dir)).expanduser() if base_dir is not None else PROJECT_ROOT
    return require_project_path(base / raw)


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
    def expand_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: expand_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand_value(item) for item in value]
        if isinstance(value, str):
            return os.path.expandvars(os.path.expanduser(value))
        return value

    return expand_value(cfg)


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
    if args.cuda_memory_limit_gb is not None:
        cfg["train"]["cuda_memory_limit_gb"] = float(args.cuda_memory_limit_gb)
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
    p = Path(os.path.expandvars(str(value))).expanduser()
    if p.is_absolute():
        return require_project_path(p)
    if base_dir is not None:
        return require_project_path(base_dir / p)
    return require_project_path(p)


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
        tried = "\n".join(f"  - {p}" for p in [train_csv, val_csv, test_csv])
        raise FileNotFoundError(f"No source CSV rows found for auto split:\n{tried}")

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


def discover_nifti_records(data_root: Path) -> List[Dict[str, Any]]:
    """Discover paired NIfTI image and mask files below data_root."""
    records: List[Dict[str, Any]] = []
    for image_path in sorted(data_root.rglob("*_img.nii.gz")):
        label_path = image_path.with_name(
            image_path.name.replace("_img.nii.gz", "_mask.nii.gz")
        )
        if not label_path.exists():
            continue
        case_id = image_path.name[: -len("_img.nii.gz")]
        match = re.search(r"LIDC-IDRI-\d+", case_id)
        patient_id = match.group(0) if match else case_id.split("_")[0]
        records.append(
            {
                "image": str(image_path.resolve()),
                "label": str(label_path.resolve()),
                "patient_id": patient_id,
                "case_id": case_id,
            }
        )
    return records


def ensure_nifti_manifest(data_cfg: Dict[str, Any], data_root: Path) -> Optional[Path]:
    all_csv = data_cfg.get("all_csv", None)
    if all_csv is not None and str(all_csv).strip():
        configured = resolve_path(all_csv, data_root)
        if configured.exists():
            return configured

    default_manifest = data_root / "all_cases.csv"
    if default_manifest.exists():
        data_cfg["all_csv"] = str(default_manifest)
        return default_manifest

    records = discover_nifti_records(data_root)
    if not records:
        return None
    split_dir = resolve_runtime_path(data_cfg.get("split_dir", "outputs/splits"))
    manifest = split_dir / "all_cases_discovered.csv"
    write_raw_csv_rows(
        manifest,
        records,
        ["image", "label", "patient_id", "case_id"],
    )
    data_cfg["all_csv"] = str(manifest)
    print(f"Discovered NIfTI cases: {len(records)} | manifest={manifest}")
    return manifest


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
    if len(positives) >= target_pos:
        sampled_pos = rng.sample(positives, target_pos)
    else:
        sampled_pos = list(positives)
        sampled_pos.extend(rng.choice(positives) for _ in range(target_pos - len(positives)))

    sampled = sampled_pos + list(negatives)
    rng.shuffle(sampled)
    print(
        "Train positive/negative sampling: "
        f"positive={len(sampled_pos)}, negative={len(negatives)}, ratio={pos_neg_ratio:.2f}:1"
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

    data_root = resolve_runtime_path(data_root_value)
    train_csv = resolve_path(train_csv_value, data_root)
    val_csv = resolve_path(val_csv_value, data_root)
    test_csv = resolve_path(test_csv_value, data_root)

    if bool(data_cfg.get("auto_split", False)):
        ensure_nifti_manifest(data_cfg, data_root)
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
    if configured in {"image_label", "nifti", "volume"}:
        return "image_label"
    if configured != "auto":
        raise ValueError(f"未知 input_format：{configured}")

    first = records[0]
    if "npz_path" in first and str(first.get("npz_path", "")).strip():
        return "npz_patch"
    if "image" in first and ("label" in first or "mask" in first):
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
        dist_map (legacy npz key "sdf" is read as dist_map)
        boundary
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

            if "dist_map" in keys:
                d["dist_map"] = _ensure_channel_first_np(npz["dist_map"], "dist_map").astype(np.float32)
            elif "sdf" in keys:
                d["dist_map"] = _ensure_channel_first_np(npz["sdf"], "sdf").astype(np.float32)
            if "boundary" in keys:
                d["boundary"] = _ensure_channel_first_np(npz["boundary"], "boundary").astype(np.float32)
            if "sample_label" in keys and "sample_label" not in d:
                d["sample_label"] = float(np.asarray(npz["sample_label"]).reshape(-1)[0])

        d.setdefault("case_id", npz_path.stem)
        d.setdefault("nodule_id", "")
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
        distance_clip: float = 20.0,
        inv_weight_alpha: float = 0.5,
        max_inv_weight: float = 8.0,
    ) -> None:
        super().__init__(keys=[label_key], allow_missing_keys=False)
        self.label_key = label_key
        self.overwrite_existing = bool(overwrite_existing)
        self.small_cc_voxels = int(small_cc_voxels)
        self.large_cc_voxels = int(large_cc_voxels)
        self.distance_clip = float(distance_clip)
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

        # ---------- signed distance map used only as boundary-loss weighting ----------
        need_dist_map = self.overwrite_existing or "dist_map" not in d
        if need_dist_map:
            if np.any(fg) and np.any(~fg):
                dist_out = distance_transform_edt(~fg)
                dist_in = distance_transform_edt(fg)
                dist_map = dist_out - dist_in
                dist_map = np.clip(dist_map, -self.distance_clip, self.distance_clip) / max(self.distance_clip, 1e-6)
                dist_map = dist_map.astype(np.float32)
            else:
                dist_map = np.zeros_like(label_np, dtype=np.float32)
            d["dist_map"] = self._add_ch(dist_map, np.float32)

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
        keys = [IMAGE_KEY, LABEL_KEY, MASK_ALIAS_KEY, "dist_map", "boundary", "component_weight", "sample_label"]
        spatial_keys = list(keys)
        transforms: List[Any] = [
            LoadNPZPatchd(npz_key="npz_path"),
        ]
        transforms += [
            EnsureMaskAliasd(label_key=LABEL_KEY),
            aux_builder,
            EnsureTyped(keys=spatial_keys, track_meta=False, allow_missing_keys=True),
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

    if input_format != "image_label":
        raise ValueError(f"未知 input_format：{input_format}")

    label_source_key = LABEL_KEY
    # 如果 CSV 里是 mask 列，需要在 records 里统一映射，后面 build_loaders 处理。

    base_keys = [IMAGE_KEY, label_source_key]
    crop_keys = [IMAGE_KEY, LABEL_KEY, MASK_ALIAS_KEY, "dist_map", "boundary", "component_weight", "sample_label"]

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


def build_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, str]:
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    data_cfg.setdefault("seed", cfg.get("seed", DEFAULT_CONFIG.get("seed", 42)))

    data_root, train_csv, val_csv = resolve_split_csv_paths(data_cfg)

    train_records = read_csv_records(train_csv, data_root)
    val_records = read_csv_records(val_csv, data_root)

    input_format = infer_input_format(train_records, data_cfg.get("input_format", "auto"))
    train_records = resample_positive_negative_records(
        train_records,
        pos_neg_ratio=float(data_cfg.get("sample_pos_neg_ratio", 3.0)),
        seed=int(data_cfg.get("seed", DEFAULT_CONFIG.get("seed", 42))),
    )
    if input_format == "image_label":
        train_records = normalize_records_for_image_label(train_records)
        val_records = normalize_records_for_image_label(val_records)

    roi_size = data_cfg.get("roi_size", data_cfg.get("patch_size", [96, 96, 96]))
    target_spacing = data_cfg.get("target_spacing", [1.0, 1.0, 1.0])
    intensity_mode = data_cfg.get("intensity_mode", "ct")
    ct_window = data_cfg.get("ct_window", [-1000.0, 400.0])
    profile_ratios = data_cfg.get("profile_ratios", [1, 1, 1, 0.5, 0.5])
    samples_per_volume = int(data_cfg.get("samples_per_volume", 1))
    train_repeat_factor = max(int(data_cfg.get("train_repeat_factor", 1) or 1), 1)
    small_cc_voxels = int(data_cfg.get("small_cc_voxels", 128))
    large_cc_voxels = int(data_cfg.get("large_cc_voxels", 4096))
    overwrite_aux = bool(data_cfg.get("overwrite_aux", False))

    train_tfms = build_transforms(
        input_format=input_format,
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
        input_format=input_format,
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
    cache_root = (
        resolve_runtime_path(data_cfg["cache_dir"])
        if data_cfg.get("cache_dir", None) is not None
        else output_dir / "persistent_cache"
    )
    ensure_dir(cache_root)

    cache_tag = input_format

    base_train_ds = PersistentDataset(
        data=train_records,
        transform=train_tfms,
        cache_dir=cache_root / f"train_{cache_tag}",
    )
    train_ds = (
        RepeatDataset(base_train_ds, train_repeat_factor)
        if train_repeat_factor > 1
        else base_train_ds
    )
    val_ds = PersistentDataset(
        data=val_records,
        transform=val_tfms,
        cache_dir=cache_root / f"val_{cache_tag}",
    )

    batch_size = int(train_cfg.get("batch_size", 1) or 1)
    if batch_size <= 0:
        raise ValueError(f"train.batch_size 必须大于 0，当前为 {batch_size}")
    num_workers = int(data_cfg.get("num_workers", 2) or 0)
    pin_memory = bool(data_cfg.get("pin_memory", False)) and torch.cuda.is_available()
    loader_kwargs: Dict[str, Any] = {}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(
            int(data_cfg.get("prefetch_factor", 1) or 1),
            1,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=bool(train_cfg.get("drop_last", False)),
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        drop_last=False,
        **loader_kwargs,
    )

    print(f"数据格式 input_format = {input_format}")
    print(
        "训练样本数："
        f"base={len(base_train_ds)}, repeat_factor={train_repeat_factor}, "
        f"loader_items={len(train_ds)}, dynamic_patches_per_epoch={len(train_ds) * samples_per_volume}，"
        f"验证样本数：{len(val_ds)}"
    )
    print(f"PersistentDataset cache_dir = {cache_root}")

    return train_loader, val_loader, input_format


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
        feature_size: int = 24,
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
                two_d_mode=str(model_cfg.get("two_d_mode", "z_axis_adjacent_triplet")),
                neighbor_radius=int(model_cfg.get("neighbor_radius", 1)),
                two_d_slice_chunk_size=int(model_cfg.get("two_d_slice_chunk_size", 8)),
                fusion_channels=int(model_cfg.get("fusion_channels", 32)),
                feature_size=int(model_cfg.get("feature_size", 48)),
                use_checkpoint=bool(model_cfg.get("use_checkpoint", True)),
                use_global_position_encoding=bool(
                    model_cfg.get(
                        "use_global_position_encoding",
                        model_cfg.get("use_absolute_position_encoding", True),
                    )
                ),
            )

    if name in {"swinunetr", "generic_swinunetr", "swin_unetr"} or HybridSwinSDFCoreNet is None:
        return GenericSwinUNETRDict(
            img_size=img_size,
            in_channels=int(model_cfg.get("in_channels", 1)),
            num_classes=num_classes,
            feature_size=int(model_cfg.get("feature_size", 24)),
            use_checkpoint=bool(model_cfg.get("use_checkpoint", True)),
        )

    raise ValueError(f"未知 model.name：{name}")


# =========================
# 动态对抗 Loss
# =========================

import losses as losses_module
from losses import AdaptiveDynamicSegLoss


def build_criterion(cfg: Dict[str, Any]) -> AdaptiveDynamicSegLoss:
    """
    构建三阶段动态复合损失。

    YAML 推荐配置：
    loss:
      dice_weight: 1.0
      tversky_weight: 0.4
      boundary_start_epoch: 40
      boundary_end_epoch: 120
      boundary_max_weight: 0.02
      tversky_alpha: 0.6
      tversky_beta: 0.4
    """
    model_cfg = cfg.get("model", {})
    loss_cfg = cfg.get("loss", {})

    criterion_kwargs = dict(
        num_classes=int(model_cfg.get("num_classes", 1)),
        include_background=bool(loss_cfg.get("include_background", False)),
        dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
        tversky_weight=float(loss_cfg.get("tversky_weight", 0.4)),
        boundary_start_epoch=int(loss_cfg.get("boundary_start_epoch", 40)),
        boundary_end_epoch=int(loss_cfg.get("boundary_end_epoch", 120)),
        boundary_max_weight=float(loss_cfg.get("boundary_max_weight", 0.02)),
        tversky_alpha=float(loss_cfg.get("tversky_alpha", 0.6)),
        tversky_beta=float(loss_cfg.get("tversky_beta", 0.4)),
        boundary_voxel_boost=float(loss_cfg.get("boundary_voxel_boost", 5.0)),
        dist_sigma=float(loss_cfg.get("dist_sigma", 0.2)),
    )
    supported = set(inspect.signature(AdaptiveDynamicSegLoss.__init__).parameters)
    missing = sorted(set(criterion_kwargs) - supported)
    if missing:
        losses_path = Path(getattr(losses_module, "__file__", "losses.py")).resolve()
        raise RuntimeError(
            "train.py and losses.py are from different code versions. "
            f"AdaptiveDynamicSegLoss is missing arguments: {missing}. "
            f"Imported losses.py: {losses_path}. "
            "Upload the matching losses.py together with train.py, then restart."
        )
    return AdaptiveDynamicSegLoss(**criterion_kwargs)


# =========================
# AMP 策略
# =========================

@dataclass
class AMPPolicy:
    enabled: bool
    dtype: torch.dtype
    scaler: Optional[torch.cuda.amp.GradScaler]


def configure_cuda_memory_limit(
    device: torch.device,
    limit_gb: Optional[float],
) -> Optional[float]:
    if device.type != "cuda" or limit_gb is None:
        return None

    limit_gb = float(limit_gb)
    if limit_gb <= 0.0:
        return None

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    props = torch.cuda.get_device_properties(device_index)
    total_gb = float(props.total_memory) / float(1024 ** 3)
    if total_gb <= 0.0:
        return None

    fraction = min(max(limit_gb / total_gb, 0.0), 1.0)
    if fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)
        print(
            "CUDA memory cap enabled: "
            f"limit={limit_gb:.2f} GB | visible_total={total_gb:.2f} GB | fraction={fraction:.4f}"
        )
    else:
        print(
            "CUDA memory cap not applied: "
            f"requested_limit={limit_gb:.2f} GB >= visible_total={total_gb:.2f} GB"
        )
    return fraction


def model_storage_gib(model: nn.Module) -> float:
    storage_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in list(model.parameters()) + list(model.buffers())
    )
    return float(storage_bytes) / float(1024 ** 3)


def enforce_host_memory_limit(limit_gb: Optional[float], context: str) -> float:
    """Stop cleanly before the process tree exhausts server RAM."""
    if limit_gb is None or float(limit_gb) <= 0.0:
        return 0.0
    try:
        import psutil

        process = psutil.Process(os.getpid())
        rss_bytes = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                rss_bytes += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except (ImportError, OSError):
        rss_bytes = 0
        if os.name == "nt":
            try:
                import ctypes
                from ctypes import wintypes

                class ProcessMemoryCounters(ctypes.Structure):
                    _fields_ = [
                        ("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]

                counters = ProcessMemoryCounters()
                counters.cb = ctypes.sizeof(counters)
                get_current_process = ctypes.windll.kernel32.GetCurrentProcess
                get_current_process.restype = wintypes.HANDLE
                get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
                get_process_memory_info.argtypes = [
                    wintypes.HANDLE,
                    ctypes.POINTER(ProcessMemoryCounters),
                    wintypes.DWORD,
                ]
                get_process_memory_info.restype = wintypes.BOOL
                handle = get_current_process()
                if get_process_memory_info(
                    handle,
                    ctypes.byref(counters),
                    counters.cb,
                ):
                    rss_bytes = int(counters.WorkingSetSize)
            except (AttributeError, OSError):
                rss_bytes = 0
        else:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            pending = [os.getpid()]
            seen = set()
            while pending:
                pid = pending.pop()
                if pid in seen:
                    continue
                seen.add(pid)
                try:
                    statm = Path(f"/proc/{pid}/statm").read_text(
                        encoding="ascii"
                    ).split()
                    rss_bytes += int(statm[1]) * page_size
                    children = Path(
                        f"/proc/{pid}/task/{pid}/children"
                    ).read_text(encoding="ascii").split()
                    pending.extend(int(child_pid) for child_pid in children)
                except (FileNotFoundError, PermissionError, IndexError, ValueError):
                    continue

    rss_gb = float(rss_bytes) / float(1024 ** 3)
    if rss_gb > float(limit_gb):
        raise MemoryError(
            f"Host memory budget exceeded during {context}: "
            f"rss={rss_gb:.2f} GB > limit={float(limit_gb):.2f} GB. "
            "Reduce data.num_workers, data.samples_per_volume, or ROI size."
        )
    return rss_gb


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


# =========================
# 后处理与指标
# =========================

def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


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
        prob = _sigmoid_np(pred) if input_is_logits else pred
        raw = prob >= float(threshold)
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
    host_memory_limit_gb: Optional[float] = None,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accum_steps = max(int(accum_steps), 1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    loss_sums = {
        "total": 0.0,
        "dice": 0.0,
        "tversky": 0.0,
        "boundary": 0.0,
        "w_dice": 0.0,
        "w_tversky": 0.0,
        "w_boundary": 0.0,
    }
    count = 0

    progress = tqdm(loader, desc=f"Epoch {epoch} [train]", dynamic_ncols=True, leave=False)

    for step, batch in enumerate(progress):
        host_rss_gb = enforce_host_memory_limit(
            host_memory_limit_gb,
            context=f"training epoch {epoch}, step {step}",
        )
        batch = move_batch_to_device(batch, device)
        images = batch[IMAGE_KEY].float()

        with autocast_context(amp_policy):
            outputs = model(images)
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

        postfix = {
            "loss": f"{float(loss.detach().cpu()):.4f}",
            "dice_l": f"{float(loss_dict['dice'].detach().cpu()):.4f}",
            "w_tv": f"{float(loss_dict.get('w_tversky', torch.tensor(0.0)).detach().cpu()):.3f}",
            "w_bnd": f"{float(loss_dict.get('w_boundary', torch.tensor(0.0)).detach().cpu()):.3f}",
        }
        if device.type == "cuda":
            postfix["gpu_gb"] = (
                f"{torch.cuda.memory_allocated(device) / (1024 ** 3):.1f}/"
                f"{torch.cuda.memory_reserved(device) / (1024 ** 3):.1f}"
            )
        if host_rss_gb > 0.0:
            postfix["ram_gb"] = f"{host_rss_gb:.1f}"
        progress.set_postfix(postfix)

    averaged = average_loss_dict(loss_sums, count)
    if device.type == "cuda":
        averaged["peak_allocated_gb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        averaged["peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    return averaged


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
    host_memory_limit_gb: Optional[float] = None,
    input_format: str = "npz_patch",
    roi_size: Sequence[int] = (96, 96, 96),
    sw_batch_size: int = 1,
    overlap: float = 0.5,
) -> Dict[str, float]:
    model.eval()
    rows: List[Dict[str, float]] = []

    progress = tqdm(loader, desc="Validation", dynamic_ncols=True, leave=False)
    for step, batch in enumerate(progress):
        enforce_host_memory_limit(
            host_memory_limit_gb,
            context=f"validation step {step}",
        )
        batch = move_batch_to_device(batch, device)
        images = batch[IMAGE_KEY].float()
        labels = batch.get(LABEL_KEY, batch.get(MASK_ALIAS_KEY))
        if labels is None:
            raise KeyError(f"验证 batch 缺少 {LABEL_KEY}/{MASK_ALIAS_KEY} 标签字段。")

        with autocast_context(amp_policy):
            if input_format == "image_label":
                def predictor(window: torch.Tensor) -> torch.Tensor:
                    window_outputs = model(window)
                    if isinstance(window_outputs, dict):
                        return window_outputs["mask_logits"]
                    return window_outputs

                mask_logits = sliding_window_inference(
                    inputs=images,
                    roi_size=tuple(int(v) for v in roi_size),
                    sw_batch_size=max(int(sw_batch_size), 1),
                    predictor=predictor,
                    overlap=float(overlap),
                    mode="gaussian",
                )
                outputs = {"mask_logits": mask_logits}
            else:
                outputs = model(images)
                if not isinstance(outputs, dict):
                    outputs = {"mask_logits": outputs}

        logits = outputs["mask_logits"].detach().float().cpu().numpy()
        gt = labels.detach().cpu().numpy()

        b = int(logits.shape[0])
        for i in range(b):
            gt_i = ensure_3d_label_np(gt[i])
            if num_classes <= 1:
                prob_i = _sigmoid_np(logits[i])
                raw_pred_i = ensure_3d_label_np(prob_i) >= float(threshold)
                soft_dice = compute_soft_binary_dice(prob_i, gt_i)
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
        if rows:
            progress.set_postfix({
                "raw_dice": f"{np.mean([r['dice'] for r in rows]):.4f}",
                "post_dice": f"{np.mean([r['post_dice'] for r in rows]):.4f}",
            })

    if not rows:
        return {
            "dice": 0.0,
            "iou": 0.0,
            "score": 0.0,
        }

    pred_voxels = float(np.mean([r["pred_voxels"] for r in rows]))
    post_pred_voxels = float(np.mean([r["post_pred_voxels"] for r in rows]))
    gt_voxels = float(np.mean([r["gt_voxels"] for r in rows]))
    pred_gt_ratio = pred_voxels / max(gt_voxels, 1e-8)
    post_pred_gt_ratio = post_pred_voxels / max(gt_voxels, 1e-8)
    return {
        "dice": float(np.mean([r["dice"] for r in rows])),
        "iou": float(np.mean([r["iou"] for r in rows])),
        "soft_dice": float(np.nanmean([r["soft_dice"] for r in rows])),
        "post_dice": float(np.mean([r["post_dice"] for r in rows])),
        "post_iou": float(np.mean([r["post_iou"] for r in rows])),
        "pred_voxels": pred_voxels,
        "post_pred_voxels": post_pred_voxels,
        "gt_voxels": gt_voxels,
        "pred_gt_ratio": float(pred_gt_ratio),
        "post_pred_gt_ratio": float(post_pred_gt_ratio),
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
        "architecture_version": getattr(
            model,
            "ARCHITECTURE_VERSION",
            "generic_swinunetr_v1",
        ),
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
    expected_version = getattr(
        model,
        "ARCHITECTURE_VERSION",
        "generic_swinunetr_v1",
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint_version = checkpoint.get("architecture_version", None)
        if checkpoint_version != expected_version:
            raise RuntimeError(
                "Checkpoint architecture is incompatible with the current model. "
                f"Expected {expected_version!r}, got {checkpoint_version!r}. "
                "The Z-axis-only feature_size=48 model must start a new experiment."
            )
        state = strip_module_prefix(checkpoint["model_state_dict"])
        model.load_state_dict(state, strict=True)
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return int(checkpoint.get("epoch", 0)), float(checkpoint.get("best_score", -1.0))

    raise RuntimeError(
        f"无法识别或拒绝旧 checkpoint 格式：{path}。"
        "当前架构要求包含 architecture_version 的完整 checkpoint。"
    )


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
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    gpu_memory_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if device.type == "cuda"
        else 0.0
    )
    print(
        f"使用设备：{device} | name={gpu_name} | memory_gb={gpu_memory_gb:.1f}"
    )
    train_loader, val_loader, input_format = build_loaders(cfg)

    if device.type == "cuda":
        torch.cuda.init()
    configure_cuda_memory_limit(device, train_cfg.get("cuda_memory_limit_gb", 20.0))

    model = build_model(cfg)
    print(f"Model parameter/buffer storage: {model_storage_gib(model):.3f} GiB")
    model = model.to(device)
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
    spacing = eval_cfg.get("spacing", data_cfg.get("target_spacing", [1.0, 1.0, 1.0]))
    min_voxels = parse_class_dict_or_scalar(eval_cfg.get("min_voxels", 16))
    min_volume_mm3 = parse_class_dict_or_scalar(eval_cfg.get("min_volume_mm3", None))
    keep_largest = parse_class_dict_or_scalar(eval_cfg.get("keep_largest", False))
    roi_size = data_cfg.get(
        "roi_size",
        data_cfg.get("patch_size", model_cfg.get("img_size", [96, 96, 96])),
    )
    val_sw_batch_size = max(
        int(eval_cfg.get("val_sw_batch_size", eval_cfg.get("sw_batch_size", 1)) or 1),
        1,
    )
    val_overlap = float(eval_cfg.get("overlap", eval_cfg.get("infer_overlap", 0.5)))
    if not 0.0 <= val_overlap < 1.0:
        raise ValueError(f"eval.overlap must be in [0, 1), got {val_overlap}")

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
    loader_batch_size = int(train_cfg.get("batch_size", 1) or 1)
    samples_per_volume = (
        int(data_cfg.get("samples_per_volume", 1) or 1)
        if input_format == "image_label"
        else 1
    )
    train_repeat_factor = max(int(data_cfg.get("train_repeat_factor", 1) or 1), 1)
    print(json.dumps({
        "architecture_version": getattr(
            model,
            "ARCHITECTURE_VERSION",
            "generic_swinunetr_v1",
        ),
        "input_format": input_format,
        "roi_size": data_cfg.get("roi_size", [96, 96, 96]),
        "feature_size": int(model_cfg.get("feature_size", 48)),
        "swin_feature_channels": int(model_cfg.get("swin_feature_channels", 16)),
        "two_d_feature_channels": int(model_cfg.get("two_d_feature_channels", 16)),
        "fusion_channels": int(model_cfg.get("fusion_channels", 32)),
        "two_d_mode": model_cfg.get("two_d_mode", "z_axis_adjacent_triplet"),
        "neighbor_radius": int(model_cfg.get("neighbor_radius", 1)),
        "epochs": epochs,
        "batch_size": loader_batch_size,
        "samples_per_volume": samples_per_volume,
        "train_repeat_factor": train_repeat_factor,
        "effective_patch_batch": loader_batch_size * samples_per_volume,
        "grad_accum_steps": accum_steps,
        "effective_optimizer_patch_batch": loader_batch_size * samples_per_volume * accum_steps,
        "dynamic_patches_per_epoch": len(train_loader.dataset) * samples_per_volume,
        "num_classes": num_classes,
        "amp": bool(train_cfg.get("amp", True)),
        "cuda_memory_limit_gb": train_cfg.get("cuda_memory_limit_gb", 20.0),
        "host_memory_limit_gb": train_cfg.get("host_memory_limit_gb", 48.0),
        "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "threshold": threshold,
        "min_voxels": min_voxels,
        "min_volume_mm3": min_volume_mm3,
        "keep_largest": keep_largest,
        "val_sw_batch_size": val_sw_batch_size,
        "val_overlap": val_overlap,
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
            host_memory_limit_gb=train_cfg.get("host_memory_limit_gb", 48.0),
        )

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
            host_memory_limit_gb=train_cfg.get("host_memory_limit_gb", 48.0),
            input_format=input_format,
            roi_size=roi_size,
            sw_batch_size=val_sw_batch_size,
            overlap=val_overlap,
        )

        if scheduler is not None:
            scheduler.step()

        current_lr = get_current_lr(optimizer)
        epoch_time = time.time() - t0
        val_score = float(val_metrics.get("score", val_metrics.get("dice", 0.0)))

        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_score=max(best_score, val_score),
            cfg=cfg,
        )

        improved = val_score > best_score
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
            "train_tversky": format_float(train_losses["tversky"]),
            "train_boundary": format_float(train_losses["boundary"]),
            "w_dice": format_float(train_losses.get("w_dice", 0.0)),
            "w_tversky": format_float(train_losses.get("w_tversky", 0.0)),
            "w_boundary": format_float(train_losses.get("w_boundary", 0.0)),
            "val_dice": format_float(val_metrics["dice"]),
            "val_iou": format_float(val_metrics["iou"]),
            "val_soft_dice": format_float(val_metrics.get("soft_dice", None)),
            "val_post_dice": format_float(val_metrics.get("post_dice", None)),
            "val_post_iou": format_float(val_metrics.get("post_iou", None)),
            "val_pred_voxels": format_float(val_metrics.get("pred_voxels", None)),
            "val_post_pred_voxels": format_float(val_metrics.get("post_pred_voxels", None)),
            "val_gt_voxels": format_float(val_metrics.get("gt_voxels", None)),
            "val_pred_gt_ratio": format_float(val_metrics.get("pred_gt_ratio", None)),
            "val_post_pred_gt_ratio": format_float(val_metrics.get("post_pred_gt_ratio", None)),
            "val_threshold": format_float(threshold),
            "val_min_voxels": min_voxels,
            "val_score": format_float(val_score),
            "epoch_time_sec": format_float(epoch_time),
        }
        append_csv_log(log_path, row)

        flag = " *best*" if improved else ""
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"loss={train_losses['total']:.5f} | "
            f"w_tv={train_losses.get('w_tversky', 0.0):.2f} | "
            f"w_bnd={train_losses.get('w_boundary', 0.0):.2f} | "
            f"val_dice={val_metrics['dice']:.5f} | "
            f"val_soft={val_metrics.get('soft_dice', 0.0):.5f} | "
            f"val_post={val_metrics.get('post_dice', 0.0):.5f} | "
            f"val_iou={val_metrics['iou']:.5f} | "
            f"pred/gt={val_metrics.get('pred_gt_ratio', 0.0):.3f} | "
            f"pred_vox={val_metrics.get('pred_voxels', 0.0):.1f} | "
            f"gt_vox={val_metrics.get('gt_voxels', 0.0):.1f} | "
            f"thr={threshold:.2f} | "
            f"min_cc={min_voxels} | "
            f"gpu_peak={train_losses.get('peak_allocated_gb', 0.0):.1f}/"
            f"{train_losses.get('peak_reserved_gb', 0.0):.1f}GB | "
            f"lr={current_lr:.3e} | "
            f"time={epoch_time:.1f}s{flag}"
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
  data_root: data/new_task
  train_csv: train.csv
  val_csv: val.csv
  test_csv: test.csv
  auto_split: true
  split_ratios: [0.8, 0.1, 0.1]
  split_dir: outputs/splits
  input_format: auto        # auto / image_label / npz_patch
  cache_dir: outputs/persistent_cache
  roi_size: [96, 96, 96]
  target_spacing: [1.0, 1.0, 1.0]
  intensity_mode: ct        # ct / percentile
  ct_window: [-1000, 400]
  profile_ratios: [1, 1, 1, 0.5, 0.5]  # class0 negative : classes1-4 positive = 1:3
  samples_per_volume: 1
  train_repeat_factor: 1
  small_cc_voxels: 128
  large_cc_voxels: 4096
  overwrite_aux: false
  num_workers: 2
  pin_memory: false
  prefetch_factor: 1

model:
  name: hybrid_swin_sdf_core   # hybrid_swin_sdf_core / generic_swinunetr
  num_classes: 1
  img_size: [96, 96, 96]
  in_channels: 1
  swin_feature_channels: 16
  two_d_feature_channels: 16
  two_d_mode: z_axis_adjacent_triplet
  neighbor_radius: 1
  two_d_slice_chunk_size: 8
  fusion_channels: 32
  feature_size: 48
  use_checkpoint: true
  use_global_position_encoding: true

loss:
  dice_weight: 1.0
  tversky_weight: 0.4
  boundary_start_epoch: 40
  boundary_end_epoch: 120
  boundary_max_weight: 0.02
  tversky_alpha: 0.6
  tversky_beta: 0.4
  boundary_voxel_boost: 5.0
  dist_sigma: 0.2

train:
  epochs: 300
  batch_size: 1
  lr: 0.0001
  min_lr: 0.000001
  weight_decay: 0.00001
  scheduler: cosine
  amp: true
  cuda_memory_limit_gb: 20.0
  host_memory_limit_gb: 48.0
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
