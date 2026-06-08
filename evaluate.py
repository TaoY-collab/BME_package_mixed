# -*- coding: utf-8 -*-
"""
src/evaluate.py

Hybrid-Swin-SDF-CoreNet 项目的测试集评估脚本。

功能：
1. 加载 configs/task_adaptive.yaml；
2. 加载 HybridSwinSDFCoreNet；
3. 加载训练好的 checkpoint；
4. 在 test.csv 上逐结节评估；
5. 保存每个结节的指标到：
   outputs/metrics/test_metrics_per_nodule.csv
6. 保存整体均值和标准差到：
   outputs/metrics/test_metrics_summary.csv
7. 保存每个样本预测为 npz：
   outputs/predictions/{case_id}_{nodule_id}_pred.npz

每个预测 npz 包含：
- image
- mask
- pred_prob
- pred_mask
- pred_sdf
- pred_core_prob
- pred_core

运行示例：
python src/evaluate.py \
    --config configs/task_adaptive.yaml \
    --checkpoint outputs/checkpoints/best_dice.pt

也可以覆盖部分配置：
python src/evaluate.py \
    --config configs/task_adaptive.yaml \
    --data-root data \
    --checkpoint outputs/checkpoints/best_dice.pt \
    --output-dir outputs \
    --batch-size 1 \
    --threshold 0.5
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from monai.inferers import sliding_window_inference
from tqdm import tqdm

try:
    from .dataset_lidc import LIDCNoduleDataset, create_dataloader
    from .metrics import compute_all_metrics
    from .model_hybrid_swin_sdf_core import HybridSwinSDFCoreNet
except ImportError:
    from dataset_lidc import LIDCNoduleDataset, create_dataloader
    from metrics import compute_all_metrics
    from model_hybrid_swin_sdf_core import HybridSwinSDFCoreNet

try:
    from train import (
        DEFAULT_CONFIG_PATH,
        DEFAULT_DATA_ROOT,
        DEFAULT_OUTPUT_DIR,
        IMAGE_KEY,
        LABEL_KEY,
        MASK_ALIAS_KEY,
        PersistentDataset,
        DataLoader,
        build_model as train_build_model,
        build_transforms,
        configure_cuda_memory_limit,
        fast_postprocess_prediction,
        infer_input_format,
        load_yaml as train_load_yaml,
        load_checkpoint as train_load_checkpoint,
        model_storage_gib,
        normalize_records_for_image_label,
        parse_class_dict_or_scalar,
        read_csv_records,
        resolve_existing_path,
        resolve_split_csv_paths,
        resolve_runtime_path,
    )
except Exception as exc:
    raise ImportError(
        "evaluate.py 需要复用 train.py 的路径、数据和模型构建逻辑。\n"
        "请确认 evaluate.py 与 train.py 在同一项目目录下运行。\n"
        f"原始错误：{repr(exc)}"
    ) from exc


METRIC_KEYS = [
    "dice",
    "iou",
    "precision",
    "recall",
    "hd95",
    "assd",
    "bf1",
    "vol_diff",
]


PER_NODULE_COLUMNS = [
    "split_name",
    "input_format",
    "case_id",
    "nodule_id",
    "threshold",
    "min_voxels",
    "dice",
    "post_dice",
    "iou",
    "post_iou",
    "precision",
    "recall",
    "hd95",
    "assd",
    "bf1",
    "vol_diff",
]


SUMMARY_COLUMNS = [
    "metric",
    "mean",
    "std",
    "num_samples",
    "num_valid",
]


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="Evaluate Hybrid-Swin-SDF-CoreNet on LIDC-IDRI test set."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"配置文件路径，默认 {DEFAULT_CONFIG_PATH}",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="覆盖配置文件中的 data.data_root",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="模型 checkpoint 路径，默认使用 output_dir/checkpoints/best_dice.pt",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="覆盖配置文件中的 output.output_dir",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="测试 batch size，默认从配置 train.batch_size 读取，若没有则为 1",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="mask 二值化阈值，默认从配置 eval.threshold 读取，若没有则为 0.5",
    )

    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--split-name", type=str, default=None)
    parser.add_argument("--input-format", choices=["auto", "npz_patch", "image_label"], default=None)

    return parser.parse_args()


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    读取 YAML 配置文件。
    """
    return train_load_yaml(config_path)


def apply_cli_overrides(
    cfg: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    使用命令行参数覆盖配置文件。
    """
    cfg.setdefault("data", {})
    cfg.setdefault("train", {})
    cfg.setdefault("eval", {})
    cfg.setdefault("output", {})

    if args.data_root is not None:
        cfg["data"]["data_root"] = args.data_root

    if args.output_dir is not None:
        cfg["output"]["output_dir"] = args.output_dir

    if args.batch_size is not None:
        cfg["train"]["batch_size"] = int(args.batch_size)

    if args.threshold is not None:
        cfg["eval"]["threshold"] = float(args.threshold)

    if args.input_format is not None:
        cfg["data"]["input_format"] = args.input_format

    return cfg


def ensure_dir(path: Path) -> None:
    """
    创建目录。
    """
    path.mkdir(parents=True, exist_ok=True)


def resolve_csv_path(
    data_root: Path,
    csv_value: Optional[str],
    default_name: str,
) -> Path:
    """
    解析 test_csv 路径。

    支持：
    1. 绝对路径；
    2. 相对于 data_root 的路径；
    3. 相对于当前工作目录的路径。

    如果 csv_value 为空，则默认使用 data_root/default_name。
    """
    if csv_value is None or str(csv_value).strip() == "":
        return data_root / default_name

    raw_path = Path(str(csv_value)).expanduser()

    if raw_path.is_absolute():
        return raw_path.resolve()

    candidate_data_root = (data_root / raw_path).resolve()
    candidate_cwd = raw_path.resolve()

    if candidate_data_root.exists():
        return candidate_data_root

    if candidate_cwd.exists():
        return candidate_cwd

    return candidate_data_root


def sanitize_filename(text: str) -> str:
    """
    清理文件名中的非法字符。
    """
    text = str(text)
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = text.strip()
    if text == "":
        text = "unknown"
    return text


def split_label_from_args(args: argparse.Namespace) -> str:
    if args.split_name is not None and str(args.split_name).strip():
        return str(args.split_name).strip()
    if args.csv is not None and str(args.csv).strip():
        return Path(str(args.csv)).stem
    return str(args.split)


def move_batch_to_device(
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """
    将 batch 中的 tensor 移动到 device。
    """
    moved: Dict[str, Any] = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value

    return moved


def get_batch_string_item(value: Any, index: int) -> str:
    """
    从 DataLoader 默认 collate 后的 case_id / nodule_id 中取出单个字符串。

    常见情况：
    - list[str]
    - tuple[str]
    - str
    """
    if isinstance(value, (list, tuple)):
        return str(value[index])

    return str(value)


def build_model(cfg: Dict[str, Any]) -> HybridSwinSDFCoreNet:
    """
    根据配置构建 HybridSwinSDFCoreNet。
    """
    model_cfg = cfg.get("model", {})

    model = HybridSwinSDFCoreNet(
        img_size=tuple(model_cfg.get("img_size", [64, 64, 64])),
        in_channels=int(model_cfg.get("in_channels", 1)),
        ct_in_channels=model_cfg.get("ct_in_channels", None),
        swin_feature_channels=int(model_cfg.get("swin_feature_channels", 16)),
        two_d_feature_channels=int(model_cfg.get("two_d_feature_channels", 16)),
        two_d_mode=str(model_cfg.get("two_d_mode", "z_axis_adjacent_triplet")),
        neighbor_radius=int(model_cfg.get("neighbor_radius", 1)),
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

    return model


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    """
    加载模型 checkpoint。

    支持以下格式：
    1. train.py 保存的完整 checkpoint：
       {
           "model_state_dict": ...
       }

    2. 直接保存的 state_dict。
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(
            f"checkpoint 格式不支持：{checkpoint_path}，类型为 {type(checkpoint)}"
        )

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if len(missing_keys) > 0:
        print("警告：加载 checkpoint 时存在 missing_keys：")
        for key in missing_keys:
            print(f"  - {key}")

    if len(unexpected_keys) > 0:
        print("警告：加载 checkpoint 时存在 unexpected_keys：")
        for key in unexpected_keys:
            print(f"  - {key}")

    print(f"已加载 checkpoint：{checkpoint_path}")


def build_test_loader(
    cfg: Dict[str, Any],
    split: str = "test",
    csv_override: Optional[str] = None,
    input_format_override: Optional[str] = None,
) -> Tuple[torch.utils.data.DataLoader, str, Path]:
    """
    构建测试集 DataLoader。

    这里复用 train.py 的 CSV 解析、input_format 推断和 MONAI transforms，
    确保同一份配置在训练、验证和测试阶段对路径的解释一致。
    """
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})

    data_root = resolve_runtime_path(data_cfg.get("data_root", DEFAULT_DATA_ROOT))
    if csv_override is not None and str(csv_override).strip():
        eval_csv = resolve_csv_path(
            data_root=data_root,
            csv_value=csv_override,
            default_name=Path(str(csv_override)).name,
        )
    elif bool(data_cfg.get("auto_split", False)):
        data_cfg.setdefault("seed", cfg.get("seed", 42))
        data_root, _, val_csv = resolve_split_csv_paths(data_cfg)
        if split == "val":
            eval_csv = val_csv
        else:
            eval_csv = resolve_csv_path(
                data_root=data_root,
                csv_value=data_cfg.get("test_csv", None),
                default_name="test.csv",
            )
    else:
        csv_key = "val_csv" if split == "val" else "test_csv"
        default_name = "val.csv" if split == "val" else "test.csv"
        eval_csv = resolve_csv_path(
            data_root=data_root,
            csv_value=data_cfg.get(csv_key, None),
            default_name=default_name,
        )

    records = read_csv_records(eval_csv, data_root)
    configured_format = input_format_override or data_cfg.get("input_format", "auto")
    input_format = infer_input_format(records, configured_format)
    if input_format == "image_label":
        records = normalize_records_for_image_label(records)

    roi_size = data_cfg.get("roi_size", data_cfg.get("patch_size", [96, 96, 96]))
    target_spacing = data_cfg.get("target_spacing", [1.0, 1.0, 1.0])
    intensity_mode = data_cfg.get("intensity_mode", "ct")
    ct_window = data_cfg.get("ct_window", [-1000.0, 400.0])
    profile_ratios = data_cfg.get("profile_ratios", [1, 2, 3, 3, 1])
    small_cc_voxels = int(data_cfg.get("small_cc_voxels", 128))
    large_cc_voxels = int(data_cfg.get("large_cc_voxels", 4096))
    overwrite_aux = bool(data_cfg.get("overwrite_aux", False))

    test_tfms = build_transforms(
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

    eval_batch_size = cfg.get("eval", {}).get("batch_size", None)
    batch_size = int(eval_batch_size if eval_batch_size is not None else train_cfg.get("batch_size", 1))
    if input_format == "image_label":
        batch_size = 1
    num_workers = int(data_cfg.get("num_workers", 4) or 0)

    test_dataset = PersistentDataset(
        data=records,
        transform=test_tfms,
        cache_dir=cache_root / f"{sanitize_filename(split)}_{cache_tag}",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    print(f"测试数据格式 input_format = {input_format}")
    print(f"测试样本数：{len(test_dataset)}")
    print(f"eval split = {split}")
    print(f"eval CSV = {eval_csv}")
    print(f"PersistentDataset cache_dir = {cache_root}")

    return test_loader, input_format, eval_csv


def extract_mask_logits(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, dict):
        if "mask_logits" not in outputs:
            raise KeyError("model outputs 缺少 mask_logits")
        return outputs["mask_logits"]
    return outputs


def run_model_for_batch(
    model: nn.Module,
    images: torch.Tensor,
    input_format: str,
    roi_size: Sequence[int],
    sw_batch_size: int,
    overlap: float,
) -> Dict[str, torch.Tensor]:
    if input_format == "image_label":
        roi = tuple(int(v) for v in roi_size)

        def predictor(window: torch.Tensor) -> torch.Tensor:
            return extract_mask_logits(model(window))

        mask_logits = sliding_window_inference(
            images,
            roi_size=roi,
            sw_batch_size=max(int(sw_batch_size), 1),
            predictor=predictor,
            overlap=float(overlap),
            mode="gaussian",
        )
        return {"mask_logits": mask_logits}

    outputs = model(images)
    if not isinstance(outputs, dict):
        return {"mask_logits": outputs}
    return outputs


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    """
    将 torch tensor 转为 numpy float32。
    """
    return x.detach().cpu().numpy().astype(np.float32, copy=False)


def save_prediction_npz(
    save_path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    pred_prob: np.ndarray,
    pred_mask: np.ndarray,
    pred_sdf: np.ndarray,
    pred_core_prob: np.ndarray,
    pred_core: np.ndarray,
) -> None:
    """
    保存单个样本的预测结果。
    """
    ensure_dir(save_path.parent)

    np.savez_compressed(
        save_path,
        image=image.astype(np.float32, copy=False),
        mask=mask.astype(np.float32, copy=False),
        pred_prob=pred_prob.astype(np.float32, copy=False),
        pred_mask=pred_mask.astype(np.float32, copy=False),
        pred_sdf=pred_sdf.astype(np.float32, copy=False),
        pred_core_prob=pred_core_prob.astype(np.float32, copy=False),
        pred_core=pred_core.astype(np.float32, copy=False),
    )


def write_per_nodule_metrics(
    csv_path: Path,
    rows: Sequence[Dict[str, Any]],
) -> None:
    """
    写入每个结节的指标 CSV。
    """
    ensure_dir(csv_path.parent)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PER_NODULE_COLUMNS)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def write_summary_metrics(
    csv_path: Path,
    rows: Sequence[Dict[str, Any]],
) -> None:
    """
    写入整体 summary CSV。
    """
    ensure_dir(csv_path.parent)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def format_metric_value(value: float) -> str:
    """
    格式化指标数值。
    """
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return "nan"
    return f"{float(value):.8f}"


def summarize_metrics(metric_rows_raw: Sequence[Dict[str, float]]) -> List[Dict[str, Any]]:
    """
    计算所有指标的均值和标准差。

    对 HD95、ASSD 这类可能出现 NaN 的指标，使用 nanmean / nanstd。
    """
    if len(metric_rows_raw) == 0:
        raise ValueError("没有任何测试样本，无法汇总指标。")

    summary_rows: List[Dict[str, Any]] = []

    for key in METRIC_KEYS:
        values = np.asarray([row[key] for row in metric_rows_raw], dtype=np.float64)
        valid_mask = ~np.isnan(values)
        valid_values = values[valid_mask]

        if valid_values.size == 0:
            mean_value = float("nan")
            std_value = float("nan")
        else:
            mean_value = float(np.nanmean(values))
            std_value = float(np.nanstd(values))

        summary_rows.append(
            {
                "metric": key,
                "mean": format_metric_value(mean_value),
                "std": format_metric_value(std_value),
                "num_samples": len(values),
                "num_valid": int(valid_values.size),
            }
        )

    return summary_rows


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
    threshold: float,
    spacing: Tuple[float, float, float],
    amp_enabled: bool,
    num_classes: int,
    min_voxels: Any,
    min_volume_mm3: Any,
    keep_largest: Any,
    split_name: str,
    input_format: str,
    roi_size: Sequence[int],
    sw_batch_size: int,
    overlap: float,
) -> None:
    """
    在测试集上进行评估。
    """
    model.eval()

    metrics_dir = output_dir / "metrics"
    safe_split = sanitize_filename(split_name)
    predictions_dir = output_dir / "predictions" / safe_split

    ensure_dir(metrics_dir)
    ensure_dir(predictions_dir)

    per_nodule_csv = metrics_dir / f"{safe_split}_metrics_per_nodule.csv"
    summary_csv = metrics_dir / f"{safe_split}_metrics_summary.csv"

    per_nodule_rows: List[Dict[str, Any]] = []
    raw_metric_rows: List[Dict[str, float]] = []
    sample_index = 0

    progress = tqdm(
        test_loader,
        desc="Evaluating",
        dynamic_ncols=True,
    )

    for batch in progress:
        batch = move_batch_to_device(batch, device)

        images = batch[IMAGE_KEY].float()
        masks = batch.get(LABEL_KEY, batch.get(MASK_ALIAS_KEY))
        if masks is None:
            raise KeyError(f"测试 batch 缺少 {LABEL_KEY}/{MASK_ALIAS_KEY} 标签字段。")

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = run_model_for_batch(
                model=model,
                images=images,
                input_format=input_format,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=overlap,
            )

        mask_logits = outputs["mask_logits"]
        if num_classes <= 1:
            pred_prob_batch = torch.sigmoid(mask_logits)
        else:
            pred_prob_batch = torch.softmax(mask_logits, dim=1)

        if "sdf" in outputs:
            pred_sdf_batch = outputs["sdf"]
        else:
            pred_sdf_batch = torch.zeros_like(pred_prob_batch)

        if "core_logits" in outputs:
            pred_core_prob_batch = torch.sigmoid(outputs["core_logits"])
            pred_core_batch = (pred_core_prob_batch >= threshold).float()
        else:
            pred_core_prob_batch = torch.zeros_like(pred_prob_batch)
            pred_core_batch = torch.zeros_like(pred_prob_batch)

        batch_size = int(images.shape[0])

        for i in range(batch_size):
            case_id_value = batch.get("case_id", f"case_{sample_index:06d}")
            nodule_id_value = batch.get("nodule_id", "")
            case_id = get_batch_string_item(case_id_value, i)
            nodule_id = get_batch_string_item(nodule_id_value, i)
            if nodule_id == "":
                nodule_id = f"sample_{sample_index:06d}"

            image_i = tensor_to_numpy(images[i])
            mask_i = tensor_to_numpy(masks[i])

            pred_prob_i = tensor_to_numpy(pred_prob_batch[i])
            logits_i = tensor_to_numpy(mask_logits[i])
            pred_mask_i = fast_postprocess_prediction(
                pred=logits_i,
                num_classes=num_classes,
                threshold=threshold,
                spacing=spacing,
                min_voxels=min_voxels,
                min_volume_mm3=min_volume_mm3,
                keep_largest=keep_largest,
                connectivity=3,
                input_is_logits=True,
            ).astype(np.float32, copy=False)
            pred_sdf_i = tensor_to_numpy(pred_sdf_batch[i])
            pred_core_prob_i = tensor_to_numpy(pred_core_prob_batch[i])
            pred_core_i = tensor_to_numpy(pred_core_batch[i])

            metrics = compute_all_metrics(
                pred=pred_mask_i,
                gt=mask_i,
                spacing=spacing,
                threshold=0.5,
            )

            raw_metric_rows.append(metrics)

            row = {
                "split_name": split_name,
                "input_format": input_format,
                "case_id": case_id,
                "nodule_id": nodule_id,
                "threshold": format_metric_value(threshold),
                "min_voxels": json.dumps(min_voxels, ensure_ascii=False),
            }

            for key in METRIC_KEYS:
                row[key] = format_metric_value(metrics[key])
            row["post_dice"] = row["dice"]
            row["post_iou"] = row["iou"]

            per_nodule_rows.append(row)

            safe_case_id = sanitize_filename(case_id)
            safe_nodule_id = sanitize_filename(nodule_id)
            pred_save_path = predictions_dir / f"{safe_case_id}_{safe_nodule_id}_pred.npz"

            save_prediction_npz(
                save_path=pred_save_path,
                image=image_i,
                mask=mask_i,
                pred_prob=pred_prob_i,
                pred_mask=pred_mask_i,
                pred_sdf=pred_sdf_i,
                pred_core_prob=pred_core_prob_i,
                pred_core=pred_core_i,
            )
            sample_index += 1

        if len(raw_metric_rows) > 0:
            current_dice = np.nanmean(
                np.asarray([m["dice"] for m in raw_metric_rows], dtype=np.float64)
            )
            current_iou = np.nanmean(
                np.asarray([m["iou"] for m in raw_metric_rows], dtype=np.float64)
            )
            progress.set_postfix(
                {
                    "dice": f"{current_dice:.4f}",
                    "iou": f"{current_iou:.4f}",
                }
            )

    if len(per_nodule_rows) == 0:
        raise RuntimeError("测试集为空，未生成任何评估结果。")

    write_per_nodule_metrics(
        csv_path=per_nodule_csv,
        rows=per_nodule_rows,
    )

    summary_rows = summarize_metrics(raw_metric_rows)

    write_summary_metrics(
        csv_path=summary_csv,
        rows=summary_rows,
    )

    print("测试集评估完成")
    print(f"每个结节指标：{per_nodule_csv}")
    print(f"整体指标汇总：{summary_csv}")
    print(f"预测结果目录：{predictions_dir}")

    print("\n测试集整体结果：")
    for row in summary_rows:
        print(
            f"{row['metric']}: "
            f"mean={row['mean']}, "
            f"std={row['std']}, "
            f"valid={row['num_valid']}/{row['num_samples']}"
        )


def main() -> None:
    """
    主入口。
    """
    args = parse_args()

    cfg = load_yaml_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    output_dir = resolve_runtime_path(
        cfg.get("output", {}).get("output_dir", DEFAULT_OUTPUT_DIR)
    )

    checkpoint_path: Path
    if args.checkpoint is None or str(args.checkpoint).strip() == "":
        checkpoint_path = output_dir / "checkpoints" / "best_dice.pt"
    else:
        checkpoint_path = resolve_existing_path(args.checkpoint, base_dirs=[output_dir / "checkpoints", output_dir])

    eval_cfg = cfg.get("eval", {})
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    threshold = float(eval_cfg.get("threshold", 0.5))

    spacing_raw = eval_cfg.get("spacing", data_cfg.get("target_spacing", [1.0, 1.0, 1.0]))
    spacing = tuple(float(v) for v in spacing_raw)

    if len(spacing) != 3:
        raise ValueError(f"eval.spacing 必须是长度为 3 的序列，但当前为 {spacing_raw}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_requested = bool(train_cfg.get("amp", True))
    amp_enabled = bool(amp_requested and device.type == "cuda")

    if amp_requested and not amp_enabled:
        print("警告：配置中 amp=True，但当前不是 CUDA 环境，已自动关闭混合精度。")

    print(f"使用设备：{device}")
    print(f"AMP 混合精度：{amp_enabled}")
    print(f"输出目录：{output_dir}")
    print(f"checkpoint：{checkpoint_path}")
    print(f"threshold：{threshold}")
    print(f"spacing：{spacing}")

    split_name = split_label_from_args(args)
    test_loader, input_format, eval_csv = build_test_loader(
        cfg,
        split=args.split,
        csv_override=args.csv,
        input_format_override=args.input_format,
    )
    roi_size = data_cfg.get("roi_size", data_cfg.get("patch_size", [96, 96, 96]))
    sw_batch_size = int(eval_cfg.get("sw_batch_size", eval_cfg.get("val_sw_batch_size", 1)) or 1)
    overlap = float(eval_cfg.get("overlap", eval_cfg.get("infer_overlap", 0.5)))

    model = train_build_model(cfg)
    print(f"Model parameter/buffer storage: {model_storage_gib(model):.3f} GiB")
    model = model.to(device)
    configure_cuda_memory_limit(
        device,
        eval_cfg.get("cuda_memory_limit_gb", train_cfg.get("cuda_memory_limit_gb", 24.0)),
    )

    train_load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=None,
        scheduler=None,
        device=device,
    )

    evaluate(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir,
        threshold=threshold,
        spacing=spacing,
        amp_enabled=amp_enabled,
        num_classes=int(model_cfg.get("num_classes", 1)),
        min_voxels=parse_class_dict_or_scalar(eval_cfg.get("min_voxels", 16)),
        min_volume_mm3=parse_class_dict_or_scalar(eval_cfg.get("min_volume_mm3", None)),
        keep_largest=parse_class_dict_or_scalar(eval_cfg.get("keep_largest", False)),
        split_name=split_name,
        input_format=input_format,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
    )


if __name__ == "__main__":
    main()
