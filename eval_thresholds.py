#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Checkpoint rescue diagnostics for task-adaptive segmentation training.

The script runs a validation threshold sweep without changing the training
public API. It is intended for early checkpoints where loss is falling but
hard-threshold validation Dice is still zero.

Example:
python eval_thresholds.py --config configs/task_adaptive.yaml \
    --checkpoint outputs/checkpoints/latest.pt \
    --thresholds 0.10 0.20 0.30 0.35 0.40 0.50 \
    --min-component-voxels 0 5 16 \
    --save-visuals 5
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from monai.inferers import sliding_window_inference

try:
    import nibabel as nib
except Exception:  # pragma: no cover - optional for non-NIfTI datasets
    nib = None


DEFAULT_THRESHOLDS = (0.10, 0.20, 0.30, 0.35, 0.40, 0.50)
DEFAULT_MIN_COMPONENT_VOXELS = (0, 5, 16)
DEFAULT_CONFIG_PATH = "configs/task_adaptive.yaml"
EPS = 1e-8

train = None
torch = None


def load_training_modules() -> Tuple[Any, Any]:
    import torch as torch_module
    import train as train_module

    return train_module, torch_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep foreground thresholds for an early segmentation checkpoint."
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV path. Overrides --split.")
    parser.add_argument("--split-name", type=str, default=None, help="Reader-facing label for a custom CSV split.")
    parser.add_argument("--input-format", choices=["auto", "npz_patch", "image_label"], default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--min-component-voxels", nargs="+", type=int, default=DEFAULT_MIN_COMPONENT_VOXELS)
    parser.add_argument("--max-cases", type=int, default=0, help="0 means all cases.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--amp", choices=["true", "false"], default=None)
    parser.add_argument("--save-visuals", type=int, default=0, help="Save N diagnostic PNGs at the selected threshold.")
    return parser.parse_args()


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg.setdefault("data", {})
    cfg.setdefault("train", {})
    cfg.setdefault("output", {})

    if args.data_root is not None:
        cfg["data"]["data_root"] = args.data_root
    if args.output_dir is not None:
        cfg["output"]["output_dir"] = args.output_dir
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = int(args.num_workers)
    if args.amp is not None:
        cfg["train"]["amp"] = args.amp.lower() == "true"
    if args.input_format is not None:
        cfg["data"]["input_format"] = args.input_format

    return cfg


def output_dir_from_cfg(cfg: Dict[str, Any]) -> Path:
    return train.resolve_runtime_path(cfg.get("output", {}).get("output_dir", train.DEFAULT_OUTPUT_DIR))


def resolve_checkpoint_path(cfg: Dict[str, Any], checkpoint_arg: Optional[str]) -> Path:
    output_dir = output_dir_from_cfg(cfg)
    ckpt_dir = output_dir / "checkpoints"
    if checkpoint_arg:
        path = train.resolve_existing_path(checkpoint_arg, base_dirs=[ckpt_dir, output_dir])
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    candidates = [ckpt_dir / "latest.pt", ckpt_dir / "best_dice.pt"]
    for path in candidates:
        if path.exists():
            return path
    tried = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"No default checkpoint found. Tried:\n{tried}")


def split_label_from_args(args: argparse.Namespace) -> str:
    if args.split_name is not None and str(args.split_name).strip():
        return str(args.split_name).strip()
    if args.csv is not None and str(args.csv).strip():
        return Path(str(args.csv)).stem
    return str(args.split)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def resolve_out_path(cfg: Dict[str, Any], args: argparse.Namespace, checkpoint_path: Path) -> Path:
    if args.out:
        return train.resolve_runtime_path(args.out)

    stem = checkpoint_path.stem.replace(" ", "_")
    split_label = safe_name(split_label_from_args(args))
    return output_dir_from_cfg(cfg) / "diagnostics" / f"threshold_scan_{split_label}_{stem}.json"


def get_split_records(
    cfg: Dict[str, Any],
    split: str,
    csv_arg: Optional[str] = None,
    input_format_arg: Optional[str] = None,
) -> Tuple[Path, List[Dict[str, Any]], str, Path]:
    data_cfg = cfg.get("data", {})
    data_cfg.setdefault("seed", cfg.get("seed", train.DEFAULT_CONFIG.get("seed", 42)))

    if csv_arg is not None and str(csv_arg).strip():
        data_root = train.resolve_runtime_path(data_cfg.get("data_root", train.DEFAULT_DATA_ROOT))
        csv_path = train.resolve_path(csv_arg, data_root)
    else:
        data_root, _train_csv, val_csv = train.resolve_split_csv_paths(data_cfg)
        if split == "val":
            csv_path = val_csv
        else:
            test_csv_value = data_cfg.get("test_csv", "test.csv")
            csv_path = train.resolve_path(test_csv_value, data_root)

    records = train.read_csv_records(csv_path, data_root)
    configured_format = input_format_arg or data_cfg.get("input_format", "auto")
    input_format = train.infer_input_format(records, configured_format)
    if input_format == "image_label":
        records = train.normalize_records_for_image_label(records)

    return data_root, records, input_format, csv_path


def build_eval_loader(
    cfg: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    input_format: str,
    split_label: str,
) -> train.DataLoader:
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})

    roi_size = data_cfg.get("roi_size", data_cfg.get("patch_size", [96, 96, 96]))
    target_spacing = data_cfg.get("target_spacing", [1.0, 1.0, 1.0])
    intensity_mode = data_cfg.get("intensity_mode", "ct")
    ct_window = data_cfg.get("ct_window", [-1000.0, 400.0])
    profile_ratios = data_cfg.get("profile_ratios", [1, 1, 1, 0.5, 0.5])
    small_cc_voxels = int(data_cfg.get("small_cc_voxels", 128))
    large_cc_voxels = int(data_cfg.get("large_cc_voxels", 4096))
    overwrite_aux = bool(data_cfg.get("overwrite_aux", False))

    eval_tfms = train.build_transforms(
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

    output_dir = output_dir_from_cfg(cfg)
    cache_root = (
        train.resolve_runtime_path(data_cfg["cache_dir"])
        if data_cfg.get("cache_dir", None) is not None
        else output_dir / "persistent_cache"
    )
    train.ensure_dir(cache_root)

    cache_tag = input_format
    dataset = train.PersistentDataset(
        data=list(records),
        transform=eval_tfms,
        cache_dir=cache_root / f"{safe_name(split_label)}_threshold_{cache_tag}",
    )

    num_workers = int(data_cfg.get("num_workers", 4) or 0)
    batch_size = int(cfg.get("eval", {}).get("batch_size", train_cfg.get("batch_size", 1)) or 1)
    if input_format == "image_label":
        batch_size = 1

    return train.DataLoader(
        dataset,
        batch_size=max(batch_size, 1),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def _record_path(value: Any) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value)).expanduser()


def _load_nifti_foreground(path: Path) -> Optional[Tuple[Tuple[int, ...], int]]:
    if nib is None:
        return None
    name = path.name.lower()
    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
        return None
    arr = np.asanyarray(nib.load(str(path)).dataobj)
    return tuple(int(x) for x in arr.shape), int(np.count_nonzero(arr > 0))


def summarize_data_integrity(records: Sequence[Dict[str, Any]], input_format: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "records": len(records),
        "input_format": input_format,
        "missing_images": 0,
        "missing_labels": 0,
        "missing_npz": 0,
        "failed_reads": 0,
        "empty_labels": 0,
        "nonempty_labels": 0,
        "foreground_unchecked": 0,
        "foreground_voxels_min": None,
        "foreground_voxels_median": None,
        "foreground_voxels_max": None,
        "issues": [],
    }
    foreground_counts: List[int] = []

    for idx, record in enumerate(records):
        try:
            if input_format == "npz_patch":
                npz_path = _record_path(record.get("npz_path"))
                if npz_path is None or not npz_path.exists():
                    summary["missing_npz"] += 1
                    summary["issues"].append({"row": idx, "issue": "missing_npz", "path": str(npz_path)})
                    continue
                with np.load(npz_path) as data:
                    mask_key = "label" if "label" in data.files else "mask"
                    if mask_key not in data.files:
                        summary["failed_reads"] += 1
                        summary["issues"].append({"row": idx, "issue": "missing_mask_key", "path": str(npz_path)})
                        continue
                    foreground = int(np.count_nonzero(np.asarray(data[mask_key]) > 0))
            else:
                image_path = _record_path(record.get("image"))
                label_path = _record_path(record.get("label", record.get("mask")))
                if image_path is None or not image_path.exists():
                    summary["missing_images"] += 1
                    summary["issues"].append({"row": idx, "issue": "missing_image", "path": str(image_path)})
                if label_path is None or not label_path.exists():
                    summary["missing_labels"] += 1
                    summary["issues"].append({"row": idx, "issue": "missing_label", "path": str(label_path)})
                    continue
                loaded = _load_nifti_foreground(label_path)
                if loaded is None:
                    summary["foreground_unchecked"] += 1
                    continue
                _shape, foreground = loaded

            foreground_counts.append(foreground)
            if foreground <= 0:
                summary["empty_labels"] += 1
                summary["issues"].append({"row": idx, "issue": "empty_label"})
            else:
                summary["nonempty_labels"] += 1
        except Exception as exc:
            summary["failed_reads"] += 1
            summary["issues"].append({"row": idx, "issue": "failed_read", "error": repr(exc)})

    if foreground_counts:
        counts = np.asarray(foreground_counts, dtype=np.float64)
        summary["foreground_voxels_min"] = int(np.min(counts))
        summary["foreground_voxels_median"] = float(np.median(counts))
        summary["foreground_voxels_max"] = int(np.max(counts))

    summary["issues"] = summary["issues"][:50]
    return summary


def load_model(cfg: Dict[str, Any], checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = train.build_model(cfg)
    print(f"Model parameter/buffer storage: {train.model_storage_gib(model):.3f} GiB")
    model = model.to(device)
    train.load_checkpoint(checkpoint_path, model=model, optimizer=None, scheduler=None, device=device)
    model.eval()
    return model


def _extract_logits(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, dict):
        if "mask_logits" not in outputs:
            raise KeyError("Model output dict is missing mask_logits")
        return outputs["mask_logits"]
    return outputs


def _predict_mask_logits(
    model: torch.nn.Module,
    images: torch.Tensor,
    input_format: str,
    roi_size: Sequence[int],
    sw_batch_size: int,
    overlap: float,
) -> torch.Tensor:
    if input_format == "image_label":
        roi = tuple(int(v) for v in roi_size)

        def predictor(window: torch.Tensor) -> torch.Tensor:
            return _extract_logits(model(window))

        return sliding_window_inference(
            images,
            roi_size=roi,
            sw_batch_size=max(int(sw_batch_size), 1),
            predictor=predictor,
            overlap=float(overlap),
            mode="gaussian",
        )

    return _extract_logits(model(images))


def _foreground_prob_np(logits_i: np.ndarray) -> np.ndarray:
    logits_i = np.asarray(logits_i)
    if logits_i.ndim == 4 and logits_i.shape[0] > 1:
        return train._softmax_np(logits_i, axis=0)[1]
    if logits_i.ndim == 4 and logits_i.shape[0] == 1:
        return train._sigmoid_np(logits_i[0])
    if logits_i.ndim == 3:
        return train._sigmoid_np(logits_i)
    squeezed = np.squeeze(logits_i)
    if squeezed.ndim != 3:
        raise ValueError(f"Cannot convert logits to foreground probability, shape={logits_i.shape}")
    return train._sigmoid_np(squeezed)


def _new_accumulator(threshold: float, min_cc: int) -> Dict[str, Any]:
    return {
        "threshold": float(threshold),
        "min_component_voxels": int(min_cc),
        "cases": 0,
        "dice_sum": 0.0,
        "iou_sum": 0.0,
        "raw_dice_sum": 0.0,
        "raw_iou_sum": 0.0,
        "soft_dice_sum": 0.0,
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "pred_voxels_total": 0.0,
        "gt_voxels_total": 0.0,
        "post_pred_voxels_total": 0.0,
        "overlap_cases": 0,
        "raw_overlap_cases": 0,
        "empty_pred_cases": 0,
        "empty_gt_cases": 0,
        "fp_components_total": 0,
    }


def _count_false_positive_components(pred_bin: np.ndarray, gt_bin: np.ndarray) -> int:
    if not np.any(pred_bin):
        return 0
    structure = train.generate_binary_structure(rank=3, connectivity=3)
    labeled, num_components = train.cc_label(pred_bin, structure=structure)
    if num_components <= 0:
        return 0
    overlap_ids = np.unique(labeled[np.logical_and(pred_bin, gt_bin)])
    overlap_ids = overlap_ids[overlap_ids > 0]
    return int(num_components - overlap_ids.size)


def _update_accumulator(acc: Dict[str, Any], prob: np.ndarray, gt: np.ndarray) -> None:
    threshold = float(acc["threshold"])
    min_cc = int(acc["min_component_voxels"])

    gt_bin = train.ensure_3d_label_np(gt) > 0
    raw_pred = train.ensure_3d_label_np(prob) >= threshold
    raw_pred_bin = np.asarray(raw_pred) > 0
    pred = train.remove_small_components_3d_vectorized(
        raw_pred,
        min_voxels=min_cc,
        keep_largest=False,
        connectivity=3,
    )
    pred_bin = np.asarray(pred) > 0

    raw_tp = float(np.logical_and(raw_pred_bin, gt_bin).sum())
    raw_pred_sum = float(raw_pred_bin.sum())
    raw_union = float(np.logical_or(raw_pred_bin, gt_bin).sum())

    tp = float(np.logical_and(pred_bin, gt_bin).sum())
    fp = float(np.logical_and(pred_bin, ~gt_bin).sum())
    fn = float(np.logical_and(~pred_bin, gt_bin).sum())
    pred_sum = float(pred_bin.sum())
    gt_sum = float(gt_bin.sum())
    union = float(np.logical_or(pred_bin, gt_bin).sum())
    raw_dice = (2.0 * raw_tp + EPS) / (raw_pred_sum + gt_sum + EPS)
    raw_iou = (raw_tp + EPS) / (raw_union + EPS)
    dice = (2.0 * tp + EPS) / (pred_sum + gt_sum + EPS)
    iou = (tp + EPS) / (union + EPS)

    acc["cases"] += 1
    acc["dice_sum"] += float(dice)
    acc["iou_sum"] += float(iou)
    acc["raw_dice_sum"] += float(raw_dice)
    acc["raw_iou_sum"] += float(raw_iou)
    acc["soft_dice_sum"] += train.compute_soft_binary_dice(prob, gt_bin)
    acc["tp"] += tp
    acc["fp"] += fp
    acc["fn"] += fn
    acc["pred_voxels_total"] += pred_sum
    acc["gt_voxels_total"] += gt_sum
    acc["post_pred_voxels_total"] += pred_sum
    acc["overlap_cases"] += 1 if tp > 0.0 else 0
    acc["raw_overlap_cases"] += 1 if raw_tp > 0.0 else 0
    acc["empty_pred_cases"] += 1 if pred_sum <= 0.0 else 0
    acc["empty_gt_cases"] += 1 if gt_sum <= 0.0 else 0
    acc["fp_components_total"] += _count_false_positive_components(pred_bin, gt_bin)


def _finalize_accumulator(acc: Dict[str, Any]) -> Dict[str, Any]:
    cases = max(int(acc["cases"]), 1)
    tp = float(acc["tp"])
    fp = float(acc["fp"])
    fn = float(acc["fn"])
    pred_total = float(acc["pred_voxels_total"])
    gt_total = float(acc["gt_voxels_total"])
    return {
        "threshold": float(acc["threshold"]),
        "min_component_voxels": int(acc["min_component_voxels"]),
        "cases": int(acc["cases"]),
        "raw_dice": float(acc["raw_dice_sum"] / cases),
        "raw_iou": float(acc["raw_iou_sum"] / cases),
        "post_dice": float(acc["dice_sum"] / cases),
        "post_iou": float(acc["iou_sum"] / cases),
        "dice": float(acc["dice_sum"] / cases),
        "iou": float(acc["iou_sum"] / cases),
        "global_dice": float((2.0 * tp + EPS) / (2.0 * tp + fp + fn + EPS)),
        "soft_dice": float(acc["soft_dice_sum"] / cases),
        "precision": float(tp / max(tp + fp, EPS)),
        "recall": float(tp / max(tp + fn, EPS)),
        "pred_gt_volume_ratio": float(pred_total / max(gt_total, EPS)),
        "pred_voxels_total": pred_total,
        "gt_voxels_total": gt_total,
        "pred_voxels_mean": float(pred_total / cases),
        "gt_voxels_mean": float(gt_total / cases),
        "overlap_cases": int(acc["overlap_cases"]),
        "raw_overlap_cases": int(acc["raw_overlap_cases"]),
        "empty_pred_cases": int(acc["empty_pred_cases"]),
        "empty_gt_cases": int(acc["empty_gt_cases"]),
        "fp_per_scan": float(acc["fp_components_total"] / cases),
    }


def run_threshold_scan(
    model: torch.nn.Module,
    loader: train.DataLoader,
    device: torch.device,
    amp_policy: train.AMPPolicy,
    input_format: str,
    roi_size: Sequence[int],
    sw_batch_size: int,
    overlap: float,
    thresholds: Sequence[float],
    min_component_voxels: Sequence[int],
) -> List[Dict[str, Any]]:
    accumulators = [
        _new_accumulator(threshold, min_cc)
        for threshold in thresholds
        for min_cc in min_component_voxels
    ]

    progress = train.tqdm(loader, desc="Threshold scan", dynamic_ncols=True, leave=False)
    with torch.inference_mode():
        for batch in progress:
            batch = train.move_batch_to_device(batch, device)
            images = batch[train.IMAGE_KEY].float()
            labels = batch.get(train.LABEL_KEY, batch.get(train.MASK_ALIAS_KEY))
            if labels is None:
                raise KeyError(f"Batch is missing {train.LABEL_KEY}/{train.MASK_ALIAS_KEY}")

            with train.autocast_context(amp_policy):
                logits_tensor = _predict_mask_logits(
                    model=model,
                    images=images,
                    input_format=input_format,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    overlap=overlap,
                )

            logits = logits_tensor.detach().float().cpu().numpy()
            gt = labels.detach().cpu().numpy()
            for batch_idx in range(int(logits.shape[0])):
                prob_i = _foreground_prob_np(logits[batch_idx])
                gt_i = train.ensure_3d_label_np(gt[batch_idx])
                for acc in accumulators:
                    _update_accumulator(acc, prob_i, gt_i)

    return [_finalize_accumulator(acc) for acc in accumulators]


def _volume_ratio_distance(row: Dict[str, Any]) -> float:
    ratio = float(row.get("pred_gt_volume_ratio", 0.0))
    if ratio <= 0.0:
        return float("inf")
    return abs(math.log(ratio))


def choose_recommended_row(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        raise RuntimeError("No threshold scan results were produced.")
    return sorted(
        results,
        key=lambda row: (
            -float(row["dice"]),
            -float(row["recall"]),
            _volume_ratio_distance(row),
            float(row["threshold"]),
            int(row["min_component_voxels"]),
        ),
    )[0]


def make_decision(results: Sequence[Dict[str, Any]], warmup_epoch: int = 20) -> Dict[str, Any]:
    low_rows = [row for row in results if float(row["threshold"]) <= 0.35]
    rows_for_signal = low_rows or list(results)
    all_pred_empty = all(float(row["pred_voxels_total"]) < 1.0 for row in results)
    any_pred_nonzero = any(float(row["pred_voxels_total"]) >= 1.0 for row in results)
    any_overlap = any(int(row["overlap_cases"]) > 0 or float(row["recall"]) > 0.0 for row in rows_for_signal)
    max_soft_dice = max(float(row["soft_dice"]) for row in results) if results else 0.0

    if all_pred_empty:
        action = "stop_empty_predictions"
        rationale = "All scanned thresholds produced nearly zero foreground voxels."
        next_step = "Inspect foreground imbalance, negative logits, loss weighting, and label loading before running to Stage2."
    elif any_overlap:
        action = "continue_with_low_threshold_monitoring"
        rationale = "At least one low-threshold setting overlaps foreground labels."
        next_step = "Continue only to the warmup checkpoint and monitor low-threshold Dice/recall plus pred_gt_volume_ratio."
    elif any_pred_nonzero:
        action = "stop_spatial_or_pipeline_mismatch"
        rationale = "Predictions are non-empty but do not overlap labels at low thresholds."
        next_step = "Inspect image/label alignment, spacing/orientation, crop behavior, and mask synchronization."
    elif max_soft_dice > 0.0:
        action = "continue_soft_signal_only"
        rationale = "Soft Dice is non-zero, but hard-threshold overlap is not visible yet."
        next_step = f"Continue only to epoch {warmup_epoch}; stop if low-threshold overlap remains zero."
    else:
        action = "inspect_uncertain"
        rationale = "No reliable early rescue signal was detected."
        next_step = "Run data integrity and visual diagnostics before spending more training time."

    return {
        "action": action,
        "rationale": rationale,
        "next_step": next_step,
        "max_soft_dice": float(max_soft_dice),
        "warmup_decision_epoch": int(warmup_epoch),
    }


def _batch_text_item(value: Any, index: int, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, (list, tuple)):
        return str(value[index]) if index < len(value) else fallback
    return str(value)


def _first_channel_3d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 5:
        arr = arr[0]
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D volume after channel selection, got shape={arr.shape}")
    return arr


def _best_slice_index(gt: np.ndarray, prob: np.ndarray) -> int:
    gt_3d = train.ensure_3d_label_np(gt) > 0
    prob_3d = train.ensure_3d_label_np(prob)
    gt_scores = gt_3d.sum(axis=(1, 2))
    if float(gt_scores.max()) > 0.0:
        return int(np.argmax(gt_scores))
    prob_scores = prob_3d.sum(axis=(1, 2))
    return int(np.argmax(prob_scores))


def save_visual_diagnostics(
    model: torch.nn.Module,
    loader: train.DataLoader,
    device: torch.device,
    amp_policy: train.AMPPolicy,
    input_format: str,
    roi_size: Sequence[int],
    sw_batch_size: int,
    overlap: float,
    threshold: float,
    min_component_voxels: int,
    output_dir: Path,
    max_cases: int,
) -> List[str]:
    if max_cases <= 0:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train.ensure_dir(output_dir)
    saved: List[str] = []

    with torch.inference_mode():
        for batch in loader:
            batch = train.move_batch_to_device(batch, device)
            images = batch[train.IMAGE_KEY].float()
            labels = batch.get(train.LABEL_KEY, batch.get(train.MASK_ALIAS_KEY))
            if labels is None:
                continue

            with train.autocast_context(amp_policy):
                logits_tensor = _predict_mask_logits(
                    model=model,
                    images=images,
                    input_format=input_format,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    overlap=overlap,
                )

            logits = logits_tensor.detach().float().cpu().numpy()
            image_np = images.detach().float().cpu().numpy()
            gt_np = labels.detach().cpu().numpy()

            for batch_idx in range(int(logits.shape[0])):
                if len(saved) >= max_cases:
                    return saved

                prob_i = _foreground_prob_np(logits[batch_idx])
                gt_i = train.ensure_3d_label_np(gt_np[batch_idx])
                pred_i = train.remove_small_components_3d_vectorized(
                    prob_i >= float(threshold),
                    min_voxels=int(min_component_voxels),
                    keep_largest=False,
                    connectivity=3,
                )
                image_i = _first_channel_3d(image_np[batch_idx])
                z = _best_slice_index(gt_i, prob_i)

                case_id = _batch_text_item(batch.get("case_id"), batch_idx, f"case_{len(saved):03d}")
                nodule_id = _batch_text_item(batch.get("nodule_id"), batch_idx, f"sample_{len(saved):03d}")
                safe_name = train.re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{case_id}_{nodule_id}")
                save_path = output_dir / f"{len(saved):03d}_{safe_name}_z{z}.png"

                fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), constrained_layout=True)
                axes[0].imshow(image_i[z], cmap="gray")
                axes[0].set_title("image")
                axes[1].imshow(gt_i[z], cmap="gray")
                axes[1].set_title("gt")
                axes[2].imshow(prob_i[z], cmap="magma", vmin=0.0, vmax=1.0)
                axes[2].set_title("prob")
                axes[3].imshow(image_i[z], cmap="gray")
                axes[3].imshow(np.ma.masked_where(gt_i[z] <= 0, gt_i[z]), cmap="Greens", alpha=0.35)
                axes[3].imshow(np.ma.masked_where(pred_i[z] <= 0, pred_i[z]), cmap="Reds", alpha=0.35)
                axes[3].set_title("gt/pred")
                for ax in axes:
                    ax.axis("off")
                fig.savefig(save_path, dpi=180)
                plt.close(fig)
                saved.append(str(save_path))

    return saved


def print_result_table(results: Sequence[Dict[str, Any]]) -> None:
    for row in sorted(results, key=lambda x: (float(x["threshold"]), int(x["min_component_voxels"]))):
        print(
            f"threshold={row['threshold']:.2f} | "
            f"min_cc={row['min_component_voxels']:>2} | "
            f"Raw={row['raw_dice']:.5f} | "
            f"Post={row['post_dice']:.5f} | "
            f"Soft={row['soft_dice']:.5f} | "
            f"Recall={row['recall']:.5f} | "
            f"Precision={row['precision']:.5f} | "
            f"FP/scan={row['fp_per_scan']:.2f} | "
            f"PredGT={row['pred_gt_volume_ratio']:.3f} | "
            f"PredVox={row['pred_voxels_mean']:.1f} | "
            f"GTVox={row['gt_voxels_mean']:.1f} | "
            f"OverlapCases={row['overlap_cases']}/{row['cases']}"
        )


def main() -> None:
    global torch, train
    args = parse_args()
    train, torch = load_training_modules()

    cfg = apply_overrides(train.load_yaml(args.config), args)
    checkpoint_path = resolve_checkpoint_path(cfg, args.checkpoint)
    out_path = resolve_out_path(cfg, args, checkpoint_path)
    train.ensure_dir(out_path.parent)

    split_label = split_label_from_args(args)
    data_root, records, input_format, csv_path = get_split_records(
        cfg,
        args.split,
        csv_arg=args.csv,
        input_format_arg=args.input_format,
    )
    data_integrity = summarize_data_integrity(records, input_format)

    scan_records = list(records)
    if args.max_cases and args.max_cases > 0:
        scan_records = scan_records[: int(args.max_cases)]

    print(
        f"[INFO] threshold scan | split={split_label} | cases={len(scan_records)}/{len(records)} | "
        f"format={input_format} | checkpoint={checkpoint_path}"
    )
    print(
        "[INFO] data check | "
        f"nonempty={data_integrity['nonempty_labels']} | empty={data_integrity['empty_labels']} | "
        f"missing_npz={data_integrity['missing_npz']} | missing_images={data_integrity['missing_images']} | "
        f"missing_labels={data_integrity['missing_labels']} | failed_reads={data_integrity['failed_reads']}"
    )

    loader = build_eval_loader(cfg, scan_records, input_format, split_label)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_policy = train.build_amp_policy(device, requested=bool(cfg.get("train", {}).get("amp", True)))
    model = load_model(cfg, checkpoint_path, device)
    train.configure_cuda_memory_limit(device, cfg.get("eval", {}).get("cuda_memory_limit_gb", cfg.get("train", {}).get("cuda_memory_limit_gb", 24.0)))
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("eval", {})
    roi_size = data_cfg.get("roi_size", data_cfg.get("patch_size", [96, 96, 96]))
    sw_batch_size = int(eval_cfg.get("sw_batch_size", eval_cfg.get("val_sw_batch_size", 1)) or 1)
    overlap = float(eval_cfg.get("overlap", eval_cfg.get("infer_overlap", 0.5)))

    results = run_threshold_scan(
        model=model,
        loader=loader,
        device=device,
        amp_policy=amp_policy,
        input_format=input_format,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        thresholds=args.thresholds,
        min_component_voxels=args.min_component_voxels,
    )
    recommended = choose_recommended_row(results)
    decision = make_decision(results)

    visuals: List[str] = []
    if args.save_visuals > 0:
        visual_dir = out_path.parent / f"{out_path.stem}_visuals"
        visuals = save_visual_diagnostics(
            model=model,
            loader=loader,
            device=device,
            amp_policy=amp_policy,
            input_format=input_format,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            threshold=float(recommended["threshold"]),
            min_component_voxels=int(recommended["min_component_voxels"]),
            output_dir=visual_dir,
            max_cases=int(args.save_visuals),
        )

    print_result_table(results)
    print(
        "[INFO] recommended | "
        f"threshold={recommended['threshold']:.2f} | "
        f"min_cc={recommended['min_component_voxels']} | "
        f"PostDice={recommended['post_dice']:.5f} | "
        f"Recall={recommended['recall']:.5f} | "
        f"PredGT={recommended['pred_gt_volume_ratio']:.3f}"
    )
    print(f"[INFO] decision | action={decision['action']} | {decision['rationale']}")
    print(f"[INFO] next step | {decision['next_step']}")

    payload = {
        "checkpoint": str(checkpoint_path),
        "config": str(Path(args.config).expanduser()),
        "split": split_label,
        "configured_split": args.split,
        "csv_path": str(csv_path),
        "data_root": str(data_root),
        "input_format": input_format,
        "roi_size": [int(v) for v in roi_size],
        "sw_batch_size": int(sw_batch_size),
        "overlap": float(overlap),
        "thresholds": [float(x) for x in args.thresholds],
        "min_component_voxels": [int(x) for x in args.min_component_voxels],
        "selection_rule": "max mean Dice, then max recall, then pred_gt_volume_ratio closest to 1",
        "recommended": recommended,
        "decision": decision,
        "data_integrity": data_integrity,
        "visuals": visuals,
        "results": results,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO] saved threshold diagnostics: {out_path}")


if __name__ == "__main__":
    main()
