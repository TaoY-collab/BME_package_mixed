# -*- coding: utf-8 -*-
"""Common 3D binary segmentation metrics."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
)


def ensure_3d(x: np.ndarray) -> np.ndarray:
    """Remove singleton batch/channel axes and return a 3D array."""
    array = np.asarray(x)
    squeezed = np.squeeze(array)
    if squeezed.ndim != 3:
        raise ValueError(
            f"Expected a 3D mask with optional singleton axes, got shape={array.shape}"
        )
    return squeezed


def _validate_spacing(spacing: Sequence[float]) -> Tuple[float, float, float]:
    values = tuple(float(value) for value in spacing)
    if len(values) != 3:
        raise ValueError(f"spacing must have three values [D,H,W], got {values}")
    if not all(np.isfinite(value) and value > 0.0 for value in values):
        raise ValueError(f"spacing values must be finite and positive, got {values}")
    return values


def to_binary(x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return ensure_3d(x) >= float(threshold)


def extract_surface(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"surface extraction expects a 3D mask, got {mask.shape}")
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    structure = generate_binary_structure(rank=3, connectivity=1)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    return mask & (~eroded)


def directed_surface_distances(
    source: np.ndarray,
    target: np.ndarray,
    spacing: Sequence[float],
) -> np.ndarray:
    """Distances from every source surface voxel to the target surface."""
    source_surface = extract_surface(source)
    target_surface = extract_surface(target)
    if not np.any(source_surface) or not np.any(target_surface):
        return np.asarray([], dtype=np.float64)

    spacing_dhw = _validate_spacing(spacing)
    target_distance = distance_transform_edt(
        ~target_surface,
        sampling=spacing_dhw,
    )
    return np.asarray(target_distance[source_surface], dtype=np.float64)


def surface_distances(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: Sequence[float],
) -> np.ndarray:
    """Return concatenated bidirectional surface distances in millimeters."""
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    pred_to_gt = directed_surface_distances(pred, gt, spacing)
    gt_to_pred = directed_surface_distances(gt, pred, spacing)
    if pred_to_gt.size == 0 or gt_to_pred.size == 0:
        return np.asarray([], dtype=np.float64)
    return np.concatenate([pred_to_gt, gt_to_pred])


def boundary_f1(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: Sequence[float],
    tolerance_mm: float = 2.0,
) -> float:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if float(tolerance_mm) < 0.0:
        raise ValueError("tolerance_mm must be non-negative")

    if not np.any(pred) and not np.any(gt):
        return 1.0
    if not np.any(pred) or not np.any(gt):
        return 0.0

    pred_to_gt = directed_surface_distances(pred, gt, spacing)
    gt_to_pred = directed_surface_distances(gt, pred, spacing)
    if pred_to_gt.size == 0 or gt_to_pred.size == 0:
        return 0.0

    precision = float(np.mean(pred_to_gt <= float(tolerance_mm)))
    recall = float(np.mean(gt_to_pred <= float(tolerance_mm)))
    denominator = precision + recall
    return 0.0 if denominator == 0.0 else 2.0 * precision * recall / denominator


def compute_all_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute overlap, surface, boundary, and signed volume metrics.

    HD95 and ASSD are measured in millimeters. ``vol_diff`` is signed
    ``predicted volume - ground-truth volume`` in cubic millimeters.
    """
    pred_bin = to_binary(pred, threshold=threshold)
    gt_bin = to_binary(gt, threshold=0.5)
    if pred_bin.shape != gt_bin.shape:
        raise ValueError(
            f"pred and gt must have the same shape, got {pred_bin.shape} and {gt_bin.shape}"
        )
    spacing_dhw = _validate_spacing(spacing)

    tp = int(np.logical_and(pred_bin, gt_bin).sum())
    fp = int(np.logical_and(pred_bin, ~gt_bin).sum())
    fn = int(np.logical_and(~pred_bin, gt_bin).sum())
    pred_sum = int(pred_bin.sum())
    gt_sum = int(gt_bin.sum())
    union = int(np.logical_or(pred_bin, gt_bin).sum())

    if pred_sum == 0 and gt_sum == 0:
        dice = 1.0
        iou = 1.0
        precision = 1.0
        recall = 1.0
        hd95 = 0.0
        assd = 0.0
        bf1 = 1.0
    else:
        dice = 2.0 * tp / float(pred_sum + gt_sum)
        iou = tp / float(union) if union > 0 else 0.0
        precision = tp / float(tp + fp) if tp + fp > 0 else 0.0
        recall = tp / float(tp + fn) if tp + fn > 0 else 0.0

        if pred_sum == 0 or gt_sum == 0:
            hd95 = float("nan")
            assd = float("nan")
            bf1 = 0.0
        else:
            pred_to_gt = directed_surface_distances(
                pred_bin,
                gt_bin,
                spacing_dhw,
            )
            gt_to_pred = directed_surface_distances(
                gt_bin,
                pred_bin,
                spacing_dhw,
            )
            if pred_to_gt.size == 0 or gt_to_pred.size == 0:
                hd95 = float("nan")
                assd = float("nan")
            else:
                bidirectional = np.concatenate([pred_to_gt, gt_to_pred])
                hd95 = float(np.percentile(bidirectional, 95))
                assd = float(
                    0.5 * (float(np.mean(pred_to_gt)) + float(np.mean(gt_to_pred)))
                )
            bf1 = boundary_f1(
                pred_bin,
                gt_bin,
                spacing=spacing_dhw,
                tolerance_mm=2.0,
            )

    voxel_volume = float(np.prod(np.asarray(spacing_dhw, dtype=np.float64)))
    vol_diff = float(pred_sum - gt_sum) * voxel_volume

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "hd95": float(hd95),
        "assd": float(assd),
        "bf1": float(bf1),
        "vol_diff": float(vol_diff),
    }


def nanmean_metric_dicts(rows):
    if not rows:
        return {}
    keys = rows[0].keys()
    output = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        output[key] = (
            float(np.nanmean(values))
            if not np.all(np.isnan(values))
            else float("nan")
        )
    return output
