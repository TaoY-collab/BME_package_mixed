# -*- coding: utf-8 -*-
"""
metrics.py

Common 3D binary segmentation metrics for Hybrid-Swin-SDF-CoreNet.

The evaluate.py script expects compute_all_metrics(pred, gt, spacing, threshold)
returning:
    dice, iou, precision, recall, hd95, assd, bf1, vol_diff
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


def ensure_3d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 5 and x.shape[0] == 1 and x.shape[1] == 1:
        return x[0, 0]
    if x.ndim == 4 and x.shape[0] == 1:
        return x[0]
    if x.ndim == 4 and x.shape[1] == 1:
        return x[0, 0]
    if x.ndim == 3:
        return x
    x = np.squeeze(x)
    if x.ndim != 3:
        raise ValueError(f"无法转换为 3D array，当前 shape={x.shape}")
    return x


def to_binary(x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (ensure_3d(x) >= float(threshold)).astype(bool)


def extract_surface(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    structure = generate_binary_structure(rank=3, connectivity=1)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    return mask & (~eroded)


def surface_distances(pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)

    if not np.any(pred) or not np.any(gt):
        return np.asarray([], dtype=np.float64)

    pred_surface = extract_surface(pred)
    gt_surface = extract_surface(gt)

    if not np.any(pred_surface) or not np.any(gt_surface):
        return np.asarray([], dtype=np.float64)

    spacing = tuple(float(v) for v in spacing)

    # distance_transform_edt samples parameter follows array axis order [D,H,W].
    dt_gt = distance_transform_edt(~gt_surface, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_surface, sampling=spacing)

    d_pred_to_gt = dt_gt[pred_surface]
    d_gt_to_pred = dt_pred[gt_surface]

    return np.concatenate([d_pred_to_gt, d_gt_to_pred]).astype(np.float64)


def boundary_f1(pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float], tolerance_mm: float = 2.0) -> float:
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)

    if not np.any(pred) and not np.any(gt):
        return 1.0
    if not np.any(pred) or not np.any(gt):
        return 0.0

    pred_surface = extract_surface(pred)
    gt_surface = extract_surface(gt)
    if not np.any(pred_surface) and not np.any(gt_surface):
        return 1.0
    if not np.any(pred_surface) or not np.any(gt_surface):
        return 0.0

    spacing = tuple(float(v) for v in spacing)
    dt_gt = distance_transform_edt(~gt_surface, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_surface, sampling=spacing)

    pred_match = dt_gt[pred_surface] <= float(tolerance_mm)
    gt_match = dt_pred[gt_surface] <= float(tolerance_mm)

    precision = float(pred_match.mean()) if pred_match.size > 0 else 0.0
    recall = float(gt_match.mean()) if gt_match.size > 0 else 0.0

    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def compute_all_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    threshold: float = 0.5,
) -> Dict[str, float]:
    pred_bin = to_binary(pred, threshold=threshold)
    gt_bin = to_binary(gt, threshold=0.5)

    tp = float(np.logical_and(pred_bin, gt_bin).sum())
    fp = float(np.logical_and(pred_bin, ~gt_bin).sum())
    fn = float(np.logical_and(~pred_bin, gt_bin).sum())

    pred_sum = float(pred_bin.sum())
    gt_sum = float(gt_bin.sum())
    union = float(np.logical_or(pred_bin, gt_bin).sum())

    dice = (2.0 * tp + 1.0) / (pred_sum + gt_sum + 1.0)
    iou = (tp + 1.0) / (union + 1.0)
    precision = (tp + 1.0) / (tp + fp + 1.0)
    recall = (tp + 1.0) / (tp + fn + 1.0)

    if pred_sum == 0.0 and gt_sum == 0.0:
        hd95 = 0.0
        assd = 0.0
        bf1 = 1.0
    elif pred_sum == 0.0 or gt_sum == 0.0:
        hd95 = float("nan")
        assd = float("nan")
        bf1 = 0.0
    else:
        dists = surface_distances(pred_bin, gt_bin, spacing=spacing)
        if dists.size == 0:
            hd95 = float("nan")
            assd = float("nan")
        else:
            hd95 = float(np.percentile(dists, 95))
            assd = float(np.mean(dists))
        bf1 = float(boundary_f1(pred_bin, gt_bin, spacing=spacing, tolerance_mm=2.0))

    voxel_volume = float(np.prod(np.asarray(spacing, dtype=np.float64)))
    pred_vol = pred_sum * voxel_volume
    gt_vol = gt_sum * voxel_volume
    vol_diff = pred_vol - gt_vol

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
    out = {}
    for key in keys:
        values = np.asarray([r[key] for r in rows], dtype=np.float64)
        out[key] = float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")
    return out
