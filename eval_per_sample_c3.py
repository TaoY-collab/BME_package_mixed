import argparse
import csv
import json
import os

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from monai.data import DataLoader, PersistentDataset, list_data_collate
from monai.inferers import sliding_window_inference
from scipy import ndimage as ndi

import train
import train_stage2_c3 as c3


DEFAULT_MODEL_PATH = c3.BEST_OVERALL_PATH
DEFAULT_OUT_JSON = os.path.join(c3.RUN_DIR, "per_sample_eval_thr0.40_cc5.json")
DEFAULT_OUT_CSV = os.path.join(c3.RUN_DIR, "per_sample_eval_thr0.40_cc5.csv")
DEFAULT_VIZ_DIR = os.path.join(c3.RUN_DIR, "per_sample_visualizations_thr0.40_cc5")
EVAL_VOXEL_VOLUME_MM3 = 1.5 * 1.5 * 2.0


def build_eval_loader(data_files, split_name):
    _, eval_trans = c3.build_transforms()
    cache_dir = os.path.join(train.CACHE_DIR, f"per_sample_eval_c3_{split_name}_roi96")
    os.makedirs(cache_dir, exist_ok=True)
    dataset = PersistentDataset(data=data_files, transform=eval_trans, cache_dir=cache_dir)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=train.VAL_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=train.VAL_NUM_WORKERS > 0,
        collate_fn=list_data_collate,
    )


def load_model(model_path, device):
    model = train.build_model(device)
    try:
        payload = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(model_path, map_location=device)
    state = train.extract_model_state(payload)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload


def safe_case_id(batch, fallback_idx):
    case_id = batch.get("id", f"case_{fallback_idx:04d}")
    if isinstance(case_id, (list, tuple)):
        return str(case_id[0])
    return str(case_id)


def dice_binary_np(pred, target):
    pred_sum = int(pred.sum())
    target_sum = int(target.sum())
    if pred_sum + target_sum == 0:
        return 1.0
    intersection = int(np.logical_and(pred, target).sum())
    return (2.0 * intersection) / max(pred_sum + target_sum, 1)


def component_count(mask):
    _, num_components = ndi.label(mask.astype(bool))
    return int(num_components)


def remove_small_components_np(pred, min_cc):
    pred = pred.astype(bool)
    if min_cc <= 0:
        return pred
    labeled, num_components = ndi.label(pred)
    keep = np.zeros_like(pred, dtype=bool)
    for component_id in range(1, num_components + 1):
        component = labeled == component_id
        if int(component.sum()) >= min_cc:
            keep |= component
    return keep


def size_component_metrics(pred, target, low_diameter_mm=None, high_diameter_mm=None):
    labeled, num_components = ndi.label(target.astype(bool))
    dice_values = []
    detected = 0
    total = 0
    pred_voxels = 0
    label_voxels = 0
    diameters = []

    for component_id in range(1, num_components + 1):
        component = labeled == component_id
        voxels = int(component.sum())
        diameter_mm = c3.equivalent_sphere_diameter_mm(voxels, EVAL_VOXEL_VOLUME_MM3)
        if low_diameter_mm is not None and diameter_mm <= low_diameter_mm:
            continue
        if high_diameter_mm is not None and diameter_mm >= high_diameter_mm:
            continue

        total += 1
        label_voxels += voxels
        diameters.append(diameter_mm)
        component_pred = np.logical_and(pred, component)
        pred_voxels += int(component_pred.sum())
        dice_values.append(dice_binary_np(component_pred, component))
        if int(component_pred.sum()) > 0:
            detected += 1

    dice_mean = float(np.mean(dice_values)) if dice_values else 0.0
    recall = detected / max(total, 1)
    pred_gt = pred_voxels / max(label_voxels, 1)
    return {
        "count": int(total),
        "dice": dice_mean,
        "recall": float(recall),
        "pred_gt": float(pred_gt),
        "label_voxels": int(label_voxels),
        "pred_voxels_in_gt": int(pred_voxels),
        "diameters_mm": [float(x) for x in diameters],
    }


def predict_case_arrays(model, batch, device, amp_enabled, amp_dtype, threshold, min_cc):
    inputs = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].long().to(device, non_blocking=True)
    valid_shape = train.get_batch_valid_shape(batch, labels)

    with torch.inference_mode():
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype if amp_enabled else None):
            outputs = sliding_window_inference(
                inputs,
                roi_size=train.ROI_SIZE,
                sw_batch_size=train.VAL_SW_BATCH_SIZE,
                predictor=model,
                overlap=train.INFER_OVERLAP,
                mode="gaussian",
            )

        foreground_probs = torch.softmax(outputs, dim=1)[:, 1]
        pred_labels = (foreground_probs >= threshold).long()
        pred_labels = train.crop_tensor_to_shape(pred_labels, valid_shape)
        labels = train.crop_tensor_to_shape(labels, valid_shape)
        inputs = train.crop_tensor_to_shape(inputs, valid_shape)

    pred_np = pred_labels[0].detach().cpu().numpy().astype(bool)
    label_np = labels[0, 0].detach().cpu().numpy().astype(bool)
    image_np = inputs[0, 0].detach().cpu().float().numpy()
    pred_np = remove_small_components_np(pred_np, min_cc)
    return image_np, pred_np, label_np


def evaluate_one_case(model, batch, case_index, device, amp_enabled, amp_dtype, threshold, min_cc):
    _, pred_np, label_np = predict_case_arrays(
        model=model,
        batch=batch,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        threshold=threshold,
        min_cc=min_cc,
    )

    tp = int(np.logical_and(pred_np, label_np).sum())
    fp = int(np.logical_and(pred_np, np.logical_not(label_np)).sum())
    fn = int(np.logical_and(np.logical_not(pred_np), label_np).sum())
    pred_fg = int(pred_np.sum())
    label_fg = int(label_np.sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    dice = dice_binary_np(pred_np, label_np)
    pred_gt = pred_fg / max(label_fg, 1)

    small = size_component_metrics(pred_np, label_np, high_diameter_mm=c3.SMALL_NODULE_DIAMETER_MM)
    medium = size_component_metrics(
        pred_np,
        label_np,
        low_diameter_mm=c3.SMALL_NODULE_DIAMETER_MM,
        high_diameter_mm=c3.LARGE_NODULE_DIAMETER_MM,
    )
    large = size_component_metrics(pred_np, label_np, low_diameter_mm=c3.LARGE_NODULE_DIAMETER_MM)

    return {
        "case_index": int(case_index),
        "id": safe_case_id(batch, case_index),
        "threshold": float(threshold),
        "min_cc": int(min_cc),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "pred_gt": float(pred_gt),
        "pred_fg": int(pred_fg),
        "label_fg": int(label_fg),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "pred_components": component_count(pred_np),
        "gt_components": component_count(label_np),
        "dice_small": small["dice"],
        "recall_small": small["recall"],
        "pred_gt_small": small["pred_gt"],
        "small_components": small["count"],
        "small_diameters_mm": small["diameters_mm"],
        "dice_medium": medium["dice"],
        "recall_medium": medium["recall"],
        "pred_gt_medium": medium["pred_gt"],
        "medium_components": medium["count"],
        "medium_diameters_mm": medium["diameters_mm"],
        "dice_large": large["dice"],
        "recall_large": large["recall"],
        "pred_gt_large": large["pred_gt"],
        "large_components": large["count"],
        "large_diameters_mm": large["diameters_mm"],
    }


def summarize(rows):
    numeric_keys = [
        "dice",
        "precision",
        "recall",
        "pred_gt",
        "pred_components",
        "gt_components",
        "dice_small",
        "recall_small",
        "pred_gt_small",
        "small_components",
        "dice_medium",
        "recall_medium",
        "pred_gt_medium",
        "medium_components",
        "dice_large",
        "recall_large",
        "pred_gt_large",
        "large_components",
    ]
    summary = {"cases": len(rows)}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows]
        summary[f"{key}_mean"] = float(np.mean(values)) if values else 0.0
        summary[f"{key}_median"] = float(np.median(values)) if values else 0.0
    summary["pred_fg_total"] = int(sum(row["pred_fg"] for row in rows))
    summary["label_fg_total"] = int(sum(row["label_fg"] for row in rows))
    summary["pred_gt_global"] = summary["pred_fg_total"] / max(summary["label_fg_total"], 1)
    return summary


def write_csv(path, rows):
    if not rows:
        return
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sanitize_name(value):
    keep = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_", "."}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep)


def small_gt_mask(label_np):
    labeled, num_components = ndi.label(label_np.astype(bool))
    small = np.zeros_like(label_np, dtype=bool)
    for component_id in range(1, num_components + 1):
        component = labeled == component_id
        diameter_mm = c3.equivalent_sphere_diameter_mm(int(component.sum()), EVAL_VOXEL_VOLUME_MM3)
        if diameter_mm < c3.SMALL_NODULE_DIAMETER_MM:
            small |= component
    return small


def choose_visual_slice(label_np, pred_np):
    small = small_gt_mask(label_np)
    small_area = small.sum(axis=(0, 1))
    if int(small_area.max()) > 0:
        return int(np.argmax(small_area)), small

    label_area = label_np.sum(axis=(0, 1))
    if int(label_area.max()) > 0:
        return int(np.argmax(label_area)), small

    pred_area = pred_np.sum(axis=(0, 1))
    if int(pred_area.max()) > 0:
        return int(np.argmax(pred_area)), small
    return int(label_np.shape[-1] // 2), small


def overlay_mask(ax, mask, color, alpha):
    if not np.any(mask):
        return
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[..., :3] = color
    rgba[..., 3] = mask.astype(np.float32) * alpha
    ax.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower")


def add_contour(ax, mask, color, label):
    if not np.any(mask):
        return
    ax.contour(mask.T.astype(np.float32), levels=[0.5], colors=[color], linewidths=1.2)
    ax.plot([], [], color=color, linewidth=1.8, label=label)


def save_case_visualization(row, image_np, pred_np, label_np, group, rank, out_dir):
    z_idx, small_mask = choose_visual_slice(label_np, pred_np)
    image_slice = image_np[:, :, z_idx]
    label_slice = label_np[:, :, z_idx]
    pred_slice = pred_np[:, :, z_idx]
    small_slice = small_mask[:, :, z_idx]

    false_positive = np.logical_and(pred_slice, np.logical_not(label_slice))
    false_negative = np.logical_and(label_slice, np.logical_not(pred_slice))
    true_positive = np.logical_and(pred_slice, label_slice)

    os.makedirs(out_dir, exist_ok=True)
    filename = (
        f"{group}_{rank:02d}_{sanitize_name(row['id'])}_"
        f"smallDice{row['dice_small']:.3f}_slice{z_idx}.png"
    )
    path = os.path.join(out_dir, filename)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=140)
    for ax in axes:
        ax.imshow(image_slice.T, cmap="gray", origin="lower")
        ax.set_axis_off()

    axes[0].set_title(f"{row['id']} | slice {z_idx}")

    overlay_mask(axes[1], label_slice, color=(0.0, 1.0, 0.0), alpha=0.28)
    overlay_mask(axes[1], pred_slice, color=(1.0, 0.0, 0.0), alpha=0.28)
    overlay_mask(axes[1], small_slice, color=(0.0, 0.7, 1.0), alpha=0.35)
    add_contour(axes[1], label_slice, "lime", "GT")
    add_contour(axes[1], pred_slice, "red", "Pred")
    add_contour(axes[1], small_slice, "cyan", "Small GT")
    axes[1].set_title("GT / Pred / Small GT")
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(loc="lower right", fontsize=7, framealpha=0.75)

    overlay_mask(axes[2], true_positive, color=(0.0, 1.0, 0.0), alpha=0.32)
    overlay_mask(axes[2], false_positive, color=(1.0, 0.0, 0.0), alpha=0.32)
    overlay_mask(axes[2], false_negative, color=(0.0, 0.4, 1.0), alpha=0.32)
    axes[2].set_title(
        f"{group} #{rank} | Dice={row['dice']:.3f} | "
        f"SmallDice={row['dice_small']:.3f} | Recall={row['recall']:.3f}"
    )

    fig.suptitle(
        f"PredGT={row['pred_gt']:.3f} | RecallSmall={row['recall_small']:.3f} | "
        f"Precision={row['precision']:.3f} | min_cc={row['min_cc']} | thr={row['threshold']:.2f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def select_small_nodule_examples(rows, per_group=2):
    candidates = [row for row in rows if int(row["small_components"]) > 0]
    if not candidates:
        candidates = list(rows)
    ordered = sorted(candidates, key=lambda row: (float(row["dice_small"]), float(row["recall_small"])))
    if not ordered:
        return []

    worst = ordered[:per_group]
    best = list(reversed(ordered[-per_group:]))
    center = len(ordered) // 2
    start = max(0, center - per_group // 2)
    median = ordered[start : start + per_group]
    if len(median) < per_group:
        median = ordered[max(0, len(ordered) - per_group) :]

    selected = []
    for group, group_rows in (("best_small", best), ("median_small", median), ("worst_small", worst)):
        for rank, row in enumerate(group_rows, start=1):
            selected.append({"group": group, "rank": rank, "id": row["id"], "row": row})
    return selected


def save_selected_visualizations(model, loader, rows, device, amp_enabled, amp_dtype, threshold, min_cc, out_dir):
    selected = select_small_nodule_examples(rows, per_group=2)
    if not selected:
        print("[WARN] no cases available for visualization")
        return []

    selected_by_id = {}
    for item in selected:
        selected_by_id.setdefault(item["id"], []).append(item)

    saved = []
    for case_index, batch in enumerate(loader):
        case_id = safe_case_id(batch, case_index)
        if case_id not in selected_by_id:
            continue
        image_np, pred_np, label_np = predict_case_arrays(
            model=model,
            batch=batch,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            threshold=threshold,
            min_cc=min_cc,
        )
        for item in selected_by_id[case_id]:
            path = save_case_visualization(
                row=item["row"],
                image_np=image_np,
                pred_np=pred_np,
                label_np=label_np,
                group=item["group"],
                rank=item["rank"],
                out_dir=out_dir,
            )
            saved.append({"id": case_id, "group": item["group"], "rank": item["rank"], "path": path})
            print(f"[INFO] saved visualization | {item['group']} #{item['rank']} | {case_id} | {path}")

    return saved


def main():
    parser = argparse.ArgumentParser(description="Per-sample validation for Stage2-C3 checkpoints.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help=f"Checkpoint path. Default: {DEFAULT_MODEL_PATH}")
    parser.add_argument("--data-dir", default=c3.DATA_ROOT, help=f"Data dir. Default: {c3.DATA_ROOT}")
    parser.add_argument("--split", choices=["val", "test", "train", "all"], default="val")
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--min-cc", type=int, default=5)
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    parser.add_argument("--viz-dir", default=DEFAULT_VIZ_DIR)
    parser.add_argument("--no-viz", action="store_true", help="Disable best/median/worst small-nodule PNG export.")
    args = parser.parse_args()

    c3.configure_train_module()
    train.PRED_THRESHOLD = float(args.threshold)
    train.MIN_COMPONENT_VOXELS = int(args.min_cc)

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    data_list = c3.build_data_list(args.data_dir)
    data_list = c3.limit_data_list(data_list, c3.MAX_DATASET_SIZE)
    if args.split == "all":
        eval_files = data_list
    else:
        train_files, val_files, test_files = train.split_data_list(data_list)
        split_map = {"train": train_files, "val": val_files, "test": test_files}
        eval_files = split_map[args.split]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, _ = train.choose_amp_dtype(device)
    amp_enabled = device.type == "cuda" and amp_dtype in (torch.bfloat16, torch.float16)
    model, payload = load_model(args.model, device)
    loader = build_eval_loader(eval_files, args.split)

    print(
        f"[INFO] per-sample eval | split={args.split} | cases={len(eval_files)} | "
        f"threshold={args.threshold:.2f} | min_cc={args.min_cc} | model={args.model}"
    )

    rows = []
    for case_index, batch in enumerate(loader):
        row = evaluate_one_case(
            model=model,
            batch=batch,
            case_index=case_index,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            threshold=args.threshold,
            min_cc=args.min_cc,
        )
        rows.append(row)
        print(
            f"{row['id']} | Dice={row['dice']:.4f} | Recall={row['recall']:.4f} | "
            f"PredGT={row['pred_gt']:.3f} | DiceSmall={row['dice_small']:.4f} | "
            f"RecallSmall={row['recall_small']:.4f} | Precision={row['precision']:.4f}"
        )

    visualizations = []
    if not args.no_viz:
        visualizations = save_selected_visualizations(
            model=model,
            loader=loader,
            rows=rows,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            threshold=args.threshold,
            min_cc=args.min_cc,
            out_dir=args.viz_dir,
        )

    result = {
        "model": args.model,
        "model_epoch": payload.get("epoch", None) if isinstance(payload, dict) else None,
        "data_dir": args.data_dir,
        "split": args.split,
        "threshold": float(args.threshold),
        "min_cc": int(args.min_cc),
        "summary": summarize(rows),
        "visualizations": visualizations,
        "cases": rows,
    }

    out_json_dir = os.path.dirname(args.out_json)
    if out_json_dir:
        os.makedirs(out_json_dir, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    write_csv(args.out_csv, rows)

    summary = result["summary"]
    print(
        f"[INFO] summary | cases={summary['cases']} | Dice={summary['dice_mean']:.4f} | "
        f"Recall={summary['recall_mean']:.4f} | PredGT={summary['pred_gt_global']:.3f} | "
        f"DiceSmall={summary['dice_small_mean']:.4f} | RecallSmall={summary['recall_small_mean']:.4f}"
    )
    print(f"[INFO] saved JSON: {args.out_json}")
    print(f"[INFO] saved CSV: {args.out_csv}")
    if not args.no_viz:
        print(f"[INFO] saved visualizations: {args.viz_dir}")


if __name__ == "__main__":
    main()
