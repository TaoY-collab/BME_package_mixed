import argparse
import json
import os

import numpy as np
import torch
from monai.data import DataLoader, PersistentDataset, list_data_collate
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
import torch.nn.functional as F

import train


DEFAULT_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
DEFAULT_MIN_COMPONENT_VOXELS = (0, 5, 10, 20, 50, 100)
DEFAULT_MODEL_PATH = (
    "/home/lembert/Desktop/BME-4/BME_package_mixed/train_runs/"
    "checkpoint_stage2_refine_epoch_110.pth"
)


def build_val_loader(val_files):
    _, eval_trans = train.build_transforms()
    val_cache = os.path.join(train.CACHE_DIR, "val_threshold_eval_roi64")
    os.makedirs(val_cache, exist_ok=True)

    val_dataset = PersistentDataset(
        data=val_files,
        transform=eval_trans,
        cache_dir=val_cache,
    )
    return DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=train.VAL_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=train.VAL_NUM_WORKERS > 0,
        collate_fn=list_data_collate,
    )


def load_model(model_path, device):
    model = train.build_model(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = train.extract_model_state(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return model


def remove_small_components(pred_labels, min_component_voxels):
    if min_component_voxels <= 0:
        return pred_labels
    if train.ndi is None:
        raise RuntimeError("scipy.ndimage is required for connected-component filtering")

    filtered = []
    for pred in pred_labels:
        pred_np = pred.detach().cpu().numpy().astype(bool)
        labeled, num_components = train.ndi.label(pred_np)
        keep = np.zeros_like(pred_np, dtype=bool)
        for component_id in range(1, num_components + 1):
            component = labeled == component_id
            if int(component.sum()) >= min_component_voxels:
                keep |= component
        filtered.append(torch.from_numpy(keep.astype(np.int64)).to(pred_labels.device))
    return torch.stack(filtered, dim=0)


def run_validation_with_postprocess(
    model,
    data_loader,
    dice_metric,
    device,
    amp_enabled,
    amp_dtype,
    threshold,
    min_component_voxels,
):
    model.eval()
    total_pred_fg = 0.0
    total_label_fg = 0.0
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_false_positive_components = 0
    scan_count = 0
    small_dice_values = []
    small_detected = 0
    small_total = 0

    with torch.inference_mode():
        for batch in data_loader:
            inputs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].long().to(device, non_blocking=True)
            valid_shape = train.get_batch_valid_shape(batch, labels)

            with torch.amp.autocast(
                "cuda",
                enabled=amp_enabled,
                dtype=amp_dtype if amp_enabled else None,
            ):
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
            pred_labels = remove_small_components(pred_labels, min_component_voxels)

            total_pred_fg += float((pred_labels > 0).sum().item())
            total_label_fg += float((labels > 0).sum().item())

            pred_binary = pred_labels > 0
            target_binary = labels.squeeze(1) > 0

            total_tp += float((pred_binary & target_binary).sum().item())
            total_fp += float((pred_binary & ~target_binary).sum().item())
            total_fn += float((~pred_binary & target_binary).sum().item())

            for batch_idx in range(pred_binary.shape[0]):
                scan_count += 1
                pred_case = pred_binary[batch_idx]
                target_case = target_binary[batch_idx]
                target_voxels = float(target_case.sum().item())
                total_false_positive_components += train.count_false_positive_components(pred_case, target_case)
                if target_voxels < train.SMALL_NODULE_VOXELS:
                    small_total += 1
                    small_dice_values.append(train.dice_from_binary(pred_case, target_case))
                    if float((pred_case & target_case).sum().item()) > 0.0:
                        small_detected += 1

            pred_onehot = F.one_hot(pred_labels, num_classes=2).permute(0, 4, 1, 2, 3).float()
            label_onehot = F.one_hot(labels.squeeze(1), num_classes=2).permute(0, 4, 1, 2, 3).float()
            dice_metric(y_pred=pred_onehot, y=label_onehot)

    dice = float(dice_metric.aggregate().item())
    dice_metric.reset()
    precision = total_tp / max(total_tp + total_fp, 1e-8)
    recall = total_tp / max(total_tp + total_fn, 1e-8)
    fp_per_scan = total_false_positive_components / max(scan_count, 1)
    pred_gt_volume_ratio = total_pred_fg / max(total_label_fg, 1e-8)
    dice_small = float(np.mean(small_dice_values)) if small_dice_values else 0.0
    recall_small = small_detected / max(small_total, 1)

    return dice, {
        "pred_fg": total_pred_fg,
        "label_fg": total_label_fg,
        "dice_small": dice_small,
        "recall_small": recall_small,
        "precision": precision,
        "recall": recall,
        "fp_per_scan": fp_per_scan,
        "pred_gt_volume_ratio": pred_gt_volume_ratio,
        "small_cases": small_total,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one checkpoint on the validation split at multiple foreground thresholds."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Checkpoint/model path. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(train.SAVE_DIR, "threshold_cc_eval_stage2_epoch_110.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Foreground probability thresholds to evaluate.",
    )
    parser.add_argument(
        "--min-component-voxels",
        nargs="+",
        type=int,
        default=DEFAULT_MIN_COMPONENT_VOXELS,
        help="Minimum connected-component sizes to keep. 0 disables filtering.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    os.makedirs(train.SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, _ = train.choose_amp_dtype(device)
    amp_enabled = device.type == "cuda" and amp_dtype in (torch.bfloat16, torch.float16)

    data_list = train.build_data_list(train.DATA_DIR)
    data_list = train.limit_data_list(data_list, train.MAX_DATASET_SIZE)
    train_files, val_files, test_files = train.split_data_list(data_list)

    print(
        f"[INFO] threshold + connected-component eval | model={args.model} | "
        f"val={len(val_files)} | test={len(test_files)} | device={device}"
    )

    val_loader = build_val_loader(val_files)
    model = load_model(args.model, device)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    results = []
    for threshold in args.thresholds:
        for min_component_voxels in args.min_component_voxels:
            dice, info = run_validation_with_postprocess(
                model=model,
                data_loader=val_loader,
                dice_metric=dice_metric,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                threshold=float(threshold),
                min_component_voxels=int(min_component_voxels),
            )
            row = {
                "threshold": float(threshold),
                "min_component_voxels": int(min_component_voxels),
                "dice": dice,
                "dice_small": info["dice_small"],
                "recall_small": info["recall_small"],
                "precision": info["precision"],
                "recall": info["recall"],
                "fp_per_scan": info["fp_per_scan"],
                "pred_gt_volume_ratio": info["pred_gt_volume_ratio"],
                "pred_fg": info["pred_fg"],
                "label_fg": info["label_fg"],
                "small_cases": info["small_cases"],
            }
            results.append(row)
            print(
                f"threshold={threshold:.2f} | "
                f"min_cc={min_component_voxels:>3} | "
                f"Dice={dice:.4f} | "
                f"DiceSmall={info['dice_small']:.4f} | "
                f"RecallSmall={info['recall_small']:.4f} | "
                f"Precision={info['precision']:.4f} | "
                f"Recall={info['recall']:.4f} | "
                f"FP/scan={info['fp_per_scan']:.2f} | "
                f"PredGT={info['pred_gt_volume_ratio']:.3f}"
            )

    best = sorted(results, key=lambda x: (-x["dice_small"], x["fp_per_scan"]))[0]
    print(
        "[INFO] selected by DiceSmall desc, then FP/scan asc | "
        f"threshold={best['threshold']:.2f} | "
        f"min_cc={best['min_component_voxels']} | "
        f"DiceSmall={best['dice_small']:.4f} | "
        f"FP/scan={best['fp_per_scan']:.2f} | "
        f"Precision={best['precision']:.4f} | "
        f"RecallSmall={best['recall_small']:.4f}"
    )

    payload = {
        "model": args.model,
        "thresholds": [float(x) for x in args.thresholds],
        "min_component_voxels": [int(x) for x in args.min_component_voxels],
        "selection_rule": "maximize dice_small, then minimize fp_per_scan",
        "recommended": best,
        "pr_points": [
            {
                "threshold": x["threshold"],
                "min_component_voxels": x["min_component_voxels"],
                "precision": x["precision"],
                "recall": x["recall"],
                "recall_small": x["recall_small"],
            }
            for x in results
        ],
        "split_sizes": {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        },
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO] saved threshold evaluation: {args.out}")


if __name__ == "__main__":
    main()
