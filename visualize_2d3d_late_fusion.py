import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference

import train
from dual_swin_fusion import SwinUnet2D
from project_config import DATA_DIR, SAVE_DIR
from train_2d_swinunet import BEST_MODEL_PATH as DEFAULT_2D_MODEL_PATH
from train_2d_swinunet import EMBED_DIM, IMG_SIZE

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None


DEFAULT_STAGE1_PATH = os.path.join(SAVE_DIR, "best_stage1.pth")
DEFAULT_OUT_DIR = os.path.join(SAVE_DIR, "testset_2d3d_fusion_eval")
DEFAULT_OUT_JSON = os.path.join(DEFAULT_OUT_DIR, "test_metrics.json")
DEFAULT_OUT_CSV = os.path.join(DEFAULT_OUT_DIR, "test_metrics.csv")
VOXEL_SPACING = (1.5, 1.5, 2.0)


def sanitize_name(value):
    keep = []
    for char in str(value):
        keep.append(char if char.isalnum() or char in {"-", "_", "."} else "_")
    return "".join(keep)


def load_2d_model(path, device):
    payload = torch.load(path, map_location=device)
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    img_size = int(config.get("img_size", IMG_SIZE))
    embed_dim = int(config.get("embed_dim", EMBED_DIM))
    model = SwinUnet2D(
        img_size=(img_size, img_size),
        in_channels=1,
        num_classes=2,
        embed_dim=embed_dim,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
    ).to(device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, img_size


def load_3d_model(path, device):
    train.MODEL_NAME = "swinunetr"
    model = train.build_model(device)
    payload = torch.load(path, map_location=device)
    state = train.extract_model_state(payload)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def shared_test_split(data_dir):
    data_list = train.build_data_list(data_dir)
    data_list = train.limit_data_list(data_list, train.MAX_DATASET_SIZE)
    _, _, test_files = train.split_data_list(data_list)
    return test_files


def preprocess_case(case):
    _, eval_transform = train.build_transforms()
    return eval_transform({"image": case["image"], "label": case["label"], "id": case["id"]})


def case_id_from_batch(batch, fallback):
    case_id = batch.get("id", fallback)
    if isinstance(case_id, (list, tuple)):
        return str(case_id[0])
    return str(case_id)


def crop_array(array, valid_shape):
    d, h, w = [int(x) for x in valid_shape]
    return array[:d, :h, :w]


def infer_2d_volume(model, volume, device, img_size, batch_size=16):
    _, _, size_x, size_y, _ = volume.shape
    slices = volume[0].permute(3, 0, 1, 2)
    resized = F.interpolate(slices, size=(img_size, img_size), mode="bilinear", align_corners=False)
    probs = []
    with torch.inference_mode():
        for start in range(0, resized.shape[0], batch_size):
            logits = model(resized[start : start + batch_size].to(device))
            prob = torch.softmax(logits, dim=1)[:, 1:2]
            prob = F.interpolate(prob, size=(size_x, size_y), mode="bilinear", align_corners=False)
            probs.append(prob.cpu())
    prob_slices = torch.cat(probs, dim=0)
    return prob_slices[:, 0].permute(1, 2, 0).contiguous().numpy()


def infer_case(case, model2d, img_size, model3d, device, amp_enabled, amp_dtype, w2d, w3d):
    batch = preprocess_case(case)
    image = batch["image"].unsqueeze(0).to(device)
    label = batch["label"].unsqueeze(0).long()
    valid_shape = train.get_batch_valid_shape(batch, label)

    with torch.inference_mode():
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype if amp_enabled else None):
            logits3d = sliding_window_inference(
                image,
                roi_size=train.ROI_SIZE,
                sw_batch_size=train.VAL_SW_BATCH_SIZE,
                predictor=model3d,
                overlap=train.INFER_OVERLAP,
                mode="gaussian",
            )
        prob3d = torch.softmax(logits3d, dim=1)[0, 1].detach().cpu().numpy()
        prob2d = infer_2d_volume(model2d, image, device, img_size)

    image_np = image[0, 0].detach().cpu().numpy()
    label_np = label[0, 0].numpy().astype(bool)
    prob2d = crop_array(prob2d, valid_shape)
    prob3d = crop_array(prob3d, valid_shape)
    image_np = crop_array(image_np, valid_shape)
    label_np = crop_array(label_np, valid_shape)
    fused = (w2d * prob2d + w3d * prob3d) / max(w2d + w3d, 1e-8)
    return image_np, label_np, {"2d": prob2d, "3d": prob3d, "fusion": fused}


def remove_small_components(mask, min_cc):
    mask = mask.astype(bool)
    if min_cc <= 0 or ndi is None:
        return mask
    labeled, count = ndi.label(mask)
    keep = np.zeros_like(mask, dtype=bool)
    for component_id in range(1, count + 1):
        component = labeled == component_id
        if int(component.sum()) >= min_cc:
            keep |= component
    return keep


def dice_np(pred, target):
    pred_sum = int(pred.sum())
    target_sum = int(target.sum())
    if pred_sum + target_sum == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, target).sum() / max(pred_sum + target_sum, 1))


def hd95_np(pred, target, spacing=VOXEL_SPACING):
    pred = pred.astype(bool)
    target = target.astype(bool)
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0 or ndi is None:
        return float("inf")
    structure = ndi.generate_binary_structure(3, 1)
    pred_border = np.logical_xor(pred, ndi.binary_erosion(pred, structure=structure, border_value=0))
    target_border = np.logical_xor(target, ndi.binary_erosion(target, structure=structure, border_value=0))
    if pred_border.sum() == 0 or target_border.sum() == 0:
        return float("inf")
    dt_target = ndi.distance_transform_edt(~target_border, sampling=spacing)
    dt_pred = ndi.distance_transform_edt(~pred_border, sampling=spacing)
    distances = np.concatenate([dt_target[pred_border], dt_pred[target_border]])
    if distances.size == 0:
        return float("inf")
    return float(np.percentile(distances, 95))


def small_component_metrics(pred, target):
    if ndi is None:
        return {"dice_small": 0.0, "recall_small": 0.0, "small_components": 0}
    labeled, count = ndi.label(target.astype(bool))
    dice_values = []
    detected = 0
    total = 0
    for component_id in range(1, count + 1):
        component = labeled == component_id
        if int(component.sum()) >= train.SMALL_NODULE_VOXELS:
            continue
        total += 1
        component_pred = np.logical_and(pred, component)
        dice_values.append(dice_np(component_pred, component))
        if int(component_pred.sum()) > 0:
            detected += 1
    return {
        "dice_small": float(np.mean(dice_values)) if dice_values else 0.0,
        "recall_small": float(detected / max(total, 1)),
        "small_components": int(total),
    }


def metrics_from_prob(prob, target, threshold, min_cc):
    pred = remove_small_components(prob >= threshold, min_cc)
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, np.logical_not(target)).sum())
    fn = int(np.logical_and(np.logical_not(pred), target).sum())
    small = small_component_metrics(pred, target)
    return {
        "pred": pred,
        "dice": dice_np(pred, target),
        "recall": float(tp / max(tp + fn, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "hd95": hd95_np(pred, target),
        "pred_fg": int(pred.sum()),
        "label_fg": int(target.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        **small,
    }


def flatten_metrics(case_id, metrics_by_name):
    row = {"id": case_id}
    for name, metrics in metrics_by_name.items():
        for key, value in metrics.items():
            if key == "pred":
                continue
            row[f"{name}_{key}"] = value
    return row


def finite_mean(values):
    finite = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("inf")


def summarize(rows):
    summary = {"cases": len(rows)}
    for prefix in ("2d", "3d", "fusion"):
        for key in ("dice", "recall", "precision", "dice_small", "recall_small", "hd95"):
            values = [row[f"{prefix}_{key}"] for row in rows]
            reducer = finite_mean if key == "hd95" else lambda xs: float(np.mean([float(x) for x in xs])) if xs else 0.0
            summary[f"{prefix}_{key}_mean"] = reducer(values)
    return summary


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def choose_slice(label, pred):
    label_area = label.sum(axis=(0, 1))
    if int(label_area.max()) > 0:
        return int(label_area.argmax())
    pred_area = pred.sum(axis=(0, 1))
    if int(pred_area.max()) > 0:
        return int(pred_area.argmax())
    return int(label.shape[-1] // 2)


def overlay(ax, mask, color, alpha=0.35):
    if not np.any(mask):
        return
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[..., :3] = color
    rgba[..., 3] = mask.astype(np.float32) * alpha
    ax.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower")


def save_case_figure(path, row, image, label, metrics_by_name, group, rank):
    fusion_pred = metrics_by_name["fusion"]["pred"]
    z = choose_slice(label, fusion_pred)
    base = image[:, :, z]
    label_slice = label[:, :, z]
    pred2d = metrics_by_name["2d"]["pred"][:, :, z]
    pred3d = metrics_by_name["3d"]["pred"][:, :, z]
    predf = fusion_pred[:, :, z]

    panels = [
        ("image + GT", label_slice),
        ("2D pred", pred2d),
        ("3D pred", pred3d),
        ("fusion pred", predf),
    ]
    size_x, size_y = base.shape
    panel_height = 4.0
    panel_width = panel_height * max(size_x / max(size_y, 1), 0.25)
    fig, axes = plt.subplots(1, 4, figsize=(max(8.0, panel_width * 4), panel_height), dpi=150)
    for ax, (title, mask) in zip(axes, panels):
        ax.imshow(base.T, cmap="gray", origin="lower", aspect="equal")
        overlay(ax, mask, (1.0, 0.1, 0.0) if title != "image + GT" else (0.0, 1.0, 0.0))
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(
        f"{group} #{rank} | {row['id']} | z={z} | "
        f"Fusion Dice={row['fusion_dice']:.3f} Recall={row['fusion_recall']:.3f} "
        f"HD95={row['fusion_hd95'] if np.isfinite(row['fusion_hd95']) else 'inf'}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def select_examples(rows, per_group=2):
    ordered = sorted(rows, key=lambda row: float(row["fusion_dice"]))
    if not ordered:
        return []
    worst = ordered[:per_group]
    best = list(reversed(ordered[-per_group:]))
    center = len(ordered) // 2
    median = ordered[max(0, center - per_group // 2) : max(0, center - per_group // 2) + per_group]
    if len(median) < per_group:
        median = ordered[max(0, len(ordered) - per_group) :]
    selected = []
    for group, group_rows in (("best", best), ("median", median), ("worst", worst)):
        for rank, row in enumerate(group_rows, start=1):
            selected.append((group, rank, row))
    return selected


def main():
    parser = argparse.ArgumentParser(description="Evaluate 2D, 3D and late-fusion predictions on the shared test split.")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--model-2d", default=DEFAULT_2D_MODEL_PATH)
    parser.add_argument("--model-3d", default=DEFAULT_STAGE1_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--min-cc", type=int, default=5)
    parser.add_argument("--w2d", type=float, default=0.5)
    parser.add_argument("--w3d", type=float, default=0.5)
    parser.add_argument("--examples-per-group", type=int, default=2)
    args = parser.parse_args()

    if not os.path.exists(args.model_2d):
        raise FileNotFoundError(f"2D model not found: {args.model_2d}")
    if not os.path.exists(args.model_3d):
        raise FileNotFoundError(f"3D model not found: {args.model_3d}")

    os.makedirs(args.out_dir, exist_ok=True)
    viz_dir = os.path.join(args.out_dir, "selected_cases")
    os.makedirs(viz_dir, exist_ok=True)

    test_files = shared_test_split(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, _ = train.choose_amp_dtype(device)
    amp_enabled = device.type == "cuda" and amp_dtype in (torch.bfloat16, torch.float16)
    model2d, img_size = load_2d_model(args.model_2d, device)
    model3d = load_3d_model(args.model_3d, device)

    rows = []
    arrays_by_id = {}
    print(f"[INFO] test eval | cases={len(test_files)} | threshold={args.threshold} | min_cc={args.min_cc}")
    for index, case in enumerate(test_files):
        case_id = case["id"]
        image, label, probs = infer_case(
            case,
            model2d=model2d,
            img_size=img_size,
            model3d=model3d,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            w2d=args.w2d,
            w3d=args.w3d,
        )
        metrics_by_name = {
            name: metrics_from_prob(prob, label, threshold=args.threshold, min_cc=args.min_cc)
            for name, prob in probs.items()
        }
        row = flatten_metrics(case_id, metrics_by_name)
        rows.append(row)
        arrays_by_id[case_id] = {"image": image, "label": label, "metrics": metrics_by_name}
        print(
            f"{index + 1:03d}/{len(test_files):03d} {case_id} | "
            f"Fusion Dice={row['fusion_dice']:.4f} Recall={row['fusion_recall']:.4f} "
            f"DiceSmall={row['fusion_dice_small']:.4f} HD95={row['fusion_hd95']}"
        )

    visualizations = []
    for group, rank, row in select_examples(rows, per_group=args.examples_per_group):
        payload = arrays_by_id[row["id"]]
        path = os.path.join(viz_dir, f"{group}_{rank:02d}_{sanitize_name(row['id'])}.png")
        save_case_figure(path, row, payload["image"], payload["label"], payload["metrics"], group, rank)
        visualizations.append({"group": group, "rank": rank, "id": row["id"], "path": path})

    result = {
        "data_dir": args.data_dir,
        "model_2d": args.model_2d,
        "model_3d": args.model_3d,
        "threshold": args.threshold,
        "min_cc": args.min_cc,
        "weights": {"w2d": args.w2d, "w3d": args.w3d},
        "summary": summarize(rows),
        "visualizations": visualizations,
        "cases": rows,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(json_safe(result), f, ensure_ascii=False, indent=2)
    write_csv(args.out_csv, rows)

    summary = result["summary"]
    print(
        "[INFO] summary | "
        f"Fusion Dice={summary['fusion_dice_mean']:.4f} | "
        f"Recall={summary['fusion_recall_mean']:.4f} | "
        f"Precision={summary['fusion_precision_mean']:.4f} | "
        f"DiceSmall={summary['fusion_dice_small_mean']:.4f} | "
        f"RecallSmall={summary['fusion_recall_small_mean']:.4f} | "
        f"HD95={summary['fusion_hd95_mean']}"
    )
    print(f"[INFO] wrote metrics: {args.out_json} | {args.out_csv}")
    print(f"[INFO] saved selected visualizations: {viz_dir}")


if __name__ == "__main__":
    main()
