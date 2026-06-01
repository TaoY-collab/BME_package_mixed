import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import json
from sklearn.model_selection import train_test_split
from scipy import ndimage

from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)
from project_config import BEST_MODEL_PATH, DATA_DIR, HISTORY_PATH, TRAINING_CURVE_PATH, VISUALIZATION_PATH


MODEL_PATH = BEST_MODEL_PATH
SAVE_FIG_PATH = VISUALIZATION_PATH
SAVE_CURVE_PATH = TRAINING_CURVE_PATH
ROI_SIZE = (96, 96, 96)
FEATURE_SIZE = 48
USE_CHECKPOINT = True
NUM_CLASSES = 2
NUM_SLICES = 5
TARGET_CASE_ID = None
CASE_SELECTION = "random_test"
TRAIN_COUNT = 700
VAL_COUNT = 150
TEST_COUNT = 150
SPLIT_SEED = 42
RANDOM_CASE_SEED = 42
INFER_OVERLAP = 0.5
APPLY_POSTPROCESS =True
MIN_COMPONENT_SIZE = 80
KEEP_LARGEST_COMPONENT_ONLY = False
MAX_COMPONENTS_TO_REPORT = 10


class RecordSpatialShapeD(MapTransform):
    def __init__(self, keys, output_key):
        super().__init__(keys)
        self.output_key = output_key

    def __call__(self, data):
        d = dict(data)
        first_key = self.keys[0]
        d[self.output_key] = np.asarray(d[first_key].shape[1:], dtype=np.int64)
        return d


def build_case_list(data_dir):
    case_list = []
    for file_name in sorted(os.listdir(data_dir)):
        if not file_name.endswith("_img.nii.gz"):
            continue

        image_path = os.path.join(data_dir, file_name)
        label_path = image_path.replace("_img.nii.gz", "_mask.nii.gz")
        if not os.path.exists(label_path):
            continue

        label_data = nib.load(label_path).get_fdata()
        if np.sum(label_data) <= 0:
            continue

        case_list.append(
            {
                "id": file_name.replace("_img.nii.gz", ""),
                "image": image_path,
                "label": label_path,
            }
        )

    return case_list


def get_case(case_list, target_case_id):
    if not case_list:
        raise RuntimeError(f"No valid image/mask pairs found in {DATA_DIR}")

    if target_case_id is None:
        return case_list[0]

    for case in case_list:
        if case["id"] == target_case_id:
            return case

    raise RuntimeError(f"Case {target_case_id} not found in {DATA_DIR}")


def split_case_list(case_list):
    if len(case_list) == TRAIN_COUNT + VAL_COUNT + TEST_COUNT:
        train_cases, temp_cases = train_test_split(
            case_list,
            train_size=TRAIN_COUNT,
            random_state=SPLIT_SEED,
        )
        val_cases, test_cases = train_test_split(
            temp_cases,
            train_size=VAL_COUNT,
            test_size=TEST_COUNT,
            random_state=SPLIT_SEED,
        )
        return train_cases, val_cases, test_cases

    train_cases, temp_cases = train_test_split(
        case_list,
        test_size=0.3,
        random_state=SPLIT_SEED,
    )
    val_cases, test_cases = train_test_split(
        temp_cases,
        test_size=0.5,
        random_state=SPLIT_SEED,
    )
    return train_cases, val_cases, test_cases


def get_best_default_case(case_list):
    if not case_list:
        raise RuntimeError(f"No valid image/mask pairs found in {DATA_DIR}")

    if TARGET_CASE_ID is not None:
        return get_case(case_list, TARGET_CASE_ID)

    if CASE_SELECTION == "first":
        return case_list[0]

    if CASE_SELECTION == "random_test":
        if len(case_list) < 2:
            return case_list[0]

        _, _, test_cases = split_case_list(case_list)

        if not test_cases:
            return case_list[0]

        rng = np.random.default_rng(RANDOM_CASE_SEED)
        selected_index = int(rng.integers(0, len(test_cases)))
        return test_cases[selected_index]

    best_case = None
    if CASE_SELECTION == "smallest_nodule":
        best_volume = float("inf")
        for case in case_list:
            label_data = nib.load(case["label"]).get_fdata()
            volume = float(np.sum(label_data > 0))
            if 0 < volume < best_volume:
                best_volume = volume
                best_case = case
        return best_case

    best_volume = -1
    for case in case_list:
        label_data = nib.load(case["label"]).get_fdata()
        volume = float(np.sum(label_data > 0))
        if volume > best_volume:
            best_volume = volume
            best_case = case
    return best_case


def build_transform():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000,
            a_max=400,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        RecordSpatialShapeD(keys=["image"], output_key="valid_shape"),
        SpatialPadd(keys=["image", "label"], spatial_size=ROI_SIZE),
        EnsureTyped(keys=["image", "label"]),
    ])


def extract_model_state(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def build_model(device):
    model = SwinUNETR(
        in_channels=1,
        out_channels=NUM_CLASSES,
        feature_size=FEATURE_SIZE,
        use_checkpoint=USE_CHECKPOINT,
    ).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(extract_model_state(checkpoint), strict=True)
    model.eval()
    return model


def compute_case_dice(pred, label):
    metric = DiceMetric(include_background=False, reduction="mean")
    pred_tensor = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).long()
    label_tensor = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).long()
    pred_onehot = torch.nn.functional.one_hot(pred_tensor.squeeze(1), num_classes=NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
    label_onehot = torch.nn.functional.one_hot(label_tensor.squeeze(1), num_classes=NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
    metric(y_pred=pred_onehot, y=label_onehot)
    return metric.aggregate().item()


def crop_array_to_shape(array, spatial_shape):
    d, h, w = [int(x) for x in spatial_shape]
    return array[:d, :h, :w]


def compute_mask_stats(mask):
    coords = np.argwhere(mask > 0)
    voxels = int(coords.shape[0])
    if voxels == 0:
        return {
            "voxels": 0,
            "centroid": None,
            "bbox_min": None,
            "bbox_max": None,
        }

    centroid = coords.mean(axis=0)
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    return {
        "voxels": voxels,
        "centroid": centroid,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
    }


def compute_alignment_stats(label, pred):
    gt_stats = compute_mask_stats(label)
    pred_stats = compute_mask_stats(pred)

    centroid_distance = None
    bbox_offset = None

    if gt_stats["centroid"] is not None and pred_stats["centroid"] is not None:
        centroid_distance = float(np.linalg.norm(gt_stats["centroid"] - pred_stats["centroid"]))
        bbox_offset = np.abs(gt_stats["bbox_min"] - pred_stats["bbox_min"]) + np.abs(
            gt_stats["bbox_max"] - pred_stats["bbox_max"]
        )

    return {
        "gt_voxels": gt_stats["voxels"],
        "pred_voxels": pred_stats["voxels"],
        "centroid_distance": centroid_distance,
        "bbox_offset": bbox_offset,
    }


def format_alignment_text(alignment_stats):
    centroid_text = (
        f"{alignment_stats['centroid_distance']:.2f}"
        if alignment_stats["centroid_distance"] is not None
        else "N/A"
    )
    bbox_text = (
        np.array2string(alignment_stats["bbox_offset"], precision=0, separator=",")
        if alignment_stats["bbox_offset"] is not None
        else "N/A"
    )
    return centroid_text, bbox_text


def postprocess_prediction(pred):
    binary = pred.astype(np.uint8)
    labeled, num_features = ndimage.label(binary)
    if num_features == 0:
        return binary

    component_sizes = ndimage.sum(binary, labeled, index=range(1, num_features + 1))
    keep_mask = np.zeros_like(binary, dtype=np.uint8)

    if KEEP_LARGEST_COMPONENT_ONLY:
        largest_idx = int(np.argmax(component_sizes)) + 1
        keep_mask[labeled == largest_idx] = 1
        return keep_mask

    for idx, size in enumerate(component_sizes, start=1):
        if size >= MIN_COMPONENT_SIZE:
            keep_mask[labeled == idx] = 1

    return keep_mask


def get_connected_components(pred):
    binary = pred.astype(np.uint8)
    labeled, num_features = ndimage.label(binary)
    components = []
    for idx in range(1, num_features + 1):
        component_mask = labeled == idx
        coords = np.argwhere(component_mask)
        if coords.size == 0:
            continue
        components.append(
            {
                "id": idx,
                "voxels": int(coords.shape[0]),
                "centroid": coords.mean(axis=0),
                "bbox_min": coords.min(axis=0),
                "bbox_max": coords.max(axis=0),
                "mask": component_mask,
            }
        )

    components.sort(key=lambda item: item["voxels"], reverse=True)
    return components


def get_slice_indices(label):
    z_nonzero = np.where(label.sum(axis=(0, 1)) > 0)[0]
    if len(z_nonzero) == 0:
        center = label.shape[-1] // 2
    else:
        center = z_nonzero[len(z_nonzero) // 2]

    start = max(center - NUM_SLICES // 2, 0)
    end = min(start + NUM_SLICES, label.shape[-1])
    start = max(end - NUM_SLICES, 0)
    return list(range(start, end))


def visualize_case(
    image,
    label,
    pred,
    case_id,
    case_dice,
    raw_pred_voxels,
    final_pred_voxels,
    alignment_stats,
    components,
):
    slice_indices = get_slice_indices(label)
    fig, axes = plt.subplots(4, len(slice_indices), figsize=(3.8 * len(slice_indices), 11))
    centroid_text, bbox_text = format_alignment_text(alignment_stats)

    if len(slice_indices) == 1:
        axes = np.expand_dims(axes, axis=1)

    for idx, z in enumerate(slice_indices):
        ct_slice = image[:, :, z]
        gt_slice = label[:, :, z]
        pred_slice = pred[:, :, z]

        axes[0, idx].imshow(ct_slice, cmap="gray")
        axes[0, idx].set_title(f"CT Slice {z}")
        axes[0, idx].axis("off")

        axes[1, idx].imshow(ct_slice, cmap="gray")
        axes[1, idx].imshow(gt_slice, cmap="Reds", alpha=0.45)
        axes[1, idx].set_title("Ground Truth")
        axes[1, idx].axis("off")

        axes[2, idx].imshow(ct_slice, cmap="gray")
        axes[2, idx].imshow(pred_slice, cmap="Blues", alpha=0.45)
        axes[2, idx].contour(gt_slice, levels=[0.5], colors="yellow", linewidths=0.8)
        axes[2, idx].set_title("Prediction")
        for comp in components[:MAX_COMPONENTS_TO_REPORT]:
            z_min = int(comp["bbox_min"][2])
            z_max = int(comp["bbox_max"][2])
            if not (z_min <= z <= z_max):
                continue
            cy, cx, _ = comp["centroid"]
            axes[2, idx].text(
                float(cx),
                float(cy),
                str(comp["id"]),
                color="lime",
                fontsize=9,
                ha="center",
                va="center",
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1},
            )
        axes[2, idx].axis("off")

        axes[3, idx].imshow(ct_slice, cmap="gray")
        axes[3, idx].contour(gt_slice, levels=[0.5], colors="yellow", linewidths=1.0)
        axes[3, idx].contour(pred_slice, levels=[0.5], colors="cyan", linewidths=1.0)
        axes[3, idx].set_title("GT(yellow) vs Pred(cyan)")
        axes[3, idx].axis("off")

    fig.suptitle(
        f"CT Lung Nodule Segmentation | Case: {case_id} | Dice: {case_dice:.4f} | "
        f"GT Voxels: {alignment_stats['gt_voxels']} | "
        f"Pred Voxels(raw/final): {raw_pred_voxels}/{final_pred_voxels} | "
        f"Centroid Dist: {centroid_text} | BBox Offset: {bbox_text}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(SAVE_FIG_PATH, dpi=200, bbox_inches="tight")
    plt.show()


def plot_training_curves():
    if not os.path.exists(HISTORY_PATH):
        print(f"[WARN] history file not found: {HISTORY_PATH}")
        return

    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        history = json.load(f)

    train_loss = history.get("train_loss", [])
    val_acc = history.get("val_acc", history.get("val_dice", []))
    boundary_weight = history.get("boundary_weight", [])

    if not train_loss:
        print("[WARN] history file exists but contains no training loss.")
        return

    epochs = list(range(1, len(train_loss) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    axes[0].plot(epochs, train_loss, color="crimson", linewidth=2, label="Train Loss")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, val_acc, color="royalblue", linewidth=2, label="Val Acc(Dice)")
    if boundary_weight:
        ax2 = axes[1].twinx()
        ax2.plot(epochs, boundary_weight, color="darkorange", linewidth=1.8, linestyle="--", label="Boundary Weight")
        ax2.set_ylabel("Boundary Weight")
        ax2.legend(loc="lower right")
    axes[1].set_title("Validation Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy / Dice")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(SAVE_CURVE_PATH, dpi=200, bbox_inches="tight")
    plt.show()


def main():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model checkpoint not found: {MODEL_PATH}")

    case_list = build_case_list(DATA_DIR)
    case = get_best_default_case(case_list)
    print(f"[INFO] visualizing case: {case['id']}")

    transform = build_transform()
    data = transform({"image": case["image"], "label": case["label"]})

    image = data["image"]
    label = data["label"]
    valid_shape = data["valid_shape"]
    input_tensor = image.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)

    with torch.inference_mode():
        pred_logits = sliding_window_inference(
            inputs=input_tensor.to(device),
            roi_size=ROI_SIZE,
            sw_batch_size=1,
            predictor=model,
            overlap=INFER_OVERLAP,
            mode="gaussian",
        )

    pred = torch.argmax(pred_logits, dim=1)[0].cpu().numpy().astype(np.uint8)
    pred = crop_array_to_shape(pred, valid_shape)
    raw_pred_voxels = int(pred.sum())
    if APPLY_POSTPROCESS:
        pred = postprocess_prediction(pred)
    final_pred_voxels = int(pred.sum())
    image_np = crop_array_to_shape(image[0].cpu().numpy(), valid_shape)
    label_np = crop_array_to_shape(label[0].cpu().numpy().astype(np.uint8), valid_shape)
    case_dice = compute_case_dice(pred, label_np)
    alignment_stats = compute_alignment_stats(label_np, pred)
    components = get_connected_components(pred)
    centroid_text, bbox_text = format_alignment_text(alignment_stats)

    print(f"[INFO] case dice: {case_dice:.4f}")
    print(
        f"[INFO] voxels | gt={alignment_stats['gt_voxels']} | "
        f"pred_raw={raw_pred_voxels} | pred_final={final_pred_voxels}"
    )
    print(f"[INFO] centroid distance: {centroid_text}")
    print(f"[INFO] bbox offset: {bbox_text}")
    print(f"[INFO] connected components: {len(components)}")
    for comp in components[:MAX_COMPONENTS_TO_REPORT]:
        centroid = np.round(comp["centroid"], 2).tolist()
        bbox_min = comp["bbox_min"].tolist()
        bbox_max = comp["bbox_max"].tolist()
        print(
            f"[INFO] component {comp['id']} | voxels={comp['voxels']} | "
            f"centroid={centroid} | bbox_min={bbox_min} | bbox_max={bbox_max}"
        )
    print(f"[INFO] output figure: {SAVE_FIG_PATH}")
    visualize_case(
        image_np,
        label_np,
        pred,
        case["id"],
        case_dice,
        raw_pred_voxels,
        final_pred_voxels,
        alignment_stats,
        components,
    )
    print(f"[INFO] output curves: {SAVE_CURVE_PATH}")
    plot_training_curves()


if __name__ == "__main__":
    main()
