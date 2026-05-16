import json
import os
from datetime import datetime

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage as ndi
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from monai.data import DataLoader, PersistentDataset, list_data_collate
from monai.losses import DiceLoss, TverskyLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    Spacingd,
    SpatialPadd,
)

import train
from project_config import CACHE_DIR, DATA_DIR, SAVE_DIR


RUN_NAME = "stage2_c3_dual_size_aware_local_loss"
RUN_ID = os.environ.get("BME_RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
RUN_DIR = os.path.join(SAVE_DIR, RUN_NAME)
BEST_MODELS_DIR = os.path.join(RUN_DIR, "best_models")

STAGE1_SOURCE_CKPT = os.environ.get("BME_STAGE2_C3_SOURCE_CKPT", os.path.join(SAVE_DIR, "best_stage1.pth"))
DEFAULT_C3_DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "data_stage2_c3")
DATA_ROOT = os.environ.get(
    "BME_STAGE2_C3_DATA_DIR",
    DEFAULT_C3_DATA_DIR if os.path.exists(DEFAULT_C3_DATA_DIR) else DATA_DIR,
)

CHECKPOINT_PATH = os.path.join(RUN_DIR, "checkpoint_stage2_c3.pth")
HISTORY_PATH = os.path.join(RUN_DIR, "history_stage2_c3.json")
FINAL_MODEL_PATH = os.path.join(RUN_DIR, "final_stage2_c3.pth")
BEST_OVERALL_PATH = os.path.join(BEST_MODELS_DIR, "best_overall.pth")
BEST_SMALL_PATH = os.path.join(BEST_MODELS_DIR, "best_small.pth")
BEST_BALANCED_PATH = os.path.join(BEST_MODELS_DIR, "best_balanced.pth")
AUX_MASK_DIR = os.path.join(RUN_DIR, "generated_aux_masks")

ROI_SIZE = (96, 96, 96)
FEATURE_SIZE = 48
MAX_EPOCHS = 100
WARMUP_EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-5
TRAIN_BATCH_SIZE = 1
ACCUMULATION_STEPS = 2
AMP_MODE = "bf16"
VAL_SW_BATCH_SIZE = 2
INFER_OVERLAP = 0.5
PRED_THRESHOLD = 0.35
MIN_COMPONENT_VOXELS = 0
VALIDATION_EPOCHS = {0, 10, 20, 30, 40, 60, 80, 100}
SNAPSHOT_INTERVAL = 20

MAX_DATASET_SIZE = 400
TRAIN_NUM_WORKERS = int(os.environ.get("BME_TRAIN_NUM_WORKERS", "4"))
VAL_NUM_WORKERS = int(os.environ.get("BME_VAL_NUM_WORKERS", "2"))

LAMBDA_DICE = 1.0
LAMBDA_TVERSKY = 0.4
ALPHA_GLOBAL = 0.60
BETA_GLOBAL = 0.40
ALPHA_SMALL = 0.45
BETA_SMALL = 0.55
ALPHA_LARGE = 0.65
BETA_LARGE = 0.35
LAMBDA_SMALL_MAX = 0.06
LAMBDA_LARGE_MAX = 0.04
LOCAL_RAMP_START_EPOCH = 10
LOCAL_RAMP_END_EPOCH = 30
SMALL_MARGIN = 8
LARGE_MARGIN = 12

SMALL_NODULE_DIAMETER_MM = 6.0
LARGE_NODULE_DIAMETER_MM = 8.0

OFFLINE_THRESHOLDS = (0.25, 0.30, 0.35, 0.40)
OFFLINE_MIN_CCS = (0, 5, 10)


def configure_train_module():
    resampled_voxel_volume = 1.5 * 1.5 * 2.0
    small_voxels_at_eval_spacing = int(round((np.pi / 6.0) * (SMALL_NODULE_DIAMETER_MM ** 3) / resampled_voxel_volume))
    large_voxels_at_eval_spacing = int(round((np.pi / 6.0) * (LARGE_NODULE_DIAMETER_MM ** 3) / resampled_voxel_volume))
    train.ROI_SIZE = ROI_SIZE
    train.FEATURE_SIZE = FEATURE_SIZE
    train.AMP_MODE = AMP_MODE
    train.VAL_SW_BATCH_SIZE = VAL_SW_BATCH_SIZE
    train.INFER_OVERLAP = INFER_OVERLAP
    train.PRED_THRESHOLD = PRED_THRESHOLD
    train.MIN_COMPONENT_VOXELS = MIN_COMPONENT_VOXELS
    train.MAX_DATASET_SIZE = MAX_DATASET_SIZE
    train.TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE
    train.ACCUMULATION_STEPS = ACCUMULATION_STEPS
    train.SMALL_NODULE_VOXELS = max(small_voxels_at_eval_spacing, 1)
    train.MEDIUM_NODULE_VOXELS = max(large_voxels_at_eval_spacing, train.SMALL_NODULE_VOXELS + 1)
    train.TRAIN_NUM_WORKERS = TRAIN_NUM_WORKERS
    train.VAL_NUM_WORKERS = VAL_NUM_WORKERS


class Stage2C3DualLocalLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
        self.global_tversky = TverskyLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
            alpha=ALPHA_GLOBAL,
            beta=BETA_GLOBAL,
        )

    @staticmethod
    def lambda_at_epoch(epoch):
        schedule_epoch = epoch + 1
        if schedule_epoch <= LOCAL_RAMP_START_EPOCH:
            return 0.0, 0.0
        if schedule_epoch >= LOCAL_RAMP_END_EPOCH:
            return LAMBDA_SMALL_MAX, LAMBDA_LARGE_MAX
        progress = (schedule_epoch - LOCAL_RAMP_START_EPOCH) / max(
            LOCAL_RAMP_END_EPOCH - LOCAL_RAMP_START_EPOCH,
            1,
        )
        return LAMBDA_SMALL_MAX * progress, LAMBDA_LARGE_MAX * progress

    @staticmethod
    def tversky_binary_from_logits(logits_roi, labels_roi, alpha, beta):
        probs = torch.softmax(logits_roi, dim=1)[:, 1:2]
        target = (labels_roi > 0).float()
        reduce_dims = tuple(range(1, probs.ndim))
        tp = torch.sum(probs * target, dim=reduce_dims)
        fp = torch.sum(probs * (1.0 - target), dim=reduce_dims)
        fn = torch.sum((1.0 - probs) * target, dim=reduce_dims)
        score = (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)
        return 1.0 - score.mean()

    @staticmethod
    def component_bboxes(mask_tensor):
        mask_np = mask_tensor.detach().cpu().numpy().astype(bool)
        labeled, num_components = ndi.label(mask_np)
        bboxes = []
        for component_id in range(1, num_components + 1):
            positions = np.argwhere(labeled == component_id)
            if positions.size == 0:
                continue
            lo = positions.min(axis=0)
            hi = positions.max(axis=0) + 1
            bboxes.append((lo, hi))
        return bboxes

    @staticmethod
    def crop_from_bbox(tensor, lo, hi, margin):
        spatial_shape = np.asarray(tensor.shape[-3:], dtype=np.int64)
        start = np.maximum(lo - margin, 0)
        stop = np.minimum(hi + margin, spatial_shape)
        return tensor[
            ...,
            int(start[0]) : int(stop[0]),
            int(start[1]) : int(stop[1]),
            int(start[2]) : int(stop[2]),
        ]

    def local_loss_for_mask(self, logits, labels, aux_mask, alpha, beta, margin):
        losses = []
        for batch_idx in range(logits.shape[0]):
            bboxes = self.component_bboxes(aux_mask[batch_idx, 0] > 0)
            if not bboxes:
                continue
            choice = int(torch.randint(len(bboxes), (1,), device=logits.device).item())
            lo, hi = bboxes[choice]
            logits_roi = self.crop_from_bbox(logits[batch_idx : batch_idx + 1], lo, hi, margin)
            labels_roi = self.crop_from_bbox(labels[batch_idx : batch_idx + 1], lo, hi, margin)
            if min(logits_roi.shape[-3:]) <= 0:
                continue
            losses.append(self.tversky_binary_from_logits(logits_roi, labels_roi, alpha, beta))
        if not losses:
            return logits.new_tensor(0.0), 0
        return torch.stack(losses).mean(), len(losses)

    def forward(self, logits, labels, small_mask, large_mask, epoch):
        dice = self.dice_loss(logits, labels)
        tversky = self.global_tversky(logits, labels)
        lambda_small, lambda_large = self.lambda_at_epoch(epoch)
        base = LAMBDA_DICE * dice + LAMBDA_TVERSKY * tversky

        small_loss = logits.new_tensor(0.0)
        large_loss = logits.new_tensor(0.0)
        small_rois = 0
        large_rois = 0
        if lambda_small > 0.0:
            small_loss, small_rois = self.local_loss_for_mask(
                logits,
                labels,
                small_mask,
                alpha=ALPHA_SMALL,
                beta=BETA_SMALL,
                margin=SMALL_MARGIN,
            )
        if lambda_large > 0.0:
            large_loss, large_rois = self.local_loss_for_mask(
                logits,
                labels,
                large_mask,
                alpha=ALPHA_LARGE,
                beta=BETA_LARGE,
                margin=LARGE_MARGIN,
            )

        total = base + lambda_small * small_loss + lambda_large * large_loss
        info = {
            "dice": float(dice.detach().item()),
            "tversky": float(tversky.detach().item()),
            "small_local": float(small_loss.detach().item()),
            "large_local": float(large_loss.detach().item()),
            "lambda_small": float(lambda_small),
            "lambda_large": float(lambda_large),
            "small_rois": int(small_rois),
            "large_rois": int(large_rois),
        }
        return total, info


def aux_path_for_label(label_path, suffix):
    existing_path = label_path.replace("_mask.nii.gz", suffix)
    if os.path.exists(existing_path):
        return existing_path
    base_name = os.path.basename(label_path).replace("_mask.nii.gz", suffix)
    return os.path.join(AUX_MASK_DIR, base_name)


def voxel_volume_from_affine(affine):
    spacing = [float(np.linalg.norm(affine[:3, axis])) for axis in range(3)]
    return float(np.prod(spacing))


def equivalent_sphere_diameter_mm(voxels, voxel_volume_mm3):
    volume_mm3 = float(voxels) * float(voxel_volume_mm3)
    if volume_mm3 <= 0.0:
        return 0.0
    return float((6.0 * volume_mm3 / np.pi) ** (1.0 / 3.0))


def classify_diameter(diameter_mm):
    if diameter_mm < SMALL_NODULE_DIAMETER_MM:
        return "small"
    if diameter_mm <= LARGE_NODULE_DIAMETER_MM:
        return "medium"
    return "large"


def label_component_diameters(label_path):
    label_nii = nib.load(label_path)
    label = np.asanyarray(label_nii.dataobj) > 0
    voxel_volume_mm3 = voxel_volume_from_affine(label_nii.affine)
    labeled, num_components = ndi.label(label)
    diameters = []
    for component_id in range(1, num_components + 1):
        voxels = int((labeled == component_id).sum())
        if voxels > 0:
            diameters.append(equivalent_sphere_diameter_mm(voxels, voxel_volume_mm3))
    return diameters


def build_data_list(data_dir):
    data_list = []
    for file_name in sorted(os.listdir(data_dir)):
        if not file_name.endswith("_img.nii.gz"):
            continue

        image_path = os.path.join(data_dir, file_name)
        label_path = image_path.replace("_img.nii.gz", "_mask.nii.gz")
        if not os.path.exists(label_path):
            continue

        try:
            mask = nib.load(label_path).get_fdata()
            diameters = label_component_diameters(label_path)
        except Exception as exc:
            print(f"[WARN] failed to read mask: {label_path} | {exc}")
            continue

        mask_sum = float(np.sum(mask > 0))
        if mask_sum <= 0:
            continue
        max_diameter = max(diameters) if diameters else 0.0

        data_list.append(
            {
                "id": file_name.replace("_img.nii.gz", ""),
                "image": image_path,
                "label": label_path,
                "mask_sum": mask_sum,
                "max_diameter_mm": float(max_diameter),
                "component_diameters_mm": [float(x) for x in diameters],
                "nodule_size": classify_diameter(max_diameter),
            }
        )
    return data_list


def describe_size_distribution(data_list, name):
    counts = {"small": 0, "medium": 0, "large": 0}
    for case in data_list:
        counts[case.get("nodule_size", "medium")] += 1
    print(
        f"[INFO] {name} diameter distribution | "
        f"small(<{SMALL_NODULE_DIAMETER_MM:g}mm)={counts['small']} | "
        f"medium({SMALL_NODULE_DIAMETER_MM:g}-{LARGE_NODULE_DIAMETER_MM:g}mm)="
        f"{counts['medium']} | large(>{LARGE_NODULE_DIAMETER_MM:g}mm)={counts['large']}"
    )
    return counts


def limit_data_list(data_list, max_count):
    if max_count <= 0 or len(data_list) <= max_count:
        return data_list

    rng = np.random.RandomState(train.SPLIT_SEED)
    groups = {"small": [], "medium": [], "large": []}
    for case in data_list:
        groups[case.get("nodule_size", "medium")].append(case)

    selected = []
    for group in groups.values():
        rng.shuffle(group)

    remaining = max_count
    non_empty_groups = [group for group in groups.values() if group]
    base_take = max_count // max(len(non_empty_groups), 1)
    for group in non_empty_groups:
        take = min(len(group), base_take, remaining)
        selected.extend(group[:take])
        del group[:take]
        remaining -= take

    leftovers = [case for group in non_empty_groups for case in group]
    rng.shuffle(leftovers)
    selected.extend(leftovers[:remaining])
    rng.shuffle(selected)

    print(f"[INFO] limited dataset for time budget: {len(data_list)} -> {len(selected)}")
    describe_size_distribution(selected, "limited dataset")
    return selected


def write_aux_masks_from_label(label_path, small_path, large_path):
    label_nii = nib.load(label_path)
    label = np.asanyarray(label_nii.dataobj) > 0
    small_mask = np.zeros(label.shape, dtype=np.uint8)
    large_mask = np.zeros(label.shape, dtype=np.uint8)
    voxel_volume_mm3 = voxel_volume_from_affine(label_nii.affine)
    labeled, num_components = ndi.label(label)
    for component_id in range(1, num_components + 1):
        component = labeled == component_id
        voxels = int(component.sum())
        diameter_mm = equivalent_sphere_diameter_mm(voxels, voxel_volume_mm3)
        size_class = classify_diameter(diameter_mm)
        if size_class == "small":
            small_mask[component] = 1
        elif size_class == "large":
            large_mask[component] = 1
    for path, arr in ((small_path, small_mask), (large_path, large_mask)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        nii = nib.Nifti1Image(arr, label_nii.affine, label_nii.header)
        nii.set_data_dtype(np.uint8)
        nib.save(nii, path)


def attach_auxiliary_masks(data_files):
    attached = []
    generated = 0
    for case in data_files:
        item = dict(case)
        small_path = aux_path_for_label(item["label"], "_small_mask.nii.gz")
        large_path = aux_path_for_label(item["label"], "_large_mask.nii.gz")
        if not os.path.exists(small_path) or not os.path.exists(large_path):
            write_aux_masks_from_label(item["label"], small_path, large_path)
            generated += 1
        item["small_mask"] = small_path
        item["large_mask"] = large_path
        attached.append(item)
    print(f"[INFO] auxiliary small/large masks ready | cases={len(attached)} | generated={generated}")
    return attached


def build_transforms():
    train_keys = ["image", "label", "small_mask", "large_mask"]
    train_trans = Compose(
        [
            LoadImaged(keys=train_keys),
            EnsureChannelFirstd(keys=train_keys),
            Orientationd(keys=train_keys, axcodes="RAS", labels=None),
            Spacingd(
                keys=train_keys,
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest", "nearest", "nearest"),
            ),
            train.ScaleIntensityRanged(
                keys=["image"],
                a_min=-1000,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            SpatialPadd(keys=train_keys, spatial_size=ROI_SIZE),
            train.ProfiledRandCropByPosNegLabeld(
                keys=train_keys,
                label_key="label",
                spatial_size=ROI_SIZE,
            ),
            EnsureTyped(keys=train_keys),
        ]
    )

    eval_trans = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            train.ScaleIntensityRanged(
                keys=["image"],
                a_min=-1000,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            train.RecordSpatialShapeD(keys=["image"], output_key="valid_shape"),
            SpatialPadd(keys=["image", "label"], spatial_size=ROI_SIZE),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    return train_trans, eval_trans


def build_loaders(train_files, val_files, test_files):
    train_trans, eval_trans = build_transforms()
    train_files = attach_auxiliary_masks(train_files)
    train_files = train.build_profiled_training_cases(train_files)

    train_cache = os.path.join(CACHE_DIR, "train_stage2_c3_dual_local_roi96")
    val_cache = os.path.join(CACHE_DIR, "val_stage2_c3_dual_local_roi96")
    test_cache = os.path.join(CACHE_DIR, "test_stage2_c3_dual_local_roi96")
    for path in (train_cache, val_cache, test_cache):
        os.makedirs(path, exist_ok=True)

    train_dataset = PersistentDataset(data=train_files, transform=train_trans, cache_dir=train_cache)
    val_dataset = PersistentDataset(data=val_files, transform=eval_trans, cache_dir=val_cache)
    test_dataset = PersistentDataset(data=test_files, transform=eval_trans, cache_dir=test_cache)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=TRAIN_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=TRAIN_NUM_WORKERS > 0,
        collate_fn=list_data_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=VAL_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=VAL_NUM_WORKERS > 0,
        collate_fn=list_data_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=VAL_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=VAL_NUM_WORKERS > 0,
        collate_fn=list_data_collate,
    )
    return train_loader, val_loader, test_loader


def run_validation_at_point(model, data_loader, dice_metric, device, amp_enabled, amp_dtype, threshold, min_cc):
    old_threshold = train.PRED_THRESHOLD
    old_min_cc = train.MIN_COMPONENT_VOXELS
    train.PRED_THRESHOLD = float(threshold)
    train.MIN_COMPONENT_VOXELS = int(min_cc)
    try:
        dice, info = train.run_validation(model, data_loader, dice_metric, device, amp_enabled, amp_dtype)
    finally:
        train.PRED_THRESHOLD = old_threshold
        train.MIN_COMPONENT_VOXELS = old_min_cc
    return {"threshold": float(threshold), "min_cc": int(min_cc), "dice": dice, **info}


def run_validation_grid(model, data_loader, dice_metric, device, amp_enabled, amp_dtype):
    grid = {}
    for threshold in OFFLINE_THRESHOLDS:
        for min_cc in OFFLINE_MIN_CCS:
            key = f"thr{threshold:.2f}_cc{min_cc}"
            grid[key] = run_validation_at_point(
                model,
                data_loader,
                dice_metric,
                device,
                amp_enabled,
                amp_dtype,
                threshold,
                min_cc,
            )
    return grid


def online_key():
    return f"thr{PRED_THRESHOLD:.2f}_cc{MIN_COMPONENT_VOXELS}"


def balanced_score(row):
    pred_gt = float(row["pred_gt_volume_ratio"])
    if pred_gt < 0.8 or pred_gt > 1.8:
        return -1.0
    return (
        float(row["dice"])
        + 0.5 * float(row["dice_small"])
        + 0.05 * float(row["precision"])
        - 0.01 * float(row["fp_per_scan"])
    )


def save_best_payload(path, model, epoch, score_name, score, validation_row, config):
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "run_id": RUN_ID,
        "score_name": score_name,
        "score": float(score),
        "validation": validation_row,
        "config": config,
    }
    torch.save(payload, path)


def load_model_payload(path, model, device):
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    return payload


def evaluate_checkpoint_grid(name, path, model, data_loader, dice_metric, device, amp_enabled, amp_dtype):
    if not os.path.exists(path):
        print(f"[WARN] skipped offline scan for {name}; checkpoint not found: {path}")
        return None
    payload = load_model_payload(path, model, device)
    grid = run_validation_grid(model, data_loader, dice_metric, device, amp_enabled, amp_dtype)
    best_dice = max(grid.values(), key=lambda row: row["dice"])
    best_small = max(grid.values(), key=lambda row: row["dice_small"])
    best_balanced = max(grid.values(), key=balanced_score)
    print(
        f"[INFO] offline scan {name} | epoch={payload.get('epoch', None)} | "
        f"best_dice=thr{best_dice['threshold']:.2f}/cc{best_dice['min_cc']}:{best_dice['dice']:.4f} | "
        f"best_small=thr{best_small['threshold']:.2f}/cc{best_small['min_cc']}:{best_small['dice_small']:.4f} | "
        f"best_balanced=thr{best_balanced['threshold']:.2f}/cc{best_balanced['min_cc']}:{balanced_score(best_balanced):.4f}"
    )
    return {
        "path": path,
        "epoch": payload.get("epoch", None) if isinstance(payload, dict) else None,
        "grid": grid,
        "best_dice": best_dice,
        "best_small": best_small,
        "best_balanced": best_balanced,
    }


def print_online_summary(prefix, row):
    print(
        f"{prefix} | Dice={row.get('dice', 0.0):.4f} | "
        f"DiceSmall={row.get('dice_small', 0.0):.4f} | "
        f"RecallSmall={row.get('recall_small', 0.0):.4f} | "
        f"DiceLarge={row.get('dice_large', 0.0):.4f} | "
        f"RecallLarge={row.get('recall_large', 0.0):.4f} | "
        f"Precision={row.get('precision', 0.0):.4f} | "
        f"Recall={row.get('recall', 0.0):.4f} | "
        f"FP/scan={row.get('fp_per_scan', 0.0):.2f} | "
        f"PredGT={row.get('pred_gt_volume_ratio', 0.0):.3f} | PredThr={PRED_THRESHOLD:.2f} | "
        f"MinCC={MIN_COMPONENT_VOXELS}"
    )


def default_val_row():
    return {
        "threshold": PRED_THRESHOLD,
        "min_cc": MIN_COMPONENT_VOXELS,
        "dice": 0.0,
        "pred_fg": 0.0,
        "label_fg": 0.0,
        "dice_small": 0.0,
        "recall_small": 0.0,
        "dice_large": 0.0,
        "recall_large": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "fp_per_scan": 0.0,
        "pred_gt_volume_ratio": 0.0,
        "pred_gt_small": 0.0,
        "pred_gt_large": 0.0,
        "small_cases": 0,
        "large_cases": 0,
    }


def main():
    configure_train_module()
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(BEST_MODELS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, scaler_enabled = train.choose_amp_dtype(device)
    amp_enabled = device.type == "cuda" and amp_dtype in (torch.bfloat16, torch.float16)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

    data_list = build_data_list(DATA_ROOT)
    describe_size_distribution(data_list, "full dataset")
    data_list = limit_data_list(data_list, MAX_DATASET_SIZE)
    train_files, val_files, test_files = train.split_data_list(data_list)
    describe_size_distribution(train_files, "train split")
    describe_size_distribution(val_files, "val split")
    describe_size_distribution(test_files, "test split")

    train_loader, val_loader, test_loader = build_loaders(train_files, val_files, test_files)

    model = train.build_model(device)
    if not train.load_stage2_source_checkpoint(model, STAGE1_SOURCE_CKPT, device):
        raise RuntimeError(
            "Stage2-C3 requires best_stage1.pth. "
            f"Set BME_STAGE2_C3_SOURCE_CKPT or place it at {STAGE1_SOURCE_CKPT}."
        )

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS),
            CosineAnnealingLR(optimizer, T_max=max(MAX_EPOCHS - WARMUP_EPOCHS, 1)),
        ],
        milestones=[WARMUP_EPOCHS],
    )
    loss_func = Stage2C3DualLocalLoss()
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    config = {
        "run_name": RUN_NAME,
        "data_root": DATA_ROOT,
        "stage1_source_ckpt": STAGE1_SOURCE_CKPT,
        "roi_size": ROI_SIZE,
        "feature_size": FEATURE_SIZE,
        "max_epochs": MAX_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_epochs": WARMUP_EPOCHS,
        "lambda_dice": LAMBDA_DICE,
        "lambda_tversky": LAMBDA_TVERSKY,
        "alpha_global": ALPHA_GLOBAL,
        "beta_global": BETA_GLOBAL,
        "alpha_small": ALPHA_SMALL,
        "beta_small": BETA_SMALL,
        "alpha_large": ALPHA_LARGE,
        "beta_large": BETA_LARGE,
        "lambda_small_max": LAMBDA_SMALL_MAX,
        "lambda_large_max": LAMBDA_LARGE_MAX,
        "local_ramp_start_epoch": LOCAL_RAMP_START_EPOCH,
        "local_ramp_end_epoch": LOCAL_RAMP_END_EPOCH,
        "small_margin": SMALL_MARGIN,
        "large_margin": LARGE_MARGIN,
        "online_threshold": PRED_THRESHOLD,
        "online_min_cc": MIN_COMPONENT_VOXELS,
        "validation_epochs": sorted(VALIDATION_EPOCHS),
        "offline_thresholds": OFFLINE_THRESHOLDS,
        "offline_min_ccs": OFFLINE_MIN_CCS,
        "small_nodule_diameter_mm": SMALL_NODULE_DIAMETER_MM,
        "large_nodule_diameter_mm": LARGE_NODULE_DIAMETER_MM,
        "size_rule": "equivalent sphere diameter: small <6 mm, medium 6-8 mm, large >8 mm",
        "split_sizes": {"train": len(train_files), "val": len(val_files), "test": len(test_files)},
    }
    history = {
        "config": config,
        "train_loss": [],
        "dice_loss": [],
        "tversky_loss": [],
        "small_local_loss": [],
        "large_local_loss": [],
        "lambda_small": [],
        "lambda_large": [],
        "lr": [],
        "validation": [],
        "best_overall": None,
        "best_small": None,
        "best_balanced": None,
    }

    val_dice, val_info = train.run_validation(model, val_loader, dice_metric, device, amp_enabled, amp_dtype)
    val_row = {"threshold": PRED_THRESHOLD, "min_cc": MIN_COMPONENT_VOXELS, "dice": val_dice, **val_info}
    print_online_summary("Epoch 0 validation", val_row)

    best_overall = float(val_row["dice"])
    best_small = float(val_row["dice_small"])
    best_balanced = balanced_score(val_row)
    save_best_payload(BEST_OVERALL_PATH, model, -1, f"Dice@{online_key()}", best_overall, val_row, config)
    save_best_payload(BEST_SMALL_PATH, model, -1, f"DiceSmall@{online_key()}", best_small, val_row, config)
    save_best_payload(BEST_BALANCED_PATH, model, -1, "balanced_score", best_balanced, val_row, config)
    history["best_overall"] = {"epoch": -1, "score": best_overall}
    history["best_small"] = {"epoch": -1, "score": best_small}
    history["best_balanced"] = {"epoch": -1, "score": best_balanced}
    history["validation"].append({"epoch": 0, "row": val_row})

    print(
        f"[INFO] Stage2-C3 | run_dir={RUN_DIR} | train/val/test={len(train_files)}/{len(val_files)}/{len(test_files)} | "
        f"source={STAGE1_SOURCE_CKPT} | online_threshold={PRED_THRESHOLD} | validation_epochs={sorted(VALIDATION_EPOCHS)}"
    )

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_tversky = 0.0
        epoch_small = 0.0
        epoch_large = 0.0
        last_loss_info = {"lambda_small": 0.0, "lambda_large": 0.0}

        progress = tqdm(train_loader, desc=f"Stage2-C3 Epoch {epoch + 1}/{MAX_EPOCHS}")
        for step, batch in enumerate(progress):
            inputs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            small_mask = batch["small_mask"].to(device, non_blocking=True)
            large_mask = batch["large_mask"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype if amp_enabled else None):
                outputs = model(inputs)
                loss, loss_info = loss_func(outputs, labels, small_mask, large_mask, epoch)
                loss_for_backward = loss / ACCUMULATION_STEPS

            if scaler_enabled:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            if ((step + 1) % ACCUMULATION_STEPS == 0) or (step + 1 == len(train_loader)):
                if scaler_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(loss.detach().item())
            epoch_dice += float(loss_info["dice"])
            epoch_tversky += float(loss_info["tversky"])
            epoch_small += float(loss_info["small_local"])
            epoch_large += float(loss_info["large_local"])
            last_loss_info = loss_info

        scheduler.step()
        num_batches = max(len(train_loader), 1)
        avg_loss = epoch_loss / num_batches
        avg_dice = epoch_dice / num_batches
        avg_tversky = epoch_tversky / num_batches
        avg_small = epoch_small / num_batches
        avg_large = epoch_large / num_batches
        current_lr = optimizer.param_groups[0]["lr"]

        display_epoch = epoch + 1
        should_validate = display_epoch in VALIDATION_EPOCHS
        val_row = default_val_row()
        if should_validate:
            val_dice, val_info = train.run_validation(model, val_loader, dice_metric, device, amp_enabled, amp_dtype)
            val_row = {"threshold": PRED_THRESHOLD, "min_cc": MIN_COMPONENT_VOXELS, "dice": val_dice, **val_info}
            print_online_summary(f"Epoch {display_epoch} validation", val_row)

            overall_score = float(val_row["dice"])
            small_score = float(val_row["dice_small"])
            balanced_epoch_score = balanced_score(val_row)
            if overall_score > best_overall:
                best_overall = overall_score
                history["best_overall"] = {"epoch": epoch, "score": best_overall}
                save_best_payload(BEST_OVERALL_PATH, model, epoch, f"Dice@{online_key()}", best_overall, val_row, config)
                print(f"[INFO] saved best_overall: {BEST_OVERALL_PATH}")
            if small_score > best_small:
                best_small = small_score
                history["best_small"] = {"epoch": epoch, "score": best_small}
                save_best_payload(BEST_SMALL_PATH, model, epoch, f"DiceSmall@{online_key()}", best_small, val_row, config)
                print(f"[INFO] saved best_small: {BEST_SMALL_PATH}")
            if balanced_epoch_score > best_balanced:
                best_balanced = balanced_epoch_score
                history["best_balanced"] = {"epoch": epoch, "score": best_balanced}
                save_best_payload(BEST_BALANCED_PATH, model, epoch, "balanced_score", best_balanced, val_row, config)
                print(f"[INFO] saved best_balanced: {BEST_BALANCED_PATH}")
            history["validation"].append({"epoch": display_epoch, "row": val_row})

        history["train_loss"].append(avg_loss)
        history["dice_loss"].append(avg_dice)
        history["tversky_loss"].append(avg_tversky)
        history["small_local_loss"].append(avg_small)
        history["large_local_loss"].append(avg_large)
        history["lambda_small"].append(last_loss_info["lambda_small"])
        history["lambda_large"].append(last_loss_info["lambda_large"])
        history["lr"].append(current_lr)

        print(
            f"Epoch {display_epoch} | Loss={avg_loss:.4f} | LDice={avg_dice:.4f} | "
            f"LTversky={avg_tversky:.4f} | LSmall={avg_small:.4f} | LLarge={avg_large:.4f} | "
            f"LambdaSmall={last_loss_info['lambda_small']:.4f} | "
            f"LambdaLarge={last_loss_info['lambda_large']:.4f} | "
            f"LR={current_lr:.6f} | validated={should_validate}"
        )

        checkpoint_payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
            "epoch": epoch,
            "run_id": RUN_ID,
            "config": config,
            "best_overall": best_overall,
            "best_small": best_small,
            "best_balanced": best_balanced,
        }
        torch.save(checkpoint_payload, CHECKPOINT_PATH)
        if display_epoch % SNAPSHOT_INTERVAL == 0:
            snapshot_path = os.path.join(RUN_DIR, f"checkpoint_stage2_c3_epoch_{display_epoch}.pth")
            torch.save(checkpoint_payload, snapshot_path)
            print(f"[INFO] saved checkpoint snapshot: {snapshot_path}")
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    torch.save({"model": model.state_dict(), "config": config, "history": history}, FINAL_MODEL_PATH)
    print(f"[INFO] saved final model: {FINAL_MODEL_PATH}")

    history["offline_scan_val"] = {
        "best_overall": evaluate_checkpoint_grid(
            "Best Overall", BEST_OVERALL_PATH, model, val_loader, dice_metric, device, amp_enabled, amp_dtype
        ),
        "best_small": evaluate_checkpoint_grid(
            "Best Small", BEST_SMALL_PATH, model, val_loader, dice_metric, device, amp_enabled, amp_dtype
        ),
        "best_balanced": evaluate_checkpoint_grid(
            "Best Balanced", BEST_BALANCED_PATH, model, val_loader, dice_metric, device, amp_enabled, amp_dtype
        ),
    }
    history["offline_scan_test"] = {
        "best_overall": evaluate_checkpoint_grid(
            "Test Best Overall", BEST_OVERALL_PATH, model, test_loader, dice_metric, device, amp_enabled, amp_dtype
        ),
        "best_small": evaluate_checkpoint_grid(
            "Test Best Small", BEST_SMALL_PATH, model, test_loader, dice_metric, device, amp_enabled, amp_dtype
        ),
        "best_balanced": evaluate_checkpoint_grid(
            "Test Best Balanced", BEST_BALANCED_PATH, model, test_loader, dice_metric, device, amp_enabled, amp_dtype
        ),
    }
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Stage2-C3 finished | run_dir={RUN_DIR}")


if __name__ == "__main__":
    main()
