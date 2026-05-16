import json
import os
from datetime import datetime

import nibabel as nib
import numpy as np
import torch
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
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)

import train
from pretrained_utils import load_pretrained_if_compatible
from project_config import CACHE_DIR, DATA_DIR, PRETRAINED_PATH, SAVE_DIR


RUN_ID = os.environ.get("BME_RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
RUN_NAME = f"from_scratch_iw_b_{RUN_ID}"
RUN_DIR = os.path.join(SAVE_DIR, RUN_NAME)
BEST_MODELS_DIR = os.path.join(RUN_DIR, "best_models")
IW_WEIGHT_DIR = os.path.join(RUN_DIR, "inverse_weight_maps")

CHECKPOINT_PATH = os.path.join(RUN_DIR, "checkpoint_from_scratch_iw_b.pth")
HISTORY_PATH = os.path.join(RUN_DIR, "history_from_scratch_iw_b.json")
FINAL_MODEL_PATH = os.path.join(RUN_DIR, "final_from_scratch_iw_b.pth")
BEST_OVERALL_PATH = os.path.join(BEST_MODELS_DIR, "best_overall.pth")
BEST_SMALL_PATH = os.path.join(BEST_MODELS_DIR, "best_small.pth")
BEST_BALANCED_PATH = os.path.join(BEST_MODELS_DIR, "best_balanced.pth")

ROI_SIZE = (96, 96, 96)
FEATURE_SIZE = 48
TRAIN_BATCH_SIZE = 1
ACCUMULATION_STEPS = 2
AMP_MODE = "bf16"
MAX_EPOCHS = 200
MAX_DATASET_SIZE = 400
WARMUP_EPOCHS = 10
VAL_INTERVAL = 2
LATE_VAL_START_EPOCH = 100
LATE_VAL_INTERVAL = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
VAL_SW_BATCH_SIZE = 2
INFER_OVERLAP = 0.5
SNAPSHOT_INTERVAL = 10
TRAIN_NUM_WORKERS = int(os.environ.get("BME_TRAIN_NUM_WORKERS", "4"))
VAL_NUM_WORKERS = int(os.environ.get("BME_VAL_NUM_WORKERS", "2"))

LAMBDA_DICE = 1.0
LAMBDA_TVERSKY = 0.4
TVERSKY_ALPHA = 0.55
TVERSKY_BETA = 0.45
IW_RAMP_START = 30
IW_RAMP_END = 60
LAMBDA_IW_MAX = 0.03
IW_GAMMA = 0.35
IW_WEIGHT_MIN = 0.7
IW_WEIGHT_MAX = 2.5
PRETRAINED_MIN_MATCH = 20

SMALL_PATCH_FRACTION = 0.25
LESION_PATCH_FRACTION = 0.25
NEGATIVE_PATCH_FRACTION = 0.50

VALIDATION_POINTS = (
    {"threshold": 0.35, "min_cc": 0},
)
PRIMARY_THRESHOLD = 0.35
SMALL_BEST_THRESHOLD = 0.35


class FromScratchIWBProgressiveLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
        self.tversky_loss = TverskyLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
            alpha=TVERSKY_ALPHA,
            beta=TVERSKY_BETA,
        )

    def iw_weight(self, epoch):
        if epoch <= IW_RAMP_START:
            return 0.0
        if epoch >= IW_RAMP_END:
            return LAMBDA_IW_MAX
        progress = (epoch - IW_RAMP_START) / max(IW_RAMP_END - IW_RAMP_START, 1)
        return LAMBDA_IW_MAX * progress

    @staticmethod
    def inverse_weighted_dice_loss(logits, labels, weights):
        probs = torch.softmax(logits, dim=1)[:, 1:2]
        target = (labels > 0).float()
        weights = weights.float()
        reduce_dims = tuple(range(1, probs.ndim))
        intersection = torch.sum(weights * probs * target, dim=reduce_dims)
        denominator = torch.sum(weights * probs, dim=reduce_dims) + torch.sum(weights * target, dim=reduce_dims)
        dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        return 1.0 - dice.mean()

    def forward(self, logits, labels, weights, epoch):
        dice = self.dice_loss(logits, labels)
        tversky = self.tversky_loss(logits, labels)
        base = LAMBDA_DICE * dice + LAMBDA_TVERSKY * tversky
        iw_dice = self.inverse_weighted_dice_loss(logits, labels, weights)
        lambda_iw = self.iw_weight(epoch)
        total = base + lambda_iw * iw_dice
        return total, {
            "dice": float(dice.detach().item()),
            "tversky": float(tversky.detach().item()),
            "base": float(base.detach().item()),
            "iw_dice": float(iw_dice.detach().item()),
            "iw_weight": float(lambda_iw),
        }


def configure_train_module():
    train.ROI_SIZE = ROI_SIZE
    train.FEATURE_SIZE = FEATURE_SIZE
    train.TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE
    train.ACCUMULATION_STEPS = ACCUMULATION_STEPS
    train.AMP_MODE = AMP_MODE
    train.MAX_EPOCHS = MAX_EPOCHS
    train.MAX_DATASET_SIZE = MAX_DATASET_SIZE
    train.WARMUP_EPOCHS = WARMUP_EPOCHS
    train.LEARNING_RATE = LEARNING_RATE
    train.WEIGHT_DECAY = WEIGHT_DECAY
    train.VAL_SW_BATCH_SIZE = VAL_SW_BATCH_SIZE
    train.INFER_OVERLAP = INFER_OVERLAP
    train.SMALL_PATCH_FRACTION = SMALL_PATCH_FRACTION
    train.MISSED_SMALL_PATCH_FRACTION = 0.0
    train.LESION_PATCH_FRACTION = LESION_PATCH_FRACTION
    train.NEGATIVE_PATCH_FRACTION = NEGATIVE_PATCH_FRACTION
    train.IW_GAMMA = IW_GAMMA
    train.IW_WEIGHT_MIN = IW_WEIGHT_MIN
    train.IW_WEIGHT_MAX = IW_WEIGHT_MAX
    train.IW_WEIGHT_DIR = IW_WEIGHT_DIR


def voxel_volume_mm3(label_nii):
    return float(np.prod(label_nii.header.get_zooms()[:3]))


def collect_train_lesion_volumes(train_files):
    volumes = []
    if train.ndi is None:
        raise RuntimeError("scipy.ndimage is required for lesion-volume weighting")

    for case in train_files:
        label_nii = nib.load(case["label"])
        voxel_volume = voxel_volume_mm3(label_nii)
        label = np.asanyarray(label_nii.dataobj) > 0
        labeled, num_components = train.ndi.label(label)
        for component_id in range(1, num_components + 1):
            voxel_count = int(np.sum(labeled == component_id))
            if voxel_count > 0:
                volume = float(voxel_count * voxel_volume)
                volumes.append(volume)

    if not volumes:
        raise RuntimeError("No lesion components found in the training split")
    return volumes


def create_inverse_weight_map(label_path, output_path, volume_median):
    label_nii = nib.load(label_path)
    voxel_volume = voxel_volume_mm3(label_nii)
    label = np.asanyarray(label_nii.dataobj) > 0
    weight = np.ones(label.shape, dtype=np.float32)

    labeled, num_components = train.ndi.label(label)
    for component_id in range(1, num_components + 1):
        component = labeled == component_id
        volume = float(np.sum(component)) * voxel_volume
        if volume <= 0:
            continue
        component_weight = (volume_median / max(volume, 1.0)) ** IW_GAMMA
        component_weight = float(np.clip(component_weight, IW_WEIGHT_MIN, IW_WEIGHT_MAX))
        weight[component] = component_weight

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    weight_nii = nib.Nifti1Image(weight, label_nii.affine, label_nii.header)
    weight_nii.set_data_dtype(np.float32)
    nib.save(weight_nii, output_path)
    return {
        "components": int(num_components),
        "weight_min": float(weight.min()),
        "weight_max": float(weight.max()),
    }


def attach_inverse_weight_maps(train_files, volume_median):
    os.makedirs(IW_WEIGHT_DIR, exist_ok=True)
    attached = []
    generated = 0
    for case in train_files:
        item = dict(case)
        base_name = os.path.basename(item["label"]).replace("_mask.nii.gz", "_iw_weight.nii.gz")
        weight_path = os.path.join(IW_WEIGHT_DIR, base_name)
        if not os.path.exists(weight_path):
            stats = create_inverse_weight_map(item["label"], weight_path, volume_median)
            generated += 1
            print(
                f"[INFO] generated IW map | id={item['id']} | "
                f"components={stats['components']} | range={stats['weight_min']:.3f}-{stats['weight_max']:.3f}"
            )
        item["weight"] = weight_path
        attached.append(item)
    print(f"[INFO] inverse weight maps ready | cases={len(attached)} | generated={generated}")
    return attached


def build_transforms():
    train_trans = Compose(
        [
            LoadImaged(keys=["image", "label", "weight"]),
            EnsureChannelFirstd(keys=["image", "label", "weight"]),
            Orientationd(keys=["image", "label", "weight"], axcodes="RAS", labels=None),
            Spacingd(
                keys=["image", "label", "weight"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1000,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            SpatialPadd(keys=["image", "label", "weight"], spatial_size=ROI_SIZE),
            train.ProfiledRandCropByPosNegLabeld(
                keys=["image", "label", "weight"],
                label_key="label",
                spatial_size=ROI_SIZE,
            ),
            EnsureTyped(keys=["image", "label", "weight"]),
        ]
    )

    eval_trans = Compose(
        [
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
            train.RecordSpatialShapeD(keys=["image"], output_key="valid_shape"),
            SpatialPadd(keys=["image", "label"], spatial_size=ROI_SIZE),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    return train_trans, eval_trans


def build_loaders(train_files, val_files, test_files, volume_median):
    train_trans, eval_trans = build_transforms()
    cache_root = os.path.join(CACHE_DIR, RUN_NAME)
    train_cache = os.path.join(cache_root, "train_iw_b_roi96")
    val_cache = os.path.join(cache_root, "val_iw_b_roi96")
    test_cache = os.path.join(cache_root, "test_iw_b_roi96")
    for path in (train_cache, val_cache, test_cache):
        os.makedirs(path, exist_ok=True)

    weighted_train_files = attach_inverse_weight_maps(train_files, volume_median)
    profiled_train_files = train.build_profiled_training_cases(weighted_train_files)

    train_dataset = PersistentDataset(data=profiled_train_files, transform=train_trans, cache_dir=train_cache)
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


def run_validation_at_threshold(model, data_loader, dice_metric, device, amp_enabled, amp_dtype, threshold, min_cc):
    old_threshold = train.PRED_THRESHOLD
    old_min_cc = train.MIN_COMPONENT_VOXELS
    train.PRED_THRESHOLD = float(threshold)
    train.MIN_COMPONENT_VOXELS = int(min_cc)
    try:
        dice, info = train.run_validation(model, data_loader, dice_metric, device, amp_enabled, amp_dtype)
    finally:
        train.PRED_THRESHOLD = old_threshold
        train.MIN_COMPONENT_VOXELS = old_min_cc
    return dice, info


def run_validation_grid(model, data_loader, dice_metric, device, amp_enabled, amp_dtype):
    results = {}
    for point in VALIDATION_POINTS:
        threshold = float(point["threshold"])
        min_cc = int(point["min_cc"])
        key = f"thr{threshold:.2f}_cc{min_cc}"
        dice, info = run_validation_at_threshold(
            model, data_loader, dice_metric, device, amp_enabled, amp_dtype, threshold, min_cc
        )
        results[key] = {"threshold": threshold, "min_cc": min_cc, "dice": dice, **info}
    return results


def primary_key():
    return f"thr{PRIMARY_THRESHOLD:.2f}_cc0"


def small_key():
    return f"thr{SMALL_BEST_THRESHOLD:.2f}_cc0"


def balanced_score(row):
    pred_gt = float(row["pred_gt_volume_ratio"])
    if pred_gt < 0.8 or pred_gt > 1.8:
        return -1.0
    return float(row["dice"]) + 0.5 * float(row["dice_small"]) + 0.05 * float(row["precision"]) - 0.01 * float(row["fp_per_scan"])


def save_best_payload(path, model, epoch, score_name, score, validation_grid, config):
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "run_id": RUN_ID,
        "score_name": score_name,
        "score": float(score),
        "validation_grid": validation_grid,
        "config": config,
    }
    torch.save(payload, path)


def print_grid_summary(prefix, validation_grid):
    print(prefix)
    for key, row in validation_grid.items():
        print(
            f"  {key} | Dice={row['dice']:.4f} | DiceSmall={row['dice_small']:.4f} | "
            f"RecallSmall={row['recall_small']:.4f} | Precision={row['precision']:.4f} | "
            f"Recall={row['recall']:.4f} | FP/scan={row['fp_per_scan']:.2f} | "
            f"PredGT={row['pred_gt_volume_ratio']:.3f}"
        )


def load_model_payload(path, model, device):
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    return payload


def evaluate_checkpoint_on_test(name, path, model, test_loader, dice_metric, device, amp_enabled, amp_dtype):
    if not os.path.exists(path):
        print(f"[WARN] skipped test for {name}; checkpoint not found: {path}")
        return None
    payload = load_model_payload(path, model, device)
    test_grid = run_validation_grid(model, test_loader, dice_metric, device, amp_enabled, amp_dtype)
    epoch = payload.get("epoch", None) if isinstance(payload, dict) else None
    label = f"Test {name}" if epoch is None else f"Test {name} (epoch {epoch})"
    print_grid_summary(label, test_grid)
    return {"path": path, "epoch": epoch, "grid": test_grid}


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

    data_list = train.build_data_list(DATA_DIR)
    train.describe_size_distribution(data_list, "full dataset")
    data_list = train.limit_data_list(data_list, MAX_DATASET_SIZE)
    train_files, val_files, test_files = train.split_data_list(data_list)
    train.describe_size_distribution(train_files, "train split")
    train.describe_size_distribution(val_files, "val split")
    train.describe_size_distribution(test_files, "test split")

    lesion_volumes = collect_train_lesion_volumes(train_files)
    volume_median = float(np.median(lesion_volumes))
    train.IW_VOLUME_REF = volume_median

    print(
        f"[INFO] from-scratch IW-B | run_dir={RUN_DIR} | pretrained={PRETRAINED_PATH} | "
        f"train/val/test={len(train_files)}/{len(val_files)}/{len(test_files)}"
    )
    print(
        f"[INFO] lesion volume stats | components={len(lesion_volumes)} | "
        f"median={volume_median:.1f} | min={min(lesion_volumes)} | max={max(lesion_volumes)}"
    )

    train_loader, val_loader, test_loader = build_loaders(train_files, val_files, test_files, volume_median)

    model = train.build_model(device)
    load_info = load_pretrained_if_compatible(
        model,
        PRETRAINED_PATH,
        device,
        min_match=PRETRAINED_MIN_MATCH,
        verbose=True,
    )
    if not load_info["loaded"]:
        print("[WARN] SWIN pretrained weights were not loaded; this run will start from random initialization")
    else:
        print(
            f"[INFO] SWIN pretrained weights accepted | "
            f"matched={load_info['matched']} | min_match={PRETRAINED_MIN_MATCH}"
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
    loss_func = FromScratchIWBProgressiveLoss()
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    config = {
        "run_name": RUN_NAME,
        "pretrained_path": PRETRAINED_PATH,
        "loads_best_stage1": False,
        "roi_size": ROI_SIZE,
        "feature_size": FEATURE_SIZE,
        "max_epochs": MAX_EPOCHS,
        "max_dataset_size": MAX_DATASET_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_epochs": WARMUP_EPOCHS,
        "lambda_dice": LAMBDA_DICE,
        "lambda_tversky": LAMBDA_TVERSKY,
        "tversky_alpha": TVERSKY_ALPHA,
        "tversky_beta": TVERSKY_BETA,
        "lambda_iw_max": LAMBDA_IW_MAX,
        "iw_ramp_start": IW_RAMP_START,
        "iw_ramp_end": IW_RAMP_END,
        "iw_gamma": IW_GAMMA,
        "iw_weight_min": IW_WEIGHT_MIN,
        "iw_weight_max": IW_WEIGHT_MAX,
        "iw_volume_median": volume_median,
        "iw_volume_unit": "mm3",
        "pretrained_min_match": PRETRAINED_MIN_MATCH,
        "pretrained_load_info": load_info,
        "small_patch_fraction": SMALL_PATCH_FRACTION,
        "lesion_patch_fraction": LESION_PATCH_FRACTION,
        "negative_patch_fraction": NEGATIVE_PATCH_FRACTION,
        "validation_points": VALIDATION_POINTS,
        "split_sizes": {"train": len(train_files), "val": len(val_files), "test": len(test_files)},
    }

    history = {
        "config": config,
        "train_loss": [],
        "dice_loss": [],
        "tversky_loss": [],
        "iw_dice_loss": [],
        "iw_weight": [],
        "lr": [],
        "validation": [],
        "best_overall": None,
        "best_small": None,
        "best_balanced": None,
    }
    best_overall = -1.0
    best_small = -1.0
    best_balanced = -1.0

    validation_grid = run_validation_grid(model, val_loader, dice_metric, device, amp_enabled, amp_dtype)
    print_grid_summary("Epoch 0 validation", validation_grid)

    initial_primary = validation_grid[primary_key()]["dice"]
    initial_small = validation_grid[small_key()]["dice_small"]
    initial_balanced = max(balanced_score(row) for row in validation_grid.values())
    best_overall = initial_primary
    best_small = initial_small
    best_balanced = initial_balanced
    save_best_payload(BEST_OVERALL_PATH, model, -1, f"Dice@{primary_key()}", best_overall, validation_grid, config)
    save_best_payload(BEST_SMALL_PATH, model, -1, f"DiceSmall@{small_key()}", best_small, validation_grid, config)
    save_best_payload(BEST_BALANCED_PATH, model, -1, "balanced_score", best_balanced, validation_grid, config)
    history["best_overall"] = {"epoch": -1, "score": best_overall}
    history["best_small"] = {"epoch": -1, "score": best_small}
    history["best_balanced"] = {"epoch": -1, "score": best_balanced}

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_tversky = 0.0
        epoch_iw_dice = 0.0
        last_loss_info = {"iw_weight": 0.0}

        progress = tqdm(train_loader, desc=f"IW-B Epoch {epoch + 1}/{MAX_EPOCHS}")
        for step, batch in enumerate(progress):
            inputs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            weights = batch["weight"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype if amp_enabled else None):
                outputs = model(inputs)
                loss, loss_info = loss_func(outputs, labels, weights, epoch)
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
            epoch_iw_dice += float(loss_info["iw_dice"])
            last_loss_info = loss_info

        scheduler.step()
        num_batches = max(len(train_loader), 1)
        avg_loss = epoch_loss / num_batches
        avg_dice = epoch_dice / num_batches
        avg_tversky = epoch_tversky / num_batches
        avg_iw_dice = epoch_iw_dice / num_batches
        current_lr = optimizer.param_groups[0]["lr"]

        current_val_interval = LATE_VAL_INTERVAL if (epoch + 1) >= LATE_VAL_START_EPOCH else VAL_INTERVAL
        should_validate = ((epoch + 1) % current_val_interval == 0) or (epoch + 1 == MAX_EPOCHS)
        validation_grid = None
        if should_validate:
            validation_grid = run_validation_grid(model, val_loader, dice_metric, device, amp_enabled, amp_dtype)
            print_grid_summary(f"Epoch {epoch + 1} validation", validation_grid)

            overall_score = validation_grid[primary_key()]["dice"]
            small_score = validation_grid[small_key()]["dice_small"]
            balanced_candidates = [(balanced_score(row), row) for row in validation_grid.values()]
            balanced_epoch_score, _ = max(balanced_candidates, key=lambda item: item[0])

            if overall_score > best_overall:
                best_overall = overall_score
                history["best_overall"] = {"epoch": epoch, "score": best_overall}
                save_best_payload(BEST_OVERALL_PATH, model, epoch, f"Dice@{primary_key()}", best_overall, validation_grid, config)
                print(f"[INFO] saved best_overall: {BEST_OVERALL_PATH}")

            if small_score > best_small:
                best_small = small_score
                history["best_small"] = {"epoch": epoch, "score": best_small}
                save_best_payload(BEST_SMALL_PATH, model, epoch, f"DiceSmall@{small_key()}", best_small, validation_grid, config)
                print(f"[INFO] saved best_small: {BEST_SMALL_PATH}")

            if balanced_epoch_score > best_balanced:
                best_balanced = balanced_epoch_score
                history["best_balanced"] = {"epoch": epoch, "score": best_balanced}
                save_best_payload(BEST_BALANCED_PATH, model, epoch, "balanced_score", best_balanced, validation_grid, config)
                print(f"[INFO] saved best_balanced: {BEST_BALANCED_PATH}")

        history["train_loss"].append(avg_loss)
        history["dice_loss"].append(avg_dice)
        history["tversky_loss"].append(avg_tversky)
        history["iw_dice_loss"].append(avg_iw_dice)
        history["iw_weight"].append(last_loss_info["iw_weight"])
        history["lr"].append(current_lr)
        if validation_grid is not None:
            history["validation"].append({"epoch": epoch, "grid": validation_grid})

        print(
            f"Epoch {epoch + 1} | Loss={avg_loss:.4f} | LDice={avg_dice:.4f} | "
            f"LTversky={avg_tversky:.4f} | LIWDice={avg_iw_dice:.4f} | "
            f"LambdaIW={last_loss_info['iw_weight']:.4f} | LR={current_lr:.6f} | "
            f"validated={should_validate}"
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
        if (epoch + 1) % SNAPSHOT_INTERVAL == 0:
            snapshot_path = os.path.join(RUN_DIR, f"checkpoint_from_scratch_iw_b_epoch_{epoch + 1}.pth")
            torch.save(checkpoint_payload, snapshot_path)
            print(f"[INFO] saved checkpoint snapshot: {snapshot_path}")

        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    torch.save({"model": model.state_dict(), "config": config, "history": history}, FINAL_MODEL_PATH)
    final_test_grid = run_validation_grid(model, test_loader, dice_metric, device, amp_enabled, amp_dtype)
    print_grid_summary("Final test validation", final_test_grid)
    history["test"] = {
        "final": {"path": FINAL_MODEL_PATH, "epoch": MAX_EPOCHS - 1, "grid": final_test_grid},
        "best_overall": evaluate_checkpoint_on_test(
            "Best Overall", BEST_OVERALL_PATH, model, test_loader, dice_metric, device, amp_enabled, amp_dtype
        ),
        "best_small": evaluate_checkpoint_on_test(
            "Best Small", BEST_SMALL_PATH, model, test_loader, dice_metric, device, amp_enabled, amp_dtype
        ),
        "best_balanced": evaluate_checkpoint_on_test(
            "Best Balanced", BEST_BALANCED_PATH, model, test_loader, dice_metric, device, amp_enabled, amp_dtype
        ),
    }
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"[INFO] saved final model: {FINAL_MODEL_PATH}")
    print(f"[INFO] run finished: {RUN_DIR}")


if __name__ == "__main__":
    main()
