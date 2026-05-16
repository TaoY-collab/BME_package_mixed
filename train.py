import json
import math
import os
import shutil
from datetime import datetime

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from monai.data import DataLoader, PersistentDataset, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss, TverskyLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)

from pretrained_utils import load_pretrained_if_compatible
from project_config import (
    BEST_MODEL_PATH,
    CACHE_DIR,
    CHECKPOINT_PATH,
    DATA_DIR,
    HISTORY_PATH,
    PRETRAINED_PATH,
    SAVE_DIR,
)

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None


# =========================
# DGX Spark config
# =========================

EXPECTED_DATASET_SIZE = 1000
MAX_DATASET_SIZE = 400
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_SEED = 42

ROI_SIZE = (96, 96, 96)
FEATURE_SIZE = 48

TRAIN_BATCH_SIZE = 1
ACCUMULATION_STEPS = 2

AMP_MODE = "bf16"
MAX_EPOCHS = 200
WARMUP_EPOCHS = 10

VAL_INTERVAL = 2
LATE_VAL_START_EPOCH = 100
LATE_VAL_INTERVAL = 1

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

VAL_SW_BATCH_SIZE = 2
INFER_OVERLAP = 0.5
PRED_THRESHOLD = 0.50
MIN_COMPONENT_VOXELS = 0
SNAPSHOT_INTERVAL = 10

USE_GRAD_CHECKPOINT = True
TRAIN_NUM_WORKERS = int(os.environ.get("BME_TRAIN_NUM_WORKERS", "4"))
VAL_NUM_WORKERS = int(os.environ.get("BME_VAL_NUM_WORKERS", "2"))

LOSS_MODE = "ours"  # "baseline", "ours", or "stage2_refine"

STAGE2_REFINEMENT = True
STAGE1_BEST_ARCHIVE_PATH = os.path.join(SAVE_DIR, "best_stage1.pth")
STAGE2_SOURCE_CKPT = STAGE1_BEST_ARCHIVE_PATH
STAGE2_CHECKPOINT_PATH = os.path.join(SAVE_DIR, "checkpoint_stage2_refine.pth")
STAGE2_BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_stage2_refine.pth")
STAGE2_HISTORY_PATH = os.path.join(SAVE_DIR, "history_stage2_refine.json")
STAGE2_FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "final_stage2_refine.pth")
STAGE2B_SOURCE_CKPT = os.environ.get("BME_STAGE2B_SOURCE_CKPT", STAGE2_BEST_MODEL_PATH)
STAGE2B_CHECKPOINT_PATH = os.path.join(SAVE_DIR, "checkpoint_stage2b.pth")
STAGE2B_BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_stage2b.pth")
STAGE2B_HISTORY_PATH = os.path.join(SAVE_DIR, "history_stage2b.json")
STAGE2B_FINAL_MODEL_PATH = os.path.join(SAVE_DIR, "final_stage2b.pth")
RUN_ID = os.environ.get("BME_RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
BEST_MODELS_DIR = os.path.join(SAVE_DIR, "best_models")

LAMBDA_DICE = 1.0
LAMBDA_TVERSKY = 0.4
LAMBDA_CE = 0.0
FOREGROUND_CE_WEIGHT = 3.0
TVERSKY_ALPHA = 0.6
TVERSKY_BETA = 0.4
FOCAL_TVERSKY_GAMMA = 0.75
LAMBDA_BOUNDARY_MAX = 0.02
BOUNDARY_RAMP_START = 40
BOUNDARY_RAMP_END = 120

STAGE2B_MAX_EPOCHS = 30
STAGE2B_LAMBDA_IW_MAX = 0.03
STAGE2B_IW_RAMP_START = 5
STAGE2B_IW_RAMP_END = 15
IW_VOLUME_REF = 2784.0
IW_GAMMA = 0.3
IW_WEIGHT_MIN = 0.7
IW_WEIGHT_MAX = 2.0
IW_WEIGHT_DIR = os.path.join(SAVE_DIR, "inverse_weight_maps_blight")
HARD_SMALL_MASK_DIR = os.path.join(SAVE_DIR, "missed_small_masks_blight")

SMALL_NODULE_VOXELS = 1500
MEDIUM_NODULE_VOXELS = 6000
SMALL_PATCH_FRACTION = 0.15
MISSED_SMALL_PATCH_FRACTION = 0.15
LESION_PATCH_FRACTION = 0.25
NEGATIVE_PATCH_FRACTION = 0.45
MISSED_SMALL_DICE_THRESHOLD = 0.05
MISSED_SMALL_IOU_THRESHOLD = 0.03


# =========================
# Utils / transforms
# =========================


def edge_map_3d(x):
    dx = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
    dy = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
    dz = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])

    dx = F.pad(dx, (0, 0, 0, 0, 0, 1))
    dy = F.pad(dy, (0, 0, 0, 1, 0, 0))
    dz = F.pad(dz, (0, 1, 0, 0, 0, 0))
    return torch.clamp(dx + dy + dz, 0.0, 1.0)


class RecordSpatialShapeD(MapTransform):
    def __init__(self, keys, output_key):
        super().__init__(keys)
        self.output_key = output_key

    def __call__(self, data):
        d = dict(data)
        first_key = self.keys[0]
        d[self.output_key] = np.asarray(d[first_key].shape[1:], dtype=np.int64)
        return d


class ProfiledRandCropByPosNegLabeld(MapTransform):
    def __init__(self, keys, label_key, spatial_size, hard_label_key=None):
        super().__init__(keys)
        self.hard_label_key = hard_label_key
        self.positive_crop = RandCropByPosNegLabeld(
            keys=keys,
            label_key=label_key,
            spatial_size=spatial_size,
            pos=1,
            neg=0,
            num_samples=1,
            allow_smaller=True,
        )
        self.negative_crop = RandCropByPosNegLabeld(
            keys=keys,
            label_key=label_key,
            spatial_size=spatial_size,
            pos=0,
            neg=1,
            num_samples=1,
            allow_smaller=True,
        )
        self.hard_positive_crop = None
        if hard_label_key is not None:
            self.hard_positive_crop = RandCropByPosNegLabeld(
                keys=keys,
                label_key=hard_label_key,
                spatial_size=spatial_size,
                pos=1,
                neg=0,
                num_samples=1,
                allow_smaller=True,
            )

    def __call__(self, data):
        profile = data.get("crop_profile", "lesion_positive")
        if isinstance(profile, (list, tuple)):
            profile = profile[0]
        if profile == "negative":
            return self.negative_crop(data)
        if profile == "missed_small_positive" and self.hard_positive_crop is not None:
            return self.hard_positive_crop(data)
        return self.positive_crop(data)


# =========================
# Losses
# =========================


class BaselineDiceCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
        )

    def forward(self, logits, labels, epoch):
        loss = self.loss(logits, labels)
        info = {
            "stage": 1,
            "dice": float(loss.detach().item()),
            "tversky": 0.0,
            "boundary": 0.0,
            "boundary_weight": 0.0,
        }
        return loss, info


class DiceTverskyBoundaryLoss(torch.nn.Module):
    def __init__(
        self,
        num_classes=2,
        lambda_dice=1.0,
        lambda_tversky=0.4,
        lambda_boundary_max=0.02,
        boundary_ramp_start=40,
        boundary_ramp_end=120,
        tversky_alpha=0.6,
        tversky_beta=0.4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_dice = lambda_dice
        self.lambda_tversky = lambda_tversky
        self.lambda_boundary_max = lambda_boundary_max
        self.boundary_ramp_start = boundary_ramp_start
        self.boundary_ramp_end = boundary_ramp_end

        self.dice_loss = DiceLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
        )
        self.tversky_loss = TverskyLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
            alpha=tversky_alpha,
            beta=tversky_beta,
        )

    def boundary_weight(self, epoch):
        if epoch < self.boundary_ramp_start:
            return 0.0
        if epoch >= self.boundary_ramp_end:
            return self.lambda_boundary_max

        progress = (epoch - self.boundary_ramp_start) / max(
            self.boundary_ramp_end - self.boundary_ramp_start,
            1,
        )
        return self.lambda_boundary_max * 0.5 * (1.0 - math.cos(math.pi * progress))

    def boundary_loss(self, logits, labels):
        probs = torch.softmax(logits, dim=1)[:, 1:2]
        target = F.one_hot(
            labels.squeeze(1).long(),
            num_classes=self.num_classes,
        ).permute(0, 4, 1, 2, 3).float()[:, 1:2]

        pred_boundary = edge_map_3d(probs)
        target_boundary = edge_map_3d(target)
        return F.l1_loss(pred_boundary, target_boundary)

    def forward(self, logits, labels, epoch):
        dice = self.dice_loss(logits, labels)
        tversky = self.tversky_loss(logits, labels)
        boundary_weight = self.boundary_weight(epoch)

        total = self.lambda_dice * dice + self.lambda_tversky * tversky

        loss_info = {
            "stage": 1 if boundary_weight == 0.0 else 2,
            "dice": float(dice.detach().item()),
            "tversky": float(tversky.detach().item()),
            "boundary": 0.0,
            "boundary_weight": float(boundary_weight),
        }

        if boundary_weight > 0.0:
            boundary = self.boundary_loss(logits, labels)
            total = total + boundary_weight * boundary
            loss_info["boundary"] = float(boundary.detach().item())

        return total, loss_info


class Stage2RefinementLoss(torch.nn.Module):
    def __init__(
        self,
        num_classes=2,
        lambda_dice=1.0,
        lambda_tversky=0.4,
        lambda_ce=0.0,
        lambda_boundary_max=0.005,
        boundary_ramp_start=30,
        boundary_ramp_end=60,
        tversky_alpha=0.55,
        tversky_beta=0.45,
        focal_tversky_gamma=0.75,
        foreground_ce_weight=3.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_dice = lambda_dice
        self.lambda_tversky = lambda_tversky
        self.lambda_ce = lambda_ce
        self.lambda_boundary_max = lambda_boundary_max
        self.boundary_ramp_start = boundary_ramp_start
        self.boundary_ramp_end = boundary_ramp_end
        self.focal_tversky_gamma = focal_tversky_gamma
        self.foreground_ce_weight = foreground_ce_weight

        self.dice_loss = DiceLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
        )
        self.tversky_loss = TverskyLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
            alpha=tversky_alpha,
            beta=tversky_beta,
        )

    def boundary_weight(self, epoch):
        if epoch < self.boundary_ramp_start:
            return 0.0
        if epoch >= self.boundary_ramp_end:
            return self.lambda_boundary_max

        progress = (epoch - self.boundary_ramp_start) / max(
            self.boundary_ramp_end - self.boundary_ramp_start,
            1,
        )
        return self.lambda_boundary_max * 0.5 * (1.0 - math.cos(math.pi * progress))

    def boundary_loss(self, logits, labels):
        probs = torch.softmax(logits, dim=1)[:, 1:2]
        target = F.one_hot(
            labels.squeeze(1).long(),
            num_classes=self.num_classes,
        ).permute(0, 4, 1, 2, 3).float()[:, 1:2]

        pred_boundary = edge_map_3d(probs)
        target_boundary = edge_map_3d(target)
        return F.l1_loss(pred_boundary, target_boundary)

    def forward(self, logits, labels, epoch):
        dice = self.dice_loss(logits, labels)
        tversky = self.tversky_loss(logits, labels)
        focal_tversky = torch.pow(torch.clamp(tversky, min=1e-8), self.focal_tversky_gamma)
        ce = logits.new_tensor(0.0)
        if self.lambda_ce > 0.0:
            ce_weight = torch.tensor(
                [1.0, self.foreground_ce_weight],
                dtype=logits.dtype,
                device=logits.device,
            )
            ce = F.cross_entropy(logits, labels.squeeze(1).long(), weight=ce_weight)
        boundary_weight = self.boundary_weight(epoch)
        total = self.lambda_dice * dice + self.lambda_tversky * focal_tversky + self.lambda_ce * ce

        loss_info = {
            "stage": 2,
            "dice": float(dice.detach().item()),
            "tversky": float(tversky.detach().item()),
            "focal_tversky": float(focal_tversky.detach().item()),
            "ce": float(ce.detach().item()),
            "boundary": 0.0,
            "boundary_weight": float(boundary_weight),
        }

        if boundary_weight > 0.0:
            boundary = self.boundary_loss(logits, labels)
            total = total + boundary_weight * boundary
            loss_info["boundary"] = float(boundary.detach().item())

        return total, loss_info


def build_loss():
    if LOSS_MODE == "baseline":
        return BaselineDiceCELoss()
    if LOSS_MODE == "ours":
        return DiceTverskyBoundaryLoss(
            num_classes=2,
            lambda_dice=LAMBDA_DICE,
            lambda_tversky=LAMBDA_TVERSKY,
            lambda_boundary_max=LAMBDA_BOUNDARY_MAX,
            boundary_ramp_start=BOUNDARY_RAMP_START,
            boundary_ramp_end=BOUNDARY_RAMP_END,
            tversky_alpha=TVERSKY_ALPHA,
            tversky_beta=TVERSKY_BETA,
        )
    if LOSS_MODE == "stage2_refine":
        return Stage2RefinementLoss(
            num_classes=2,
            lambda_dice=LAMBDA_DICE,
            lambda_tversky=LAMBDA_TVERSKY,
            lambda_ce=LAMBDA_CE,
            lambda_boundary_max=LAMBDA_BOUNDARY_MAX,
            boundary_ramp_start=BOUNDARY_RAMP_START,
            boundary_ramp_end=BOUNDARY_RAMP_END,
            tversky_alpha=TVERSKY_ALPHA,
            tversky_beta=TVERSKY_BETA,
            focal_tversky_gamma=FOCAL_TVERSKY_GAMMA,
            foreground_ce_weight=FOREGROUND_CE_WEIGHT,
        )
    raise ValueError(f"Unsupported LOSS_MODE: {LOSS_MODE}")


class Stage2BInverseWeightedLoss(torch.nn.Module):
    def __init__(
        self,
        lambda_iw_max=0.05,
        iw_ramp_start=5,
        iw_ramp_end=15,
        stage1_loss=None,
    ):
        super().__init__()
        self.lambda_iw_max = lambda_iw_max
        self.iw_ramp_start = iw_ramp_start
        self.iw_ramp_end = iw_ramp_end
        self.stage1_loss = stage1_loss or DiceTverskyBoundaryLoss(
            num_classes=2,
            lambda_dice=LAMBDA_DICE,
            lambda_tversky=LAMBDA_TVERSKY,
            lambda_boundary_max=LAMBDA_BOUNDARY_MAX,
            boundary_ramp_start=BOUNDARY_RAMP_START,
            boundary_ramp_end=BOUNDARY_RAMP_END,
            tversky_alpha=TVERSKY_ALPHA,
            tversky_beta=TVERSKY_BETA,
        )

    def iw_weight(self, epoch):
        if epoch < self.iw_ramp_start:
            return 0.0
        if epoch >= self.iw_ramp_end:
            return self.lambda_iw_max
        progress = (epoch - self.iw_ramp_start) / max(self.iw_ramp_end - self.iw_ramp_start, 1)
        return self.lambda_iw_max * progress

    def inverse_weighted_dice_loss(self, logits, labels, weights):
        probs = torch.softmax(logits, dim=1)[:, 1:2]
        target = (labels > 0).float()
        weights = weights.float()

        reduce_dims = tuple(range(1, probs.ndim))
        intersection = torch.sum(weights * probs * target, dim=reduce_dims)
        denominator = torch.sum(weights * probs, dim=reduce_dims) + torch.sum(weights * target, dim=reduce_dims)
        dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        return 1.0 - dice.mean()

    def forward(self, logits, labels, weights, epoch):
        stage1, loss_info = self.stage1_loss(logits, labels, epoch)
        iw_weight = self.iw_weight(epoch)
        iw_dice = self.inverse_weighted_dice_loss(logits, labels, weights)
        total = stage1 + iw_weight * iw_dice

        loss_info = dict(loss_info)
        loss_info["iw_dice"] = float(iw_dice.detach().item())
        loss_info["iw_weight"] = float(iw_weight)
        return total, loss_info


# =========================
# Data helpers
# =========================


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
        except Exception as exc:
            print(f"[WARN] failed to read mask: {label_path} | {exc}")
            continue

        mask_sum = float(np.sum(mask > 0))
        if mask_sum <= 0:
            continue

        data_list.append(
            {
                "id": file_name.replace("_img.nii.gz", ""),
                "image": image_path,
                "label": label_path,
                "mask_sum": mask_sum,
                "nodule_size": classify_nodule_size(mask_sum),
            }
        )
    return data_list


def classify_nodule_size(mask_sum):
    mask_sum = float(mask_sum)
    if mask_sum < SMALL_NODULE_VOXELS:
        return "small"
    if mask_sum <= MEDIUM_NODULE_VOXELS:
        return "medium"
    return "large"


def describe_size_distribution(data_list, name):
    counts = {"small": 0, "medium": 0, "large": 0}
    for case in data_list:
        counts[classify_nodule_size(case.get("mask_sum", 0.0))] += 1
    print(
        f"[INFO] {name} size distribution | "
        f"small={counts['small']} | medium={counts['medium']} | large={counts['large']}"
    )
    return counts


def limit_data_list(data_list, max_count):
    if max_count <= 0 or len(data_list) <= max_count:
        return data_list

    rng = np.random.RandomState(SPLIT_SEED)
    groups = {"small": [], "medium": [], "large": []}
    for case in data_list:
        groups[classify_nodule_size(case.get("mask_sum", 0.0))].append(case)

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


def split_data_list(data_list):
    data_count = len(data_list)
    if data_count < 10:
        raise RuntimeError(f"Not enough valid samples in {DATA_DIR}, found only {data_count}")

    test_size = max(VAL_RATIO + TEST_RATIO, 1.0 / data_count)
    stratify = [case["nodule_size"] for case in data_list]
    if min(stratify.count(x) for x in set(stratify)) < 2:
        stratify = None

    train_files, temp_files = train_test_split(
        data_list,
        test_size=test_size,
        random_state=SPLIT_SEED,
        stratify=stratify,
    )
    temp_stratify = [case["nodule_size"] for case in temp_files]
    if min(temp_stratify.count(x) for x in set(temp_stratify)) < 2:
        temp_stratify = None

    val_fraction_of_temp = VAL_RATIO / max(VAL_RATIO + TEST_RATIO, 1e-8)
    val_files, test_files = train_test_split(
        temp_files,
        train_size=val_fraction_of_temp,
        random_state=SPLIT_SEED,
        stratify=temp_stratify,
    )
    return train_files, val_files, test_files


def repeat_cases_to_count(cases, count, rng, crop_profile):
    if count <= 0 or not cases:
        return []

    repeated = []
    while len(repeated) < count:
        shuffled = list(cases)
        rng.shuffle(shuffled)
        for case in shuffled:
            item = dict(case)
            item["crop_profile"] = crop_profile
            repeated.append(item)
            if len(repeated) >= count:
                break
    return repeated


def build_profiled_training_cases(train_files):
    if not train_files:
        return train_files

    rng = np.random.RandomState(SPLIT_SEED)
    small_cases = [case for case in train_files if case["nodule_size"] == "small"]
    missed_small_cases = [case for case in small_cases if case.get("missed_small", False)]
    ordinary_small_cases = [case for case in small_cases if not case.get("missed_small", False)]
    lesion_cases = [case for case in train_files if case["nodule_size"] in ("medium", "large")]
    all_cases = list(train_files)

    target_count = len(train_files)
    small_count = int(round(target_count * SMALL_PATCH_FRACTION))
    missed_small_count = int(round(target_count * MISSED_SMALL_PATCH_FRACTION))
    lesion_count = int(round(target_count * LESION_PATCH_FRACTION))
    negative_count = max(target_count - small_count - missed_small_count - lesion_count, 0)

    if not ordinary_small_cases:
        if small_cases:
            print("[WARN] all small-nodule cases are mined as missed; ordinary-small quota will reuse small cases")
            ordinary_small_cases = small_cases
        else:
            print("[WARN] no small-nodule cases found; reallocating small-positive quota to all lesion cases")
            ordinary_small_cases = all_cases
    if not missed_small_cases:
        print("[WARN] no missed-small cases found; reallocating missed-small quota to ordinary small-positive")
        small_count += missed_small_count
        missed_small_count = 0
    if not lesion_cases:
        print("[WARN] no medium/large cases found; reallocating lesion-positive quota to all lesion cases")
        lesion_cases = all_cases

    profiled = []
    profiled.extend(repeat_cases_to_count(ordinary_small_cases, small_count, rng, "small_positive"))
    profiled.extend(repeat_cases_to_count(missed_small_cases, missed_small_count, rng, "missed_small_positive"))
    profiled.extend(repeat_cases_to_count(lesion_cases, lesion_count, rng, "lesion_positive"))
    profiled.extend(repeat_cases_to_count(all_cases, negative_count, rng, "negative"))
    rng.shuffle(profiled)

    profile_counts = {
        "small_positive": 0,
        "missed_small_positive": 0,
        "lesion_positive": 0,
        "negative": 0,
    }
    for case in profiled:
        profile_counts[case["crop_profile"]] += 1
    print(
        f"[INFO] profiled crop sampling | small_positive={profile_counts['small_positive']} | "
        f"missed_small_positive={profile_counts['missed_small_positive']} | "
        f"lesion_positive={profile_counts['lesion_positive']} | negative={profile_counts['negative']} | "
        f"train_profiled={len(profiled)}"
    )
    return profiled


def inverse_weight_path_for_label(label_path):
    base_name = os.path.basename(label_path).replace("_mask.nii.gz", "_iw_weight.nii.gz")
    return os.path.join(IW_WEIGHT_DIR, base_name)


def create_inverse_weight_map(label_path, output_path):
    label_nii = nib.load(label_path)
    label = np.asanyarray(label_nii.dataobj) > 0
    weight = np.ones(label.shape, dtype=np.float32)

    if ndi is None:
        raise RuntimeError("scipy.ndimage is required to generate inverse lesion-volume weight maps")

    labeled, num_components = ndi.label(label)
    for component_id in range(1, num_components + 1):
        component = labeled == component_id
        volume = float(np.sum(component))
        if volume <= 0:
            continue

        component_weight = (IW_VOLUME_REF / max(volume, 1.0)) ** IW_GAMMA
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
        "weight_mean": float(weight.mean()),
    }


def attach_inverse_weight_maps(data_files):
    os.makedirs(IW_WEIGHT_DIR, exist_ok=True)
    attached = []
    generated = 0
    for case in data_files:
        item = dict(case)
        weight_path = inverse_weight_path_for_label(item["label"])
        if not os.path.exists(weight_path):
            stats = create_inverse_weight_map(item["label"], weight_path)
            generated += 1
            print(
                f"[INFO] generated IW weight map | id={item['id']} | "
                f"components={stats['components']} | "
                f"range={stats['weight_min']:.3f}-{stats['weight_max']:.3f}"
            )
        item["weight"] = weight_path
        attached.append(item)

    print(f"[INFO] inverse weight maps ready | cases={len(attached)} | generated={generated}")
    return attached


def hard_small_mask_path_for_label(label_path):
    base_name = os.path.basename(label_path).replace("_mask.nii.gz", "_hard_small_mask.nii.gz")
    return os.path.join(HARD_SMALL_MASK_DIR, base_name)


def write_hard_small_mask(label_path, output_path, use_gt_mask):
    label_nii = nib.load(label_path)
    label = np.asanyarray(label_nii.dataobj)
    if use_gt_mask:
        hard_mask = (label > 0).astype(np.uint8)
    else:
        hard_mask = np.zeros(label.shape, dtype=np.uint8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    hard_nii = nib.Nifti1Image(hard_mask, label_nii.affine, label_nii.header)
    hard_nii.set_data_dtype(np.uint8)
    nib.save(hard_nii, output_path)


def attach_hard_small_masks(data_files):
    os.makedirs(HARD_SMALL_MASK_DIR, exist_ok=True)
    attached = []
    missed_count = 0
    for case in data_files:
        item = dict(case)
        hard_path = hard_small_mask_path_for_label(item["label"])
        use_gt_mask = bool(item.get("missed_small", False))
        if use_gt_mask:
            missed_count += 1
        write_hard_small_mask(item["label"], hard_path, use_gt_mask)
        item["hard_label"] = hard_path
        attached.append(item)

    print(
        f"[INFO] hard-small masks ready | cases={len(attached)} | "
        f"missed_small_cases={missed_count}"
    )
    return attached


def iou_from_binary(pred, target):
    union = float((pred | target).sum().item())
    if union <= 0:
        return 1.0
    intersection = float((pred & target).sum().item())
    return intersection / (union + 1e-8)


def build_eval_loader_for_mining(data_files):
    _, eval_trans = build_transforms()
    cache_dir = os.path.join(CACHE_DIR, "mine_missed_small_blight_roi96")
    os.makedirs(cache_dir, exist_ok=True)
    dataset = PersistentDataset(data=data_files, transform=eval_trans, cache_dir=cache_dir)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=VAL_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=VAL_NUM_WORKERS > 0,
        collate_fn=list_data_collate,
    )


def mine_missed_small_cases(model, train_files, device, amp_enabled, amp_dtype):
    if not train_files:
        return train_files

    small_ids = {case["id"] for case in train_files if case["nodule_size"] == "small"}
    if not small_ids:
        print("[WARN] no small training cases available for missed-small mining")
        return [dict(case, missed_small=False) for case in train_files]

    model.eval()
    loader = build_eval_loader_for_mining(train_files)
    missed_ids = set()
    checked_small = 0

    with torch.inference_mode():
        for batch in loader:
            case_id = batch["id"][0] if isinstance(batch["id"], (list, tuple)) else batch["id"]
            if case_id not in small_ids:
                continue

            inputs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].long().to(device, non_blocking=True)
            valid_shape = get_batch_valid_shape(batch, labels)

            with torch.amp.autocast(
                "cuda",
                enabled=amp_enabled,
                dtype=amp_dtype if amp_enabled else None,
            ):
                outputs = sliding_window_inference(
                    inputs,
                    roi_size=ROI_SIZE,
                    sw_batch_size=VAL_SW_BATCH_SIZE,
                    predictor=model,
                    overlap=INFER_OVERLAP,
                    mode="gaussian",
                )

            foreground_probs = torch.softmax(outputs, dim=1)[:, 1]
            pred_labels = (foreground_probs >= PRED_THRESHOLD).long()
            pred_labels = crop_tensor_to_shape(pred_labels, valid_shape)
            labels = crop_tensor_to_shape(labels, valid_shape)
            pred_labels = remove_small_components(pred_labels, MIN_COMPONENT_VOXELS)

            pred_case = pred_labels[0] > 0
            target_case = labels[0, 0] > 0
            checked_small += 1

            overlap = float((pred_case & target_case).sum().item())
            dice = dice_from_binary(pred_case, target_case)
            iou = iou_from_binary(pred_case, target_case)
            if overlap <= 0.0 or dice < MISSED_SMALL_DICE_THRESHOLD or iou < MISSED_SMALL_IOU_THRESHOLD:
                missed_ids.add(case_id)

    mined = []
    for case in train_files:
        item = dict(case)
        item["missed_small"] = item["id"] in missed_ids
        mined.append(item)

    print(
        f"[INFO] missed-small mining | checked_small={checked_small} | "
        f"missed_small={len(missed_ids)} | dice_thr={MISSED_SMALL_DICE_THRESHOLD} | "
        f"iou_thr={MISSED_SMALL_IOU_THRESHOLD}"
    )
    return mined


# =========================
# Dataloaders / transforms
# =========================


def build_transforms():
    train_trans = Compose(
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
            SpatialPadd(keys=["image", "label"], spatial_size=ROI_SIZE),
            ProfiledRandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=ROI_SIZE,
            ),
            EnsureTyped(keys=["image", "label"]),
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
            RecordSpatialShapeD(keys=["image"], output_key="valid_shape"),
            SpatialPadd(keys=["image", "label"], spatial_size=ROI_SIZE),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    return train_trans, eval_trans


def build_loaders(train_files, val_files, test_files):
    train_trans, eval_trans = build_transforms()

    os.makedirs(CACHE_DIR, exist_ok=True)
    train_cache = os.path.join(CACHE_DIR, "train_stage2_profiled_roi64")
    val_cache = os.path.join(CACHE_DIR, "val_stage2_roi64")
    test_cache = os.path.join(CACHE_DIR, "test_stage2_roi64")
    for path in (train_cache, val_cache, test_cache):
        os.makedirs(path, exist_ok=True)

    profiled_train_files = build_profiled_training_cases(train_files)

    train_dataset = PersistentDataset(
        data=profiled_train_files,
        transform=train_trans,
        cache_dir=train_cache,
    )
    val_dataset = PersistentDataset(
        data=val_files,
        transform=eval_trans,
        cache_dir=val_cache,
    )
    test_dataset = PersistentDataset(
        data=test_files,
        transform=eval_trans,
        cache_dir=test_cache,
    )

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


def build_stage2b_transforms():
    train_trans = Compose(
        [
            LoadImaged(keys=["image", "label", "weight", "hard_label"]),
            EnsureChannelFirstd(keys=["image", "label", "weight", "hard_label"]),
            Orientationd(keys=["image", "label", "weight", "hard_label"], axcodes="RAS", labels=None),
            Spacingd(
                keys=["image", "label", "weight", "hard_label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest", "nearest", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1000,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            SpatialPadd(keys=["image", "label", "weight", "hard_label"], spatial_size=ROI_SIZE),
            ProfiledRandCropByPosNegLabeld(
                keys=["image", "label", "weight", "hard_label"],
                label_key="label",
                spatial_size=ROI_SIZE,
                hard_label_key="hard_label",
            ),
            EnsureTyped(keys=["image", "label", "weight", "hard_label"]),
        ]
    )

    _, eval_trans = build_transforms()
    return train_trans, eval_trans


def build_stage2b_loaders(train_files, val_files, test_files, model=None, device=None, amp_enabled=False, amp_dtype=torch.float32):
    train_trans, eval_trans = build_stage2b_transforms()

    os.makedirs(CACHE_DIR, exist_ok=True)
    train_cache = os.path.join(CACHE_DIR, "train_stage2b_blight_hard_small_roi96")
    val_cache = os.path.join(CACHE_DIR, "val_stage2b_blight_roi96")
    test_cache = os.path.join(CACHE_DIR, "test_stage2b_blight_roi96")
    for path in (train_cache, val_cache, test_cache):
        os.makedirs(path, exist_ok=True)

    if model is not None and device is not None:
        train_files = mine_missed_small_cases(model, train_files, device, amp_enabled, amp_dtype)
    else:
        train_files = [dict(case, missed_small=False) for case in train_files]

    weighted_train_files = attach_inverse_weight_maps(train_files)
    hard_train_files = attach_hard_small_masks(weighted_train_files)
    profiled_train_files = build_profiled_training_cases(hard_train_files)

    train_dataset = PersistentDataset(
        data=profiled_train_files,
        transform=train_trans,
        cache_dir=train_cache,
    )
    val_dataset = PersistentDataset(
        data=val_files,
        transform=eval_trans,
        cache_dir=val_cache,
    )
    test_dataset = PersistentDataset(
        data=test_files,
        transform=eval_trans,
        cache_dir=test_cache,
    )

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


# =========================
# Validation
# =========================


def crop_tensor_to_shape(tensor, spatial_shape):
    d, h, w = [int(x) for x in spatial_shape]
    return tensor[..., :d, :h, :w]


def get_batch_valid_shape(batch, labels):
    valid_shape = batch.get("valid_shape")
    if valid_shape is None:
        return labels.shape[-3:]

    if isinstance(valid_shape, torch.Tensor):
        valid_shape = valid_shape[0].tolist()
    elif isinstance(valid_shape, np.ndarray):
        valid_shape = valid_shape.tolist()
    elif isinstance(valid_shape[0], (list, tuple, np.ndarray)):
        valid_shape = list(valid_shape[0])
    else:
        valid_shape = list(valid_shape)

    return valid_shape


def dice_from_binary(pred, target):
    pred_sum = float(pred.sum().item())
    target_sum = float(target.sum().item())
    if pred_sum + target_sum <= 0:
        return 1.0
    intersection = float((pred & target).sum().item())
    return (2.0 * intersection) / (pred_sum + target_sum + 1e-8)


def count_false_positive_components(pred, target):
    pred_np = pred.detach().cpu().numpy().astype(bool)
    target_np = target.detach().cpu().numpy().astype(bool)

    if ndi is None:
        return 1 if (pred_np & ~target_np).any() else 0

    labeled, num_components = ndi.label(pred_np)
    false_positive_count = 0
    for component_id in range(1, num_components + 1):
        component = labeled == component_id
        if not np.any(component & target_np):
            false_positive_count += 1
    return false_positive_count


def remove_small_components(pred_labels, min_component_voxels):
    if min_component_voxels <= 0:
        return pred_labels
    if ndi is None:
        print("[WARN] scipy.ndimage is unavailable; skip connected-component filtering")
        return pred_labels

    filtered = []
    for pred in pred_labels:
        pred_np = pred.detach().cpu().numpy().astype(bool)
        labeled, num_components = ndi.label(pred_np)
        keep = np.zeros_like(pred_np, dtype=bool)
        for component_id in range(1, num_components + 1):
            component = labeled == component_id
            if int(component.sum()) >= min_component_voxels:
                keep |= component
        filtered.append(torch.from_numpy(keep.astype(np.int64)).to(pred_labels.device))
    return torch.stack(filtered, dim=0)


def run_validation(model, data_loader, dice_metric, device, amp_enabled, amp_dtype):
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
    large_dice_values = []
    large_detected = 0
    large_total = 0
    small_pred_fg = 0.0
    small_label_fg = 0.0
    large_pred_fg = 0.0
    large_label_fg = 0.0

    with torch.inference_mode():
        for batch in data_loader:
            inputs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].long().to(device, non_blocking=True)
            valid_shape = get_batch_valid_shape(batch, labels)

            with torch.amp.autocast(
                "cuda",
                enabled=amp_enabled,
                dtype=amp_dtype if amp_enabled else None,
            ):
                outputs = sliding_window_inference(
                    inputs,
                    roi_size=ROI_SIZE,
                    sw_batch_size=VAL_SW_BATCH_SIZE,
                    predictor=model,
                    overlap=INFER_OVERLAP,
                    mode="gaussian",
                )

            foreground_probs = torch.softmax(outputs, dim=1)[:, 1]
            pred_labels = (foreground_probs >= PRED_THRESHOLD).long()
            pred_labels = crop_tensor_to_shape(pred_labels, valid_shape)
            labels = crop_tensor_to_shape(labels, valid_shape)
            pred_labels = remove_small_components(pred_labels, MIN_COMPONENT_VOXELS)

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
                total_false_positive_components += count_false_positive_components(pred_case, target_case)
                if target_voxels < SMALL_NODULE_VOXELS:
                    small_total += 1
                    small_dice_values.append(dice_from_binary(pred_case, target_case))
                    small_pred_fg += float(pred_case.sum().item())
                    small_label_fg += target_voxels
                    if float((pred_case & target_case).sum().item()) > 0.0:
                        small_detected += 1
                elif target_voxels > MEDIUM_NODULE_VOXELS:
                    large_total += 1
                    large_dice_values.append(dice_from_binary(pred_case, target_case))
                    large_pred_fg += float(pred_case.sum().item())
                    large_label_fg += target_voxels
                    if float((pred_case & target_case).sum().item()) > 0.0:
                        large_detected += 1

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
    dice_large = float(np.mean(large_dice_values)) if large_dice_values else 0.0
    recall_large = large_detected / max(large_total, 1)
    pred_gt_small = small_pred_fg / max(small_label_fg, 1e-8)
    pred_gt_large = large_pred_fg / max(large_label_fg, 1e-8)

    return dice, {
        "pred_fg": total_pred_fg,
        "label_fg": total_label_fg,
        "dice_small": dice_small,
        "recall_small": recall_small,
        "dice_large": dice_large,
        "recall_large": recall_large,
        "precision": precision,
        "recall": recall,
        "fp_per_scan": fp_per_scan,
        "pred_gt_volume_ratio": pred_gt_volume_ratio,
        "pred_gt_small": pred_gt_small,
        "pred_gt_large": pred_gt_large,
        "small_cases": small_total,
        "large_cases": large_total,
    }


# =========================
# Training
# =========================


def build_model(device):
    model = SwinUNETR(
        in_channels=1,
        out_channels=2,
        feature_size=FEATURE_SIZE,
        use_checkpoint=USE_GRAD_CHECKPOINT,
    ).to(device)
    return model


def choose_amp_dtype(device):
    if device.type != "cuda":
        return torch.float32, False

    if AMP_MODE == "bf16":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16, False

        print("[WARN] bf16 is not reported as supported. Falling back to fp16 AMP.")
        return torch.float16, True

    if AMP_MODE == "fp16":
        return torch.float16, True

    return torch.float32, False


def extract_model_state(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_stage2_source_checkpoint(model, source_path, device):
    if not os.path.exists(source_path):
        print(f"[WARN] stage-2 source checkpoint not found: {source_path}")
        return False

    checkpoint = torch.load(source_path, map_location=device)
    state_dict = extract_model_state(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    print(f"[INFO] loaded stage-2 source checkpoint: {source_path}")
    return True


def archive_stage1_best_if_needed():
    if not STAGE2_REFINEMENT:
        return
    if os.path.exists(STAGE1_BEST_ARCHIVE_PATH):
        print(f"[INFO] using fixed stage-1 source checkpoint: {STAGE1_BEST_ARCHIVE_PATH}")
        return

    print(
        f"[WARN] required stage-1 source checkpoint is missing: {STAGE1_BEST_ARCHIVE_PATH}. "
        "This script will not fall back to best_model.pth."
    )


def print_validation_summary(prefix, dice, val_info):
    print(
        f"{prefix} | "
        f"Dice={dice:.4f} | "
        f"DiceSmall={val_info['dice_small']:.4f} | "
        f"RecallSmall={val_info['recall_small']:.4f} | "
        f"DiceLarge={val_info['dice_large']:.4f} | "
        f"RecallLarge={val_info['recall_large']:.4f} | "
        f"Precision={val_info['precision']:.4f} | "
        f"Recall={val_info['recall']:.4f} | "
        f"FP/scan={val_info['fp_per_scan']:.2f} | "
        f"PredGT={val_info['pred_gt_volume_ratio']:.3f} | "
        f"PredGT_small={val_info['pred_gt_small']:.3f} | "
        f"PredGT_large={val_info['pred_gt_large']:.3f} | "
        f"PredThr={PRED_THRESHOLD:.2f} | "
        f"MinCC={MIN_COMPONENT_VOXELS}"
    )


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(BEST_MODELS_DIR, exist_ok=True)
    archive_stage1_best_if_needed()

    checkpoint_path = STAGE2_CHECKPOINT_PATH if STAGE2_REFINEMENT else CHECKPOINT_PATH
    best_model_path = STAGE2_BEST_MODEL_PATH if STAGE2_REFINEMENT else BEST_MODEL_PATH
    run_best_model_path = os.path.join(BEST_MODELS_DIR, f"best_{RUN_ID}.pth")
    history_path = STAGE2_HISTORY_PATH if STAGE2_REFINEMENT else HISTORY_PATH
    final_model_path = STAGE2_FINAL_MODEL_PATH if STAGE2_REFINEMENT else os.path.join(SAVE_DIR, "final_model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    amp_dtype, scaler_enabled = choose_amp_dtype(device)
    amp_enabled = amp_enabled and amp_dtype in (torch.bfloat16, torch.float16)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

    data_list = build_data_list(DATA_DIR)
    describe_size_distribution(data_list, "full dataset")
    data_list = limit_data_list(data_list, MAX_DATASET_SIZE)
    train_files, val_files, test_files = split_data_list(data_list)
    describe_size_distribution(train_files, "train split")
    describe_size_distribution(val_files, "val split")
    describe_size_distribution(test_files, "test split")

    print(
        f"[INFO] valid samples: {len(data_list)} | "
        f"train: {len(train_files)} | val: {len(val_files)} | test: {len(test_files)}"
    )
    print(
        f"[INFO] config | model=SwinUNETR | roi={ROI_SIZE} | feature_size={FEATURE_SIZE} | "
        f"batch_size={TRAIN_BATCH_SIZE} | accumulation_steps={ACCUMULATION_STEPS} | "
        f"amp={AMP_MODE} | optimizer=AdamW | lr={LEARNING_RATE} | "
        f"weight_decay={WEIGHT_DECAY} | max_epochs={MAX_EPOCHS} | "
        f"warmup_epochs={WARMUP_EPOCHS} | infer_overlap={INFER_OVERLAP} | "
        f"pred_threshold={PRED_THRESHOLD} | max_dataset_size={MAX_DATASET_SIZE} | loss_mode={LOSS_MODE}"
    )
    if STAGE2_REFINEMENT:
        print(
            f"[INFO] stage-2 refinement | source={STAGE2_SOURCE_CKPT} | "
            f"best_out={best_model_path} | run_best_out={run_best_model_path} | "
            f"checkpoint_out={checkpoint_path}"
        )

    train_loader, val_loader, test_loader = build_loaders(train_files, val_files, test_files)

    model = build_model(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=WARMUP_EPOCHS,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(MAX_EPOCHS - WARMUP_EPOCHS, 1),
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS],
    )

    loss_func = build_loss()
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    history = {
        "train_loss": [],
        "val_dice": [],
        "val_acc": [],
        "lr": [],
        "stage": [],
        "boundary_weight": [],
        "ce": [],
        "val_pred_fg": [],
        "val_label_fg": [],
        "val_dice_small": [],
        "val_recall_small": [],
        "val_dice_large": [],
        "val_recall_large": [],
        "val_precision": [],
        "val_recall": [],
        "val_fp_per_scan": [],
        "val_pred_gt_volume_ratio": [],
        "val_pred_gt_small": [],
        "val_pred_gt_large": [],
        "loss_mode": LOSS_MODE,
        "run_id": RUN_ID,
        "run_best_model_path": run_best_model_path,
        "pred_threshold": PRED_THRESHOLD,
        "tversky_alpha": TVERSKY_ALPHA,
        "tversky_beta": TVERSKY_BETA,
        "focal_tversky_gamma": FOCAL_TVERSKY_GAMMA,
        "lambda_ce": LAMBDA_CE,
        "foreground_ce_weight": FOREGROUND_CE_WEIGHT,
        "min_component_voxels": MIN_COMPONENT_VOXELS,
        "max_dataset_size": MAX_DATASET_SIZE,
        "small_nodule_voxels": SMALL_NODULE_VOXELS,
        "medium_nodule_voxels": MEDIUM_NODULE_VOXELS,
        "stage2_refinement": STAGE2_REFINEMENT,
        "stage2_source_ckpt": STAGE2_SOURCE_CKPT if STAGE2_REFINEMENT else "",
        "split_sizes": {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        },
    }

    best_dice = -1.0
    start_epoch = 0
    end_epoch = MAX_EPOCHS

    if os.path.exists(checkpoint_path):
        print(
            f"[INFO] old run checkpoint exists but stage-2 starts a fresh refinement schedule: "
            f"{checkpoint_path}"
        )

    loaded_stage2_source = False
    if STAGE2_REFINEMENT:
        loaded_stage2_source = load_stage2_source_checkpoint(model, STAGE2_SOURCE_CKPT, device)
        if not loaded_stage2_source:
            raise RuntimeError(
                "Stage-2 refinement requires an existing checkpoint. "
                f"Put the current best model at {STAGE2_SOURCE_CKPT} or update STAGE2_SOURCE_CKPT."
            )

    if not loaded_stage2_source:
        load_info = load_pretrained_if_compatible(
            model,
            PRETRAINED_PATH,
            device,
            min_match=1,
            verbose=True,
        )
        if not load_info["loaded"]:
            print("[WARN] training will start from random initialization")

    print(
        f"[INFO] epoch range in this run: {start_epoch} -> {end_epoch - 1} | "
        f"single run to MAX_EPOCHS={MAX_EPOCHS}"
    )

    val_dice_epoch0, val_info_epoch0 = run_validation(
        model=model,
        data_loader=val_loader,
        dice_metric=dice_metric,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    best_dice = val_dice_epoch0
    print_validation_summary("Epoch 0 validation", val_dice_epoch0, val_info_epoch0)
    initial_best_payload = {
        "model": model.state_dict(),
        "best_dice": best_dice,
        "epoch": -1,
        "run_id": RUN_ID,
        "loss_mode": LOSS_MODE,
            "pred_threshold": PRED_THRESHOLD,
            "min_component_voxels": MIN_COMPONENT_VOXELS,
            "val_info": val_info_epoch0,
        "config": {
            "max_epochs": MAX_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "lambda_dice": LAMBDA_DICE,
            "lambda_tversky": LAMBDA_TVERSKY,
            "lambda_ce": LAMBDA_CE,
            "tversky_alpha": TVERSKY_ALPHA,
            "tversky_beta": TVERSKY_BETA,
            "focal_tversky_gamma": FOCAL_TVERSKY_GAMMA,
            "lambda_boundary_max": LAMBDA_BOUNDARY_MAX,
            "min_component_voxels": MIN_COMPONENT_VOXELS,
            "small_patch_fraction": SMALL_PATCH_FRACTION,
            "missed_small_patch_fraction": MISSED_SMALL_PATCH_FRACTION,
            "lesion_patch_fraction": LESION_PATCH_FRACTION,
            "negative_patch_fraction": NEGATIVE_PATCH_FRACTION,
        },
    }
    torch.save(model.state_dict(), best_model_path)
    torch.save(initial_best_payload, run_best_model_path)
    print(f"[INFO] saved epoch-0 run best model: {run_best_model_path}")

    for epoch in range(start_epoch, end_epoch):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_tversky = 0.0
        epoch_ce = 0.0
        epoch_boundary = 0.0
        last_loss_info = {
            "stage": 1,
            "dice": 0.0,
            "tversky": 0.0,
            "ce": 0.0,
            "boundary": 0.0,
            "boundary_weight": 0.0,
        }

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{MAX_EPOCHS}")
        for step, batch in enumerate(progress):
            inputs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast(
                "cuda",
                enabled=amp_enabled,
                dtype=amp_dtype if amp_enabled else None,
            ):
                outputs = model(inputs)
                loss, loss_info = loss_func(outputs, labels, epoch)
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
            epoch_ce += float(loss_info.get("ce", 0.0))
            epoch_boundary += float(loss_info["boundary"])
            last_loss_info = loss_info

        scheduler.step()

        num_batches = max(len(train_loader), 1)
        avg_loss = epoch_loss / num_batches
        avg_dice_loss = epoch_dice / num_batches
        avg_tversky_loss = epoch_tversky / num_batches
        avg_ce_loss = epoch_ce / num_batches
        avg_boundary_loss = epoch_boundary / num_batches
        current_lr = optimizer.param_groups[0]["lr"]

        current_val_interval = LATE_VAL_INTERVAL if (epoch + 1) >= LATE_VAL_START_EPOCH else VAL_INTERVAL
        should_validate = ((epoch + 1) % current_val_interval == 0)

        val_dice = history["val_dice"][-1] if history["val_dice"] else 0.0
        val_info = {
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
        if should_validate:
            val_dice, val_info = run_validation(
                model=model,
                data_loader=val_loader,
                dice_metric=dice_metric,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )

        history["train_loss"].append(avg_loss)
        history["val_dice"].append(val_dice)
        history["val_acc"].append(val_dice)
        history["lr"].append(current_lr)
        history["stage"].append(last_loss_info["stage"])
        history["boundary_weight"].append(last_loss_info["boundary_weight"])
        history["ce"].append(avg_ce_loss)
        history["val_pred_fg"].append(val_info["pred_fg"])
        history["val_label_fg"].append(val_info["label_fg"])
        history["val_dice_small"].append(val_info["dice_small"])
        history["val_recall_small"].append(val_info["recall_small"])
        history["val_dice_large"].append(val_info["dice_large"])
        history["val_recall_large"].append(val_info["recall_large"])
        history["val_precision"].append(val_info["precision"])
        history["val_recall"].append(val_info["recall"])
        history["val_fp_per_scan"].append(val_info["fp_per_scan"])
        history["val_pred_gt_volume_ratio"].append(val_info["pred_gt_volume_ratio"])
        history["val_pred_gt_small"].append(val_info["pred_gt_small"])
        history["val_pred_gt_large"].append(val_info["pred_gt_large"])

        print(
            f"Epoch {epoch + 1} | "
            f"Stage={last_loss_info['stage']} | "
            f"Loss={avg_loss:.4f} | "
            f"Dice={val_dice:.4f} | "
            f"LDice={avg_dice_loss:.4f} | "
            f"LTversky={avg_tversky_loss:.4f} | "
            f"LCE={avg_ce_loss:.4f} | "
            f"LBoundary={avg_boundary_loss:.4f} | "
            f"LambdaBoundary={last_loss_info['boundary_weight']:.4f} | "
            f"DiceSmall={val_info['dice_small']:.4f} | "
            f"RecallSmall={val_info['recall_small']:.4f} | "
            f"DiceLarge={val_info['dice_large']:.4f} | "
            f"RecallLarge={val_info['recall_large']:.4f} | "
            f"Precision={val_info['precision']:.4f} | "
            f"Recall={val_info['recall']:.4f} | "
            f"FP/scan={val_info['fp_per_scan']:.2f} | "
            f"PredGT={val_info['pred_gt_volume_ratio']:.3f} | "
            f"PredGT_small={val_info['pred_gt_small']:.3f} | "
            f"PredGT_large={val_info['pred_gt_large']:.3f} | "
            f"PredThr={PRED_THRESHOLD:.2f} | "
            f"MinCC={MIN_COMPONENT_VOXELS} | "
            f"LR={current_lr:.6f} | "
            f"val_interval={current_val_interval} | "
            f"validated={should_validate}"
        )

        if should_validate and val_dice > best_dice:
            best_dice = val_dice
            best_payload = {
                "model": model.state_dict(),
                "best_dice": best_dice,
                "epoch": epoch,
                "run_id": RUN_ID,
                "loss_mode": LOSS_MODE,
                "pred_threshold": PRED_THRESHOLD,
                "min_component_voxels": MIN_COMPONENT_VOXELS,
                "val_info": val_info,
                "config": {
                    "max_epochs": MAX_EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "lambda_dice": LAMBDA_DICE,
                    "lambda_tversky": LAMBDA_TVERSKY,
                    "lambda_ce": LAMBDA_CE,
                    "tversky_alpha": TVERSKY_ALPHA,
                    "tversky_beta": TVERSKY_BETA,
                    "focal_tversky_gamma": FOCAL_TVERSKY_GAMMA,
                    "lambda_boundary_max": LAMBDA_BOUNDARY_MAX,
                    "min_component_voxels": MIN_COMPONENT_VOXELS,
                    "small_patch_fraction": SMALL_PATCH_FRACTION,
                    "missed_small_patch_fraction": MISSED_SMALL_PATCH_FRACTION,
                    "lesion_patch_fraction": LESION_PATCH_FRACTION,
                    "negative_patch_fraction": NEGATIVE_PATCH_FRACTION,
                },
            }
            torch.save(model.state_dict(), best_model_path)
            torch.save(best_payload, run_best_model_path)
            print(f"[INFO] saved latest best model: {best_model_path}")
            print(f"[INFO] saved run best model: {run_best_model_path}")

        checkpoint_payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
            "best_dice": best_dice,
            "epoch": epoch,
            "split_sizes": history["split_sizes"],
            "loss_mode": LOSS_MODE,
            "run_id": RUN_ID,
            "run_best_model_path": run_best_model_path,
            "pred_threshold": PRED_THRESHOLD,
            "min_component_voxels": MIN_COMPONENT_VOXELS,
            "max_dataset_size": MAX_DATASET_SIZE,
            "small_nodule_voxels": SMALL_NODULE_VOXELS,
            "medium_nodule_voxels": MEDIUM_NODULE_VOXELS,
            "stage2_refinement": STAGE2_REFINEMENT,
            "stage2_source_ckpt": STAGE2_SOURCE_CKPT if STAGE2_REFINEMENT else "",
        }
        torch.save(checkpoint_payload, checkpoint_path)

        if (epoch + 1) % SNAPSHOT_INTERVAL == 0:
            snapshot_prefix = "checkpoint_stage2_refine" if STAGE2_REFINEMENT else "checkpoint"
            snapshot_path = os.path.join(SAVE_DIR, f"{snapshot_prefix}_epoch_{epoch + 1}.pth")
            torch.save(checkpoint_payload, snapshot_path)
            print(f"[INFO] saved checkpoint snapshot: {snapshot_path}")

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    torch.save(model.state_dict(), final_model_path)
    print(f"[INFO] saved final model: {final_model_path}")
    print("[INFO] training finished")
    print(f"[INFO] best dice = {best_dice:.5f}")


def train_stage2b():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(BEST_MODELS_DIR, exist_ok=True)

    run_best_model_path = os.path.join(BEST_MODELS_DIR, f"best_stage2b_{RUN_ID}.pth")
    checkpoint_path = STAGE2B_CHECKPOINT_PATH
    best_model_path = STAGE2B_BEST_MODEL_PATH
    history_path = STAGE2B_HISTORY_PATH
    final_model_path = STAGE2B_FINAL_MODEL_PATH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    amp_dtype, scaler_enabled = choose_amp_dtype(device)
    amp_enabled = amp_enabled and amp_dtype in (torch.bfloat16, torch.float16)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

    data_list = build_data_list(DATA_DIR)
    describe_size_distribution(data_list, "full dataset")
    data_list = limit_data_list(data_list, MAX_DATASET_SIZE)
    train_files, val_files, test_files = split_data_list(data_list)
    describe_size_distribution(train_files, "train split")
    describe_size_distribution(val_files, "val split")
    describe_size_distribution(test_files, "test split")

    print(
        f"[INFO] stage2-B inverse weighting | source={STAGE2B_SOURCE_CKPT} | "
        f"best_out={best_model_path} | run_best_out={run_best_model_path} | "
        f"checkpoint_out={checkpoint_path}"
    )
    print(
        f"[INFO] stage2-B config | epochs={STAGE2B_MAX_EPOCHS} | "
        f"lambda_iw_max={STAGE2B_LAMBDA_IW_MAX} | iw_ramp={STAGE2B_IW_RAMP_START}->{STAGE2B_IW_RAMP_END} | "
        f"iw_ref={IW_VOLUME_REF} | iw_gamma={IW_GAMMA} | iw_clip={IW_WEIGHT_MIN}-{IW_WEIGHT_MAX} | "
        f"sampling=lesion:{LESION_PATCH_FRACTION}/small:{SMALL_PATCH_FRACTION}/"
        f"missed_small:{MISSED_SMALL_PATCH_FRACTION}/negative:{NEGATIVE_PATCH_FRACTION} | "
        f"missed_small_thr=dice<{MISSED_SMALL_DICE_THRESHOLD},iou<{MISSED_SMALL_IOU_THRESHOLD}"
    )

    model = build_model(device)
    if not load_stage2_source_checkpoint(model, STAGE2B_SOURCE_CKPT, device):
        raise RuntimeError(
            "Stage2-B requires a Stage2-A best checkpoint. "
            f"Set BME_STAGE2B_SOURCE_CKPT or put it at {STAGE2B_SOURCE_CKPT}."
        )

    train_loader, val_loader, test_loader = build_stage2b_loaders(
        train_files,
        val_files,
        test_files,
        model=model,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=min(WARMUP_EPOCHS, STAGE2B_MAX_EPOCHS),
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(STAGE2B_MAX_EPOCHS - min(WARMUP_EPOCHS, STAGE2B_MAX_EPOCHS), 1),
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[min(WARMUP_EPOCHS, STAGE2B_MAX_EPOCHS)],
    )

    loss_func = Stage2BInverseWeightedLoss(
        lambda_iw_max=STAGE2B_LAMBDA_IW_MAX,
        iw_ramp_start=STAGE2B_IW_RAMP_START,
        iw_ramp_end=STAGE2B_IW_RAMP_END,
    )
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    history = {
        "train_loss": [],
        "val_dice": [],
        "val_acc": [],
        "lr": [],
        "stage": [],
        "boundary_weight": [],
        "iw_dice": [],
        "iw_weight": [],
        "val_pred_fg": [],
        "val_label_fg": [],
        "val_dice_small": [],
        "val_recall_small": [],
        "val_dice_large": [],
        "val_recall_large": [],
        "val_precision": [],
        "val_recall": [],
        "val_fp_per_scan": [],
        "val_pred_gt_volume_ratio": [],
        "val_pred_gt_small": [],
        "val_pred_gt_large": [],
        "loss_mode": "stage2b_inverse_weighted",
        "run_id": RUN_ID,
        "run_best_model_path": run_best_model_path,
        "source_ckpt": STAGE2B_SOURCE_CKPT,
        "lambda_iw_max": STAGE2B_LAMBDA_IW_MAX,
        "iw_ramp_start": STAGE2B_IW_RAMP_START,
        "iw_ramp_end": STAGE2B_IW_RAMP_END,
        "iw_volume_ref": IW_VOLUME_REF,
        "iw_gamma": IW_GAMMA,
        "iw_weight_min": IW_WEIGHT_MIN,
        "iw_weight_max": IW_WEIGHT_MAX,
        "small_patch_fraction": SMALL_PATCH_FRACTION,
        "missed_small_patch_fraction": MISSED_SMALL_PATCH_FRACTION,
        "lesion_patch_fraction": LESION_PATCH_FRACTION,
        "negative_patch_fraction": NEGATIVE_PATCH_FRACTION,
        "missed_small_dice_threshold": MISSED_SMALL_DICE_THRESHOLD,
        "missed_small_iou_threshold": MISSED_SMALL_IOU_THRESHOLD,
        "split_sizes": {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        },
    }

    val_dice_epoch0, val_info_epoch0 = run_validation(
        model=model,
        data_loader=val_loader,
        dice_metric=dice_metric,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    best_dice = val_dice_epoch0
    print_validation_summary("Stage2-B epoch 0 validation", val_dice_epoch0, val_info_epoch0)
    torch.save(
        {
            "model": model.state_dict(),
            "best_dice": best_dice,
            "epoch": -1,
            "run_id": RUN_ID,
            "loss_mode": "stage2b_inverse_weighted",
            "source_ckpt": STAGE2B_SOURCE_CKPT,
            "val_info": val_info_epoch0,
        },
        run_best_model_path,
    )
    torch.save(model.state_dict(), best_model_path)

    for epoch in range(STAGE2B_MAX_EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_tversky = 0.0
        epoch_boundary = 0.0
        epoch_iw_dice = 0.0
        last_loss_info = {
            "stage": 1,
            "dice": 0.0,
            "tversky": 0.0,
            "boundary": 0.0,
            "boundary_weight": 0.0,
            "iw_dice": 0.0,
            "iw_weight": 0.0,
        }

        progress = tqdm(train_loader, desc=f"Stage2-B Epoch {epoch + 1}/{STAGE2B_MAX_EPOCHS}")
        for step, batch in enumerate(progress):
            inputs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            weights = batch["weight"].to(device, non_blocking=True)

            with torch.amp.autocast(
                "cuda",
                enabled=amp_enabled,
                dtype=amp_dtype if amp_enabled else None,
            ):
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
            epoch_boundary += float(loss_info["boundary"])
            epoch_iw_dice += float(loss_info["iw_dice"])
            last_loss_info = loss_info

        scheduler.step()

        num_batches = max(len(train_loader), 1)
        avg_loss = epoch_loss / num_batches
        avg_dice_loss = epoch_dice / num_batches
        avg_tversky_loss = epoch_tversky / num_batches
        avg_boundary_loss = epoch_boundary / num_batches
        avg_iw_dice = epoch_iw_dice / num_batches
        current_lr = optimizer.param_groups[0]["lr"]

        should_validate = ((epoch + 1) % VAL_INTERVAL == 0) or (epoch + 1 == STAGE2B_MAX_EPOCHS)
        val_dice = history["val_dice"][-1] if history["val_dice"] else best_dice
        val_info = {
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
        if should_validate:
            val_dice, val_info = run_validation(
                model=model,
                data_loader=val_loader,
                dice_metric=dice_metric,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )

        history["train_loss"].append(avg_loss)
        history["val_dice"].append(val_dice)
        history["val_acc"].append(val_dice)
        history["lr"].append(current_lr)
        history["stage"].append(last_loss_info["stage"])
        history["boundary_weight"].append(last_loss_info["boundary_weight"])
        history["iw_dice"].append(avg_iw_dice)
        history["iw_weight"].append(last_loss_info["iw_weight"])
        history["val_pred_fg"].append(val_info["pred_fg"])
        history["val_label_fg"].append(val_info["label_fg"])
        history["val_dice_small"].append(val_info["dice_small"])
        history["val_recall_small"].append(val_info["recall_small"])
        history["val_dice_large"].append(val_info["dice_large"])
        history["val_recall_large"].append(val_info["recall_large"])
        history["val_precision"].append(val_info["precision"])
        history["val_recall"].append(val_info["recall"])
        history["val_fp_per_scan"].append(val_info["fp_per_scan"])
        history["val_pred_gt_volume_ratio"].append(val_info["pred_gt_volume_ratio"])
        history["val_pred_gt_small"].append(val_info["pred_gt_small"])
        history["val_pred_gt_large"].append(val_info["pred_gt_large"])

        print(
            f"Stage2-B Epoch {epoch + 1} | "
            f"Loss={avg_loss:.4f} | "
            f"Dice={val_dice:.4f} | "
            f"LDice={avg_dice_loss:.4f} | "
            f"LTversky={avg_tversky_loss:.4f} | "
            f"LIWDice={avg_iw_dice:.4f} | "
            f"LambdaIW={last_loss_info['iw_weight']:.4f} | "
            f"LBoundary={avg_boundary_loss:.4f} | "
            f"DiceSmall={val_info['dice_small']:.4f} | "
            f"RecallSmall={val_info['recall_small']:.4f} | "
            f"DiceLarge={val_info['dice_large']:.4f} | "
            f"RecallLarge={val_info['recall_large']:.4f} | "
            f"Precision={val_info['precision']:.4f} | "
            f"FP/scan={val_info['fp_per_scan']:.2f} | "
            f"PredGT={val_info['pred_gt_volume_ratio']:.3f} | "
            f"PredGT_small={val_info['pred_gt_small']:.3f} | "
            f"PredGT_large={val_info['pred_gt_large']:.3f} | "
            f"LR={current_lr:.6f} | "
            f"validated={should_validate}"
        )

        if should_validate and val_dice > best_dice:
            best_dice = val_dice
            best_payload = {
                "model": model.state_dict(),
                "best_dice": best_dice,
                "epoch": epoch,
                "run_id": RUN_ID,
                "loss_mode": "stage2b_inverse_weighted",
                "source_ckpt": STAGE2B_SOURCE_CKPT,
                "val_info": val_info,
                "lambda_iw": last_loss_info["iw_weight"],
            }
            torch.save(model.state_dict(), best_model_path)
            torch.save(best_payload, run_best_model_path)
            print(f"[INFO] saved stage2-B latest best model: {best_model_path}")
            print(f"[INFO] saved stage2-B run best model: {run_best_model_path}")

        checkpoint_payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
            "best_dice": best_dice,
            "epoch": epoch,
            "loss_mode": "stage2b_inverse_weighted",
            "source_ckpt": STAGE2B_SOURCE_CKPT,
        }
        torch.save(checkpoint_payload, checkpoint_path)

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    torch.save(model.state_dict(), final_model_path)
    print(f"[INFO] saved stage2-B final model: {final_model_path}")
    print(f"[INFO] stage2-B training finished | best dice = {best_dice:.5f}")


if __name__ == "__main__":
    if os.environ.get("BME_TRAIN_MODE", "").lower() == "stage2b":
        train_stage2b()
    else:
        main()
