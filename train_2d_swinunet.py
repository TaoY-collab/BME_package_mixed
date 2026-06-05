import argparse
import json
import os
import random
import time
from datetime import datetime

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from dual_swin_fusion import SlicePropagationSwinUnet2D, SwinUnet2D
from project_config import DATA_DIR, SAVE_DIR

try:
    from scipy.ndimage import zoom
except ImportError:
    zoom = None


RUN_NAME = "swinunet2d_lung_nodule"
RUN_ID = os.environ.get("BME_RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
RUN_DIR = os.path.join(SAVE_DIR, RUN_NAME)
SLICE_DIR = os.path.join(RUN_DIR, "slices_2d")
MANIFEST_PATH = os.path.join(RUN_DIR, "manifest_2d.json")
BEST_MODEL_PATH = os.path.join(RUN_DIR, "best_swinunet2d.pth")
FINAL_MODEL_PATH = os.path.join(RUN_DIR, "final_swinunet2d.pth")
HISTORY_PATH = os.path.join(RUN_DIR, "history_swinunet2d.json")
DEFAULT_PRETRAINED_PATH = os.path.join(SAVE_DIR, "pretrained_2d", "swin_tiny_patch4_window7_224.pth")

IMG_SIZE = int(os.environ.get("BME_2D_IMG_SIZE", "224"))
EMBED_DIM = int(os.environ.get("BME_2D_EMBED_DIM", "96"))
WINDOW_SIZE = int(os.environ.get("BME_2D_WINDOW_SIZE", "0"))
BATCH_SIZE = int(os.environ.get("BME_2D_BATCH_SIZE", "8"))
CONTEXT_SLICES = int(os.environ.get("BME_2D_CONTEXT_SLICES", "3"))
MAX_EPOCHS = int(os.environ.get("BME_2D_MAX_EPOCHS", "60"))
MAX_TRAIN_SECONDS = int(os.environ.get("BME_2D_MAX_TRAIN_SECONDS", "0"))
LEARNING_RATE = float(os.environ.get("BME_2D_LR", "1e-4"))
WEIGHT_DECAY = float(os.environ.get("BME_2D_WEIGHT_DECAY", "1e-5"))
VAL_RATIO = float(os.environ.get("BME_2D_VAL_RATIO", "0.34"))
SPLIT_SEED = int(os.environ.get("BME_2D_SPLIT_SEED", "42"))
NEGATIVE_PER_POSITIVE = float(os.environ.get("BME_2D_NEG_PER_POS", "1.0"))
MAX_POSITIVE_SLICES_PER_CASE = int(os.environ.get("BME_2D_MAX_POS_PER_CASE", "96"))
BALANCED_SAMPLING = os.environ.get("BME_2D_BALANCED_SAMPLING", "1") != "0"
SAMPLES_PER_EPOCH_MULTIPLIER = float(os.environ.get("BME_2D_SAMPLES_PER_EPOCH_MULT", "2.0"))
NUM_WORKERS = int(os.environ.get("BME_2D_NUM_WORKERS", "2"))
PRED_THRESHOLD = float(os.environ.get("BME_2D_THRESHOLD", "0.5"))
EARLY_STOPPING = os.environ.get("BME_2D_EARLY_STOPPING", "0") == "1"
PATIENCE = int(os.environ.get("BME_2D_PATIENCE", "15"))
HU_MIN = float(os.environ.get("BME_2D_HU_MIN", "-1000"))
HU_MAX = float(os.environ.get("BME_2D_HU_MAX", "400"))
PRETRAINED_PATH = os.environ.get("BME_2D_PRETRAINED_PATH", DEFAULT_PRETRAINED_PATH)
LOAD_PRETRAINED = os.environ.get("BME_2D_LOAD_PRETRAINED", "1") != "0"
PLANE_SPEC = os.environ.get("BME_2D_PLANES", "xyz")
BALANCE_BY_PLANE = os.environ.get("BME_2D_BALANCE_BY_PLANE", "1") != "0"
AMP_MODE = os.environ.get("BME_2D_AMP_MODE", "bf16").lower()
USE_SLICE_PROPAGATION = os.environ.get("BME_2D_SLICE_PROPAGATION", "1") != "0"
Z_ENCODER_BIAS = float(os.environ.get("BME_2D_Z_ENCODER_BIAS", "1.75"))
PROPAGATION_DEPTH = int(os.environ.get("BME_2D_PROP_DEPTH", "1"))
PROPAGATION_HEADS = int(os.environ.get("BME_2D_PROP_HEADS", "4"))
PROPAGATION_DROPOUT = float(os.environ.get("BME_2D_PROP_DROPOUT", "0.0"))


PLANE_INFO = {
    "x": {"axis": 0, "name": "sagittal"},
    "y": {"axis": 1, "name": "coronal"},
    "z": {"axis": 2, "name": "axial"},
}
PLANE_ALIASES = {
    "x": "x",
    "sagittal": "x",
    "yz": "x",
    "y": "y",
    "coronal": "y",
    "xz": "y",
    "z": "z",
    "axial": "z",
    "xy": "z",
}
ALL_PLANE_ALIASES = {"all", "xyz", "x_y_z", "triplanar", "tri_planar", "3plane", "3planes"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resize_2d(array, size, order):
    if array.shape == (size, size):
        return array
    if zoom is None:
        raise RuntimeError("scipy is required to resize 2D slices during mask generation")
    factors = (size / array.shape[0], size / array.shape[1])
    return zoom(array, factors, order=order)


def normalize_hu(image):
    image = np.clip(image.astype(np.float32), HU_MIN, HU_MAX)
    return (image - HU_MIN) / max(HU_MAX - HU_MIN, 1e-6)


def normalize_context_slices(value):
    context_slices = int(value)
    if context_slices < 1:
        raise ValueError("BME_2D_CONTEXT_SLICES must be >= 1.")
    if context_slices % 2 == 0:
        raise ValueError("BME_2D_CONTEXT_SLICES must be odd so one center slice is defined.")
    return context_slices


def context_indices(center_index, num_slices, context_slices):
    radius = context_slices // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.int32)
    return np.clip(center_index + offsets, 0, num_slices - 1)


def make_context_image(image_stack, center_index, context_slices):
    indices = context_indices(center_index, image_stack.shape[0], context_slices)
    slices = normalize_hu(image_stack[indices])
    return np.stack(
        [resize_2d(slice_image, IMG_SIZE, order=1).astype(np.float32) for slice_image in slices],
        axis=0,
    )


def parse_plane_spec(value):
    value = str(value).strip().lower().replace("-", "_")
    if value in ALL_PLANE_ALIASES:
        return ("x", "y", "z")
    for separator in ("+", "/", "|", ";", " "):
        value = value.replace(separator, ",")
    planes = []
    for token in [item.strip() for item in value.split(",") if item.strip()]:
        if token in ALL_PLANE_ALIASES:
            candidates = ("x", "y", "z")
        else:
            if token not in PLANE_ALIASES:
                raise ValueError(f"Unsupported BME_2D_PLANES token: {token}. Use x/y/z, xyz, or axial/coronal/sagittal.")
            candidates = (PLANE_ALIASES[token],)
        for plane in candidates:
            if plane not in planes:
                planes.append(plane)
    if not planes:
        raise ValueError("BME_2D_PLANES must select at least one plane.")
    return tuple(planes)


def plane_stack(volume, plane):
    info = PLANE_INFO[plane]
    return np.moveaxis(volume, info["axis"], 0)


def generation_config(data_dir, planes, context_slices):
    return {
        "version": 3,
        "img_size": IMG_SIZE,
        "context_slices": int(context_slices),
        "in_channels": int(context_slices),
        "planes": list(planes),
        "plane_names": {plane: PLANE_INFO[plane]["name"] for plane in planes},
        "negative_per_positive": NEGATIVE_PER_POSITIVE,
        "max_positive_slices_per_case": MAX_POSITIVE_SLICES_PER_CASE,
        "hu_min": HU_MIN,
        "hu_max": HU_MAX,
        "data_dir": os.path.abspath(data_dir),
    }


def manifest_matches_config(manifest, expected_config):
    config = manifest.get("config", {})
    keys = (
        "version",
        "img_size",
        "context_slices",
        "in_channels",
        "planes",
        "negative_per_positive",
        "max_positive_slices_per_case",
        "hu_min",
        "hu_max",
        "data_dir",
    )
    return all(config.get(key) == expected_config.get(key) for key in keys)


def resolve_window_size(img_size):
    if img_size % 32 != 0:
        raise ValueError("BME_2D_IMG_SIZE must be divisible by 32 for the 4-stage Swin-Unet.")
    patch_resolution = img_size // 4
    if WINDOW_SIZE > 0:
        candidates = (WINDOW_SIZE,)
    else:
        candidates = range(min(7, patch_resolution), 0, -1)
    for candidate in candidates:
        valid = True
        for layer_idx in range(4):
            resolution = patch_resolution // (2 ** layer_idx)
            window = min(candidate, resolution)
            if resolution % window != 0:
                valid = False
                break
        if valid:
            return int(candidate)
    raise ValueError(
        f"BME_2D_WINDOW_SIZE={WINDOW_SIZE} is incompatible with BME_2D_IMG_SIZE={img_size}."
    )


def find_cases(data_dir):
    cases = []
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith("_img.nii.gz"):
            continue
        image_path = os.path.join(data_dir, name)
        label_path = image_path.replace("_img.nii.gz", "_mask.nii.gz")
        if not os.path.exists(label_path):
            continue
        case_id = name.replace("_img.nii.gz", "")
        cases.append({"id": case_id, "image": image_path, "label": label_path})
    if not cases:
        raise RuntimeError(f"No *_img.nii.gz / *_mask.nii.gz pairs found in {data_dir}")
    return cases


def choose_slices(mask, rng):
    positive = np.where(mask.reshape(mask.shape[0], -1).sum(axis=1) > 0)[0].tolist()
    if len(positive) > MAX_POSITIVE_SLICES_PER_CASE:
        positive = sorted(rng.choice(positive, size=MAX_POSITIVE_SLICES_PER_CASE, replace=False).tolist())

    positive_set = set(positive)
    candidate_negative = set()
    for z in positive:
        for offset in (-3, -2, -1, 1, 2, 3):
            nz = z + offset
            if 0 <= nz < mask.shape[0] and nz not in positive_set:
                candidate_negative.add(nz)

    target_negatives = int(round(len(positive) * NEGATIVE_PER_POSITIVE))
    candidate_negative = sorted(candidate_negative)
    if target_negatives > 0 and len(candidate_negative) > target_negatives:
        candidate_negative = sorted(rng.choice(candidate_negative, size=target_negatives, replace=False).tolist())

    return positive, candidate_negative


def generate_2d_slices(data_dir, force=False, plane_spec=PLANE_SPEC, context_slices=CONTEXT_SLICES):
    os.makedirs(SLICE_DIR, exist_ok=True)
    os.makedirs(RUN_DIR, exist_ok=True)
    planes = parse_plane_spec(plane_spec)
    context_slices = normalize_context_slices(context_slices)
    expected_config = generation_config(data_dir, planes, context_slices)
    if os.path.exists(MANIFEST_PATH) and not force:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest_matches_config(manifest, expected_config):
            return manifest
        print("[INFO] existing 2D manifest config differs; regenerating selected-plane slice manifest")

    cases = find_cases(data_dir)
    rng = np.random.default_rng(SPLIT_SEED)
    records = []
    for case in tqdm(cases, desc=f"Generating 2D {','.join(planes)} image/mask slices"):
        image = nib.load(case["image"]).get_fdata().astype(np.float32)
        mask = (nib.load(case["label"]).get_fdata() > 0).astype(np.uint8)

        for plane in planes:
            image_stack = plane_stack(image, plane)
            mask_stack = plane_stack(mask, plane)
            positive_slices, negative_slices = choose_slices(mask_stack, rng)
            selected = [(idx, 1) for idx in positive_slices] + [(idx, 0) for idx in negative_slices]

            for slice_index, has_nodule in selected:
                image_context = make_context_image(image_stack, slice_index, context_slices)
                mask_slice = mask_stack[slice_index].astype(np.uint8)
                mask_slice = (resize_2d(mask_slice, IMG_SIZE, order=0) > 0).astype(np.uint8)

                out_name = f"{case['id']}_{plane}{slice_index:04d}.npz"
                out_path = os.path.join(SLICE_DIR, out_name)
                np.savez_compressed(
                    out_path,
                    image=image_context,
                    mask=mask_slice[None],
                    case_id=case["id"],
                    plane=plane,
                    plane_name=PLANE_INFO[plane]["name"],
                    axis=np.asarray(PLANE_INFO[plane]["axis"], dtype=np.int32),
                    slice_index=np.asarray(slice_index, dtype=np.int32),
                    context_slices=np.asarray(context_slices, dtype=np.int32),
                    has_nodule=np.asarray(has_nodule, dtype=np.uint8),
                )
                records.append(
                    {
                        "path": out_path,
                        "case_id": case["id"],
                        "plane": plane,
                        "plane_name": PLANE_INFO[plane]["name"],
                        "axis": int(PLANE_INFO[plane]["axis"]),
                        "slice_index": int(slice_index),
                        "context_slices": int(context_slices),
                        "has_nodule": int(has_nodule),
                        "mask_pixels": int(mask_slice.sum()),
                    }
                )

    plane_counts = {
        plane: {
            "records": sum(1 for record in records if record["plane"] == plane),
            "positive": sum(1 for record in records if record["plane"] == plane and int(record["has_nodule"]) == 1),
            "negative": sum(1 for record in records if record["plane"] == plane and int(record["has_nodule"]) == 0),
        }
        for plane in planes
    }

    manifest = {
        "config": expected_config,
        "plane_counts": plane_counts,
        "records": records,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def build_shared_3d_split(data_dir):
    import train as train3d

    data_list = train3d.build_data_list(data_dir)
    data_list = train3d.limit_data_list(data_list, train3d.MAX_DATASET_SIZE)
    train_files, val_files, test_files = train3d.split_data_list(data_list)
    return train_files, val_files, test_files


def split_records_by_shared_3d_split(records, data_dir):
    train_files, val_files, test_files = build_shared_3d_split(data_dir)
    train_cases = {case["id"] for case in train_files}
    val_cases = {case["id"] for case in val_files}
    test_cases = {case["id"] for case in test_files}

    train_records = [record for record in records if record["case_id"] in train_cases]
    val_records = [record for record in records if record["case_id"] in val_cases]
    test_records = [record for record in records if record["case_id"] in test_cases]
    return (
        train_records,
        val_records,
        test_records,
        sorted(train_cases),
        sorted(val_cases),
        sorted(test_cases),
    )


def count_pos_neg(records):
    positives = sum(1 for record in records if int(record["has_nodule"]) == 1)
    negatives = len(records) - positives
    return positives, negatives


def count_pos_neg_by_plane(records):
    counts = {}
    for record in records:
        plane = record.get("plane", "z")
        if plane not in counts:
            counts[plane] = {"records": 0, "positive": 0, "negative": 0}
        counts[plane]["records"] += 1
        if int(record["has_nodule"]) == 1:
            counts[plane]["positive"] += 1
        else:
            counts[plane]["negative"] += 1
    return counts


def format_plane_counts(counts):
    parts = []
    for plane in ("x", "y", "z"):
        if plane not in counts:
            continue
        row = counts[plane]
        parts.append(f"{plane}:{row['records']}(pos={row['positive']},neg={row['negative']})")
    return ", ".join(parts) if parts else "none"


def format_time_budget(max_train_seconds):
    return f"{max_train_seconds}s" if max_train_seconds > 0 else "disabled"


class SliceDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records = records
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        data = np.load(record["path"])
        image = torch.from_numpy(data["image"].astype(np.float32))
        mask = torch.from_numpy(data["mask"].astype(np.int64))

        if self.augment:
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=[1])
                mask = torch.flip(mask, dims=[1])
            if torch.rand(()) < 0.5:
                image = torch.flip(image, dims=[2])
                mask = torch.flip(mask, dims=[2])
            if torch.rand(()) < 0.25:
                k = int(torch.randint(1, 4, (1,)).item())
                image = torch.rot90(image, k=k, dims=[1, 2])
                mask = torch.rot90(mask, k=k, dims=[1, 2])
            if torch.rand(()) < 0.2:
                image = torch.clamp(image * (0.9 + 0.2 * torch.rand(())) + (torch.rand(()) - 0.5) * 0.1, 0.0, 1.0)

        return {
            "image": image,
            "mask": mask,
            "has_nodule": int(record["has_nodule"]),
            "plane": record.get("plane", "z"),
        }


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, foreground_weight=3.0):
        super().__init__()
        self.register_buffer("ce_weight", torch.tensor([1.0, foreground_weight], dtype=torch.float32))

    def forward(self, logits, mask, sample_weight=None):
        target = mask.squeeze(1).long()
        ce_map = F.cross_entropy(logits, target, weight=self.ce_weight.to(logits.device), reduction="none")
        if sample_weight is None:
            sample_weight = torch.ones(logits.shape[0], device=logits.device, dtype=logits.dtype)
        sample_weight = sample_weight.to(logits.device, dtype=logits.dtype).view(-1)
        weight_norm = torch.clamp(sample_weight.sum(), min=1e-6)
        ce = (ce_map.mean(dim=(1, 2)) * sample_weight).sum() / weight_norm
        probs = torch.softmax(logits, dim=1)[:, 1]
        target_fg = (target > 0).float()
        reduce_dims = tuple(range(1, probs.ndim))
        intersection = torch.sum(probs * target_fg, dim=reduce_dims)
        denominator = torch.sum(probs, dim=reduce_dims) + torch.sum(target_fg, dim=reduce_dims)
        dice_per_sample = 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        dice = (dice_per_sample * sample_weight).sum() / weight_norm
        return dice + ce, {"dice_loss": float(dice.detach().item()), "ce": float(ce.detach().item())}


def plane_sample_weights(planes, device):
    weights = []
    for plane in planes:
        weights.append(Z_ENCODER_BIAS if str(plane).lower() == "z" else 1.0)
    return torch.tensor(weights, device=device, dtype=torch.float32)


def model_forward(model, images, planes=None):
    if getattr(model, "uses_plane_heads", False):
        return model(images, planes=planes)
    return model(images)


def build_model(device, context_slices=CONTEXT_SLICES, planes=("x", "y", "z")):
    context_slices = normalize_context_slices(context_slices)
    window_size = resolve_window_size(IMG_SIZE)
    if USE_SLICE_PROPAGATION:
        return SlicePropagationSwinUnet2D(
            img_size=(IMG_SIZE, IMG_SIZE),
            context_slices=context_slices,
            planes=planes,
            num_classes=2,
            embed_dim=EMBED_DIM,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            window_size=window_size,
            drop_path_rate=0.1,
            propagation_depth=PROPAGATION_DEPTH,
            propagation_heads=PROPAGATION_HEADS,
            propagation_dropout=PROPAGATION_DROPOUT,
        ).to(device)
    return SwinUnet2D(
        img_size=(IMG_SIZE, IMG_SIZE),
        in_channels=context_slices,
        num_classes=2,
        embed_dim=EMBED_DIM,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=window_size,
        drop_path_rate=0.1,
    ).to(device)


def extract_pretrained_state(payload):
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "network", "module"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported pretrained checkpoint type: {type(payload)}")


def normalize_pretrained_key(key):
    for prefix in ("module.", "model."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    return key


def candidate_pretrained_keys(key):
    candidates = [key]
    if USE_SLICE_PROPAGATION and key.startswith(("patch_embed.", "layers.", "norm.")):
        candidates.append(f"shared_encoder.{key}")
    return candidates


def load_pretrained_if_available(model, path, device):
    if not LOAD_PRETRAINED:
        print("[INFO] 2D pretrained loading disabled by BME_2D_LOAD_PRETRAINED=0")
        return {"loaded": False, "path": path, "matched": 0, "missing": len(model.state_dict())}
    if not path or not os.path.exists(path):
        print(f"[WARN] 2D pretrained checkpoint not found: {path}")
        print("[WARN] run: python download_2d_swinunet_weights.py")
        return {"loaded": False, "path": path, "matched": 0, "missing": len(model.state_dict())}

    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)

    pretrained_state = extract_pretrained_state(payload)
    model_state = model.state_dict()
    matched = {}
    mismatched = []
    unexpected = []
    for raw_key, value in pretrained_state.items():
        key = normalize_pretrained_key(raw_key)
        target_key = None
        for candidate in candidate_pretrained_keys(key):
            if candidate in model_state:
                target_key = candidate
                break
        if target_key is None:
            unexpected.append(raw_key)
            continue
        if tuple(value.shape) != tuple(model_state[target_key].shape):
            if (
                target_key == "shared_encoder.patch_embed.proj.weight"
                and value.ndim == 4
                and model_state[target_key].ndim == 4
                and value.shape[0] == model_state[target_key].shape[0]
                and tuple(value.shape[2:]) == tuple(model_state[target_key].shape[2:])
            ):
                value = value.mean(dim=1, keepdim=True)
            else:
                mismatched.append(target_key)
                continue
        if tuple(value.shape) != tuple(model_state[target_key].shape):
            mismatched.append(target_key)
            continue
        matched[target_key] = value

    updated = dict(model_state)
    updated.update(matched)
    model.load_state_dict(updated, strict=False)
    info = {
        "loaded": bool(matched),
        "path": path,
        "matched": len(matched),
        "missing": len(model_state) - len(matched),
        "unexpected": len(unexpected),
        "mismatched": len(mismatched),
        "sample_mismatched": mismatched[:10],
        "sample_unexpected": unexpected[:10],
    }
    print(
        "[INFO] 2D pretrained load | "
        f"path={path} | matched={info['matched']} | missing={info['missing']} | "
        f"unexpected={info['unexpected']} | mismatched={info['mismatched']}"
    )
    if info["matched"] == 0:
        print("[WARN] no pretrained tensors matched. For official Swin-T weights, use BME_2D_EMBED_DIM=96.")
    return info


def choose_amp_dtype(device):
    if device.type != "cuda" or AMP_MODE in {"0", "off", "false", "no", "fp32"}:
        return None, False, False

    if AMP_MODE == "bf16":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16, True, False
        print("[WARN] bf16 is not reported as supported. Falling back to fp16 AMP.")
        return torch.float16, True, True

    if AMP_MODE == "fp16":
        return torch.float16, True, True

    print(f"[WARN] unsupported BME_2D_AMP_MODE={AMP_MODE}; falling back to fp16 AMP.")
    return torch.float16, True, True


def dice_from_logits(logits, mask, threshold=PRED_THRESHOLD):
    probs = torch.softmax(logits, dim=1)[:, 1]
    pred = probs >= threshold
    target = mask.squeeze(1) > 0
    intersection = (pred & target).sum(dim=(1, 2)).float()
    denom = pred.sum(dim=(1, 2)).float() + target.sum(dim=(1, 2)).float()
    dice = torch.where(denom > 0, (2.0 * intersection) / torch.clamp(denom, min=1.0), torch.ones_like(denom))
    return dice


def evaluate(model, loader, device, amp_enabled, amp_dtype):
    model.eval()
    dice_values = []
    positive_dice = []
    by_plane = {}
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            planes = batch.get("plane", ["z"] * int(images.shape[0]))
            with torch.amp.autocast(
                "cuda",
                enabled=amp_enabled,
                dtype=amp_dtype if amp_enabled else None,
            ):
                logits = model_forward(model, images, planes=planes)
            dice = dice_from_logits(logits, masks)
            dice_cpu = dice.detach().cpu()
            dice_values.extend(dice_cpu.tolist())
            has_nodule = batch["has_nodule"].bool()
            if has_nodule.any():
                positive_dice.extend(dice_cpu[has_nodule].tolist())
            if isinstance(planes, str):
                planes = [planes]
            for plane, dice_value, positive in zip(planes, dice_cpu.tolist(), has_nodule.cpu().tolist()):
                if plane not in by_plane:
                    by_plane[plane] = {"dice": [], "positive_dice": []}
                by_plane[plane]["dice"].append(float(dice_value))
                if positive:
                    by_plane[plane]["positive_dice"].append(float(dice_value))
    by_plane_summary = {
        plane: {
            "dice": float(np.mean(values["dice"])) if values["dice"] else 0.0,
            "positive_dice": float(np.mean(values["positive_dice"])) if values["positive_dice"] else 0.0,
            "num_slices": len(values["dice"]),
            "num_positive_slices": len(values["positive_dice"]),
        }
        for plane, values in sorted(by_plane.items())
    }
    return {
        "dice": float(np.mean(dice_values)) if dice_values else 0.0,
        "positive_dice": float(np.mean(positive_dice)) if positive_dice else 0.0,
        "num_slices": len(dice_values),
        "num_positive_slices": len(positive_dice),
        "by_plane": by_plane_summary,
    }


def make_sampler(records):
    positives, negatives = count_pos_neg(records)
    if not BALANCED_SAMPLING or positives == 0 or negatives == 0:
        return None

    if BALANCE_BY_PLANE:
        bucket_counts = {}
        for record in records:
            bucket = (record.get("plane", "z"), int(record["has_nodule"]))
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        weights = [1.0 / (len(bucket_counts) * bucket_counts[(record.get("plane", "z"), int(record["has_nodule"]))]) for record in records]
        samples_per_epoch = int(round(SAMPLES_PER_EPOCH_MULTIPLIER * max(bucket_counts.values()) * len(bucket_counts)))
    else:
        pos_weight = 0.5 / positives
        neg_weight = 0.5 / negatives
        weights = [pos_weight if record["has_nodule"] else neg_weight for record in records]
        samples_per_epoch = int(round(SAMPLES_PER_EPOCH_MULTIPLIER * max(positives, negatives)))
    samples_per_epoch = max(samples_per_epoch, len(records))
    return WeightedRandomSampler(weights, num_samples=samples_per_epoch, replacement=True)


def save_checkpoint(path, model, epoch, score, config, history):
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "score": float(score),
            "model_name": "slice_propagation_swinunet2d" if USE_SLICE_PROPAGATION else "swinunet2d",
            "config": config,
            "history": history,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train 2D Swin-Unet on generated lung nodule slice masks.")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--force-generate", action="store_true", help="Regenerate 2D image/mask npz files.")
    parser.add_argument(
        "--planes",
        default=PLANE_SPEC,
        help="2D slice directions to train on. Use xyz, x,y,z, axial/coronal/sagittal, or a single plane.",
    )
    parser.add_argument(
        "--context-slices",
        type=int,
        default=CONTEXT_SLICES,
        help="Odd number of adjacent slices used as 2.5D input channels; 1 restores plain 2D.",
    )
    args = parser.parse_args()

    set_seed(SPLIT_SEED)
    planes = parse_plane_spec(args.planes)
    context_slices = normalize_context_slices(args.context_slices)
    resolved_window_size = resolve_window_size(IMG_SIZE)
    manifest = generate_2d_slices(
        args.data_dir,
        force=args.force_generate,
        plane_spec=args.planes,
        context_slices=context_slices,
    )
    records = manifest["records"]
    train_records, val_records, test_records, train_cases, val_cases, test_cases = split_records_by_shared_3d_split(
        records,
        args.data_dir,
    )
    if not train_records or not val_records:
        raise RuntimeError("2D train/val split is empty; check generated masks and shared 3D split.")
    train_pos, train_neg = count_pos_neg(train_records)
    val_pos, val_neg = count_pos_neg(val_records)
    test_pos, test_neg = count_pos_neg(test_records)
    train_plane_counts = count_pos_neg_by_plane(train_records)
    val_plane_counts = count_pos_neg_by_plane(val_records)
    test_plane_counts = count_pos_neg_by_plane(test_records)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, amp_enabled, scaler_enabled = choose_amp_dtype(device)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

    train_sampler = make_sampler(train_records)
    train_loader = DataLoader(
        SliceDataset(train_records, augment=True),
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        SliceDataset(val_records, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )

    model = build_model(device, context_slices=context_slices, planes=planes)
    pretrained_info = load_pretrained_if_available(model, PRETRAINED_PATH, device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(MAX_EPOCHS, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    loss_fn = DiceCrossEntropyLoss()

    config = {
        "run_id": RUN_ID,
        "run_dir": RUN_DIR,
        "planes": list(planes),
        "plane_names": {plane: PLANE_INFO[plane]["name"] for plane in planes},
        "context_slices": context_slices,
        "in_channels": context_slices,
        "img_size": IMG_SIZE,
        "embed_dim": EMBED_DIM,
        "window_size": resolved_window_size,
        "use_slice_propagation": USE_SLICE_PROPAGATION,
        "z_encoder_bias": Z_ENCODER_BIAS,
        "propagation_depth": PROPAGATION_DEPTH,
        "propagation_heads": PROPAGATION_HEADS,
        "propagation_dropout": PROPAGATION_DROPOUT,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "max_train_seconds": MAX_TRAIN_SECONDS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "negative_per_positive": NEGATIVE_PER_POSITIVE,
        "max_positive_slices_per_case": MAX_POSITIVE_SLICES_PER_CASE,
        "balanced_sampling": BALANCED_SAMPLING,
        "balance_by_plane": BALANCE_BY_PLANE,
        "samples_per_epoch_multiplier": SAMPLES_PER_EPOCH_MULTIPLIER,
        "amp_mode": AMP_MODE,
        "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "fp32",
        "scaler_enabled": scaler_enabled,
        "early_stopping": EARLY_STOPPING,
        "train_cases": train_cases,
        "val_cases": val_cases,
        "test_cases": test_cases,
        "train_slices": len(train_records),
        "train_positive_slices": train_pos,
        "train_negative_slices": train_neg,
        "train_plane_counts": train_plane_counts,
        "val_slices": len(val_records),
        "val_positive_slices": val_pos,
        "val_negative_slices": val_neg,
        "val_plane_counts": val_plane_counts,
        "test_slices": len(test_records),
        "test_positive_slices": test_pos,
        "test_negative_slices": test_neg,
        "test_plane_counts": test_plane_counts,
        "split_source": "train.py shared 3D split",
        "pretrained_path": PRETRAINED_PATH,
        "load_pretrained": LOAD_PRETRAINED,
        "pretrained_info": pretrained_info,
    }
    print(
        f"[INFO] 2D Swin-Unet | planes={list(planes)} | train_slices={len(train_records)} "
        f"(pos={train_pos}, neg={train_neg}) | val_slices={len(val_records)} "
        f"(pos={val_pos}, neg={val_neg}) | test_slices={len(test_records)} "
        f"(pos={test_pos}, neg={test_neg}) | train_cases={train_cases} | "
        f"val_cases={val_cases} | test_cases={test_cases} | "
        f"context_slices={context_slices} | "
        f"balanced_sampling={BALANCED_SAMPLING} | balance_by_plane={BALANCE_BY_PLANE} | "
        f"samples_per_epoch={len(train_loader.dataset) if train_sampler is None else train_sampler.num_samples} | "
        f"window={resolved_window_size} | amp={AMP_MODE}/{config['amp_dtype']} | "
        f"slice_propagation={USE_SLICE_PROPAGATION} | z_encoder_bias={Z_ENCODER_BIAS:.2f} | "
        f"early_stopping={EARLY_STOPPING} | device={device} | max_time={format_time_budget(MAX_TRAIN_SECONDS)}"
    )
    print(
        f"[INFO] plane counts | train={format_plane_counts(train_plane_counts)} | "
        f"val={format_plane_counts(val_plane_counts)} | test={format_plane_counts(test_plane_counts)}"
    )

    os.makedirs(RUN_DIR, exist_ok=True)
    history = {"config": config, "epochs": []}
    best_score = -1.0
    best_epoch = -1
    epoch = -1
    start_time = time.time()

    for epoch in range(MAX_EPOCHS):
        if MAX_TRAIN_SECONDS > 0 and time.time() - start_time > MAX_TRAIN_SECONDS:
            print(f"[INFO] stopping because the configured training budget was reached: {MAX_TRAIN_SECONDS}s")
            break

        model.train()
        total_loss = 0.0
        total_dice_loss = 0.0
        total_ce = 0.0
        progress = tqdm(train_loader, desc=f"2D Swin-Unet Epoch {epoch + 1}/{MAX_EPOCHS}")
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            planes_batch = batch.get("plane", ["z"] * int(images.shape[0]))
            sample_weight = plane_sample_weights(planes_batch, device) if USE_SLICE_PROPAGATION else None
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda",
                enabled=amp_enabled,
                dtype=amp_dtype if amp_enabled else None,
            ):
                logits = model_forward(model, images, planes=planes_batch)
                loss, loss_info = loss_fn(logits, masks, sample_weight=sample_weight)

            if scaler_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.detach().item())
            total_dice_loss += loss_info["dice_loss"]
            total_ce += loss_info["ce"]
            progress.set_postfix(loss=f"{float(loss.detach().item()):.4f}")

        scheduler.step()
        train_batches = max(len(train_loader), 1)
        val_info = evaluate(model, val_loader, device, amp_enabled, amp_dtype)
        elapsed = int(time.time() - start_time)
        row = {
            "epoch": epoch,
            "loss": total_loss / train_batches,
            "dice_loss": total_dice_loss / train_batches,
            "ce": total_ce / train_batches,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": elapsed,
            **val_info,
        }
        history["epochs"].append(row)
        plane_summary = " | ".join(
            f"{plane}:Dice={metrics['dice']:.4f},Pos={metrics['positive_dice']:.4f}"
            for plane, metrics in row["by_plane"].items()
        )
        print(
            f"Epoch {epoch + 1} | Loss={row['loss']:.4f} | ValDice={row['dice']:.4f} | "
            f"ValPositiveDice={row['positive_dice']:.4f} | {plane_summary} | "
            f"LR={row['lr']:.6f} | elapsed={elapsed}s"
        )

        score = row["positive_dice"] if row["num_positive_slices"] > 0 else row["dice"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(BEST_MODEL_PATH, model, epoch, score, config, history)
            print(f"[INFO] saved best 2D model: {BEST_MODEL_PATH}")

        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        if EARLY_STOPPING and epoch - best_epoch >= PATIENCE:
            print(f"[INFO] early stopping after {PATIENCE} epochs without improvement")
            break

    save_checkpoint(FINAL_MODEL_PATH, model, epoch, best_score, config, history)
    print(f"[INFO] saved final 2D model: {FINAL_MODEL_PATH}")
    print(f"[INFO] best 2D score={best_score:.5f} at epoch={best_epoch + 1}")


if __name__ == "__main__":
    main()
