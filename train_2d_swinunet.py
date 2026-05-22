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

from dual_swin_fusion import SwinUnet2D
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
BATCH_SIZE = int(os.environ.get("BME_2D_BATCH_SIZE", "8"))
MAX_EPOCHS = int(os.environ.get("BME_2D_MAX_EPOCHS", "60"))
MAX_TRAIN_SECONDS = int(os.environ.get("BME_2D_MAX_TRAIN_SECONDS", str(4 * 60 * 60)))
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


def generate_2d_slices(data_dir, force=False):
    os.makedirs(SLICE_DIR, exist_ok=True)
    os.makedirs(RUN_DIR, exist_ok=True)
    if os.path.exists(MANIFEST_PATH) and not force:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    cases = find_cases(data_dir)
    rng = np.random.default_rng(SPLIT_SEED)
    records = []
    for case in tqdm(cases, desc="Generating 2D image/mask slices"):
        image = nib.load(case["image"]).get_fdata().astype(np.float32)
        mask = (nib.load(case["label"]).get_fdata() > 0).astype(np.uint8)
        image = np.moveaxis(image, -1, 0)
        mask = np.moveaxis(mask, -1, 0)

        positive_slices, negative_slices = choose_slices(mask, rng)
        selected = [(z, 1) for z in positive_slices] + [(z, 0) for z in negative_slices]

        for z, has_nodule in selected:
            image_slice = normalize_hu(image[z])
            mask_slice = mask[z].astype(np.uint8)
            image_slice = resize_2d(image_slice, IMG_SIZE, order=1).astype(np.float32)
            mask_slice = (resize_2d(mask_slice, IMG_SIZE, order=0) > 0).astype(np.uint8)

            out_name = f"{case['id']}_z{z:04d}.npz"
            out_path = os.path.join(SLICE_DIR, out_name)
            np.savez_compressed(
                out_path,
                image=image_slice[None],
                mask=mask_slice[None],
                case_id=case["id"],
                slice_index=np.asarray(z, dtype=np.int32),
                has_nodule=np.asarray(has_nodule, dtype=np.uint8),
            )
            records.append(
                {
                    "path": out_path,
                    "case_id": case["id"],
                    "slice_index": int(z),
                    "has_nodule": int(has_nodule),
                    "mask_pixels": int(mask_slice.sum()),
                }
            )

    manifest = {
        "config": {
            "img_size": IMG_SIZE,
            "negative_per_positive": NEGATIVE_PER_POSITIVE,
            "max_positive_slices_per_case": MAX_POSITIVE_SLICES_PER_CASE,
            "hu_min": HU_MIN,
            "hu_max": HU_MAX,
            "data_dir": data_dir,
        },
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

        return {"image": image, "mask": mask, "has_nodule": int(record["has_nodule"])}


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, foreground_weight=3.0):
        super().__init__()
        self.register_buffer("ce_weight", torch.tensor([1.0, foreground_weight], dtype=torch.float32))

    def forward(self, logits, mask):
        target = mask.squeeze(1).long()
        ce = F.cross_entropy(logits, target, weight=self.ce_weight.to(logits.device))
        probs = torch.softmax(logits, dim=1)[:, 1]
        target_fg = (target > 0).float()
        reduce_dims = tuple(range(1, probs.ndim))
        intersection = torch.sum(probs * target_fg, dim=reduce_dims)
        denominator = torch.sum(probs, dim=reduce_dims) + torch.sum(target_fg, dim=reduce_dims)
        dice = 1.0 - torch.mean((2.0 * intersection + 1e-6) / (denominator + 1e-6))
        return dice + ce, {"dice_loss": float(dice.detach().item()), "ce": float(ce.detach().item())}


def build_model(device):
    return SwinUnet2D(
        img_size=(IMG_SIZE, IMG_SIZE),
        in_channels=1,
        num_classes=2,
        embed_dim=EMBED_DIM,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
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
        if key not in model_state:
            unexpected.append(raw_key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            mismatched.append(key)
            continue
        matched[key] = value

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


def dice_from_logits(logits, mask, threshold=PRED_THRESHOLD):
    probs = torch.softmax(logits, dim=1)[:, 1]
    pred = probs >= threshold
    target = mask.squeeze(1) > 0
    intersection = (pred & target).sum(dim=(1, 2)).float()
    denom = pred.sum(dim=(1, 2)).float() + target.sum(dim=(1, 2)).float()
    dice = torch.where(denom > 0, (2.0 * intersection) / torch.clamp(denom, min=1.0), torch.ones_like(denom))
    return dice


def evaluate(model, loader, device, amp_enabled):
    model.eval()
    dice_values = []
    positive_dice = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images)
            dice = dice_from_logits(logits, masks)
            dice_values.extend(dice.detach().cpu().tolist())
            has_nodule = batch["has_nodule"].bool()
            if has_nodule.any():
                positive_dice.extend(dice.detach().cpu()[has_nodule].tolist())
    return {
        "dice": float(np.mean(dice_values)) if dice_values else 0.0,
        "positive_dice": float(np.mean(positive_dice)) if positive_dice else 0.0,
        "num_slices": len(dice_values),
        "num_positive_slices": len(positive_dice),
    }


def make_sampler(records):
    positives, negatives = count_pos_neg(records)
    if not BALANCED_SAMPLING or positives == 0 or negatives == 0:
        return None

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
            "model_name": "swinunet2d",
            "config": config,
            "history": history,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train 2D Swin-Unet on generated lung nodule slice masks.")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--force-generate", action="store_true", help="Regenerate 2D image/mask npz files.")
    args = parser.parse_args()

    set_seed(SPLIT_SEED)
    manifest = generate_2d_slices(args.data_dir, force=args.force_generate)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

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

    model = build_model(device)
    pretrained_info = load_pretrained_if_available(model, PRETRAINED_PATH, device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(MAX_EPOCHS, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loss_fn = DiceCrossEntropyLoss()

    config = {
        "run_id": RUN_ID,
        "run_dir": RUN_DIR,
        "img_size": IMG_SIZE,
        "embed_dim": EMBED_DIM,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "max_train_seconds": MAX_TRAIN_SECONDS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "negative_per_positive": NEGATIVE_PER_POSITIVE,
        "max_positive_slices_per_case": MAX_POSITIVE_SLICES_PER_CASE,
        "balanced_sampling": BALANCED_SAMPLING,
        "samples_per_epoch_multiplier": SAMPLES_PER_EPOCH_MULTIPLIER,
        "early_stopping": EARLY_STOPPING,
        "train_cases": train_cases,
        "val_cases": val_cases,
        "test_cases": test_cases,
        "train_slices": len(train_records),
        "train_positive_slices": train_pos,
        "train_negative_slices": train_neg,
        "val_slices": len(val_records),
        "val_positive_slices": val_pos,
        "val_negative_slices": val_neg,
        "test_slices": len(test_records),
        "test_positive_slices": test_pos,
        "test_negative_slices": test_neg,
        "split_source": "train.py shared 3D split",
        "pretrained_path": PRETRAINED_PATH,
        "load_pretrained": LOAD_PRETRAINED,
        "pretrained_info": pretrained_info,
    }
    print(
        f"[INFO] 2D Swin-Unet | train_slices={len(train_records)} "
        f"(pos={train_pos}, neg={train_neg}) | val_slices={len(val_records)} "
        f"(pos={val_pos}, neg={val_neg}) | test_slices={len(test_records)} "
        f"(pos={test_pos}, neg={test_neg}) | train_cases={train_cases} | "
        f"val_cases={val_cases} | test_cases={test_cases} | "
        f"balanced_sampling={BALANCED_SAMPLING} | samples_per_epoch={len(train_loader.dataset) if train_sampler is None else train_sampler.num_samples} | "
        f"early_stopping={EARLY_STOPPING} | device={device} | max_time={MAX_TRAIN_SECONDS}s"
    )

    os.makedirs(RUN_DIR, exist_ok=True)
    history = {"config": config, "epochs": []}
    best_score = -1.0
    best_epoch = -1
    start_time = time.time()

    for epoch in range(MAX_EPOCHS):
        if time.time() - start_time > MAX_TRAIN_SECONDS:
            print("[INFO] stopping because the 4-hour training budget was reached")
            break

        model.train()
        total_loss = 0.0
        total_dice_loss = 0.0
        total_ce = 0.0
        progress = tqdm(train_loader, desc=f"2D Swin-Unet Epoch {epoch + 1}/{MAX_EPOCHS}")
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images)
                loss, loss_info = loss_fn(logits, masks)

            if amp_enabled:
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
        val_info = evaluate(model, val_loader, device, amp_enabled)
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
        print(
            f"Epoch {epoch + 1} | Loss={row['loss']:.4f} | ValDice={row['dice']:.4f} | "
            f"ValPositiveDice={row['positive_dice']:.4f} | LR={row['lr']:.6f} | elapsed={elapsed}s"
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
