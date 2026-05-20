import argparse
import os

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


DEFAULT_STAGE1_PATH = os.path.join(SAVE_DIR, "best_stage1.pth")
DEFAULT_OUT_DIR = os.path.join(SAVE_DIR, "late_fusion_visualization")


def find_case(data_dir, case_id=None):
    cases = []
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith("_img.nii.gz"):
            continue
        image_path = os.path.join(data_dir, name)
        label_path = image_path.replace("_img.nii.gz", "_mask.nii.gz")
        if not os.path.exists(label_path):
            continue
        cases.append(
            {
                "id": name.replace("_img.nii.gz", ""),
                "image": image_path,
                "label": label_path,
            }
        )
    if not cases:
        raise RuntimeError(f"No image/mask pairs found in {data_dir}")
    if case_id is None:
        return cases[0]
    for case in cases:
        if case["id"] == case_id:
            return case
    raise RuntimeError(f"Case {case_id} not found in {data_dir}")


def load_2d_model(path, device):
    model = SwinUnet2D(
        img_size=(IMG_SIZE, IMG_SIZE),
        in_channels=1,
        num_classes=2,
        embed_dim=EMBED_DIM,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
    ).to(device)
    payload = torch.load(path, map_location=device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_3d_stage1_model(path, device):
    train.MODEL_NAME = "swinunetr"
    model = train.build_model(device)
    payload = torch.load(path, map_location=device)
    state = train.extract_model_state(payload)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def preprocess_case(case):
    _, eval_transform = train.build_transforms()
    return eval_transform({"image": case["image"], "label": case["label"], "id": case["id"]})


def get_valid_shape(batch, labels):
    valid_shape = batch.get("valid_shape")
    if valid_shape is None:
        return labels.shape[-3:]
    if isinstance(valid_shape, torch.Tensor):
        return [int(x) for x in valid_shape.tolist()]
    if isinstance(valid_shape, np.ndarray):
        return [int(x) for x in valid_shape.tolist()]
    return [int(x) for x in valid_shape]


def infer_2d_volume(model, volume, device, batch_size=16):
    _, _, size_x, size_y, size_z = volume.shape
    slices = volume[0].permute(3, 0, 1, 2)
    resized = F.interpolate(slices, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    probs = []
    with torch.inference_mode():
        for start in range(0, resized.shape[0], batch_size):
            logits = model(resized[start : start + batch_size].to(device))
            prob = torch.softmax(logits, dim=1)[:, 1:2]
            prob = F.interpolate(prob, size=(size_x, size_y), mode="bilinear", align_corners=False)
            probs.append(prob.cpu())
    prob_slices = torch.cat(probs, dim=0)
    return prob_slices[:, 0].permute(1, 2, 0).contiguous()


def save_nifti(path, array):
    nib.save(nib.Nifti1Image(array.astype(np.float32), np.eye(4)), path)


def choose_visual_slice(label, fused):
    label_sum = label.sum(axis=(0, 1))
    if label_sum.max() > 0:
        return int(label_sum.argmax())
    fused_sum = fused.sum(axis=(0, 1))
    return int(fused_sum.argmax()) if fused_sum.max() > 0 else fused.shape[2] // 2


def save_png(path, image, label, pred2d, pred3d, fused):
    z = choose_visual_slice(label, fused)
    base_slice = image[:, :, z]
    panels = [
        ("image", base_slice, None),
        ("label", base_slice, label[:, :, z]),
        ("2D", base_slice, pred2d[:, :, z]),
        ("3D stage1", base_slice, pred3d[:, :, z]),
        ("fused", base_slice, fused[:, :, z]),
    ]
    size_x, size_y = base_slice.shape
    panel_height = 4.0
    panel_width = panel_height * max(size_x / max(size_y, 1), 0.25)
    fig_width = max(panel_width * len(panels), 8.0)
    fig, axes = plt.subplots(1, len(panels), figsize=(fig_width, panel_height))
    for ax, (title, base, overlay) in zip(axes, panels):
        ax.imshow(base.T, cmap="gray", origin="lower", aspect="equal")
        if overlay is not None:
            overlay = overlay.T
            ax.imshow(
                np.ma.masked_where(overlay <= 0, overlay),
                cmap="autumn",
                alpha=0.45,
                origin="lower",
                aspect="equal",
            )
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"axial slice z={z} | shape={size_x}x{size_y}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize 2D Swin-Unet + 3D SwinUNETR stage1 late fusion.")
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--model-2d", default=DEFAULT_2D_MODEL_PATH)
    parser.add_argument("--model-3d", default=DEFAULT_STAGE1_PATH)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--w2d", type=float, default=0.5)
    parser.add_argument("--w3d", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not os.path.exists(args.model_2d):
        raise FileNotFoundError(f"2D model not found: {args.model_2d}")
    if not os.path.exists(args.model_3d):
        raise FileNotFoundError(f"3D stage1 model not found: {args.model_3d}")

    os.makedirs(args.out_dir, exist_ok=True)
    case = find_case(args.data_dir, args.case_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, _ = train.choose_amp_dtype(device)
    amp_enabled = device.type == "cuda" and amp_dtype in (torch.bfloat16, torch.float16)

    model2d = load_2d_model(args.model_2d, device)
    model3d = load_3d_stage1_model(args.model_3d, device)
    batch = preprocess_case(case)
    image = batch["image"].unsqueeze(0).to(device)
    label = batch["label"].unsqueeze(0).long()
    valid_shape = get_valid_shape(batch, label)

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
        prob3d = torch.softmax(logits3d, dim=1)[0, 1].cpu()
        prob2d = infer_2d_volume(model2d, image, device)

    d, h, w = valid_shape
    prob2d = prob2d[:d, :h, :w].numpy()
    prob3d = prob3d[:d, :h, :w].numpy()
    image_np = image[0, 0, :d, :h, :w].detach().cpu().numpy()
    label_np = label[0, 0, :d, :h, :w].numpy().astype(np.uint8)

    fused_prob = (args.w2d * prob2d + args.w3d * prob3d) / max(args.w2d + args.w3d, 1e-8)
    pred2d = (prob2d >= args.threshold).astype(np.uint8)
    pred3d = (prob3d >= args.threshold).astype(np.uint8)
    fused = (fused_prob >= args.threshold).astype(np.uint8)

    prefix = os.path.join(args.out_dir, case["id"])
    save_nifti(prefix + "_prob2d.nii.gz", prob2d)
    save_nifti(prefix + "_prob3d_stage1.nii.gz", prob3d)
    save_nifti(prefix + "_probfused.nii.gz", fused_prob)
    save_nifti(prefix + "_pred2d.nii.gz", pred2d)
    save_nifti(prefix + "_pred3d_stage1.nii.gz", pred3d)
    save_nifti(prefix + "_predfused.nii.gz", fused)
    save_png(prefix + "_comparison.png", image_np, label_np, pred2d, pred3d, fused)
    print(f"[INFO] saved late-fusion outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
