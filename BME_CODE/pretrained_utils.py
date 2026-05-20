import os
import urllib.request

import torch


MONAI_SSL_PRETRAINED_URL = (
    "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/ssl_pretrained_weights.pth"
)


def download_file(url, dst_path):
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    urllib.request.urlretrieve(url, dst_path)
    return dst_path


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "network_weights"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")


def remap_swinunetr_pretrained_key(key):
    if key.startswith("module."):
        key = key[7:]

    if key in {
        "encoder.mask_token",
        "encoder.norm.weight",
        "encoder.norm.bias",
        "out.conv.conv.weight",
        "out.conv.conv.bias",
    }:
        return None

    if key.startswith("encoder."):
        if key[8:19] == "patch_embed":
            new_key = "swinViT." + key[8:]
        else:
            new_key = "swinViT." + key[8:18] + key[20:]
    elif key.startswith("patch_embed.") or key.startswith("layers") or key.startswith("norm."):
        new_key = "swinViT." + key
    else:
        new_key = key

    new_key = new_key.replace(".fc1.", ".linear1.")
    new_key = new_key.replace(".fc2.", ".linear2.")
    return new_key


def prepare_pretrained_state_dict(checkpoint):
    raw_state = extract_state_dict(checkpoint)
    remapped = {}
    for key, value in raw_state.items():
        new_key = remap_swinunetr_pretrained_key(key)
        if new_key is not None:
            remapped[new_key] = value
    return remapped


def inspect_pretrained_match(model, checkpoint):
    prepared = prepare_pretrained_state_dict(checkpoint)
    model_state = model.state_dict()

    matched = {}
    mismatched_shape = []
    unexpected = []

    for key, value in prepared.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            mismatched_shape.append(key)
            continue
        matched[key] = value

    missing = [key for key in model_state.keys() if key not in matched]
    return {
        "matched_state": matched,
        "matched_keys": list(matched.keys()),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_shape_keys": mismatched_shape,
        "prepared_keys": list(prepared.keys()),
    }


def load_pretrained_if_compatible(model, checkpoint_path, device, min_match=1, verbose=True):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        if verbose:
            print(f"[WARN] pretrained weights not found: {checkpoint_path}")
        return {
            "loaded": False,
            "matched": 0,
            "missing": len(model.state_dict()),
            "unexpected": 0,
            "mismatched_shape": 0,
            "path": checkpoint_path,
        }

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    target_model = model.pretrained_target() if hasattr(model, "pretrained_target") else model
    info = inspect_pretrained_match(target_model, checkpoint)
    matched_state = info["matched_state"]

    if verbose:
        print(f"[INFO] inspected pretrained weights: {checkpoint_path}")
        print(
            "[INFO] pretrained compatibility | "
            f"matched={len(info['matched_keys'])} | "
            f"missing={len(info['missing_keys'])} | "
            f"unexpected={len(info['unexpected_keys'])} | "
            f"mismatched_shape={len(info['mismatched_shape_keys'])}"
        )

    if len(matched_state) < min_match:
        if verbose:
            print(
                "[WARN] pretrained weights rejected because compatible parameter count "
                f"is below min_match={min_match}"
            )
        return {
            "loaded": False,
            "matched": len(info["matched_keys"]),
            "missing": len(info["missing_keys"]),
            "unexpected": len(info["unexpected_keys"]),
            "mismatched_shape": len(info["mismatched_shape_keys"]),
            "path": checkpoint_path,
            "sample_missing": info["missing_keys"][:10],
            "sample_unexpected": info["unexpected_keys"][:10],
            "sample_mismatched_shape": info["mismatched_shape_keys"][:10],
        }

    model_state = target_model.state_dict()
    model_state.update(matched_state)
    target_model.load_state_dict(model_state, strict=False)

    if verbose:
        print(f"[INFO] loaded compatible pretrained parameters: {len(matched_state)}")

    return {
        "loaded": True,
        "matched": len(info["matched_keys"]),
        "missing": len(info["missing_keys"]),
        "unexpected": len(info["unexpected_keys"]),
        "mismatched_shape": len(info["mismatched_shape_keys"]),
        "path": checkpoint_path,
        "sample_missing": info["missing_keys"][:10],
        "sample_unexpected": info["unexpected_keys"][:10],
        "sample_mismatched_shape": info["mismatched_shape_keys"][:10],
    }
