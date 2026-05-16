import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from project_config import DATA_DIR


def find_mask_files(data_dir):
    data_path = Path(data_dir)
    return sorted(data_path.glob("*_mask.nii.gz"))


def check_mask(mask_path):
    mask = nib.load(str(mask_path)).get_fdata()
    foreground_voxels = int(np.count_nonzero(mask > 0))
    return {
        "path": mask_path,
        "shape": tuple(int(x) for x in mask.shape),
        "foreground_voxels": foreground_voxels,
        "is_empty": foreground_voxels == 0,
    }


def paired_image_path(mask_path):
    image_name = mask_path.name.replace("_mask.nii.gz", "_img.nii.gz")
    return mask_path.with_name(image_name)


def summarize(results):
    voxel_counts = [item["foreground_voxels"] for item in results]
    if not voxel_counts:
        return None

    return {
        "min": int(np.min(voxel_counts)),
        "median": float(np.median(voxel_counts)),
        "max": int(np.max(voxel_counts)),
        "mean": float(np.mean(voxel_counts)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check whether NIfTI mask files are empty."
    )
    parser.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help=f"Directory containing *_mask.nii.gz files. Default: {DATA_DIR}",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Print every mask, not only empty or problematic masks.",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        raise RuntimeError(f"Data directory not found: {data_dir}")

    mask_files = find_mask_files(data_dir)
    if not mask_files:
        raise RuntimeError(f"No *_mask.nii.gz files found in: {data_dir}")

    results = []
    failed = []
    missing_images = []

    for mask_path in mask_files:
        image_path = paired_image_path(mask_path)
        if not image_path.exists():
            missing_images.append(mask_path)

        try:
            result = check_mask(mask_path)
            results.append(result)
        except Exception as exc:
            failed.append((mask_path, exc))

    empty_masks = [item for item in results if item["is_empty"]]
    non_empty_masks = [item for item in results if not item["is_empty"]]
    stats = summarize(results)

    print(f"[INFO] data_dir: {data_dir}")
    print(f"[INFO] masks found: {len(mask_files)}")
    print(f"[INFO] masks checked: {len(results)}")
    print(f"[INFO] non-empty masks: {len(non_empty_masks)}")
    print(f"[INFO] empty masks: {len(empty_masks)}")
    print(f"[INFO] failed reads: {len(failed)}")
    print(f"[INFO] missing paired images: {len(missing_images)}")

    if stats is not None:
        print(
            "[INFO] foreground voxels | "
            f"min={stats['min']} | median={stats['median']:.1f} | "
            f"mean={stats['mean']:.1f} | max={stats['max']}"
        )

    if args.show_all:
        print("\n[DETAIL] all masks:")
        for item in results:
            status = "EMPTY" if item["is_empty"] else "OK"
            print(
                f"[{status}] {item['path'].name} | "
                f"shape={item['shape']} | voxels={item['foreground_voxels']}"
            )

    if empty_masks:
        print("\n[WARN] empty masks:")
        for item in empty_masks:
            print(f"[EMPTY] {item['path']}")

    if missing_images:
        print("\n[WARN] masks with missing paired image:")
        for mask_path in missing_images:
            print(f"[MISSING_IMG] {mask_path}")

    if failed:
        print("\n[WARN] masks failed to read:")
        for mask_path, exc in failed:
            print(f"[FAILED] {mask_path} | {exc}")

    if empty_masks or failed or missing_images:
        raise SystemExit(1)

    print("\n[INFO] all masks are non-empty and readable.")


if __name__ == "__main__":
    main()
