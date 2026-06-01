#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build an image,label CSV for whole-CT visualization.

Example:
    python make_volume_csv.py \
        --input-root /home/lembert/Desktop/BME-4/BME-WYZ/old1/processed_data \
        --output /home/lembert/Desktop/BME-4/BME-WYZ/data_processed/LIDC-IDRI/val_volume.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


SUPPORTED_SUFFIXES = (
    ".nii.gz",
    ".nii",
    ".nrrd",
    ".mha",
    ".mhd",
    ".npy",
    ".npz",
)

LABEL_TOKENS = (
    "mask",
    "label",
    "labels",
    "seg",
    "segmentation",
    "gt",
    "truth",
)

IMAGE_TOKENS = (
    "image",
    "img",
    "ct",
    "volume",
    "scan",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a whole-volume image,label CSV.")
    parser.add_argument("--input-root", required=True, help="Case directory or parent directory to scan.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--relative-to",
        default=None,
        help="Write paths relative to this directory. Defaults to the output CSV parent.",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Write absolute image/label paths instead of relative paths.",
    )
    return parser.parse_args()


def has_supported_suffix(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)


def token_match(path: Path, tokens: Sequence[str]) -> bool:
    text = " ".join(part.lower() for part in path.parts)
    return any(token in text for token in tokens)


def case_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".bt_ready"):
        return name[: -len(".bt_ready")]
    for part in reversed(path.parts):
        if part.endswith(".bt_ready"):
            return part[: -len(".bt_ready")]
        if part.startswith("LIDC-IDRI-"):
            return part
    return path.stem


def find_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root] if has_supported_suffix(root) else []
    return sorted(p for p in root.rglob("*") if p.is_file() and has_supported_suffix(p))


def choose_pair(files: Sequence[Path], case_id: Optional[str] = None) -> Optional[Tuple[Path, Path]]:
    if case_id:
        files = [
            p for p in files
            if p.name.startswith(case_id)
            and not p.name.endswith(".bt_ready")
            and "_dist" not in p.name.lower()
        ]

    labels = [p for p in files if token_match(p, LABEL_TOKENS)]
    images = [p for p in files if p not in labels and token_match(p, IMAGE_TOKENS)]

    if not images:
        images = [p for p in files if p not in labels]

    if not images or not labels:
        return None

    images.sort(key=lambda p: (len(p.name), str(p)))
    labels.sort(key=lambda p: (len(p.name), str(p)))
    return images[0], labels[0]


def discover_case_dirs(input_root: Path) -> List[Path]:
    if input_root.is_file():
        return [input_root]
    if input_root.name.endswith(".bt_ready"):
        return [input_root]

    ready_markers = sorted(p for p in input_root.rglob("*.bt_ready"))
    if ready_markers:
        return ready_markers
    return [input_root]


def format_path(path: Path, *, base: Path, absolute: bool) -> str:
    path = path.resolve()
    if absolute:
        return str(path)
    try:
        return path.relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    base = Path(args.relative_to).expanduser().resolve() if args.relative_to else output.parent

    if not input_root.exists():
        raise FileNotFoundError(f"input root not found: {input_root}")

    rows = []
    skipped = []
    for case_dir in discover_case_dirs(input_root):
        if case_dir.name.endswith(".bt_ready"):
            case_id = case_id_from_path(case_dir)
            search_root = case_dir if case_dir.is_dir() else case_dir.parent
        else:
            case_id = case_id_from_path(case_dir)
            search_root = case_dir

        pair = choose_pair(find_files(search_root), case_id=case_id)
        if pair is None:
            skipped.append(str(case_dir))
            continue

        image, label = pair
        rows.append(
            {
                "image": format_path(image, base=base, absolute=args.absolute),
                "label": format_path(label, base=base, absolute=args.absolute),
                "case_id": case_id,
            }
        )

    if not rows:
        raise RuntimeError(
            "No image/label pairs found. Check file names contain image-like "
            "tokens such as image/ct/volume and label-like tokens such as mask/label/seg."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label", "case_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] wrote {len(rows)} rows: {output}")
    if skipped:
        print(f"[WARN] skipped {len(skipped)} case dirs without image/label pair")
        for item in skipped[:20]:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
