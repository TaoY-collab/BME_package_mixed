import argparse
import csv
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np

from preprocess_lidc import (
    get_all_scans,
    get_patient_id,
    get_scan_identifier,
    import_pylidc,
    make_consensus_mask_hwd,
    prepare_pylidc_config,
)
from project_config import DATA_DIR, RAW_ROOT


HU_MIN = -1000.0
HU_MAX = 400.0
# Keep CT volumes in HU. Training and visualization apply the HU window.
NORMALIZE_IMAGE = False


def safe_name(value: str) -> str:
    return str(value).replace("/", "_").replace("\\", "_").replace(":", "_")


def _dicom_z_position(ds: Any) -> float:
    position = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError(f"Invalid ImagePositionPatient: {position}")
    return float(position[2])


def _dicom_instance_number(ds: Any) -> float:
    try:
        return float(getattr(ds, "InstanceNumber"))
    except (TypeError, ValueError, AttributeError):
        return float("inf")


def sort_dicom_slices_like_pylidc(dicom_images: Sequence[Any]) -> List[Any]:
    """
    Mirror pylidc 0.2.3 ordering: one slice per z position, then ascending z.

    The consensus annotation bbox indexes the same z-ordered volume. Using a
    different orientation-based ordering here could silently misalign CT and
    masks, so this intentionally follows pylidc's Scan implementation.
    """
    selected_by_z: Dict[float, Any] = {}
    for ds in dicom_images:
        z_position = _dicom_z_position(ds)
        current = selected_by_z.get(z_position)
        if current is None or _dicom_instance_number(ds) < _dicom_instance_number(
            current
        ):
            selected_by_z[z_position] = ds
    return [selected_by_z[z] for z in sorted(selected_by_z)]


def validate_pylidc_slice_alignment(
    scan: Any,
    dicom_images: Sequence[Any],
    atol: float = 1e-3,
) -> None:
    if not dicom_images:
        raise ValueError("Cannot validate an empty DICOM series.")

    actual_z = np.asarray(
        [_dicom_z_position(ds) for ds in dicom_images],
        dtype=np.float64,
    )
    if actual_z.size > 1 and not np.all(np.diff(actual_z) > 0):
        raise ValueError(f"DICOM z positions are not strictly increasing: {actual_z}")

    expected_z = getattr(scan, "slice_zvals", None)
    if expected_z is not None:
        expected_z = np.asarray(expected_z, dtype=np.float64)
        if expected_z.shape != actual_z.shape or not np.allclose(
            expected_z,
            actual_z,
            rtol=0.0,
            atol=float(atol),
        ):
            raise ValueError(
                "pylidc slice order mismatch: "
                f"scan.slice_zvals={expected_z.tolist()}, "
                f"dicom_z={actual_z.tolist()}"
            )


def _validate_in_plane_geometry(dicom_images: Sequence[Any]) -> None:
    first = dicom_images[0]
    first_orientation = np.asarray(
        first.ImageOrientationPatient,
        dtype=np.float64,
    )
    first_spacing = np.asarray(first.PixelSpacing, dtype=np.float64)
    if first_orientation.shape != (6,) or not np.isfinite(first_orientation).all():
        raise ValueError(f"Invalid ImageOrientationPatient: {first_orientation}")
    if first_spacing.shape != (2,) or not np.all(
        np.isfinite(first_spacing) & (first_spacing > 0)
    ):
        raise ValueError(f"Invalid PixelSpacing: {first_spacing}")

    row_cosine = first_orientation[:3]
    col_cosine = first_orientation[3:]
    if not np.isclose(np.linalg.norm(row_cosine), 1.0, atol=1e-3):
        raise ValueError(f"Invalid row direction cosine: {row_cosine}")
    if not np.isclose(np.linalg.norm(col_cosine), 1.0, atol=1e-3):
        raise ValueError(f"Invalid column direction cosine: {col_cosine}")
    if not np.isclose(np.dot(row_cosine, col_cosine), 0.0, atol=1e-3):
        raise ValueError("DICOM row and column direction cosines are not orthogonal.")

    for index, ds in enumerate(dicom_images[1:], start=1):
        orientation = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
        spacing = np.asarray(ds.PixelSpacing, dtype=np.float64)
        if not np.allclose(orientation, first_orientation, rtol=0.0, atol=1e-4):
            raise ValueError(f"ImageOrientationPatient changes at slice {index}.")
        if not np.allclose(spacing, first_spacing, rtol=0.0, atol=1e-4):
            raise ValueError(f"PixelSpacing changes at slice {index}.")


def build_affine(dicom_images: Sequence[Any]) -> np.ndarray:
    if not dicom_images:
        raise ValueError("Cannot build affine from an empty DICOM series.")

    _validate_in_plane_geometry(dicom_images)
    first = dicom_images[0]
    orientation = np.asarray(first.ImageOrientationPatient, dtype=np.float64)
    row_cosine = orientation[:3]
    col_cosine = orientation[3:]
    pixel_spacing = np.asarray(first.PixelSpacing, dtype=np.float64)
    row_spacing = float(pixel_spacing[0])
    col_spacing = float(pixel_spacing[1])
    origin = np.asarray(first.ImagePositionPatient, dtype=np.float64)

    if len(dicom_images) > 1:
        positions = np.stack(
            [
                np.asarray(ds.ImagePositionPatient, dtype=np.float64)
                for ds in dicom_images
            ],
            axis=0,
        )
        deltas = np.diff(positions, axis=0)
        slice_vector = np.median(deltas, axis=0)
        if not np.isfinite(slice_vector).all() or np.linalg.norm(slice_vector) <= 0:
            raise ValueError(f"Invalid slice displacement vector: {slice_vector}")
        expected_positions = origin[None, :] + np.arange(
            len(dicom_images),
            dtype=np.float64,
        )[:, None] * slice_vector[None, :]
        max_position_error = float(
            np.max(np.linalg.norm(positions - expected_positions, axis=1))
        )
        tolerance = max(1e-3, float(np.linalg.norm(slice_vector)) * 1e-3)
        if max_position_error > tolerance:
            raise ValueError(
                "Irregular slice positions cannot be represented by one NIfTI "
                f"affine: max_error={max_position_error:.6f} mm, "
                f"tolerance={tolerance:.6f} mm"
            )

        slice_normal = np.cross(row_cosine, col_cosine)
        alignment = abs(
            float(
                np.dot(
                    slice_vector / np.linalg.norm(slice_vector),
                    slice_normal / np.linalg.norm(slice_normal),
                )
            )
        )
        if alignment < 0.99:
            raise ValueError(
                f"Slice displacement is inconsistent with orientation: {alignment:.6f}"
            )
    else:
        slice_normal = np.cross(row_cosine, col_cosine)
        slice_spacing = float(getattr(first, "SliceThickness", 1.0))
        if not np.isfinite(slice_spacing) or slice_spacing <= 0:
            raise ValueError(f"Invalid SliceThickness: {slice_spacing}")
        slice_vector = slice_normal * slice_spacing

    affine = np.eye(4, dtype=np.float64)
    # Array axes are (row, column, slice).
    affine[:3, 0] = col_cosine * row_spacing
    affine[:3, 1] = row_cosine * col_spacing
    affine[:3, 2] = slice_vector
    affine[:3, 3] = origin
    if not np.isfinite(affine).all() or abs(float(np.linalg.det(affine[:3, :3]))) <= 0:
        raise ValueError(f"Invalid NIfTI affine:\n{affine}")
    return affine


def load_pylidc_scan(scan: Any) -> Tuple[np.ndarray, np.ndarray]:
    dicom_images = scan.load_all_dicom_images(verbose=False)
    if not dicom_images:
        raise RuntimeError("pylidc returned no DICOM images for this scan.")
    dicom_images = sort_dicom_slices_like_pylidc(dicom_images)
    validate_pylidc_slice_alignment(scan, dicom_images)

    slices = []
    for ds in dicom_images:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        slices.append(ds.pixel_array.astype(np.float32) * slope + intercept)

    image_hu = np.stack(slices, axis=-1).astype(np.float32)
    affine = build_affine(dicom_images)
    if not np.isfinite(image_hu).all():
        raise ValueError("CT volume contains NaN or Inf.")
    return image_hu, affine


def normalize_hu(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, HU_MIN, HU_MAX)
    return ((image - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def create_pylidc_mask(
    scan: Any,
    image_shape: Sequence[int],
    consensus_fn: Any,
    min_annotations: int,
) -> Tuple[np.ndarray, int]:
    """Merge pylidc multi-reader consensus masks into one scan-level mask."""
    try:
        clusters = scan.cluster_annotations(verbose=False)
    except TypeError:
        clusters = scan.cluster_annotations()

    mask = np.zeros(tuple(int(v) for v in image_shape), dtype=np.uint8)
    accepted_nodules = 0
    for nodule_index, cluster in enumerate(clusters, start=1):
        if len(cluster) < int(min_annotations):
            continue
        try:
            nodule_mask = make_consensus_mask_hwd(
                cluster=cluster,
                volume_shape_hwd=image_shape,
                min_annotations=int(min_annotations),
                consensus_fn=consensus_fn,
            )
        except Exception as exc:
            print(
                f"[WARN] consensus failed: patient={get_patient_id(scan)} | "
                f"scan={get_scan_identifier(scan)} | nodule={nodule_index} | {exc}"
            )
            continue
        nodule_mask = np.asarray(nodule_mask)
        if nodule_mask.shape != tuple(int(v) for v in image_shape):
            raise ValueError(
                f"Mask shape mismatch: image_shape={tuple(image_shape)}, "
                f"nodule_mask.shape={nodule_mask.shape}"
            )
        if not np.isfinite(nodule_mask).all():
            raise ValueError(
                f"Nodule mask contains NaN or Inf: nodule={nodule_index}"
            )
        if np.count_nonzero(nodule_mask) == 0:
            print(
                f"[WARN] empty nodule mask: patient={get_patient_id(scan)} | "
                f"scan={get_scan_identifier(scan)} | nodule={nodule_index}"
            )
            continue
        mask = np.logical_or(mask > 0, nodule_mask > 0).astype(np.uint8)
        accepted_nodules += 1

    return mask, accepted_nodules


def save_nifti(data: np.ndarray, affine: np.ndarray, path: Path, dtype: Any) -> None:
    array = data.astype(dtype, copy=False)
    nib.save(nib.Nifti1Image(array, affine), str(path))


def voxel_spacing_from_affine(affine: np.ndarray) -> Tuple[float, float, float]:
    spacing = tuple(
        float(np.linalg.norm(np.asarray(affine, dtype=np.float64)[:3, axis]))
        for axis in range(3)
    )
    if not all(np.isfinite(value) and value > 0 for value in spacing):
        raise ValueError(f"Invalid voxel spacing derived from affine: {spacing}")
    return spacing


def validate_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    affine: np.ndarray,
) -> None:
    if image.shape != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {image.shape} vs {mask.shape}")
    if image.ndim != 3:
        raise ValueError(f"Image and mask must be 3D, got shape={image.shape}")
    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or Inf.")
    if not np.isfinite(mask).all():
        raise ValueError("Mask contains NaN or Inf.")
    unique_mask_values = np.unique(mask)
    if not np.all(np.isin(unique_mask_values, [0, 1])):
        raise ValueError(f"Mask is not binary: values={unique_mask_values.tolist()}")
    voxel_spacing_from_affine(affine)


def validate_saved_nifti_pair(
    image_path: Path,
    label_path: Path,
    expected_shape: Sequence[int],
    expected_affine: np.ndarray,
) -> None:
    image_nii = nib.load(str(image_path))
    label_nii = nib.load(str(label_path))
    expected_shape = tuple(int(value) for value in expected_shape)
    if tuple(image_nii.shape) != expected_shape:
        raise ValueError(
            f"Saved image shape mismatch: {image_nii.shape} vs {expected_shape}"
        )
    if tuple(label_nii.shape) != expected_shape:
        raise ValueError(
            f"Saved mask shape mismatch: {label_nii.shape} vs {expected_shape}"
        )
    if not np.allclose(image_nii.affine, expected_affine, rtol=0.0, atol=1e-5):
        raise ValueError("Saved image affine does not match the source affine.")
    if not np.allclose(label_nii.affine, expected_affine, rtol=0.0, atol=1e-5):
        raise ValueError("Saved mask affine does not match the source affine.")
    if not np.allclose(image_nii.affine, label_nii.affine, rtol=0.0, atol=1e-5):
        raise ValueError("Saved image and mask affines do not match.")
    saved_mask = np.asanyarray(label_nii.dataobj)
    if not np.all(np.isin(np.unique(saved_mask), [0, 1])):
        raise ValueError("Saved mask is not binary.")


def save_nifti_pair(
    image: np.ndarray,
    mask: np.ndarray,
    affine: np.ndarray,
    image_path: Path,
    label_path: Path,
    overwrite: bool,
    verify_after_save: bool,
) -> None:
    existing = [path for path in (image_path, label_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output file already exists; use --overwrite to replace it: "
            + ", ".join(str(path) for path in existing)
        )

    token = uuid.uuid4().hex
    temp_image = image_path.with_name(f".{image_path.stem}.{token}.tmp.nii.gz")
    temp_label = label_path.with_name(f".{label_path.stem}.{token}.tmp.nii.gz")
    try:
        save_nifti(image, affine, temp_image, np.float32)
        save_nifti(mask, affine, temp_label, np.uint8)
        if verify_after_save:
            validate_saved_nifti_pair(
                temp_image,
                temp_label,
                expected_shape=image.shape,
                expected_affine=affine,
            )
        os.replace(temp_image, image_path)
        os.replace(temp_label, label_path)
    finally:
        for temp_path in (temp_image, temp_label):
            if temp_path.exists():
                temp_path.unlink()


def _series_uid_suffix(series_uid: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9]+", "", str(series_uid))
    return compact[-12:] if compact else "unknownseries"


def validate_lidc_dicom_root(dicom_root: Path) -> None:
    """
    Ensure the path matches the directory layout expected by pylidc.

    Converted NIfTI folders are deliberately rejected: pylidc needs original
    DICOM files below direct ``LIDC-IDRI-dddd`` patient directories.
    """
    if not dicom_root.exists() or not dicom_root.is_dir():
        raise FileNotFoundError(f"LIDC DICOM root does not exist: {dicom_root}")

    patient_dirs = sorted(
        path
        for path in dicom_root.glob("LIDC-IDRI-*")
        if path.is_dir()
    )
    if not patient_dirs:
        raise ValueError(
            "The selected root has no LIDC-IDRI patient directories. "
            f"Expected {dicom_root / 'LIDC-IDRI-0001'} (or similar). "
            "Pass the original TCIA DICOM root, not processed_nifti or "
            "converted_nifti."
        )

    has_dicom = any(
        next(patient_dir.rglob("*.dcm"), None) is not None
        for patient_dir in patient_dirs[:10]
    )
    if not has_dicom:
        raise ValueError(
            "No .dcm files were found under the first LIDC-IDRI patient "
            f"directories in {dicom_root}. pylidc cannot operate on NIfTI-only data."
        )


def process_scan(
    scan: Any,
    consensus_fn: Any,
    out_dir: Path,
    min_annotations: int,
    overwrite: bool = False,
    verify_after_save: bool = True,
) -> Optional[Dict[str, Any]]:
    patient_id = get_patient_id(scan)
    scan_id = get_scan_identifier(scan)
    series_uid = str(getattr(scan, "series_instance_uid", scan_id))

    image_hu, affine = load_pylidc_scan(scan)
    mask, nodule_count = create_pylidc_mask(
        scan=scan,
        image_shape=image_hu.shape,
        consensus_fn=consensus_fn,
        min_annotations=min_annotations,
    )
    mask_voxels = int(np.count_nonzero(mask))
    if nodule_count == 0 or mask_voxels == 0:
        print(
            f"[WARN] no qualifying pylidc consensus mask: "
            f"patient={patient_id} | scan={scan_id}"
        )
        return None

    image = normalize_hu(image_hu) if NORMALIZE_IMAGE else image_hu
    validate_image_and_mask(image, mask, affine)
    spacing = voxel_spacing_from_affine(affine)
    case_id = safe_name(
        f"{patient_id}_{scan_id}_{_series_uid_suffix(series_uid)}"
    )
    image_name = f"{case_id}_img.nii.gz"
    label_name = f"{case_id}_mask.nii.gz"
    save_nifti_pair(
        image=image,
        mask=mask,
        affine=affine,
        image_path=out_dir / image_name,
        label_path=out_dir / label_name,
        overwrite=overwrite,
        verify_after_save=verify_after_save,
    )

    print(
        f"[INFO] saved {case_id} | series={series_uid} | shape={image.shape} | "
        f"spacing={spacing} | nodules={nodule_count} | mask_voxels={mask_voxels}"
    )
    return {
        "image": image_name,
        "label": label_name,
        "patient_id": patient_id,
        "series_uid": series_uid,
        "case_id": case_id,
        "shape": json.dumps([int(value) for value in image.shape]),
        "spacing": json.dumps([float(value) for value in spacing]),
        "mask_voxels": mask_voxels,
        "nodule_count": nodule_count,
    }


def write_manifest(records: Sequence[Dict[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "label",
        "patient_id",
        "series_uid",
        "case_id",
        "shape",
        "spacing",
        "mask_voxels",
        "nodule_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)


def write_failures(records: Sequence[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patient_id",
        "scan_id",
        "series_uid",
        "error_type",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def process(
    root_dir: str,
    out_dir: str,
    manifest_path: Optional[str] = None,
    min_annotations: int = 3,
    max_scans: Optional[int] = None,
    fail_fast: bool = False,
    overwrite: bool = False,
    verify_after_save: bool = True,
) -> List[Dict[str, Any]]:
    if int(min_annotations) <= 0:
        raise ValueError("min_annotations must be greater than zero.")
    if max_scans is not None and int(max_scans) < 0:
        raise ValueError("max_scans must be non-negative.")

    dicom_root = Path(root_dir).expanduser().resolve()
    output_dir = Path(out_dir).expanduser().resolve()

    validate_lidc_dicom_root(dicom_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = prepare_pylidc_config(dicom_root)
    if config_path is not None:
        print(f"[INFO] pylidc DICOM root: {dicom_root}")
        print(f"[INFO] pylidc config: {config_path}")

    pl, consensus_fn = import_pylidc()
    scans = get_all_scans(pl)
    if max_scans is not None:
        scans = scans[: max(int(max_scans), 0)]

    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for index, scan in enumerate(scans, start=1):
        print(
            f"\n[INFO] scan {index}/{len(scans)}: "
            f"{get_patient_id(scan)} / {get_scan_identifier(scan)}"
        )
        try:
            record = process_scan(
                scan=scan,
                consensus_fn=consensus_fn,
                out_dir=output_dir,
                min_annotations=min_annotations,
                overwrite=overwrite,
                verify_after_save=verify_after_save,
            )
        except Exception as exc:
            failure = {
                "patient_id": get_patient_id(scan),
                "scan_id": get_scan_identifier(scan),
                "series_uid": str(
                    getattr(scan, "series_instance_uid", "")
                ),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            print(
                f"[WARN] failed scan {get_patient_id(scan)} / "
                f"{get_scan_identifier(scan)}: {exc}"
            )
            if fail_fast:
                write_failures(failures, output_dir / "failed_cases.csv")
                raise
            continue
        if record is not None:
            records.append(record)

    manifest = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else output_dir / "all_cases.csv"
    )
    write_manifest(records, manifest)
    failure_path = output_dir / "failed_cases.csv"
    write_failures(failures, failure_path)
    print(f"[INFO] manifest: {manifest} | cases={len(records)}")
    print(f"[INFO] failures: {failure_path} | cases={len(failures)}")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paired NIfTI CT/masks from pylidc annotations."
    )
    parser.add_argument("--raw-root", default=RAW_ROOT)
    parser.add_argument("--output-dir", default=DATA_DIR)
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--min-annotations",
        type=int,
        default=3,
        help="Minimum agreeing pylidc reader annotations per nodule.",
    )
    parser.add_argument(
        "--max-scans",
        type=int,
        default=None,
        help="Optional scan limit for a small preprocessing smoke run.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed scan after writing failed_cases.csv.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output NIfTI files.",
    )
    parser.add_argument(
        "--no-verify-after-save",
        dest="verify_after_save",
        action="store_false",
        help="Skip reloading saved NIfTI files for shape/affine checks.",
    )
    parser.set_defaults(verify_after_save=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process(
        os.path.abspath(os.path.expandvars(os.path.expanduser(args.raw_root))),
        os.path.abspath(os.path.expandvars(os.path.expanduser(args.output_dir))),
        None
        if args.manifest is None
        else os.path.abspath(os.path.expandvars(os.path.expanduser(args.manifest))),
        min_annotations=args.min_annotations,
        max_scans=args.max_scans,
        fail_fast=args.fail_fast,
        overwrite=args.overwrite,
        verify_after_save=args.verify_after_save,
    )
