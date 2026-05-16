import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from scipy import ndimage as ndi
from skimage.draw import polygon

from project_config import BASE_DIR, DATA_DIR


RAW_ROOT = BASE_DIR
OUT_DIR = os.environ.get("BME_C3_DATA_DIR", os.path.join(BASE_DIR, "data_stage2_c3"))

HU_MIN = -1000.0
HU_MAX = 400.0
NORMALIZE_IMAGE = False

SMALL_NODULE_DIAMETER_MM = 6.0
LARGE_NODULE_DIAMETER_MM = 8.0
DUPLICATE_SLICE_TOL_MM = 1e-3
SLICE_MATCH_TOL_MM = 1.0
PATIENT_DIR_PREFIX = "LIDC-IDRI-"

LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"ns": "http://www.nih.gov"}

    series_uid = None
    series_uid_elem = root.find(".//ns:SeriesInstanceUid", ns)
    if series_uid_elem is None:
        series_uid_elem = root.find(".//ns:SeriesInstanceUID", ns)
    if series_uid_elem is not None and series_uid_elem.text:
        series_uid = series_uid_elem.text.strip()

    nodules = []
    for nodule in root.findall(".//ns:unblindedReadNodule", ns):
        roi_list = []
        for roi in nodule.findall(".//ns:roi", ns):
            z_elem = roi.find("ns:imageZposition", ns)
            if z_elem is None or z_elem.text is None:
                continue

            xs, ys = [], []
            for edge in roi.findall("ns:edgeMap", ns):
                x = edge.find("ns:xCoord", ns)
                y = edge.find("ns:yCoord", ns)
                if x is not None and y is not None and x.text is not None and y.text is not None:
                    xs.append(float(x.text))
                    ys.append(float(y.text))

            if len(xs) >= 3:
                roi_list.append(
                    {
                        "z": float(z_elem.text),
                        "x": np.asarray(xs, dtype=np.float32),
                        "y": np.asarray(ys, dtype=np.float32),
                    }
                )
        if roi_list:
            nodules.append(roi_list)

    small_points = []
    for small_tag in ("nonNodule", "smallNodule"):
        for item in root.findall(f".//ns:{small_tag}", ns):
            z_elem = item.find("ns:imageZposition", ns)
            x_elem = item.find("ns:locus/ns:xCoord", ns)
            y_elem = item.find("ns:locus/ns:yCoord", ns)
            if x_elem is None:
                x_elem = item.find(".//ns:xCoord", ns)
            if y_elem is None:
                y_elem = item.find(".//ns:yCoord", ns)
            if (
                z_elem is not None
                and x_elem is not None
                and y_elem is not None
                and z_elem.text is not None
                and x_elem.text is not None
                and y_elem.text is not None
            ):
                small_points.append(
                    {
                        "z": float(z_elem.text),
                        "x": float(x_elem.text),
                        "y": float(y_elem.text),
                        "source": small_tag,
                    }
                )

    return {"series_uid": series_uid, "nodules": nodules, "small_points": small_points}


def read_dicom_header(path):
    try:
        return pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception:
        return None


def collect_ct_dicoms(patient_dir):
    series_map = {}
    for root, _, files in os.walk(patient_dir):
        for file_name in files:
            path = os.path.join(root, file_name)
            ds = read_dicom_header(path)
            if ds is None or getattr(ds, "Modality", None) != "CT":
                continue
            series_uid = getattr(ds, "SeriesInstanceUID", None)
            if not series_uid:
                continue
            series_map.setdefault(str(series_uid), []).append(path)
    return series_map


def choose_series_uid(series_map, target_uid=None):
    if target_uid and target_uid in series_map:
        return target_uid
    if target_uid:
        print(f"[WARN] XML series UID not found in DICOM tree: {target_uid}")
    if not series_map:
        raise ValueError("No CT DICOM series found")
    return max(series_map.keys(), key=lambda uid: len(series_map[uid]))


def get_orientation_vectors(ds):
    orientation = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
    row_cosine = orientation[:3]
    col_cosine = orientation[3:]
    slice_normal = np.cross(row_cosine, col_cosine)
    slice_normal = slice_normal / max(np.linalg.norm(slice_normal), 1e-8)
    return row_cosine, col_cosine, slice_normal


def get_slice_position(ds, slice_normal):
    ipp = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
    return float(np.dot(ipp, slice_normal))


def sort_and_deduplicate_slices(slice_paths):
    headers = []
    for path in slice_paths:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        if not hasattr(ds, "ImageOrientationPatient") or not hasattr(ds, "ImagePositionPatient"):
            continue
        headers.append((path, ds))
    if not headers:
        raise ValueError("No CT slices with ImageOrientationPatient/ImagePositionPatient")

    _, _, slice_normal = get_orientation_vectors(headers[0][1])

    sortable = []
    for path, ds in headers:
        projection = get_slice_position(ds, slice_normal)
        instance = int(getattr(ds, "InstanceNumber", 0))
        sop_uid = str(getattr(ds, "SOPInstanceUID", ""))
        sortable.append((projection, instance, sop_uid, path))
    sortable.sort(key=lambda row: (row[0], row[1], row[2]))

    kept_paths = []
    kept_positions = []
    duplicate_records = []
    for projection, instance, sop_uid, path in sortable:
        if kept_positions and abs(projection - kept_positions[-1]) <= DUPLICATE_SLICE_TOL_MM:
            duplicate_records.append(
                {
                    "kept": kept_paths[-1],
                    "dropped": path,
                    "projection": projection,
                    "instance_number": instance,
                    "sop_instance_uid": sop_uid,
                }
            )
            continue
        kept_paths.append(path)
        kept_positions.append(projection)

    slices = [pydicom.dcmread(path, force=True) for path in kept_paths]
    return slices, slice_normal, duplicate_records


def infer_slice_spacing(slices, slice_normal):
    if len(slices) < 2:
        return float(getattr(slices[0], "SliceThickness", 1.0))
    positions = np.asarray([get_slice_position(ds, slice_normal) for ds in slices], dtype=np.float64)
    diffs = np.diff(positions)
    diffs = np.abs(diffs[np.abs(diffs) > DUPLICATE_SLICE_TOL_MM])
    if diffs.size == 0:
        return float(getattr(slices[0], "SliceThickness", 1.0))
    return float(np.median(diffs))


def build_lps_affine(slices, slice_normal):
    first = slices[0]
    row_cosine, col_cosine, _ = get_orientation_vectors(first)
    pixel_spacing = np.asarray(first.PixelSpacing, dtype=np.float64)
    row_spacing = float(pixel_spacing[0])
    col_spacing = float(pixel_spacing[1])
    slice_spacing = infer_slice_spacing(slices, slice_normal)
    origin_lps = np.asarray(first.ImagePositionPatient, dtype=np.float64)

    affine_lps = np.eye(4, dtype=np.float64)
    # NumPy data are indexed as [row, col, slice]. DICOM XML points are
    # (xCoord, yCoord), so rasterization uses row=y and col=x.
    affine_lps[:3, 0] = col_cosine * row_spacing
    affine_lps[:3, 1] = row_cosine * col_spacing
    affine_lps[:3, 2] = slice_normal * slice_spacing
    affine_lps[:3, 3] = origin_lps
    return affine_lps


def load_dicom_series(series_paths):
    slices, slice_normal, duplicate_records = sort_and_deduplicate_slices(series_paths)

    slice_images = []
    for ds in slices:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        slice_image = ds.pixel_array.astype(np.float32) * slope + intercept
        slice_images.append(slice_image)

    image_hu = np.stack(slice_images, axis=-1)
    affine_lps = build_lps_affine(slices, slice_normal)
    affine_ras = LPS_TO_RAS @ affine_lps
    return image_hu, affine_ras, affine_lps, slices, slice_normal, duplicate_records


def normalize_hu(image):
    image = np.clip(image, HU_MIN, HU_MAX)
    return ((image - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def find_slice_index_for_xml_z(xml_z, slices):
    z_positions = np.asarray([float(ds.ImagePositionPatient[2]) for ds in slices], dtype=np.float64)
    distances = np.abs(z_positions - float(xml_z))
    idx = int(np.argmin(distances))
    if float(distances[idx]) > SLICE_MATCH_TOL_MM:
        print(
            f"[WARN] XML z did not closely match a kept DICOM slice | "
            f"xml_z={xml_z:.4f} | closest_z={z_positions[idx]:.4f} | diff={distances[idx]:.4f}"
        )
    return idx


def rasterize_nodule_mask(image_shape, nodules, slices):
    mask = np.zeros(image_shape, dtype=np.uint8)
    for nodule in nodules:
        for roi in nodule:
            z_idx = find_slice_index_for_xml_z(roi["z"], slices)

            col = np.round(roi["x"]).astype(np.int64)
            row = np.round(roi["y"]).astype(np.int64)
            valid = (
                (row >= 0)
                & (row < image_shape[0])
                & (col >= 0)
                & (col < image_shape[1])
            )
            row = row[valid]
            col = col[valid]
            if len(row) < 3:
                continue
            rr, cc = polygon(row, col, shape=image_shape[:2])
            mask[rr, cc, z_idx] = 1
    return mask


def rasterize_small_points(image_shape, small_points, slices):
    point_mask = np.zeros(image_shape, dtype=np.uint8)
    records = []
    for point in small_points:
        z_idx = find_slice_index_for_xml_z(point["z"], slices)
        col = int(round(point["x"]))
        row = int(round(point["y"]))
        if 0 <= row < image_shape[0] and 0 <= col < image_shape[1]:
            point_mask[row, col, z_idx] = 1
            records.append({"row": row, "col": col, "slice": z_idx, **point})
    return point_mask, records


def voxel_volume_from_affine(affine):
    spacing = [float(np.linalg.norm(affine[:3, axis])) for axis in range(3)]
    return float(np.prod(spacing)), spacing


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


def split_small_large_masks(mask, affine_ras):
    small_mask = np.zeros(mask.shape, dtype=np.uint8)
    large_mask = np.zeros(mask.shape, dtype=np.uint8)
    labeled, num_components = ndi.label(mask > 0)
    components = []
    voxel_volume_mm3, spacing_mm = voxel_volume_from_affine(affine_ras)

    for component_id in range(1, num_components + 1):
        component = labeled == component_id
        voxels = int(component.sum())
        if voxels <= 0:
            continue
        diameter_mm = equivalent_sphere_diameter_mm(voxels, voxel_volume_mm3)
        size_class = classify_diameter(diameter_mm)
        if size_class == "small":
            size_class = "small"
            small_mask[component] = 1
        elif size_class == "large":
            large_mask[component] = 1
        components.append(
            {
                "component_id": component_id,
                "voxels": voxels,
                "volume_mm3": voxels * voxel_volume_mm3,
                "equivalent_diameter_mm": diameter_mm,
                "size_class": size_class,
            }
        )

    return small_mask, large_mask, components, voxel_volume_mm3, spacing_mm


def save_nifti(data, affine_ras, path, dtype):
    arr = data.astype(dtype, copy=False)
    nii = nib.Nifti1Image(arr, affine_ras)
    nii.set_data_dtype(dtype)
    nii.set_qform(affine_ras, code=1)
    nii.set_sform(affine_ras, code=1)
    nib.save(nii, path)


def save_sidecar(path, metadata):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def process_patient(patient_dir, out_dir):
    patient = os.path.basename(patient_dir)
    series_map = collect_ct_dicoms(patient_dir)

    xml_paths = []
    for root, _, files in os.walk(patient_dir):
        for file_name in files:
            if file_name.lower().endswith(".xml"):
                xml_paths.append(os.path.join(root, file_name))

    if not xml_paths:
        print(f"[WARN] no XML found: {patient}")
        return

    seen_series_uids = {}
    for xml_path in xml_paths:
        parsed = parse_xml(xml_path)
        if not parsed["nodules"] and not parsed["small_points"]:
            continue

        series_uid = choose_series_uid(series_map, parsed["series_uid"])
        seen_series_uids[series_uid] = seen_series_uids.get(series_uid, 0) + 1
        if seen_series_uids[series_uid] > 1:
            print(
                f"[WARN] duplicate XML for same series: patient={patient} | "
                f"series={series_uid} | count={seen_series_uids[series_uid]}"
            )

        image_hu, affine_ras, affine_lps, slices, slice_normal, duplicates = load_dicom_series(series_map[series_uid])
        mask = rasterize_nodule_mask(image_hu.shape, parsed["nodules"], slices)
        small_point_mask, point_records = rasterize_small_points(image_hu.shape, parsed["small_points"], slices)
        small_mask, large_mask, components, voxel_volume_mm3, spacing_mm = split_small_large_masks(mask, affine_ras)

        mask_voxels = int(np.sum(mask > 0))
        if mask_voxels <= 0 and not point_records:
            print(f"[WARN] empty contour and point masks after rasterization: {patient} | {xml_path}")
            continue

        image = normalize_hu(image_hu) if NORMALIZE_IMAGE else image_hu.astype(np.float32)
        name = f"{patient}_{Path(xml_path).stem}"

        save_nifti(image, affine_ras, os.path.join(out_dir, name + "_img.nii.gz"), np.float32)
        save_nifti(mask, affine_ras, os.path.join(out_dir, name + "_mask.nii.gz"), np.uint8)
        save_nifti(small_mask, affine_ras, os.path.join(out_dir, name + "_small_mask.nii.gz"), np.uint8)
        save_nifti(large_mask, affine_ras, os.path.join(out_dir, name + "_large_mask.nii.gz"), np.uint8)
        save_nifti(small_point_mask, affine_ras, os.path.join(out_dir, name + "_lt3mm_points_mask.nii.gz"), np.uint8)

        metadata = {
            "patient": patient,
            "xml_path": xml_path,
            "series_uid": series_uid,
            "shape_row_col_slice": list(image.shape),
            "affine_ras": affine_ras.tolist(),
            "affine_lps_dicom": affine_lps.tolist(),
            "slice_normal_lps": np.asarray(slice_normal, dtype=float).tolist(),
            "duplicate_projected_slices_dropped": duplicates,
            "components": components,
            "lt3mm_points": point_records,
            "voxel_spacing_mm": spacing_mm,
            "voxel_volume_mm3": voxel_volume_mm3,
            "small_nodule_diameter_mm": SMALL_NODULE_DIAMETER_MM,
            "large_nodule_diameter_mm": LARGE_NODULE_DIAMETER_MM,
            "size_rule": "equivalent sphere diameter: small <6 mm, medium 6-8 mm, large >8 mm",
            "notes": [
                "XML coordinates are rasterized as row=yCoord, col=xCoord.",
                "DICOM slices are sorted by ImagePositionPatient projected onto the slice normal.",
                "NIfTI affine is converted from DICOM LPS to RAS for downstream tools.",
                "<3 mm point annotations are saved separately and are not merged into the segmentation label.",
                "Small/large auxiliary masks use physical equivalent diameter rather than raw voxel count.",
            ],
        }
        save_sidecar(os.path.join(out_dir, name + "_geometry.json"), metadata)

        print(
            f"[INFO] saved {name} | series={series_uid} | shape={image.shape} | "
            f"mask_voxels={mask_voxels} | small_voxels={int(small_mask.sum())} | "
            f"large_voxels={int(large_mask.sum())} | lt3mm_points={len(point_records)} | "
            f"dropped_duplicate_slices={len(duplicates)}"
        )


def discover_patient_dirs(root_dir):
    root_dir = os.path.abspath(root_dir)
    if os.path.basename(root_dir).startswith(PATIENT_DIR_PREFIX):
        return [root_dir]

    patient_dirs = []
    for current_dir, dir_names, _ in os.walk(root_dir):
        dir_names[:] = [
            name
            for name in dir_names
            if name not in {"data", "data_stage2_c3", "train_runs", "__pycache__"}
        ]
        for dir_name in dir_names:
            if dir_name.startswith(PATIENT_DIR_PREFIX):
                patient_dirs.append(os.path.join(current_dir, dir_name))

    if patient_dirs:
        return sorted(set(patient_dirs))

    return [
        os.path.join(root_dir, name)
        for name in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, name))
    ]


def process(root_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    patient_dirs = discover_patient_dirs(root_dir)
    print(f"[INFO] discovered patient dirs: {len(patient_dirs)}")
    for patient_dir in patient_dirs:
        patient = os.path.basename(patient_dir)
        print(f"\n[INFO] patient: {patient}")
        try:
            process_patient(patient_dir, out_dir)
        except Exception as exc:
            print(f"[WARN] failed patient {patient}: {exc}")


if __name__ == "__main__":
    process(RAW_ROOT, OUT_DIR)
