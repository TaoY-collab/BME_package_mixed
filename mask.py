import os
import xml.etree.ElementTree as ET
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from skimage.draw import polygon

from project_config import BASE_DIR, DATA_DIR


RAW_ROOT = BASE_DIR
HU_MIN = -1000.0
HU_MAX = 400.0
# Keep CT volumes in HU. Training and visualization apply the HU window
# consistently via ScaleIntensityRanged.
NORMALIZE_IMAGE = False


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
    nodules = []
    for nodule in root.findall(".//ns:unblindedReadNodule", ns):
        roi_list = []
        for roi in nodule.findall(".//ns:roi", ns):
            z_elem = roi.find("ns:imageZposition", ns)
            if z_elem is None:
                continue

            xs, ys = [], []
            for edge in roi.findall("ns:edgeMap", ns):
                x = edge.find("ns:xCoord", ns)
                y = edge.find("ns:yCoord", ns)
                if x is not None and y is not None:
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

    return {
        "series_uid": series_uid,
        "nodules": nodules,
    }


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

    # Fall back to the largest CT series, which is usually the axial diagnostic CT.
    return max(series_map.keys(), key=lambda uid: len(series_map[uid]))


def get_slice_normal(ds):
    orientation = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
    row_cosine = orientation[:3]
    col_cosine = orientation[3:]
    return np.cross(row_cosine, col_cosine)


def get_slice_position(ds, slice_normal):
    ipp = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
    return float(np.dot(ipp, slice_normal))


def sort_slices(slices):
    orientation = np.asarray(slices[0].ImageOrientationPatient, dtype=np.float64)
    row_cosine = orientation[:3]
    col_cosine = orientation[3:]
    slice_normal = np.cross(row_cosine, col_cosine)

    return sorted(slices, key=lambda ds: get_slice_position(ds, slice_normal)), slice_normal


def build_affine(slices, slice_normal):
    first = slices[0]
    orientation = np.asarray(first.ImageOrientationPatient, dtype=np.float64)
    row_cosine = orientation[:3]
    col_cosine = orientation[3:]

    pixel_spacing = np.asarray(first.PixelSpacing, dtype=np.float64)
    row_spacing = float(pixel_spacing[0])
    col_spacing = float(pixel_spacing[1])

    if len(slices) > 1:
        first_pos = np.asarray(slices[0].ImagePositionPatient, dtype=np.float64)
        last_pos = np.asarray(slices[-1].ImagePositionPatient, dtype=np.float64)
        slice_spacing = float(
            abs(np.dot(last_pos - first_pos, slice_normal)) / max(len(slices) - 1, 1)
        )
    else:
        slice_spacing = float(getattr(first, "SliceThickness", 1.0))

    origin = np.asarray(first.ImagePositionPatient, dtype=np.float64)

    affine = np.eye(4, dtype=np.float64)
    # Data is stored as numpy (row, col, slice). Axis 0 follows the DICOM
    # column direction, and axis 1 follows the DICOM row direction.
    affine[:3, 0] = col_cosine * row_spacing
    affine[:3, 1] = row_cosine * col_spacing
    affine[:3, 2] = slice_normal * slice_spacing
    affine[:3, 3] = origin
    return affine


def load_dicom_series(series_paths):
    slices = [pydicom.dcmread(path, force=True) for path in series_paths]
    slices, slice_normal = sort_slices(slices)

    slice_images = []
    for ds in slices:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        slice_image = ds.pixel_array.astype(np.float32) * slope + intercept
        slice_images.append(slice_image)

    image = np.stack(slice_images, axis=-1)

    affine = build_affine(slices, slice_normal)
    return image, affine, slices, slice_normal


def normalize_hu(image):
    image = np.clip(image, HU_MIN, HU_MAX)
    return ((image - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def create_mask(image_shape, nodules, slices, slice_normal):
    mask = np.zeros(image_shape, dtype=np.uint8)

    z_positions = np.asarray(
        [float(ds.ImagePositionPatient[2]) for ds in slices],
        dtype=np.float64,
    )
    slice_positions = np.asarray(
        [get_slice_position(ds, slice_normal) for ds in slices],
        dtype=np.float64,
    )

    order = np.argsort(z_positions)
    sorted_z = z_positions[order]
    sorted_projected = slice_positions[order]

    for nodule in nodules:
        for roi in nodule:
            roi_position = float(
                np.interp(
                    float(roi["z"]),
                    sorted_z,
                    sorted_projected,
                    left=sorted_projected[0],
                    right=sorted_projected[-1],
                )
            )
            z_idx = int(np.argmin(np.abs(slice_positions - roi_position)))

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


def save_nifti(data, affine, path, dtype):
    arr = data.astype(dtype, copy=False)
    nii = nib.Nifti1Image(arr, affine)
    nib.save(nii, path)


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
        nodules = parsed["nodules"]
        if not nodules:
            continue

        series_uid = choose_series_uid(series_map, parsed["series_uid"])
        seen_series_uids[series_uid] = seen_series_uids.get(series_uid, 0) + 1
        if seen_series_uids[series_uid] > 1:
            print(
                f"[WARN] duplicate XML for same series: patient={patient} | "
                f"series={series_uid} | count={seen_series_uids[series_uid]}"
            )

        image_hu, affine, slices, slice_normal = load_dicom_series(series_map[series_uid])
        mask = create_mask(image_hu.shape, nodules, slices, slice_normal)

        mask_voxels = int(np.sum(mask > 0))
        if mask_voxels <= 0:
            print(f"[WARN] empty mask after rasterization: {patient} | {xml_path}")
            continue

        image = normalize_hu(image_hu) if NORMALIZE_IMAGE else image_hu.astype(np.float32)
        name = f"{patient}_{Path(xml_path).stem}"

        save_nifti(image, affine, os.path.join(out_dir, name + "_img.nii.gz"), np.float32)
        save_nifti(mask, affine, os.path.join(out_dir, name + "_mask.nii.gz"), np.uint8)

        print(
            f"[INFO] saved {name} | series={series_uid} | "
            f"shape={image.shape} | mask_voxels={mask_voxels}"
        )


def process(root_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for patient in sorted(os.listdir(root_dir)):
        patient_dir = os.path.join(root_dir, patient)
        if not os.path.isdir(patient_dir):
            continue

        print(f"\n[INFO] patient: {patient}")
        try:
            process_patient(patient_dir, out_dir)
        except Exception as exc:
            print(f"[WARN] failed patient {patient}: {exc}")


if __name__ == "__main__":
    process(RAW_ROOT, DATA_DIR)
