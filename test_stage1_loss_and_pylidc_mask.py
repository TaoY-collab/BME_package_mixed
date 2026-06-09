import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from losses import AdaptiveDynamicSegLoss
from mask import (
    build_affine,
    create_pylidc_mask,
    save_nifti_pair,
    sort_dicom_slices_like_pylidc,
    validate_lidc_dicom_root,
    validate_pylidc_slice_alignment,
)


class Stage1LossTests(unittest.TestCase):
    def test_boundary_weight_cosine_schedule(self) -> None:
        criterion = AdaptiveDynamicSegLoss(num_classes=1)

        self.assertEqual(criterion.get_weights(39).boundary, 0.0)
        self.assertEqual(criterion.get_weights(40).boundary, 0.0)
        self.assertAlmostEqual(criterion.get_weights(80).boundary, 0.01)
        self.assertAlmostEqual(criterion.get_weights(120).boundary, 0.02)
        self.assertAlmostEqual(criterion.get_weights(121).boundary, 0.02)

    def test_total_matches_stage1_formula_and_backpropagates(self) -> None:
        criterion = AdaptiveDynamicSegLoss(num_classes=1)
        logits = torch.zeros((1, 1, 8, 8, 8), requires_grad=True)
        label = torch.zeros((1, 1, 8, 8, 8))
        label[:, :, 2:6, 2:6, 2:6] = 1

        total, parts = criterion(
            {"mask_logits": logits},
            {"label": label},
            epoch=80,
        )
        expected = (
            parts["dice"]
            + 0.4 * parts["tversky"]
            + 0.01 * parts["boundary"]
        )
        torch.testing.assert_close(total.detach(), expected)

        total.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


class PylidcMaskTests(unittest.TestCase):
    class FakeDicom:
        ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        PixelSpacing = [0.7, 0.8]
        RescaleSlope = 1.0
        RescaleIntercept = 0.0

        def __init__(self, z, instance_number):
            self.ImagePositionPatient = [0.0, 0.0, float(z)]
            self.InstanceNumber = int(instance_number)
            self.SliceThickness = 2.0
            self.pixel_array = np.full((3, 4), z, dtype=np.int16)

    def test_slice_order_matches_pylidc_and_deduplicates(self) -> None:
        slices = [
            self.FakeDicom(2.0, 3),
            self.FakeDicom(0.0, 2),
            self.FakeDicom(0.0, 1),
            self.FakeDicom(1.0, 4),
        ]
        ordered = sort_dicom_slices_like_pylidc(slices)

        self.assertEqual(
            [float(ds.ImagePositionPatient[2]) for ds in ordered],
            [0.0, 1.0, 2.0],
        )
        self.assertEqual(ordered[0].InstanceNumber, 1)

        class FakeScan:
            slice_zvals = np.asarray([0.0, 1.0, 2.0])

        validate_pylidc_slice_alignment(FakeScan(), ordered)

    def test_affine_uses_array_axis_order(self) -> None:
        slices = [
            self.FakeDicom(0.0, 1),
            self.FakeDicom(2.0, 2),
            self.FakeDicom(4.0, 3),
        ]
        affine = build_affine(slices)

        np.testing.assert_allclose(affine[:3, 0], [0.0, 0.7, 0.0])
        np.testing.assert_allclose(affine[:3, 1], [0.8, 0.0, 0.0])
        np.testing.assert_allclose(affine[:3, 2], [0.0, 0.0, 2.0])

    def test_consensus_mask_uses_only_qualifying_clusters(self) -> None:
        class FakeScan:
            patient_id = "LIDC-IDRI-TEST"
            id = 7

            def cluster_annotations(self, verbose: bool = False):
                return [
                    [object(), object(), object()],
                    [object(), object()],
                ]

        def fake_consensus(cluster, clevel=0.5, pad=0):
            masks = [
                np.ones((2, 3, 2), dtype=np.uint8)
                for _ in cluster
            ]
            bbox = (slice(1, 3), slice(2, 5), slice(3, 5))
            return np.ones((2, 3, 2), dtype=np.uint8), bbox, masks

        mask, count = create_pylidc_mask(
            scan=FakeScan(),
            image_shape=(5, 7, 8),
            consensus_fn=fake_consensus,
            min_annotations=3,
        )

        self.assertEqual(count, 1)
        self.assertEqual(mask.shape, (5, 7, 8))
        self.assertEqual(int(mask.sum()), 12)
        self.assertTrue(np.all(mask[1:3, 2:5, 3:5] == 1))

    def test_empty_consensus_mask_is_not_counted(self) -> None:
        class FakeScan:
            patient_id = "LIDC-IDRI-TEST"
            id = 8

            def cluster_annotations(self, verbose: bool = False):
                return [[object(), object(), object()]]

        with patch(
            "mask.make_consensus_mask_hwd",
            return_value=np.zeros((5, 7, 8), dtype=np.uint8),
        ):
            result, count = create_pylidc_mask(
                scan=FakeScan(),
                image_shape=(5, 7, 8),
                consensus_fn=object(),
                min_annotations=3,
            )

        self.assertEqual(count, 0)
        self.assertEqual(int(result.sum()), 0)

    def test_consensus_shape_mismatch_fails(self) -> None:
        class FakeScan:
            patient_id = "LIDC-IDRI-TEST"
            id = 9

            def cluster_annotations(self, verbose: bool = False):
                return [[object(), object(), object()]]

        with patch(
            "mask.make_consensus_mask_hwd",
            return_value=np.ones((4, 7, 8), dtype=np.uint8),
        ):
            with self.assertRaisesRegex(ValueError, "Mask shape mismatch"):
                create_pylidc_mask(
                    scan=FakeScan(),
                    image_shape=(5, 7, 8),
                    consensus_fn=object(),
                    min_annotations=3,
                )

    def test_nifti_round_trip_and_overwrite_protection(self) -> None:
        image = np.zeros((4, 5, 6), dtype=np.float32)
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[1:3, 2:4, 3:5] = 1
        affine = np.diag([0.7, 0.8, 2.0, 1.0])

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "case_img.nii.gz"
            label_path = Path(tmp) / "case_mask.nii.gz"
            save_nifti_pair(
                image=image,
                mask=mask,
                affine=affine,
                image_path=image_path,
                label_path=label_path,
                overwrite=False,
                verify_after_save=True,
            )

            self.assertEqual(nib.load(str(image_path)).shape, image.shape)
            np.testing.assert_array_equal(
                np.asanyarray(nib.load(str(label_path)).dataobj),
                mask,
            )
            with self.assertRaises(FileExistsError):
                save_nifti_pair(
                    image=image,
                    mask=mask,
                    affine=affine,
                    image_path=image_path,
                    label_path=label_path,
                    overwrite=False,
                    verify_after_save=True,
                )

    def test_dicom_root_rejects_nifti_only_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "case_img.nii.gz").touch()
            with self.assertRaisesRegex(ValueError, "no LIDC-IDRI patient"):
                validate_lidc_dicom_root(root)

            patient = root / "LIDC-IDRI-0001" / "study" / "series"
            patient.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "No .dcm files"):
                validate_lidc_dicom_root(root)

            (patient / "slice001.dcm").touch()
            validate_lidc_dicom_root(root)


if __name__ == "__main__":
    unittest.main()
