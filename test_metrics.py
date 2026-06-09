import unittest

import numpy as np

from metrics import compute_all_metrics, ensure_3d


class SegmentationMetricTests(unittest.TestCase):
    def test_identical_masks_are_perfect(self) -> None:
        mask = np.zeros((5, 5, 5), dtype=np.uint8)
        mask[1:4, 1:4, 1:4] = 1

        metrics = compute_all_metrics(mask, mask)

        for key in ("dice", "iou", "precision", "recall", "bf1"):
            self.assertEqual(metrics[key], 1.0)
        self.assertEqual(metrics["hd95"], 0.0)
        self.assertEqual(metrics["assd"], 0.0)
        self.assertEqual(metrics["vol_diff"], 0.0)

    def test_partial_overlap_matches_hand_calculation(self) -> None:
        pred = np.zeros((5, 5, 5), dtype=np.uint8)
        gt = np.zeros_like(pred)
        pred[1, 1, 1] = 1
        pred[2, 1, 1] = 1
        gt[2, 1, 1] = 1
        gt[3, 1, 1] = 1

        metrics = compute_all_metrics(pred, gt)

        self.assertAlmostEqual(metrics["dice"], 0.5)
        self.assertAlmostEqual(metrics["iou"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["vol_diff"], 0.0)

    def test_disjoint_masks_have_zero_overlap(self) -> None:
        pred = np.zeros((5, 5, 5), dtype=np.uint8)
        gt = np.zeros_like(pred)
        pred[1, 1, 1] = 1
        gt[3, 1, 1] = 1

        metrics = compute_all_metrics(pred, gt)

        for key in ("dice", "iou", "precision", "recall"):
            self.assertEqual(metrics[key], 0.0)

    def test_surface_distances_respect_anisotropic_spacing(self) -> None:
        pred = np.zeros((5, 5, 5), dtype=np.uint8)
        gt = np.zeros_like(pred)
        pred[1, 2, 2] = 1
        gt[2, 2, 2] = 1

        metrics = compute_all_metrics(pred, gt, spacing=(2.0, 1.0, 1.0))

        self.assertAlmostEqual(metrics["hd95"], 2.0)
        self.assertAlmostEqual(metrics["assd"], 2.0)
        self.assertAlmostEqual(metrics["vol_diff"], 0.0)

    def test_empty_mask_conventions(self) -> None:
        empty = np.zeros((5, 5, 5), dtype=np.uint8)
        nonempty = empty.copy()
        nonempty[2, 2, 2] = 1

        both_empty = compute_all_metrics(empty, empty)
        self.assertEqual(both_empty["dice"], 1.0)
        self.assertEqual(both_empty["iou"], 1.0)
        self.assertEqual(both_empty["hd95"], 0.0)
        self.assertEqual(both_empty["assd"], 0.0)

        missed = compute_all_metrics(empty, nonempty)
        for key in ("dice", "iou", "precision", "recall", "bf1"):
            self.assertEqual(missed[key], 0.0)
        self.assertTrue(np.isnan(missed["hd95"]))
        self.assertTrue(np.isnan(missed["assd"]))
        self.assertEqual(missed["vol_diff"], -1.0)

    def test_rejects_invalid_shapes_and_spacing(self) -> None:
        with self.assertRaises(ValueError):
            ensure_3d(np.zeros((2, 2), dtype=np.uint8))
        with self.assertRaises(ValueError):
            compute_all_metrics(
                np.zeros((3, 3, 3), dtype=np.uint8),
                np.zeros((4, 4, 4), dtype=np.uint8),
            )
        with self.assertRaises(ValueError):
            compute_all_metrics(
                np.zeros((3, 3, 3), dtype=np.uint8),
                np.zeros((3, 3, 3), dtype=np.uint8),
                spacing=(1.0, 0.0, 1.0),
            )


if __name__ == "__main__":
    unittest.main()
