import math
import unittest

import numpy as np
import torch

from losses import AdaptiveDynamicSegLoss
from train import BuildAuxTargetsd, inverse_equivalent_diameter_weight


def sphere_volume_mm3(diameter_mm: float) -> float:
    return math.pi * float(diameter_mm) ** 3 / 6.0


class SizeWeightMapTests(unittest.TestCase):
    def test_inverse_diameter_weights_for_3_6_10_mm(self) -> None:
        self.assertAlmostEqual(
            inverse_equivalent_diameter_weight(sphere_volume_mm3(3.0)),
            10.0 / 3.0,
            places=6,
        )
        self.assertAlmostEqual(
            inverse_equivalent_diameter_weight(sphere_volume_mm3(6.0)),
            10.0 / 6.0,
            places=6,
        )
        self.assertAlmostEqual(
            inverse_equivalent_diameter_weight(sphere_volume_mm3(10.0)),
            1.0,
            places=6,
        )
        self.assertEqual(
            inverse_equivalent_diameter_weight(sphere_volume_mm3(1.0)),
            4.0,
        )

    def test_single_small_component_gets_weight_above_one(self) -> None:
        label = np.zeros((1, 8, 8, 8), dtype=np.uint8)
        label[0, 3, 3, 3] = 1
        result = BuildAuxTargetsd()({"label": label})
        weight = np.asarray(result["component_weight"])[0]

        self.assertEqual(float(weight[3, 3, 3]), 4.0)
        self.assertTrue(np.all(weight[label[0] == 0] == 1.0))

    def test_multiple_components_receive_independent_weights(self) -> None:
        label = np.zeros((1, 14, 14, 14), dtype=np.uint8)
        label[0, 1, 1, 1] = 1
        label[0, 7:11, 7:11, 7:11] = 1
        result = BuildAuxTargetsd()({"label": label})
        weight = np.asarray(result["component_weight"])[0]

        small_weight = float(weight[1, 1, 1])
        large_weight = float(weight[8, 8, 8])
        self.assertEqual(small_weight, 4.0)
        self.assertGreater(large_weight, 1.0)
        self.assertGreater(small_weight, large_weight)

    def test_empty_label_keeps_unit_weight(self) -> None:
        label = np.zeros((1, 8, 8, 8), dtype=np.uint8)
        result = BuildAuxTargetsd()({"label": label})
        np.testing.assert_array_equal(
            result["component_weight"],
            np.ones_like(label, dtype=np.float32),
        )


class SizeWeightedLossTests(unittest.TestCase):
    @staticmethod
    def make_batch(
        label: torch.Tensor,
        component_weight: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "label": label,
            "boundary": torch.ones_like(label),
            "dist_map": torch.full_like(label, 0.5),
            "component_weight": component_weight,
        }

    @staticmethod
    def make_criterion() -> AdaptiveDynamicSegLoss:
        return AdaptiveDynamicSegLoss(
            num_classes=1,
            boundary_start_epoch=0,
            boundary_end_epoch=0,
            boundary_max_weight=1.0,
            sdf_start_epoch=0,
            sdf_end_epoch=0,
            sdf_max_weight=1.0,
            size_weight_enabled=True,
            size_weight_start_epoch=0,
            size_weight_end_epoch=20,
        )

    def test_size_weight_cosine_ramp(self) -> None:
        criterion = self.make_criterion()
        label = torch.zeros(1, 1, 2, 2, 2)
        label[..., 0, 0, 0] = 1
        component_weight = torch.ones_like(label)
        component_weight[..., 0, 0, 0] = 4
        batch = self.make_batch(label, component_weight)
        outputs = {
            "mask_logits": torch.zeros_like(label),
            "sdf": torch.zeros_like(label),
        }

        _, parts_0 = criterion(outputs, batch, epoch=0)
        _, parts_10 = criterion(outputs, batch, epoch=10)
        _, parts_20 = criterion(outputs, batch, epoch=20)

        self.assertAlmostEqual(float(parts_0["size_weight_ramp"]), 0.0)
        self.assertAlmostEqual(float(parts_0["size_weight_fg_mean"]), 1.0)
        self.assertAlmostEqual(float(parts_10["size_weight_ramp"]), 0.5)
        self.assertAlmostEqual(float(parts_10["size_weight_fg_mean"]), 2.5)
        self.assertAlmostEqual(float(parts_20["size_weight_ramp"]), 1.0)
        self.assertAlmostEqual(float(parts_20["size_weight_fg_mean"]), 4.0)

    def test_enabled_size_weight_requires_map(self) -> None:
        criterion = self.make_criterion()
        label = torch.zeros(1, 1, 2, 2, 2)
        with self.assertRaisesRegex(KeyError, "component_weight"):
            criterion(
                {"mask_logits": torch.zeros_like(label)},
                {"label": label},
                epoch=20,
            )

    def test_small_nodule_error_contributes_more_to_segmentation_losses(self) -> None:
        criterion = self.make_criterion()
        label = torch.zeros(1, 1, 1, 1, 6)
        label[..., 1] = 1
        label[..., 4] = 1
        component_weight = torch.ones_like(label)
        component_weight[..., 1] = 4
        batch = self.make_batch(label, component_weight)

        perfect_logits = torch.full_like(label, -8.0)
        perfect_logits[label > 0] = 8.0
        small_error_logits = perfect_logits.clone()
        small_error_logits[..., 1] = -8.0
        large_error_logits = perfect_logits.clone()
        large_error_logits[..., 4] = -8.0

        _, small_parts = criterion(
            {
                "mask_logits": small_error_logits,
                "sdf": torch.zeros_like(label),
            },
            batch,
            epoch=20,
        )
        _, large_parts = criterion(
            {
                "mask_logits": large_error_logits,
                "sdf": torch.zeros_like(label),
            },
            batch,
            epoch=20,
        )

        self.assertGreater(float(small_parts["dice"]), float(large_parts["dice"]))
        self.assertGreater(
            float(small_parts["tversky"]),
            float(large_parts["tversky"]),
        )
        self.assertGreater(
            float(small_parts["boundary"]),
            float(large_parts["boundary"]),
        )
        self.assertEqual(float(small_parts["sdf"]), float(large_parts["sdf"]))

    def test_sdf_loss_is_independent_of_component_weight(self) -> None:
        criterion = self.make_criterion()
        label = torch.ones(1, 1, 2, 2, 2)
        outputs = {
            "mask_logits": torch.zeros_like(label),
            "sdf": torch.zeros_like(label),
        }
        unit_batch = self.make_batch(label, torch.ones_like(label))
        weighted_batch = self.make_batch(label, torch.full_like(label, 4.0))

        _, unit_parts = criterion(outputs, unit_batch, epoch=20)
        _, weighted_parts = criterion(outputs, weighted_batch, epoch=20)

        self.assertGreater(float(unit_parts["sdf"]), 0.0)
        self.assertEqual(float(unit_parts["sdf"]), float(weighted_parts["sdf"]))


if __name__ == "__main__":
    unittest.main()
