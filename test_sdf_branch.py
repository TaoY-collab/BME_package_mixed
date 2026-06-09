import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

import model_hybrid_swin_sdf_core as model_module
from losses import AdaptiveDynamicSegLoss


class _DummyProjector(nn.Module):
    def __init__(self, out_channels: int, **_: object) -> None:
        super().__init__()
        self.out_channels = int(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(1, self.out_channels, 1, 1, 1)


class _DummyFusion(nn.Module):
    def __init__(self, **_: object) -> None:
        super().__init__()

    def forward(self, f3d: torch.Tensor, f2d: torch.Tensor) -> torch.Tensor:
        return f3d + f2d[:, :1].mean(dim=1, keepdim=True)


class SDFBranchTests(unittest.TestCase):
    def _build_model(self, use_sdf_branch: bool) -> nn.Module:
        def make_backbone(
            img_size: object,
            in_channels: int,
            out_channels: int,
            feature_size: int,
            use_checkpoint: bool,
        ) -> nn.Module:
            del img_size, feature_size, use_checkpoint
            return nn.Conv3d(in_channels, out_channels, kernel_size=1)

        with (
            patch.object(model_module, "_create_swin_unetr_compatible", make_backbone),
            patch.object(model_module, "ZAxisAdjacent2DProjector", _DummyProjector),
            patch.object(model_module, "GatedFusion3D", _DummyFusion),
        ):
            return model_module.HybridSwinSDFCoreNet(
                img_size=(8, 8, 8),
                in_channels=1,
                swin_feature_channels=2,
                two_d_feature_channels=2,
                fusion_channels=2,
                feature_size=12,
                use_checkpoint=False,
                use_sdf_branch=use_sdf_branch,
            )

    def test_model_outputs_bounded_sdf(self) -> None:
        model = self._build_model(use_sdf_branch=True)
        outputs = model(torch.randn(2, 1, 8, 8, 8))

        self.assertEqual(set(outputs), {"mask_logits", "sdf"})
        self.assertEqual(outputs["mask_logits"].shape, (2, 1, 8, 8, 8))
        self.assertEqual(outputs["sdf"].shape, (2, 1, 8, 8, 8))
        self.assertLessEqual(float(outputs["sdf"].max()), 1.0)
        self.assertGreaterEqual(float(outputs["sdf"].min()), -1.0)

    def test_sdf_branch_can_be_disabled(self) -> None:
        model = self._build_model(use_sdf_branch=False)
        outputs = model(torch.randn(1, 1, 8, 8, 8))
        self.assertEqual(set(outputs), {"mask_logits"})

    def test_model_has_no_global_position_encoding_module(self) -> None:
        model = self._build_model(use_sdf_branch=True)
        self.assertFalse(hasattr(model, "global_position_encoding"))
        self.assertFalse(hasattr(model_module, "GlobalPositionEncoding3D"))

    def test_sdf_regression_is_added_to_total_loss(self) -> None:
        outputs = {
            "mask_logits": torch.zeros(1, 1, 4, 4, 4),
            "sdf": torch.zeros(1, 1, 4, 4, 4),
        }
        batch = {
            "label": torch.zeros(1, 1, 4, 4, 4),
            "dist_map": torch.ones(1, 1, 4, 4, 4),
        }
        criterion_with_sdf = AdaptiveDynamicSegLoss(
            num_classes=1,
            boundary_max_weight=0.0,
            sdf_start_epoch=0,
            sdf_end_epoch=0,
            sdf_max_weight=0.2,
        )
        criterion_without_sdf = AdaptiveDynamicSegLoss(
            num_classes=1,
            boundary_max_weight=0.0,
            sdf_max_weight=0.0,
        )

        total_with_sdf, parts = criterion_with_sdf(outputs, batch, epoch=1)
        total_without_sdf, _ = criterion_without_sdf(outputs, batch, epoch=1)

        self.assertAlmostEqual(float(parts["sdf"]), 0.5, places=6)
        self.assertAlmostEqual(float(parts["w_sdf"]), 0.2, places=6)
        self.assertAlmostEqual(
            float(total_with_sdf - total_without_sdf),
            0.1,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
