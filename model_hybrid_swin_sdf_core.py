# -*- coding: utf-8 -*-
"""
src/model_hybrid_swin_sdf_core.py

Hybrid-Swin-SDF-CoreNet 主模型文件。

模型名称：
    HybridSwinSDFCoreNet

模型输入：
    x: [B, 1, 64, 64, 64]

模型输出：
    {
        "mask_logits": [B, 1, 64, 64, 64],
        "sdf": [B, 1, 64, 64, 64]
    }

整体结构：
1. 3D 分支：
   使用 MONAI SwinUNETR 作为 3D 主干。
   不直接将 SwinUNETR 作为最终分割器，而是设置：
       out_channels = swin_feature_channels
   使其输出 full-resolution 3D feature。

2. 2D 分支：
   只保留 z-axis adjacent triplet 2D texture branch，并还原成 3D feature。

3. 融合模块：
   使用 GatedFusion3D 融合 3D SwinUNETR feature 和 z-axis 2D feature。

4. 输出头：
   - mask_head 输出 mask logits；
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from monai.networks.nets import SwinUNETR
except Exception as e:
    SwinUNETR = None
    _MONAI_IMPORT_ERROR = e
else:
    _MONAI_IMPORT_ERROR = None

try:
    from .fusion import GatedFusion3D, ZAxisAdjacent2DProjector
except ImportError:
    from fusion import GatedFusion3D, ZAxisAdjacent2DProjector


def _to_3tuple(value: Sequence[int] | int) -> Tuple[int, int, int]:
    """
    将 img_size 统一转换为 3 元组。

    参数
    ----
    value:
        可以是 int，也可以是长度为 3 的序列。

    返回
    ----
    tuple_value:
        形如 (D, H, W) 的三元组。
    """
    if isinstance(value, int):
        return (value, value, value)

    if len(value) != 3:
        raise ValueError(f"img_size 必须是 int 或长度为 3 的序列，但当前为 {value}")

    return tuple(int(v) for v in value)


def _create_swin_unetr_compatible(
    img_size: Tuple[int, int, int],
    in_channels: int,
    out_channels: int,
    feature_size: int,
    use_checkpoint: bool,
) -> nn.Module:
    """
    创建兼容不同 MONAI 版本的 SwinUNETR。

    背景
    ----
    不同 MONAI 版本中 SwinUNETR 的 img_size 参数存在差异：
    1. 旧版本需要 img_size；
    2. 新版本可能不再需要 img_size；
    3. 有些版本传入 img_size 会触发 deprecation 或 removed 报错。

    策略
    ----
    先尝试不传 img_size 的新版写法；
    如果失败，再尝试传入 img_size 的旧版写法。

    返回
    ----
    model:
        MONAI SwinUNETR 实例。
    """
    if SwinUNETR is None:
        raise ImportError(
            "无法导入 MONAI 的 SwinUNETR。请先安装 MONAI，例如：\n"
            "pip install monai\n"
            f"原始错误信息：{_MONAI_IMPORT_ERROR}"
        )

    common_kwargs = dict(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
        norm_name="instance",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
    )

    # 新版 MONAI：img_size 可能已被移除。
    try:
        return SwinUNETR(**common_kwargs)
    except Exception as err_without_img_size:
        # 旧版 MONAI：img_size 是必需参数。
        try:
            return SwinUNETR(
                img_size=img_size,
                **common_kwargs,
            )
        except Exception as err_with_img_size:
            raise RuntimeError(
                "创建 SwinUNETR 失败。已经分别尝试：\n"
                "1. 不传 img_size 的新版 MONAI 写法；\n"
                "2. 传入 img_size 的旧版 MONAI 写法。\n\n"
                f"不传 img_size 时的错误：{repr(err_without_img_size)}\n"
                f"传入 img_size 时的错误：{repr(err_with_img_size)}"
            ) from err_with_img_size


class ConvNormAct3D(nn.Module):
    """
    3D Conv + InstanceNorm3d + LeakyReLU 模块。

    用于将 SwinUNETR 输出的 swin_feature_channels
    投影到 fusion_channels，便于后续门控融合。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=False,
            ),
            nn.InstanceNorm3d(
                num_features=out_channels,
                affine=True,
            ),
            nn.LeakyReLU(
                negative_slope=negative_slope,
                inplace=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        """
        return self.block(x)


class PredictionHead3D(nn.Module):
    """
    轻量级 3D 预测头。

    结构：
        Conv3d -> InstanceNorm3d -> LeakyReLU -> Conv3d

    输出：
        [B, out_channels, D, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        hidden_channels: int | None = None,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()

        if hidden_channels is None:
            hidden_channels = in_channels

        self.head = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(
                num_features=hidden_channels,
                affine=True,
            ),
            nn.LeakyReLU(
                negative_slope=negative_slope,
                inplace=True,
            ),
            nn.Conv3d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        """
        return self.head(x)


class HybridSwinSDFCoreNet(nn.Module):
    ARCHITECTURE_VERSION = "hybrid_swin_z_axis_2d_mask_sdf_no_global_pos_v1"
    """
    Hybrid-Swin-SDF-CoreNet 主模型。

    参数
    ----
    img_size:
        输入 patch 尺寸，默认 (64, 64, 64)。

    in_channels:
        输入通道数，LIDC-IDRI CT patch 默认为 1。

    swin_feature_channels:
        SwinUNETR 输出的 full-resolution 3D feature 通道数。

    two_d_feature_channels:
        z-axis 2D 分支还原到 3D 后的输出通道数。

    fusion_channels:
        门控融合阶段使用的主特征通道数。

    feature_size:
        MONAI SwinUNETR 的 feature_size。
        注意 MONAI 通常要求 feature_size 能被 12 整除。
        默认 24。

    use_checkpoint:
        是否启用 SwinUNETR 的 gradient checkpointing。
    """

    def __init__(
        self,
        img_size: Sequence[int] | int = (64, 64, 64),
        in_channels: int = 1,
        ct_in_channels: int | None = None,
        swin_feature_channels: int = 16,
        two_d_feature_channels: int = 16,
        fusion_channels: int = 32,
        feature_size: int = 48,
        use_checkpoint: bool = True,
        use_sdf_branch: bool = True,
        two_d_mode: str = "z_axis_adjacent_triplet",
        neighbor_radius: int = 1,
        two_d_slice_chunk_size: int = 8,
    ) -> None:
        super().__init__()

        self.img_size = _to_3tuple(img_size)
        self.in_channels = int(in_channels)
        self.ct_in_channels = self.in_channels if ct_in_channels is None else int(ct_in_channels)
        self.swin_feature_channels = int(swin_feature_channels)
        self.two_d_feature_channels = int(two_d_feature_channels)
        self.fusion_channels = int(fusion_channels)
        self.feature_size = int(feature_size)
        self.use_checkpoint = bool(use_checkpoint)
        self.two_d_mode = str(two_d_mode)
        self.neighbor_radius = int(neighbor_radius)
        self.two_d_slice_chunk_size = max(int(two_d_slice_chunk_size), 1)
        self.use_sdf_branch = bool(use_sdf_branch)

        if self.in_channels <= 0:
            raise ValueError(f"in_channels 必须大于 0，但当前为 {self.in_channels}")

        if self.ct_in_channels <= 0:
            raise ValueError(f"ct_in_channels must be positive, got {self.ct_in_channels}")

        if self.ct_in_channels != self.in_channels:
            raise ValueError(
                "Auxiliary input channels have been removed; "
                f"ct_in_channels must match in_channels, got {self.ct_in_channels} vs {self.in_channels}"
            )

        if self.swin_feature_channels <= 0:
            raise ValueError(
                f"swin_feature_channels 必须大于 0，"
                f"但当前为 {self.swin_feature_channels}"
            )

        if self.two_d_feature_channels <= 0:
            raise ValueError(
                f"two_d_feature_channels 必须大于 0，"
                f"但当前为 {self.two_d_feature_channels}"
            )

        if self.fusion_channels <= 0:
            raise ValueError(
                f"fusion_channels 必须大于 0，但当前为 {self.fusion_channels}"
            )

        # 3D 分支：SwinUNETR 输出 full-resolution 3D feature。
        # 这里 out_channels 不是最终类别数，而是 3D feature 通道数。
        self.swin3d = _create_swin_unetr_compatible(
            img_size=self.img_size,
            in_channels=self.in_channels,
            out_channels=self.swin_feature_channels,
            feature_size=self.feature_size,
            use_checkpoint=self.use_checkpoint,
        )

        # 将 SwinUNETR 输出的 swin_feature_channels 投影到 fusion_channels。
        self.swin_feature_proj = ConvNormAct3D(
            in_channels=self.swin_feature_channels,
            out_channels=self.fusion_channels,
            kernel_size=1,
            padding=0,
        )

        if self.two_d_mode == "adjacent_axial_triplet":
            self.two_d_mode = "z_axis_adjacent_triplet"
        if self.two_d_mode != "z_axis_adjacent_triplet":
            raise ValueError(
                "This architecture only supports two_d_mode='z_axis_adjacent_triplet', "
                f"got {self.two_d_mode!r}"
            )

        self.projector2d = ZAxisAdjacent2DProjector(
            in_channels=self.in_channels,
            encoder_base_channels=self.two_d_feature_channels,
            encoder_out_channels=self.two_d_feature_channels,
            out_channels=self.two_d_feature_channels,
            num_blocks=3,
            neighbor_radius=self.neighbor_radius,
            slice_chunk_size=self.two_d_slice_chunk_size,
            use_checkpoint=self.use_checkpoint,
        )

        # 2D-3D 门控融合。
        self.gated_fusion = GatedFusion3D(
            channels_3d=self.fusion_channels,
            channels_2d=self.two_d_feature_channels,
        )

        # Segmentation and signed-distance output heads.
        self.mask_head = PredictionHead3D(
            in_channels=self.fusion_channels,
            out_channels=1,
            hidden_channels=self.fusion_channels,
        )
        self.sdf_head = (
            PredictionHead3D(
                in_channels=self.fusion_channels,
                out_channels=1,
                hidden_channels=self.fusion_channels,
            )
            if self.use_sdf_branch
            else None
        )

    def _check_input(self, x: torch.Tensor) -> None:
        """
        检查输入张量是否合法。
        """
        if x.ndim != 5:
            raise ValueError(
                f"模型输入必须是 [B, C, D, H, W]，但当前 shape = {tuple(x.shape)}"
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"模型期望输入通道数为 {self.in_channels}，"
                f"但当前输入通道数为 {x.shape[1]}"
            )

    def _resize_to_input(
        self,
        feat: torch.Tensor,
        target_size: Tuple[int, int, int],
        name: str,
    ) -> torch.Tensor:
        """
        如果特征空间尺寸和输入不一致，则插值回输入尺寸。

        参数
        ----
        feat:
            输入特征，shape 为 [B, C, D, H, W]。

        target_size:
            目标空间尺寸，一般为输入 x 的 [D, H, W]。

        name:
            特征名称，用于报错提示。
        """
        if feat.ndim != 5:
            raise ValueError(
                f"{name} 必须是 5D 张量 [B, C, D, H, W]，"
                f"但当前 shape = {tuple(feat.shape)}"
            )

        if feat.shape[2:] != target_size:
            feat = F.interpolate(
                feat,
                size=target_size,
                mode="trilinear",
                align_corners=False,
            )

        return feat

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播。

        参数
        ----
        x:
            输入 CT patch，shape 为 [B, 1, 64, 64, 64]。

        返回
        ----
        outputs:
            {
                "mask_logits": [B, 1, 64, 64, 64],
                "sdf":         [B, 1, 64, 64, 64],
            }
        """
        self._check_input(x)

        target_size = tuple(x.shape[2:])
        ct_x = x

        # 3D SwinUNETR 分支。
        f3d = self.swin3d(ct_x)

        # 某些情况下 SwinUNETR 输出尺寸可能和输入不一致，这里统一插值回输入尺寸。
        f3d = self._resize_to_input(
            feat=f3d,
            target_size=target_size,
            name="f3d",
        )

        # 投影到 fusion_channels。
        f3d = self.swin_feature_proj(f3d)

        # z-axis adjacent triplet 2D texture branch.
        f2d = self.projector2d(ct_x)

        # 理论上 ZAxisAdjacent2DProjector 输出与输入尺寸一致；
        # 这里仍然做一次保险处理。
        f2d = self._resize_to_input(
            feat=f2d,
            target_size=target_size,
            name="f2d",
        )

        # 2D-3D 门控融合。
        fused = self.gated_fusion(f3d=f3d, f2d=f2d)

        mask_logits = self.mask_head(fused)
        # Ensure the output returns to the input spatial size.
        mask_logits = self._resize_to_input(
            feat=mask_logits,
            target_size=target_size,
            name="mask_logits",
        )

        outputs = {
            "mask_logits": mask_logits,
        }
        if self.sdf_head is not None:
            sdf = torch.tanh(self.sdf_head(fused))
            outputs["sdf"] = self._resize_to_input(
                feat=sdf,
                target_size=target_size,
                name="sdf",
            )

        return outputs


if __name__ == "__main__":
    # 简单自检：
    # 输入 [2, 1, 64, 64, 64]，检查三个输出的 shape。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridSwinSDFCoreNet(
        img_size=(64, 64, 64),
        in_channels=1,
        swin_feature_channels=16,
        two_d_feature_channels=16,
        fusion_channels=32,
        feature_size=48,
        use_checkpoint=True,
    ).to(device)

    x = torch.randn(2, 1, 64, 64, 64).to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(x)

    print("输入 x shape:", tuple(x.shape))
    print("mask_logits shape:", tuple(outputs["mask_logits"].shape))
    print("sdf shape:", tuple(outputs["sdf"].shape))
    print("sdf range:", float(outputs["sdf"].min()), float(outputs["sdf"].max()))
