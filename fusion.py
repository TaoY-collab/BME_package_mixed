# -*- coding: utf-8 -*-
"""
src/fusion.py

Hybrid-Swin-SDF-CoreNet 项目的 2D-3D 多视角融合模块。

主要功能：
1. 将 3D 体数据切成 axial、coronal、sagittal 三个方向的 2D 切片；
2. 使用独立或共享的 2D encoder 提取三视角纹理特征；
3. 将三视角 2D 特征还原为 3D 特征；
4. 将三视角特征 concat 后用 1x1x1 Conv3d 压缩通道；
5. 使用 GatedFusion3D 将 2D 纹理特征和 3D 主干特征进行门控融合。

张量约定：
- 3D 输入统一为 [B, C, D, H, W]
- axial 2D 切片为 [B*D, C, H, W]
- coronal 2D 切片为 [B*H, C, D, W]
- sagittal 2D 切片为 [B*W, C, D, H]
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .modules2d import AdjacentSliceGatedEncoder, Simple2DTextureEncoder
except ImportError:
    from modules2d import AdjacentSliceGatedEncoder, Simple2DTextureEncoder


def _check_5d_tensor(x: torch.Tensor, name: str = "x") -> None:
    """
    检查输入是否为 5D 体数据张量 [B, C, D, H, W]。
    """
    if x.ndim != 5:
        raise ValueError(
            f"{name} 必须是 5D 张量 [B, C, D, H, W]，"
            f"但当前 shape = {tuple(x.shape)}"
        )


def _check_4d_tensor(x: torch.Tensor, name: str = "x") -> None:
    """
    检查输入是否为 4D 2D 特征张量 [N, C, H, W]。
    """
    if x.ndim != 4:
        raise ValueError(
            f"{name} 必须是 4D 张量 [N, C, H, W]，"
            f"但当前 shape = {tuple(x.shape)}"
        )


def axial_2d_to_3d(
    feat_2d: torch.Tensor,
    batch_size: int,
    depth: int,
) -> torch.Tensor:
    """
    将 axial 方向 2D 特征还原为 3D 特征。

    参数
    ----
    feat_2d:
        axial 方向 2D 特征，shape 为 [B*D, C, H, W]。

    batch_size:
        B。

    depth:
        D。

    返回
    ----
    feat_3d:
        还原后的 3D 特征，shape 为 [B, C, D, H, W]。
    """
    _check_4d_tensor(feat_2d, name="feat_2d")

    if batch_size <= 0:
        raise ValueError(f"batch_size 必须大于 0，但当前为 {batch_size}")

    if depth <= 0:
        raise ValueError(f"depth 必须大于 0，但当前为 {depth}")

    n, channels, height, width = feat_2d.shape
    expected_n = batch_size * depth

    if n != expected_n:
        raise ValueError(
            f"axial feat_2d 的第 0 维应为 B*D={expected_n}，"
            f"但当前为 {n}，feat_2d shape = {tuple(feat_2d.shape)}"
        )

    feat_3d = feat_2d.reshape(batch_size, depth, channels, height, width)
    feat_3d = feat_3d.permute(0, 2, 1, 3, 4).contiguous()

    return feat_3d


def coronal_2d_to_3d(
    feat_2d: torch.Tensor,
    batch_size: int,
    height: int,
) -> torch.Tensor:
    """
    将 coronal 方向 2D 特征还原为 3D 特征。

    参数
    ----
    feat_2d:
        coronal 方向 2D 特征，shape 为 [B*H, C, D, W]。

    batch_size:
        B。

    height:
        H。

    返回
    ----
    feat_3d:
        还原后的 3D 特征，shape 为 [B, C, D, H, W]。
    """
    _check_4d_tensor(feat_2d, name="feat_2d")

    if batch_size <= 0:
        raise ValueError(f"batch_size 必须大于 0，但当前为 {batch_size}")

    if height <= 0:
        raise ValueError(f"height 必须大于 0，但当前为 {height}")

    n, channels, depth, width = feat_2d.shape
    expected_n = batch_size * height

    if n != expected_n:
        raise ValueError(
            f"coronal feat_2d 的第 0 维应为 B*H={expected_n}，"
            f"但当前为 {n}，feat_2d shape = {tuple(feat_2d.shape)}"
        )

    feat_3d = feat_2d.reshape(batch_size, height, channels, depth, width)
    feat_3d = feat_3d.permute(0, 2, 3, 1, 4).contiguous()

    return feat_3d


def sagittal_2d_to_3d(
    feat_2d: torch.Tensor,
    batch_size: int,
    width: int,
) -> torch.Tensor:
    """
    将 sagittal 方向 2D 特征还原为 3D 特征。

    参数
    ----
    feat_2d:
        sagittal 方向 2D 特征，shape 为 [B*W, C, D, H]。

    batch_size:
        B。

    width:
        W。

    返回
    ----
    feat_3d:
        还原后的 3D 特征，shape 为 [B, C, D, H, W]。
    """
    _check_4d_tensor(feat_2d, name="feat_2d")

    if batch_size <= 0:
        raise ValueError(f"batch_size 必须大于 0，但当前为 {batch_size}")

    if width <= 0:
        raise ValueError(f"width 必须大于 0，但当前为 {width}")

    n, channels, depth, height = feat_2d.shape
    expected_n = batch_size * width

    if n != expected_n:
        raise ValueError(
            f"sagittal feat_2d 的第 0 维应为 B*W={expected_n}，"
            f"但当前为 {n}，feat_2d shape = {tuple(feat_2d.shape)}"
        )

    feat_3d = feat_2d.reshape(batch_size, width, channels, depth, height)
    feat_3d = feat_3d.permute(0, 2, 3, 4, 1).contiguous()

    return feat_3d


def make_multiview_slices(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从 3D 输入中构造 axial、coronal、sagittal 三个方向的 2D 切片。

    参数
    ----
    x:
        输入 3D 图像，shape 为 [B, 1, D, H, W]。
        这里也兼容 [B, C, D, H, W]，但项目默认 C=1。

    返回
    ----
    axial_slices:
        axial 方向切片，shape 为 [B*D, C, H, W]。

    coronal_slices:
        coronal 方向切片，shape 为 [B*H, C, D, W]。

    sagittal_slices:
        sagittal 方向切片，shape 为 [B*W, C, D, H]。
    """
    _check_5d_tensor(x, name="x")

    batch_size, channels, depth, height, width = x.shape

    # axial: 沿 D 方向切片，每张切片大小为 H x W。
    axial_slices = x.permute(0, 2, 1, 3, 4).contiguous()
    axial_slices = axial_slices.reshape(batch_size * depth, channels, height, width)

    # coronal: 沿 H 方向切片，每张切片大小为 D x W。
    coronal_slices = x.permute(0, 3, 1, 2, 4).contiguous()
    coronal_slices = coronal_slices.reshape(batch_size * height, channels, depth, width)

    # sagittal: 沿 W 方向切片，每张切片大小为 D x H。
    sagittal_slices = x.permute(0, 4, 1, 2, 3).contiguous()
    sagittal_slices = sagittal_slices.reshape(batch_size * width, channels, depth, height)

    return axial_slices, coronal_slices, sagittal_slices


class ConvBlock3D(nn.Module):
    """
    3D 基础卷积块。

    结构：
        Conv3d -> InstanceNorm3d -> LeakyReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        negative_slope: float = 0.01,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
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


class ResidualRefinement3D(nn.Module):
    """
    3D residual-style refinement 模块。

    结构：
        identity = x
        out = Conv3d -> InstanceNorm3d -> LeakyReLU
        out = Conv3d -> InstanceNorm3d
        out = LeakyReLU(out + identity)

    作用：
    在门控融合之后，对融合特征进行轻量级 3D 局部细化。
    """

    def __init__(
        self,
        channels: int,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.InstanceNorm3d(
            num_features=channels,
            affine=True,
        )
        self.act1 = nn.LeakyReLU(
            negative_slope=negative_slope,
            inplace=True,
        )

        self.conv2 = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.InstanceNorm3d(
            num_features=channels,
            affine=True,
        )
        self.act2 = nn.LeakyReLU(
            negative_slope=negative_slope,
            inplace=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        """
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity
        out = self.act2(out)

        return out


class MultiView2DProjector(nn.Module):
    """
    三视角 2D 特征投影器。

    功能：
    1. 将输入 3D 图像切成 axial、coronal、sagittal 三个方向的 2D 切片；
    2. 使用独立或共享的 2D encoder 提取三个方向的 2D 纹理特征；
    3. 将三方向 2D 特征还原为 [B, C, D, H, W]；
    4. 将三个方向的 3D 特征 concat；
    5. 使用 1x1x1 Conv3d 压缩通道；
    6. 输出统一的 3D 多视角纹理特征。

    输入：
        x: [B, in_channels, D, H, W]

    输出：
        out: [B, out_channels, D, H, W]
    """

    def __init__(
        self,
        in_channels: int = 1,
        encoder_base_channels: int = 16,
        encoder_out_channels: int = 16,
        out_channels: int = 16,
        num_blocks: int = 3,
        negative_slope: float = 0.01,
        separate_plane_encoders: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = int(in_channels)
        self.encoder_base_channels = int(encoder_base_channels)
        self.encoder_out_channels = int(encoder_out_channels)
        self.out_channels = int(out_channels)
        self.num_blocks = int(num_blocks)
        self.separate_plane_encoders = bool(separate_plane_encoders)

        encoder_kwargs = dict(
            in_channels=in_channels,
            base_channels=encoder_base_channels,
            out_channels=encoder_out_channels,
            num_blocks=num_blocks,
            negative_slope=negative_slope,
        )
        if self.separate_plane_encoders:
            self.encoder2d_axial = Simple2DTextureEncoder(**encoder_kwargs)
            self.encoder2d_coronal = Simple2DTextureEncoder(**encoder_kwargs)
            self.encoder2d_sagittal = Simple2DTextureEncoder(**encoder_kwargs)
            self.encoder2d = None
        else:
            self.encoder2d = Simple2DTextureEncoder(**encoder_kwargs)
            self.encoder2d_axial = None
            self.encoder2d_coronal = None
            self.encoder2d_sagittal = None

        # 三视角 concat 后通道数为 3 * encoder_out_channels。
        self.channel_compress = nn.Sequential(
            nn.Conv3d(
                in_channels=encoder_out_channels * 3,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
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

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        if self.separate_plane_encoders:
            shared_prefix = prefix + "encoder2d."
            shared_keys = [key for key in state_dict if key.startswith(shared_prefix)]
            for shared_key in shared_keys:
                suffix = shared_key[len(shared_prefix) :]
                for plane_name in ("axial", "coronal", "sagittal"):
                    target_key = prefix + f"encoder2d_{plane_name}." + suffix
                    state_dict.setdefault(target_key, state_dict[shared_key].clone())
                state_dict.pop(shared_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数
        ----
        x:
            输入 3D 图像，shape 为 [B, in_channels, D, H, W]。

        返回
        ----
        out:
            输出 3D 多视角纹理特征，shape 为 [B, out_channels, D, H, W]。
        """
        _check_5d_tensor(x, name="x")

        batch_size, channels, depth, height, width = x.shape

        if channels != self.in_channels:
            raise ValueError(
                f"MultiView2DProjector 期望输入通道数为 {self.in_channels}，"
                f"但当前输入通道数为 {channels}"
            )

        axial_slices, coronal_slices, sagittal_slices = make_multiview_slices(x)

        if self.separate_plane_encoders:
            axial_feat_2d = self.encoder2d_axial(axial_slices)
            coronal_feat_2d = self.encoder2d_coronal(coronal_slices)
            sagittal_feat_2d = self.encoder2d_sagittal(sagittal_slices)
        else:
            axial_feat_2d = self.encoder2d(axial_slices)
            coronal_feat_2d = self.encoder2d(coronal_slices)
            sagittal_feat_2d = self.encoder2d(sagittal_slices)

        axial_feat_3d = axial_2d_to_3d(
            feat_2d=axial_feat_2d,
            batch_size=batch_size,
            depth=depth,
        )

        coronal_feat_3d = coronal_2d_to_3d(
            feat_2d=coronal_feat_2d,
            batch_size=batch_size,
            height=height,
        )

        sagittal_feat_3d = sagittal_2d_to_3d(
            feat_2d=sagittal_feat_2d,
            batch_size=batch_size,
            width=width,
        )

        multiview_feat = torch.cat(
            [axial_feat_3d, coronal_feat_3d, sagittal_feat_3d],
            dim=1,
        )

        out = self.channel_compress(multiview_feat)

        return out


def make_adjacent_z_triplets(
    x: torch.Tensor,
    neighbor_radius: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return z/depth-axis triplets as [B*D, C, H, W]."""
    _check_5d_tensor(x, name="x")
    radius = int(neighbor_radius)
    if radius <= 0:
        raise ValueError(f"neighbor_radius must be positive, got {radius}")

    batch_size, channels, depth, height, width = x.shape
    indices = torch.arange(depth, device=x.device)
    previous_indices = torch.clamp(indices - radius, min=0)
    following_indices = torch.clamp(indices + radius, max=depth - 1)

    def flatten_z(value: torch.Tensor) -> torch.Tensor:
        return value.permute(0, 2, 1, 3, 4).contiguous().reshape(
            batch_size * depth,
            channels,
            height,
            width,
        )

    previous = flatten_z(torch.index_select(x, 2, previous_indices))
    center = flatten_z(x)
    following = flatten_z(torch.index_select(x, 2, following_indices))
    return previous, center, following


class ZAxisAdjacent2DProjector(nn.Module):
    """Build a 3D feature volume from gated z-1/z/z+1 axial slice features."""

    def __init__(
        self,
        in_channels: int = 1,
        encoder_base_channels: int = 16,
        encoder_out_channels: int = 16,
        out_channels: int = 16,
        num_blocks: int = 3,
        neighbor_radius: int = 1,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.encoder_out_channels = int(encoder_out_channels)
        self.out_channels = int(out_channels)
        self.neighbor_radius = int(neighbor_radius)
        self.triplet_encoder = AdjacentSliceGatedEncoder(
            in_channels=self.in_channels,
            base_channels=int(encoder_base_channels),
            out_channels=self.encoder_out_channels,
            num_blocks=int(num_blocks),
            negative_slope=float(negative_slope),
        )
        self.output_projection = nn.Sequential(
            nn.Conv3d(
                self.encoder_out_channels,
                self.out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.InstanceNorm3d(self.out_channels, affine=True),
            nn.LeakyReLU(negative_slope=float(negative_slope), inplace=True),
        )
        self.last_gate_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _check_5d_tensor(x, name="x")
        batch_size, channels, depth, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(
                f"ZAxisAdjacent2DProjector expected {self.in_channels} channels, got {channels}"
            )
        previous, center, following = make_adjacent_z_triplets(
            x,
            neighbor_radius=self.neighbor_radius,
        )
        fused_2d, weights = self.triplet_encoder(
            previous,
            center,
            following,
            return_weights=True,
        )
        self.last_gate_weights = weights
        fused_3d = axial_2d_to_3d(
            fused_2d,
            batch_size=batch_size,
            depth=depth,
        )
        if fused_3d.shape[2:] != (depth, height, width):
            raise RuntimeError(
                f"adjacent slice feature shape mismatch: {tuple(fused_3d.shape)}"
            )
        return self.output_projection(fused_3d)


AdjacentAxial2DProjector = ZAxisAdjacent2DProjector


class GatedFusion3D(nn.Module):
    """
    3D 主干特征与 2D 多视角特征的门控融合模块。

    输入：
        f3d: [B, C3, D, H, W]
        f2d: [B, C2, D, H, W]

    输出：
        out: [B, C3, D, H, W]

    融合公式：
        projected_f2d = Conv3d(f2d)
        gate = sigmoid(Conv3d(concat(f3d, projected_f2d)))
        fused = f3d + gate * projected_f2d
        out = ResidualRefinement3D(fused)
    """

    def __init__(
        self,
        channels_3d: int,
        channels_2d: int,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()

        if channels_3d <= 0:
            raise ValueError(f"channels_3d 必须大于 0，但当前为 {channels_3d}")

        if channels_2d <= 0:
            raise ValueError(f"channels_2d 必须大于 0，但当前为 {channels_2d}")

        self.channels_3d = int(channels_3d)
        self.channels_2d = int(channels_2d)

        # 将 2D 多视角特征投影到 3D 主干特征通道数。
        self.project_2d = nn.Sequential(
            nn.Conv3d(
                in_channels=channels_2d,
                out_channels=channels_3d,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.InstanceNorm3d(
                num_features=channels_3d,
                affine=True,
            ),
            nn.LeakyReLU(
                negative_slope=negative_slope,
                inplace=True,
            ),
        )

        # 根据 f3d 和 projected_f2d 生成空间-通道门控权重。
        self.gate_conv = nn.Conv3d(
            in_channels=channels_3d * 2,
            out_channels=channels_3d,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # 融合后的残差细化模块。
        self.refinement = ResidualRefinement3D(
            channels=channels_3d,
            negative_slope=negative_slope,
        )

    def forward(self, f3d: torch.Tensor, f2d: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数
        ----
        f3d:
            3D 主干特征，shape 为 [B, C3, D, H, W]。

        f2d:
            2D 多视角还原后的 3D 特征，shape 为 [B, C2, D, H, W]。
            如果空间尺寸与 f3d 不一致，会自动 trilinear interpolate 到 f3d 尺寸。

        返回
        ----
        out:
            融合后的 3D 特征，shape 为 [B, C3, D, H, W]。
        """
        _check_5d_tensor(f3d, name="f3d")
        _check_5d_tensor(f2d, name="f2d")

        if f3d.shape[1] != self.channels_3d:
            raise ValueError(
                f"f3d 通道数应为 {self.channels_3d}，"
                f"但当前为 {f3d.shape[1]}"
            )

        if f2d.shape[1] != self.channels_2d:
            raise ValueError(
                f"f2d 通道数应为 {self.channels_2d}，"
                f"但当前为 {f2d.shape[1]}"
            )

        target_size = f3d.shape[2:]

        # 如果空间尺寸不一致，将 2D 多视角特征插值到 f3d 的 D/H/W。
        if f2d.shape[2:] != target_size:
            f2d = F.interpolate(
                f2d,
                size=target_size,
                mode="trilinear",
                align_corners=False,
            )

        projected_f2d = self.project_2d(f2d)

        gate_input = torch.cat([f3d, projected_f2d], dim=1)
        gate = torch.sigmoid(self.gate_conv(gate_input))

        fused = f3d + gate * projected_f2d
        out = self.refinement(fused)

        return out


if __name__ == "__main__":
    # 简单自检：验证三视角切片、2D->3D 还原、多视角投影和门控融合是否能正常运行。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 2
    in_channels = 1
    depth = 64
    height = 64
    width = 64

    x = torch.randn(batch_size, in_channels, depth, height, width).to(device)

    axial_slices, coronal_slices, sagittal_slices = make_multiview_slices(x)

    print("输入 x shape:", tuple(x.shape))
    print("axial_slices shape:", tuple(axial_slices.shape))
    print("coronal_slices shape:", tuple(coronal_slices.shape))
    print("sagittal_slices shape:", tuple(sagittal_slices.shape))

    # 测试三种 2D 特征还原函数。
    dummy_axial_feat = torch.randn(batch_size * depth, 16, height, width).to(device)
    dummy_coronal_feat = torch.randn(batch_size * height, 16, depth, width).to(device)
    dummy_sagittal_feat = torch.randn(batch_size * width, 16, depth, height).to(device)

    axial_3d = axial_2d_to_3d(
        feat_2d=dummy_axial_feat,
        batch_size=batch_size,
        depth=depth,
    )
    coronal_3d = coronal_2d_to_3d(
        feat_2d=dummy_coronal_feat,
        batch_size=batch_size,
        height=height,
    )
    sagittal_3d = sagittal_2d_to_3d(
        feat_2d=dummy_sagittal_feat,
        batch_size=batch_size,
        width=width,
    )

    print("axial_3d shape:", tuple(axial_3d.shape))
    print("coronal_3d shape:", tuple(coronal_3d.shape))
    print("sagittal_3d shape:", tuple(sagittal_3d.shape))

    # 测试 MultiView2DProjector。
    projector = MultiView2DProjector(
        in_channels=1,
        encoder_base_channels=16,
        encoder_out_channels=16,
        out_channels=16,
        num_blocks=3,
    ).to(device)

    with torch.no_grad():
        f2d = projector(x)

    print("MultiView2DProjector 输出 f2d shape:", tuple(f2d.shape))

    # 测试 GatedFusion3D。
    f3d = torch.randn(batch_size, 32, depth, height, width).to(device)

    fusion = GatedFusion3D(
        channels_3d=32,
        channels_2d=16,
    ).to(device)

    with torch.no_grad():
        fused = fusion(f3d, f2d)

    print("GatedFusion3D 输出 fused shape:", tuple(fused.shape))

    # 测试 f2d 空间尺寸不一致时的自动插值。
    f2d_small = torch.randn(batch_size, 16, 32, 32, 32).to(device)

    with torch.no_grad():
        fused_interp = fusion(f3d, f2d_small)

    print("GatedFusion3D 插值融合输出 fused_interp shape:", tuple(fused_interp.shape))
