# -*- coding: utf-8 -*-
"""
src/fusion.py

Memory-light 2D-3D fusion modules.

Only the z-axis adjacent-slice 2D path is kept:
1. Build z-1/z/z+1 axial triplets from [B, C, D, H, W].
2. Encode each triplet with AdjacentSliceGatedEncoder.
3. Restore [B*D, C, H, W] features to [B, C, D, H, W].
4. Fuse z-axis 2D texture features with the 3D backbone feature via GatedFusion3D.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from .modules2d import AdjacentSliceGatedEncoder
except ImportError:
    from modules2d import AdjacentSliceGatedEncoder


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
        slice_chunk_size: int = 8,
        use_checkpoint: bool = True,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.encoder_out_channels = int(encoder_out_channels)
        self.out_channels = int(out_channels)
        self.neighbor_radius = int(neighbor_radius)
        self.slice_chunk_size = max(int(slice_chunk_size), 1)
        self.use_checkpoint = bool(use_checkpoint)
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
        feature_chunks = []
        weight_chunks = []
        for start in range(0, depth, self.slice_chunk_size):
            end = min(start + self.slice_chunk_size, depth)
            indices = torch.arange(start, end, device=x.device)
            previous_indices = torch.clamp(indices - self.neighbor_radius, min=0)
            following_indices = torch.clamp(
                indices + self.neighbor_radius,
                max=depth - 1,
            )

            def flatten_chunk(selected: torch.Tensor) -> torch.Tensor:
                return selected.permute(0, 2, 1, 3, 4).contiguous().reshape(
                    batch_size * (end - start),
                    channels,
                    height,
                    width,
                )

            previous = flatten_chunk(torch.index_select(x, 2, previous_indices))
            center = flatten_chunk(torch.index_select(x, 2, indices))
            following = flatten_chunk(torch.index_select(x, 2, following_indices))

            def encode_triplet(
                previous_chunk: torch.Tensor,
                center_chunk: torch.Tensor,
                following_chunk: torch.Tensor,
            ):
                return self.triplet_encoder(
                    previous_chunk,
                    center_chunk,
                    following_chunk,
                    return_weights=True,
                )

            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                fused_2d, weights = checkpoint(
                    encode_triplet,
                    previous,
                    center,
                    following,
                    use_reentrant=False,
                )
            else:
                fused_2d, weights = encode_triplet(previous, center, following)

            chunk_depth = end - start
            feature_chunks.append(
                fused_2d.reshape(
                    batch_size,
                    chunk_depth,
                    self.encoder_out_channels,
                    height,
                    width,
                )
            )
            weight_chunks.append(
                weights.detach().reshape(
                    batch_size,
                    chunk_depth,
                    weights.shape[1],
                    height,
                    width,
                )
            )

        fused_depth_first = torch.cat(feature_chunks, dim=1)
        fused_3d = fused_depth_first.permute(0, 2, 1, 3, 4).contiguous()
        self.last_gate_weights = torch.cat(weight_chunks, dim=1).reshape(
            batch_size * depth,
            weight_chunks[0].shape[2],
            height,
            width,
        )
        if fused_3d.shape[2:] != (depth, height, width):
            raise RuntimeError(
                f"adjacent slice feature shape mismatch: {tuple(fused_3d.shape)}"
            )
        return self.output_projection(fused_3d)


AdjacentAxial2DProjector = ZAxisAdjacent2DProjector


class GatedFusion3D(nn.Module):
    """
    3D 主干特征与 z-axis 2D 特征的门控融合模块。

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

        # 将 z-axis 2D 特征投影到 3D 主干特征通道数。
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
            z-axis 2D 还原后的 3D 特征，shape 为 [B, C2, D, H, W]。
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

        # 如果空间尺寸不一致，将 z-axis 2D 特征插值到 f3d 的 D/H/W。
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    depth = height = width = 64
    x = torch.randn(batch_size, 1, depth, height, width, device=device)

    projector = ZAxisAdjacent2DProjector(
        in_channels=1,
        encoder_base_channels=16,
        encoder_out_channels=16,
        out_channels=16,
        num_blocks=3,
    ).to(device)
    fusion = GatedFusion3D(channels_3d=32, channels_2d=16).to(device)
    f3d = torch.randn(batch_size, 32, depth, height, width, device=device)

    with torch.no_grad():
        f2d = projector(x)
        fused = fusion(f3d, f2d)

    print("input x shape:", tuple(x.shape))
    print("z-axis f2d shape:", tuple(f2d.shape))
    print("fused shape:", tuple(fused.shape))
