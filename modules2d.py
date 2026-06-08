# -*- coding: utf-8 -*-
"""Lightweight 2D texture encoder used by the multiview fusion branch."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvNormAct2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                int(in_channels),
                int(out_channels),
                kernel_size=int(kernel_size),
                stride=1,
                padding=int(padding),
                bias=False,
            ),
            nn.InstanceNorm2d(int(out_channels), affine=True),
            nn.LeakyReLU(negative_slope=float(negative_slope), inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock2D(nn.Module):
    def __init__(self, channels: int, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.conv1 = ConvNormAct2D(channels, channels, negative_slope=negative_slope)
        self.conv2 = nn.Sequential(
            nn.Conv2d(int(channels), int(channels), kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(int(channels), affine=True),
        )
        self.act = nn.LeakyReLU(negative_slope=float(negative_slope), inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + x)


class Simple2DTextureEncoder(nn.Module):
    """
    Lightweight 2D encoder for one anatomical plane.

    Input:  [N, in_channels, H, W]
    Output: [N, out_channels, H, W]
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        out_channels: int = 16,
        num_blocks: int = 3,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        base_channels = int(base_channels)
        out_channels = int(out_channels)
        num_blocks = max(int(num_blocks), 1)

        blocks = [ConvNormAct2D(int(in_channels), base_channels, negative_slope=negative_slope)]
        blocks.extend(ResidualBlock2D(base_channels, negative_slope=negative_slope) for _ in range(num_blocks - 1))
        if out_channels != base_channels:
            blocks.append(ConvNormAct2D(base_channels, out_channels, kernel_size=1, padding=0, negative_slope=negative_slope))

        self.encoder = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must be [N, C, H, W], got shape={tuple(x.shape)}")
        return self.encoder(x)


class AdjacentSliceGatedEncoder(nn.Module):
    """Encode previous, center, and next slices with one shared encoder."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        out_channels: int = 16,
        num_blocks: int = 3,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.encoder = Simple2DTextureEncoder(
            in_channels=self.in_channels,
            base_channels=int(base_channels),
            out_channels=self.out_channels,
            num_blocks=int(num_blocks),
            negative_slope=float(negative_slope),
        )
        self.gate = nn.Conv2d(self.out_channels * 3, 3, kernel_size=1, bias=True)
        self.refine = ResidualBlock2D(self.out_channels, negative_slope=negative_slope)

    def forward(
        self,
        previous: torch.Tensor,
        center: torch.Tensor,
        following: torch.Tensor,
        return_weights: bool = False,
    ):
        for name, value in (
            ("previous", previous),
            ("center", center),
            ("following", following),
        ):
            if value.ndim != 4:
                raise ValueError(f"{name} must be [N, C, H, W], got {tuple(value.shape)}")
            if value.shape[1] != self.in_channels:
                raise ValueError(
                    f"{name} channels must be {self.in_channels}, got {value.shape[1]}"
                )
        if previous.shape != center.shape or following.shape != center.shape:
            raise ValueError(
                "previous, center, and following slices must have identical shapes"
            )

        previous_feature = self.encoder(previous)
        center_feature = self.encoder(center)
        following_feature = self.encoder(following)
        stacked = torch.stack(
            [previous_feature, center_feature, following_feature],
            dim=1,
        )
        gate_input = torch.cat(
            [previous_feature, center_feature, following_feature],
            dim=1,
        )
        weights = torch.softmax(self.gate(gate_input), dim=1)
        fused = torch.sum(stacked * weights.unsqueeze(2), dim=1)
        fused = self.refine(fused)
        if return_weights:
            return fused, weights
        return fused
