# -*- coding: utf-8 -*-
"""Stage-1 Dice, Tversky, and boundary loss for 3D segmentation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LABEL_KEY = "label"
MASK_ALIAS_KEY = "mask"


@dataclass(frozen=True)
class DynamicLossWeights:
    dice: float
    tversky: float
    boundary: float
    sdf: float


def _as_target_long(label: torch.Tensor) -> torch.Tensor:
    if label.ndim == 5 and label.shape[1] == 1:
        label = label[:, 0]
    return label.long()


def _one_hot(label: torch.Tensor, num_classes: int) -> torch.Tensor:
    label = label.clamp(min=0, max=int(num_classes) - 1)
    one_hot = F.one_hot(label.long(), num_classes=int(num_classes))
    return one_hot.permute(0, 4, 1, 2, 3).contiguous().float()


def _schedule_boundary_cosine(
    epoch: int,
    start_epoch: int,
    end_epoch: int,
    max_weight: float,
) -> float:
    """Cosine ramp: 0 before start, max_weight after end."""
    if int(epoch) < int(start_epoch):
        return 0.0
    if int(epoch) > int(end_epoch) or int(end_epoch) <= int(start_epoch):
        return float(max_weight)
    progress = (float(epoch) - float(start_epoch)) / float(end_epoch - start_epoch)
    return float(max_weight) * (1.0 - math.cos(math.pi * progress)) / 2.0


def _schedule_size_weight_cosine(
    epoch: int,
    start_epoch: int,
    end_epoch: int,
) -> float:
    """Cosine ramp from no size weighting (0) to full weighting (1)."""
    if int(epoch) < int(start_epoch):
        return 0.0
    if int(epoch) >= int(end_epoch) or int(end_epoch) <= int(start_epoch):
        return 1.0
    progress = (float(epoch) - float(start_epoch)) / float(end_epoch - start_epoch)
    return (1.0 - math.cos(math.pi * progress)) / 2.0


class AdaptiveDynamicSegLoss(nn.Module):
    """
    L = L_Dice + 0.4 * L_Tversky + lambda_boundary(epoch) * L_Boundary.

    Tversky uses alpha for false positives and beta for false negatives.
    Boundary weight follows the cosine schedule shown in the stage-1 design.
    """

    def __init__(
        self,
        num_classes: int,
        include_background: bool = False,
        dice_weight: float = 1.0,
        tversky_weight: float = 0.4,
        boundary_start_epoch: int = 40,
        boundary_end_epoch: int = 120,
        boundary_max_weight: float = 0.02,
        sdf_start_epoch: int = 20,
        sdf_end_epoch: int = 80,
        sdf_max_weight: float = 0.1,
        size_weight_enabled: bool = False,
        size_weight_start_epoch: int = 0,
        size_weight_end_epoch: int = 20,
        tversky_alpha: float = 0.6,
        tversky_beta: float = 0.4,
        boundary_voxel_boost: float = 5.0,
        dist_sigma: float = 0.2,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.include_background = bool(include_background)
        self.dice_weight = float(dice_weight)
        self.tversky_weight = float(tversky_weight)
        self.boundary_start_epoch = int(boundary_start_epoch)
        self.boundary_end_epoch = int(boundary_end_epoch)
        self.boundary_max_weight = float(boundary_max_weight)
        self.sdf_start_epoch = int(sdf_start_epoch)
        self.sdf_end_epoch = int(sdf_end_epoch)
        self.sdf_max_weight = float(sdf_max_weight)
        self.size_weight_enabled = bool(size_weight_enabled)
        self.size_weight_start_epoch = int(size_weight_start_epoch)
        self.size_weight_end_epoch = int(size_weight_end_epoch)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)
        self.boundary_voxel_boost = float(boundary_voxel_boost)
        self.dist_sigma = float(dist_sigma)
        self.eps = float(eps)

    def get_weights(self, epoch: int) -> DynamicLossWeights:
        return DynamicLossWeights(
            dice=self.dice_weight,
            tversky=self.tversky_weight,
            boundary=_schedule_boundary_cosine(
                epoch=epoch,
                start_epoch=self.boundary_start_epoch,
                end_epoch=self.boundary_end_epoch,
                max_weight=self.boundary_max_weight,
            ),
            sdf=_schedule_boundary_cosine(
                epoch=epoch,
                start_epoch=self.sdf_start_epoch,
                end_epoch=self.sdf_end_epoch,
                max_weight=self.sdf_max_weight,
            ),
        )

    def _get_probs_and_target(
        self,
        logits: torch.Tensor,
        label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_long = _as_target_long(label)
        if self.num_classes <= 1:
            probs = torch.sigmoid(logits)
            target_bin = (target_long > 0).float().unsqueeze(1)
            return probs, target_bin, target_long
        probs = torch.softmax(logits, dim=1)
        return probs, _one_hot(target_long, self.num_classes), target_long

    def _select_channels(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_classes <= 1 or self.include_background:
            return x
        return x[:, 1:]

    def _dice_loss(
        self,
        probs: torch.Tensor,
        target: torch.Tensor,
        size_weight: torch.Tensor,
    ) -> torch.Tensor:
        probs = self._select_channels(probs.float())
        target = self._select_channels(target.float())
        dims = (2, 3, 4)
        intersection = torch.sum(size_weight * probs * target, dim=dims)
        denominator = torch.sum(size_weight * (probs + target), dim=dims)
        score = (2.0 * intersection + 1.0) / (denominator + 1.0 + self.eps)
        return 1.0 - score.mean()

    def _tversky_loss(
        self,
        probs: torch.Tensor,
        target: torch.Tensor,
        size_weight: torch.Tensor,
    ) -> torch.Tensor:
        probs = self._select_channels(probs.float())
        target = self._select_channels(target.float())
        dims = (2, 3, 4)
        tp = torch.sum(size_weight * probs * target, dim=dims)
        fp = torch.sum(size_weight * probs * (1.0 - target), dim=dims)
        fn = torch.sum(size_weight * (1.0 - probs) * target, dim=dims)
        score = (tp + 1.0) / (
            tp
            + self.tversky_alpha * fp
            + self.tversky_beta * fn
            + 1.0
            + self.eps
        )
        return 1.0 - score.mean()

    def _bce_or_ce_map(
        self,
        logits: torch.Tensor,
        target_long: torch.Tensor,
    ) -> torch.Tensor:
        if self.num_classes <= 1:
            target = (target_long > 0).float().unsqueeze(1)
            return F.binary_cross_entropy_with_logits(
                logits.float(),
                target,
                reduction="none",
            )
        return F.cross_entropy(
            logits.float(),
            target_long,
            reduction="none",
        ).unsqueeze(1)

    def _boundary_loss(
        self,
        logits: torch.Tensor,
        target_long: torch.Tensor,
        boundary: Optional[torch.Tensor],
        dist_map: Optional[torch.Tensor],
        size_weight: torch.Tensor,
    ) -> torch.Tensor:
        if boundary is None:
            target = (target_long > 0).float().unsqueeze(1)
            dilated = F.max_pool3d(target, kernel_size=3, stride=1, padding=1)
            eroded = -F.max_pool3d(-target, kernel_size=3, stride=1, padding=1)
            boundary = (dilated - eroded) > 0

        if boundary.ndim == 4:
            boundary = boundary.unsqueeze(1)
        if tuple(boundary.shape[2:]) != tuple(logits.shape[2:]):
            boundary = F.interpolate(
                boundary.float(),
                size=logits.shape[2:],
                mode="nearest",
            )

        weight = self.boundary_voxel_boost * (boundary.float() > 0).float()
        if dist_map is not None:
            if dist_map.ndim == 4:
                dist_map = dist_map.unsqueeze(1)
            if tuple(dist_map.shape[2:]) != tuple(logits.shape[2:]):
                dist_map = F.interpolate(
                    dist_map.float(),
                    size=logits.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )
            weight = weight + torch.exp(
                -torch.abs(dist_map.float()) / max(self.dist_sigma, self.eps)
            )

        loss_map = self._bce_or_ce_map(logits, target_long)
        weight = weight * size_weight
        return (loss_map * weight).sum() / weight.sum().clamp_min(self.eps)

    def _get_size_weight(
        self,
        batch: Dict[str, torch.Tensor],
        logits: torch.Tensor,
        target_long: torch.Tensor,
        epoch: int,
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        if not self.size_weight_enabled:
            unit = logits.new_ones((logits.shape[0], 1, *logits.shape[2:]))
            return unit, 0.0, logits.new_tensor(1.0)

        raw_weight = batch.get("component_weight", None)
        if raw_weight is None:
            raise KeyError(
                "size weighting is enabled but batch is missing 'component_weight'"
            )
        if raw_weight.ndim == 4:
            raw_weight = raw_weight.unsqueeze(1)
        if raw_weight.ndim != 5 or raw_weight.shape[1] != 1:
            raise ValueError(
                "component_weight must have shape [B,1,D,H,W] or [B,D,H,W]"
            )
        if tuple(raw_weight.shape[2:]) != tuple(logits.shape[2:]):
            raw_weight = F.interpolate(
                raw_weight.float(),
                size=logits.shape[2:],
                mode="nearest",
            )
        raw_weight = raw_weight.to(device=logits.device, dtype=torch.float32)
        if not torch.isfinite(raw_weight).all():
            raise ValueError("component_weight contains NaN or Inf")
        if torch.any(raw_weight < 1.0):
            raise ValueError("component_weight values must be >= 1")

        ramp = _schedule_size_weight_cosine(
            epoch=epoch,
            start_epoch=self.size_weight_start_epoch,
            end_epoch=self.size_weight_end_epoch,
        )
        effective_weight = 1.0 + float(ramp) * (raw_weight - 1.0)
        foreground = target_long > 0
        if torch.any(foreground):
            foreground_mean = effective_weight[:, 0][foreground].mean()
        else:
            foreground_mean = logits.new_tensor(1.0)
        return effective_weight, float(ramp), foreground_mean

    def _sdf_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        sdf_pred = outputs.get("sdf", None)
        if sdf_pred is None:
            return outputs["mask_logits"].new_tensor(0.0)

        sdf_target = batch.get("dist_map", batch.get("sdf", None))
        if sdf_target is None:
            return sdf_pred.new_tensor(0.0)
        if sdf_target.ndim == 4:
            sdf_target = sdf_target.unsqueeze(1)
        if tuple(sdf_target.shape[2:]) != tuple(sdf_pred.shape[2:]):
            sdf_target = F.interpolate(
                sdf_target.float(),
                size=sdf_pred.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        return F.smooth_l1_loss(
            sdf_pred.float(),
            sdf_target.float().clamp(-1.0, 1.0),
            reduction="mean",
        )

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if "mask_logits" not in outputs:
            raise KeyError("outputs must contain 'mask_logits'")
        logits = outputs["mask_logits"]
        label = batch.get(LABEL_KEY, batch.get(MASK_ALIAS_KEY, None))
        if label is None:
            raise KeyError("batch must contain label/mask")
        if label.ndim == 5 and tuple(label.shape[2:]) != tuple(logits.shape[2:]):
            label = F.interpolate(
                label.float(),
                size=logits.shape[2:],
                mode="nearest",
            ).long()

        weights = self.get_weights(epoch)
        probs, target, target_long = self._get_probs_and_target(logits, label)
        size_weight, size_weight_ramp, size_weight_fg_mean = self._get_size_weight(
            batch=batch,
            logits=logits,
            target_long=target_long,
            epoch=epoch,
        )
        dice = self._dice_loss(probs, target, size_weight)
        tversky = self._tversky_loss(probs, target, size_weight)
        boundary = self._boundary_loss(
            logits=logits,
            target_long=target_long,
            boundary=batch.get("boundary", None),
            dist_map=batch.get("dist_map", None),
            size_weight=size_weight,
        )
        sdf = self._sdf_loss(outputs, batch)
        total = (
            weights.dice * dice
            + weights.tversky * tversky
            + weights.boundary * boundary
            + weights.sdf * sdf
        )
        return total, {
            "total": total.detach(),
            "dice": dice.detach(),
            "tversky": tversky.detach(),
            "boundary": boundary.detach(),
            "sdf": sdf.detach(),
            "w_dice": logits.new_tensor(weights.dice),
            "w_tversky": logits.new_tensor(weights.tversky),
            "w_boundary": logits.new_tensor(weights.boundary),
            "w_sdf": logits.new_tensor(weights.sdf),
            "size_weight_ramp": logits.new_tensor(size_weight_ramp),
            "size_weight_fg_mean": size_weight_fg_mean.detach(),
        }


DynamicAntagonisticLoss = AdaptiveDynamicSegLoss
