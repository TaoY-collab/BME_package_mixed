# -*- coding: utf-8 -*-
"""
losses_dynamic.py

Task-Adaptive 3D Medical Segmentation Dynamic Losses

核心思想：
1. Warmup 定位期：只使用 Dice + CE/BCE，避免早期边界/SDF/Tversky 过强导致训练不稳定；
2. 结构强化期：Dice 与 CE/BCE 权重逐步下降，Tversky / Boundary / SDF / Core 权重逐步上升；
3. 精修 + 难例挖掘期：启用 component_weight，对小连通域/难例区域进行反向加权；
4. 支持二分类与多分类；
5. 支持 mask_logits / sdf / core_logits 多任务输出。

模型输出 outputs 推荐格式：
{
    "mask_logits": [B, 1或C, D, H, W],
    "sdf":         [B, 1, D, H, W],          # 可选
    "core_logits": [B, 1, D, H, W],          # 可选
}

batch 推荐格式：
{
    "image":            [B, 1, D, H, W],
    "label" 或 "mask":  [B, 1, D, H, W],
    "boundary":         [B, 1, D, H, W],      # 可选，但建议提供
    "sdf" 或 "dist_map": [B, 1, D, H, W],     # 可选，但建议提供
    "core":             [B, 1, D, H, W],      # 可选
    "component_weight": [B, 1, D, H, W],      # Stage2 可选
}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LABEL_KEY = "label"
MASK_ALIAS_KEY = "mask"


@dataclass(frozen=True)
class DynamicLossWeights:
    """当前 epoch 实际使用的动态权重。"""

    dice: float
    ce: float
    tversky: float
    boundary: float
    sdf: float
    core: float
    stage2_active: bool


def _as_target_long(label: torch.Tensor) -> torch.Tensor:
    """
    将 label 统一为 [B, D, H, W] 的 long tensor。

    输入支持：
    - [B, 1, D, H, W]
    - [B, D, H, W]
    """
    if label.ndim == 5 and label.shape[1] == 1:
        label = label[:, 0]
    return label.long()


def _one_hot(label: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    label: [B, D, H, W]
    return: [B, C, D, H, W]
    """
    label = label.clamp(min=0, max=int(num_classes) - 1)
    one_hot = F.one_hot(label.long(), num_classes=int(num_classes))
    return one_hot.permute(0, 4, 1, 2, 3).contiguous().float()


def _schedule_linear(
    epoch: int,
    warmup_epochs: int,
    max_epochs: int,
    start: float,
    end: float,
) -> float:
    """warmup 后从 start 线性变化到 end。"""
    if int(epoch) < int(warmup_epochs):
        return float(start)

    denom = max(float(max_epochs - warmup_epochs), 1.0)
    p = float(epoch - warmup_epochs) / denom
    p = max(0.0, min(1.0, p))
    return float(start + (end - start) * p)


class AdaptiveDynamicSegLoss(nn.Module):
    """
    三阶段动态复合损失。

    阶段 1：Warmup 定位期
        total = 1.0 * Dice + 1.0 * CE/BCE

    阶段 2 / 3：结构强化与精修期
        total =
            w_dice(epoch)    * Dice
          + w_ce(epoch)      * CE/BCE
          + w_tversky(epoch) * Tversky
          + w_boundary(epoch)* Boundary-aware CE/BCE
          + w_sdf(epoch)     * SDF regression
          + w_core(epoch)    * Core auxiliary loss

    其中 Stage2 开始后，如果 batch 中包含 component_weight，则 Dice/CE/Tversky 会使用小连通域反向加权。
    """

    def __init__(
        self,
        num_classes: int,
        max_epochs: int,
        warmup_epochs: int = 15,
        stage2_start_epoch: int = 80,
        include_background: bool = False,
        dice_weight_start: float = 1.0,
        dice_weight_end: float = 0.35,
        ce_weight_start: float = 1.0,
        ce_weight_end: float = 0.25,
        tversky_weight_start: float = 0.0,
        tversky_weight_end: float = 1.2,
        boundary_weight_start: float = 0.0,
        boundary_weight_end: float = 0.8,
        sdf_weight_start: float = 0.0,
        sdf_weight_end: float = 0.3,
        core_weight_start: float = 0.0,
        core_weight_end: float = 0.3,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        boundary_voxel_boost: float = 5.0,
        dist_sigma: float = 0.2,
        stage2_weight_power: float = 1.0,
        max_stage2_weight: float = 8.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.num_classes = int(num_classes)
        self.max_epochs = int(max_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.stage2_start_epoch = int(stage2_start_epoch)
        self.include_background = bool(include_background)

        self.dice_weight_start = float(dice_weight_start)
        self.dice_weight_end = float(dice_weight_end)
        self.ce_weight_start = float(ce_weight_start)
        self.ce_weight_end = float(ce_weight_end)
        self.tversky_weight_start = float(tversky_weight_start)
        self.tversky_weight_end = float(tversky_weight_end)
        self.boundary_weight_start = float(boundary_weight_start)
        self.boundary_weight_end = float(boundary_weight_end)
        self.sdf_weight_start = float(sdf_weight_start)
        self.sdf_weight_end = float(sdf_weight_end)
        self.core_weight_start = float(core_weight_start)
        self.core_weight_end = float(core_weight_end)

        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)
        self.boundary_voxel_boost = float(boundary_voxel_boost)
        self.dist_sigma = float(dist_sigma)
        self.stage2_weight_power = float(stage2_weight_power)
        self.max_stage2_weight = float(max_stage2_weight)
        self.eps = float(eps)

    def get_weights(self, epoch: int) -> DynamicLossWeights:
        """返回当前 epoch 的动态权重。"""
        if int(epoch) < self.warmup_epochs:
            return DynamicLossWeights(
                dice=self.dice_weight_start,
                ce=self.ce_weight_start,
                tversky=0.0,
                boundary=0.0,
                sdf=0.0,
                core=0.0,
                stage2_active=False,
            )

        return DynamicLossWeights(
            dice=_schedule_linear(epoch, self.warmup_epochs, self.max_epochs, self.dice_weight_start, self.dice_weight_end),
            ce=_schedule_linear(epoch, self.warmup_epochs, self.max_epochs, self.ce_weight_start, self.ce_weight_end),
            tversky=_schedule_linear(epoch, self.warmup_epochs, self.max_epochs, self.tversky_weight_start, self.tversky_weight_end),
            boundary=_schedule_linear(epoch, self.warmup_epochs, self.max_epochs, self.boundary_weight_start, self.boundary_weight_end),
            sdf=_schedule_linear(epoch, self.warmup_epochs, self.max_epochs, self.sdf_weight_start, self.sdf_weight_end),
            core=_schedule_linear(epoch, self.warmup_epochs, self.max_epochs, self.core_weight_start, self.core_weight_end),
            stage2_active=int(epoch) >= self.stage2_start_epoch,
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
        target_oh = _one_hot(target_long, self.num_classes)
        return probs, target_oh, target_long

    def _select_channels(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_classes <= 1:
            return x
        if self.include_background:
            return x
        return x[:, 1:]

    def _dice_loss(
        self,
        probs: torch.Tensor,
        target: torch.Tensor,
        voxel_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        probs = self._select_channels(probs.float())
        target = self._select_channels(target.float())

        if voxel_weight is None:
            voxel_weight = torch.ones_like(probs[:, :1])

        dims = (2, 3, 4)
        intersection = torch.sum(probs * target * voxel_weight, dim=dims)
        denom = torch.sum((probs + target) * voxel_weight, dim=dims)
        dice = (2.0 * intersection + 1.0) / (denom + 1.0 + self.eps)
        return 1.0 - dice.mean()

    def _tversky_loss(
        self,
        probs: torch.Tensor,
        target: torch.Tensor,
        voxel_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        probs = self._select_channels(probs.float())
        target = self._select_channels(target.float())

        if voxel_weight is None:
            voxel_weight = torch.ones_like(probs[:, :1])

        dims = (2, 3, 4)
        tp = torch.sum(probs * target * voxel_weight, dim=dims)
        fp = torch.sum(probs * (1.0 - target) * voxel_weight, dim=dims)
        fn = torch.sum((1.0 - probs) * target * voxel_weight, dim=dims)
        score = (tp + 1.0) / (tp + self.tversky_alpha * fp + self.tversky_beta * fn + 1.0 + self.eps)
        return 1.0 - score.mean()

    def _ce_or_bce_loss(
        self,
        logits: torch.Tensor,
        target_long: torch.Tensor,
        voxel_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.num_classes <= 1:
            target_bin = (target_long > 0).float().unsqueeze(1)
            loss_map = F.binary_cross_entropy_with_logits(
                logits.float(),
                target_bin,
                reduction="none",
            )
        else:
            loss_map = F.cross_entropy(
                logits.float(),
                target_long,
                reduction="none",
            ).unsqueeze(1)

        if voxel_weight is not None:
            loss_map = loss_map * voxel_weight

        return loss_map.mean()

    def _boundary_loss(
        self,
        logits: torch.Tensor,
        target_long: torch.Tensor,
        boundary: Optional[torch.Tensor],
        dist_map: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        边界感知损失：
        - boundary map 直接提高边界体素权重；
        - dist_map 使用 exp(-abs(dist_map)/sigma)，越靠近零等值面权重越高。
        """
        if boundary is None and dist_map is None:
            return logits.new_tensor(0.0)

        weight = torch.ones_like(logits[:, :1], dtype=torch.float32)

        if boundary is not None:
            if boundary.ndim == 4:
                boundary = boundary.unsqueeze(1)
            if tuple(boundary.shape[2:]) != tuple(logits.shape[2:]):
                boundary = F.interpolate(boundary.float(), size=logits.shape[2:], mode="nearest")
            weight = weight + self.boundary_voxel_boost * boundary.float().clamp(0.0, 1.0)

        if dist_map is not None:
            if dist_map.ndim == 4:
                dist_map = dist_map.unsqueeze(1)
            if tuple(dist_map.shape[2:]) != tuple(logits.shape[2:]):
                dist_map = F.interpolate(dist_map.float(), size=logits.shape[2:], mode="trilinear", align_corners=False)
            weight = weight + torch.exp(-torch.abs(dist_map.float()) / max(self.dist_sigma, self.eps))

        return self._ce_or_bce_loss(logits, target_long, voxel_weight=weight)

    def _sdf_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if "sdf" not in outputs:
            return next(iter(outputs.values())).new_tensor(0.0)

        sdf_target = batch.get("sdf", batch.get("dist_map", None))
        if sdf_target is None:
            return outputs["sdf"].new_tensor(0.0)

        sdf_pred = outputs["sdf"]
        if sdf_target.ndim == 4:
            sdf_target = sdf_target.unsqueeze(1)
        if tuple(sdf_pred.shape[2:]) != tuple(sdf_target.shape[2:]):
            sdf_target = F.interpolate(
                sdf_target.float(),
                size=sdf_pred.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        return F.smooth_l1_loss(sdf_pred.float(), sdf_target.float(), reduction="mean")

    def _sdf_mask_consistency_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        可选的一致性约束：从 sdf_pred 反推 mask probability。
        默认并入 sdf loss 使用，权重仍由 w_sdf 控制。
        """
        if "sdf" not in outputs:
            return next(iter(outputs.values())).new_tensor(0.0)

        sdf_pred = outputs["sdf"].float()
        if target.ndim == 4:
            target = target.unsqueeze(1)
        if tuple(target.shape[2:]) != tuple(sdf_pred.shape[2:]):
            target = F.interpolate(target.float(), size=sdf_pred.shape[2:], mode="nearest")
        target = (target > 0).float()

        # SDF 约定：内部为负，外部为正；因此 -sdf 越大越像前景。
        prob_from_sdf = torch.sigmoid(-10.0 * sdf_pred)
        dims = tuple(range(1, prob_from_sdf.ndim))
        inter = torch.sum(prob_from_sdf * target, dim=dims)
        denom = torch.sum(prob_from_sdf + target, dim=dims)
        dice = (2.0 * inter + 1.0) / (denom + 1.0 + self.eps)
        return 1.0 - dice.mean()

    def _core_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if "core_logits" not in outputs or "core" not in batch:
            return next(iter(outputs.values())).new_tensor(0.0)

        core_logits = outputs["core_logits"]
        core = batch["core"].float()
        if core.ndim == 4:
            core = core.unsqueeze(1)
        if tuple(core_logits.shape[2:]) != tuple(core.shape[2:]):
            core = F.interpolate(core, size=core_logits.shape[2:], mode="nearest")

        bce = F.binary_cross_entropy_with_logits(core_logits.float(), core, reduction="mean")
        prob = torch.sigmoid(core_logits.float())
        inter = torch.sum(prob * core, dim=(1, 2, 3, 4))
        denom = torch.sum(prob + core, dim=(1, 2, 3, 4))
        dice = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0 + self.eps)).mean()
        return bce + dice

    def _stage2_weight(
        self,
        batch: Dict[str, torch.Tensor],
        epoch: int,
        ref: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if int(epoch) < self.stage2_start_epoch:
            return None
        if "component_weight" not in batch:
            return None

        w = batch["component_weight"].float()
        if w.ndim == 4:
            w = w.unsqueeze(1)
        if tuple(w.shape[2:]) != tuple(ref.shape[2:]):
            w = F.interpolate(w.float(), size=ref.shape[2:], mode="nearest")

        w = torch.clamp(w, min=1.0, max=self.max_stage2_weight)
        w = w ** self.stage2_weight_power
        return w.to(device=ref.device, dtype=ref.dtype)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if "mask_logits" not in outputs:
            raise KeyError("outputs 缺少必要字段：'mask_logits'")

        logits = outputs["mask_logits"]
        label = batch.get(LABEL_KEY, batch.get(MASK_ALIAS_KEY, None))
        if label is None:
            raise KeyError("batch 缺少 label/mask")

        if label.ndim == 5 and tuple(label.shape[2:]) != tuple(logits.shape[2:]):
            label = F.interpolate(label.float(), size=logits.shape[2:], mode="nearest").long()

        weights = self.get_weights(epoch)

        probs, target, target_long = self._get_probs_and_target(logits, label)
        voxel_weight = self._stage2_weight(batch, epoch, logits)

        dice = self._dice_loss(probs, target, voxel_weight)
        ce = self._ce_or_bce_loss(logits, target_long, voxel_weight)

        zero = logits.new_tensor(0.0)

        if int(epoch) < self.warmup_epochs:
            tversky = zero
            boundary = zero
            sdf = zero
            core = zero
        else:
            tversky = self._tversky_loss(probs, target, voxel_weight)
            boundary = self._boundary_loss(
                logits=logits,
                target_long=target_long,
                boundary=batch.get("boundary", None),
                dist_map=batch.get("dist_map", batch.get("sdf", None)),
            )
            # SDF 回归 + SDF-mask 一致性，统一受 w_sdf 控制。
            sdf = self._sdf_loss(outputs, batch) + self._sdf_mask_consistency_loss(outputs, target)
            core = self._core_loss(outputs, batch)

        total = (
            weights.dice * dice
            + weights.ce * ce
            + weights.tversky * tversky
            + weights.boundary * boundary
            + weights.sdf * sdf
            + weights.core * core
        )

        return total, {
            "total": total.detach(),
            "dice": dice.detach(),
            "ce": ce.detach(),
            "tversky": tversky.detach(),
            "boundary": boundary.detach(),
            "sdf": sdf.detach(),
            "core": core.detach(),
            "w_dice": logits.new_tensor(weights.dice),
            "w_ce": logits.new_tensor(weights.ce),
            "w_tversky": logits.new_tensor(weights.tversky),
            "w_boundary": logits.new_tensor(weights.boundary),
            "w_sdf": logits.new_tensor(weights.sdf),
            "w_core": logits.new_tensor(weights.core),
            "stage2_active": logits.new_tensor(1.0 if weights.stage2_active else 0.0),
        }


# 兼容旧代码中可能使用的名字。
DynamicAntagonisticLoss = AdaptiveDynamicSegLoss
