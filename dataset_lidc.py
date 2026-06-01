# -*- coding: utf-8 -*-
"""
src/dataset_lidc.py

Hybrid-Swin-SDF-CoreNet 项目的 LIDC-IDRI 肺结节 patch 数据集读取模块。

预处理后的数据结构约定：

data_root/
├── train.csv
├── val.csv
├── test.csv
└── patches/
    ├── LIDC-IDRI-0001_nodule001.npz
    ├── LIDC-IDRI-0002_nodule003.npz
    └── ...

每个 npz 文件包含：
- image:    [1, 64, 64, 64]
- mask:     [1, 64, 64, 64]
- sdf:      [1, 64, 64, 64]
- boundary: [1, 64, 64, 64]
- core:     [1, 64, 64, 64]

CSV 至少包含：
- case_id
- nodule_id
- npz_path

__getitem__ 返回：
{
    "image": torch.float32,
    "mask": torch.float32,
    "sdf": torch.float32,
    "boundary": torch.float32,
    "core": torch.float32,
    "case_id": str,
    "nodule_id": str
}
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ArrayDict = Dict[str, np.ndarray]


class LIDCNoduleDataset(Dataset):
    """
    LIDC-IDRI 肺结节 3D patch 数据集。

    参数
    ----
    data_root:
        数据根目录。

    mode:
        数据模式，支持 "train"、"val"、"test"。
        当 csv_path=None 时，会默认读取：
        - train: data_root/train.csv
        - val:   data_root/val.csv
        - test:  data_root/test.csv

    csv_path:
        可选的 CSV 路径。
        支持绝对路径，或相对于 data_root 的路径。
        若不传入，则根据 mode 自动选择。

    augment:
        是否使用训练增强。
        如果为 None，则 mode="train" 时自动启用，val/test 时关闭。

    expected_shape:
        期望每个 npz 字段的形状，默认 [1, 64, 64, 64]。

    flip_prob:
        训练时每个空间维度执行随机翻转的概率。

    intensity_scale_range:
        image 随机强度缩放范围。

    intensity_shift_range:
        image 随机强度平移范围。

    noise_prob:
        image 添加随机高斯噪声的概率。

    noise_std:
        高斯噪声标准差。
    """

    REQUIRED_CSV_COLUMNS = ("case_id", "nodule_id", "npz_path")
    REQUIRED_NPZ_KEYS = ("image", "mask", "sdf", "boundary", "core")

    def __init__(
        self,
        data_root: Union[str, Path],
        mode: str = "train",
        csv_path: Optional[Union[str, Path]] = None,
        augment: Optional[bool] = None,
        expected_shape: Sequence[int] = (1, 64, 64, 64),
        flip_prob: float = 0.5,
        intensity_scale_range: Tuple[float, float] = (0.9, 1.1),
        intensity_shift_range: Tuple[float, float] = (-0.1, 0.1),
        noise_prob: float = 0.5,
        noise_std: float = 0.01,
    ) -> None:
        super().__init__()

        self.data_root = Path(data_root).expanduser().resolve()
        self.mode = str(mode).lower()
        self.expected_shape = tuple(int(v) for v in expected_shape)

        if self.mode not in ("train", "val", "test"):
            raise ValueError(
                f"mode 只支持 'train'、'val'、'test'，但当前 mode = {mode}"
            )

        if augment is None:
            self.augment = self.mode == "train"
        else:
            self.augment = bool(augment)

        self.flip_prob = float(flip_prob)
        self.intensity_scale_range = intensity_scale_range
        self.intensity_shift_range = intensity_shift_range
        self.noise_prob = float(noise_prob)
        self.noise_std = float(noise_std)

        self._check_augmentation_params()

        self.csv_path = self._resolve_csv_path(csv_path)
        self.samples = self._load_csv(self.csv_path)

        if len(self.samples) == 0:
            raise RuntimeError(f"CSV 文件中没有任何样本：{self.csv_path}")

    def _check_augmentation_params(self) -> None:
        """
        检查数据增强参数是否合法。
        """
        if not 0.0 <= self.flip_prob <= 1.0:
            raise ValueError(f"flip_prob 必须在 [0, 1] 内，但当前为 {self.flip_prob}")

        if not 0.0 <= self.noise_prob <= 1.0:
            raise ValueError(f"noise_prob 必须在 [0, 1] 内，但当前为 {self.noise_prob}")

        if self.noise_std < 0:
            raise ValueError(f"noise_std 不能为负数，但当前为 {self.noise_std}")

        scale_min, scale_max = self.intensity_scale_range
        shift_min, shift_max = self.intensity_shift_range

        if scale_min > scale_max:
            raise ValueError(
                f"intensity_scale_range 不合法：{self.intensity_scale_range}"
            )

        if shift_min > shift_max:
            raise ValueError(
                f"intensity_shift_range 不合法：{self.intensity_shift_range}"
            )

    def _resolve_csv_path(self, csv_path: Optional[Union[str, Path]]) -> Path:
        """
        解析 CSV 路径。

        如果 csv_path=None，则默认使用 data_root/{mode}.csv。
        如果 csv_path 是绝对路径，则直接使用。
        如果 csv_path 是相对路径，则认为它是相对于 data_root 的路径。
        """
        if csv_path is None:
            path = self.data_root / f"{self.mode}.csv"
        else:
            path = Path(csv_path).expanduser()
            if not path.is_absolute():
                path = self.data_root / path

        path = path.resolve()

        if not path.exists():
            raise FileNotFoundError(f"找不到 CSV 文件：{path}")

        return path

    def _load_csv(self, csv_path: Path) -> List[Dict[str, str]]:
        """
        读取 CSV 文件，并检查必要字段。
        """
        samples: List[Dict[str, str]] = []

        with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                raise ValueError(f"CSV 文件没有表头：{csv_path}")

            missing_columns = [
                col for col in self.REQUIRED_CSV_COLUMNS if col not in reader.fieldnames
            ]
            if missing_columns:
                raise ValueError(
                    f"CSV 文件缺少必要列 {missing_columns}，"
                    f"当前列为 {reader.fieldnames}，文件路径：{csv_path}"
                )

            for row_index, row in enumerate(reader):
                sample = {}

                for col in self.REQUIRED_CSV_COLUMNS:
                    value = row.get(col, "")
                    value = "" if value is None else str(value).strip()

                    if value == "":
                        raise ValueError(
                            f"CSV 第 {row_index + 2} 行字段 '{col}' 为空，"
                            f"文件路径：{csv_path}"
                        )

                    sample[col] = value

                samples.append(sample)

        return samples

    def _resolve_npz_path(self, npz_path_value: str) -> Path:
        """
        解析 npz_path。

        支持：
        1. 绝对路径；
        2. 相对于 data_root 的路径。

        例如：
        - /abs/path/to/LIDC-IDRI-0001_nodule001.npz
        - patches/LIDC-IDRI-0001_nodule001.npz
        """
        npz_path = Path(npz_path_value).expanduser()

        if not npz_path.is_absolute():
            npz_path = self.data_root / npz_path

        npz_path = npz_path.resolve()

        if not npz_path.exists():
            raise FileNotFoundError(
                f"找不到 npz 文件：{npz_path}。"
                f"请检查 CSV 中的 npz_path 是否为绝对路径，"
                f"或是否为相对于 data_root 的路径。"
            )

        return npz_path

    def _load_npz(self, npz_path: Path) -> ArrayDict:
        """
        读取单个 npz 文件，并检查必要字段和 shape。
        """
        arrays: ArrayDict = {}

        try:
            with np.load(npz_path) as data:
                available_keys = set(data.files)

                missing_keys = [
                    key for key in self.REQUIRED_NPZ_KEYS if key not in available_keys
                ]
                if missing_keys:
                    raise KeyError(
                        f"npz 文件缺少必要字段 {missing_keys}，"
                        f"当前字段为 {sorted(list(available_keys))}，"
                        f"文件路径：{npz_path}"
                    )

                for key in self.REQUIRED_NPZ_KEYS:
                    array = np.asarray(data[key])

                    if array.shape != self.expected_shape:
                        raise ValueError(
                            f"npz 文件字段 '{key}' 的 shape 不符合要求，"
                            f"期望 shape = {self.expected_shape}，"
                            f"实际 shape = {array.shape}，"
                            f"文件路径：{npz_path}"
                        )

                    arrays[key] = array.astype(np.float32, copy=False)

        except Exception as e:
            raise RuntimeError(f"读取 npz 文件失败：{npz_path}，错误信息：{e}") from e

        return arrays

    def _random_flip(self, arrays: ArrayDict) -> ArrayDict:
        """
        对 image、mask、sdf、boundary、core 同步执行随机 3D 翻转。

        输入 shape 为 [1, D, H, W]，空间维度对应：
        - axis=1: D 方向
        - axis=2: H 方向
        - axis=3: W 方向

        注意：
        np.flip 会产生负 stride 数组，后续 torch.from_numpy 不支持负 stride，
        因此最后需要使用 np.ascontiguousarray。
        """
        spatial_axes = (1, 2, 3)

        for axis in spatial_axes:
            if np.random.rand() < self.flip_prob:
                for key in self.REQUIRED_NPZ_KEYS:
                    arrays[key] = np.flip(arrays[key], axis=axis)

        for key in self.REQUIRED_NPZ_KEYS:
            arrays[key] = np.ascontiguousarray(arrays[key])

        return arrays

    def _random_intensity_scale_shift(self, image: np.ndarray) -> np.ndarray:
        """
        对 image 执行随机强度缩放和平移。

        只作用于 image，不作用于 mask、sdf、boundary、core。
        """
        scale_min, scale_max = self.intensity_scale_range
        shift_min, shift_max = self.intensity_shift_range

        scale = np.random.uniform(scale_min, scale_max)
        shift = np.random.uniform(shift_min, shift_max)

        image = image * np.float32(scale) + np.float32(shift)
        return image.astype(np.float32, copy=False)

    def _random_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """
        对 image 添加随机高斯噪声。

        只作用于 image，不作用于 mask、sdf、boundary、core。
        """
        if self.noise_std <= 0:
            return image

        if np.random.rand() >= self.noise_prob:
            return image

        noise = np.random.normal(
            loc=0.0,
            scale=self.noise_std,
            size=image.shape,
        ).astype(np.float32)

        image = image + noise
        return image.astype(np.float32, copy=False)

    def _apply_train_augmentation(self, arrays: ArrayDict) -> ArrayDict:
        """
        训练阶段数据增强。

        空间增强：
        - random flip 同步作用于 image、mask、sdf、boundary、core。

        强度增强：
        - random intensity scale 只作用于 image；
        - random intensity shift 只作用于 image；
        - random gaussian noise 只作用于 image。
        """
        arrays = self._random_flip(arrays)

        arrays["image"] = self._random_intensity_scale_shift(arrays["image"])
        arrays["image"] = self._random_gaussian_noise(arrays["image"])
        arrays["image"] = np.ascontiguousarray(arrays["image"])

        return arrays

    def __len__(self) -> int:
        """
        返回样本数量。
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        读取单个样本。

        返回字段：
        - image: torch.float32, [1, 64, 64, 64]
        - mask: torch.float32, [1, 64, 64, 64]
        - sdf: torch.float32, [1, 64, 64, 64]
        - boundary: torch.float32, [1, 64, 64, 64]
        - core: torch.float32, [1, 64, 64, 64]
        - case_id: str
        - nodule_id: str
        """
        sample = self.samples[index]

        case_id = sample["case_id"]
        nodule_id = sample["nodule_id"]
        npz_path = self._resolve_npz_path(sample["npz_path"])

        arrays = self._load_npz(npz_path)

        if self.augment:
            arrays = self._apply_train_augmentation(arrays)

        output: Dict[str, Union[torch.Tensor, str]] = {
            "image": torch.from_numpy(np.ascontiguousarray(arrays["image"])).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(arrays["mask"])).float(),
            "sdf": torch.from_numpy(np.ascontiguousarray(arrays["sdf"])).float(),
            "boundary": torch.from_numpy(
                np.ascontiguousarray(arrays["boundary"])
            ).float(),
            "core": torch.from_numpy(np.ascontiguousarray(arrays["core"])).float(),
            "case_id": str(case_id),
            "nodule_id": str(nodule_id),
        }

        return output

    def __repr__(self) -> str:
        """
        打印数据集基本信息，方便调试。
        """
        return (
            f"{self.__class__.__name__}("
            f"data_root='{self.data_root}', "
            f"mode='{self.mode}', "
            f"csv_path='{self.csv_path}', "
            f"num_samples={len(self)}, "
            f"augment={self.augment}, "
            f"expected_shape={self.expected_shape}"
            f")"
        )


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """
    创建 PyTorch DataLoader。

    参数
    ----
    dataset:
        PyTorch Dataset 实例。

    batch_size:
        batch 大小。

    shuffle:
        是否打乱数据。
        训练集通常为 True，验证集和测试集通常为 False。

    num_workers:
        DataLoader 使用的进程数。

    返回
    ----
    dataloader:
        PyTorch DataLoader。
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须大于 0，但当前 batch_size = {batch_size}")

    if num_workers < 0:
        raise ValueError(
            f"num_workers 不能为负数，但当前 num_workers = {num_workers}"
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return dataloader


if __name__ == "__main__":
    """
    简单使用示例。

    注意：
    运行前请确保 data_root 下存在 train.csv/val.csv/test.csv，
    并且 CSV 中的 npz_path 能够正确指向 patches 目录中的 npz 文件。
    """
    demo_data_root = "data"

    try:
        train_dataset = LIDCNoduleDataset(
            data_root=demo_data_root,
            mode="train",
            augment=True,
        )

        print(train_dataset)

        train_loader = create_dataloader(
            dataset=train_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
        )

        batch = next(iter(train_loader))

        print("image:", batch["image"].shape, batch["image"].dtype)
        print("mask:", batch["mask"].shape, batch["mask"].dtype)
        print("sdf:", batch["sdf"].shape, batch["sdf"].dtype)
        print("boundary:", batch["boundary"].shape, batch["boundary"].dtype)
        print("core:", batch["core"].shape, batch["core"].dtype)
        print("case_id:", batch["case_id"])
        print("nodule_id:", batch["nodule_id"])

    except Exception as err:
        print(f"Dataset 自检未通过：{err}")