import csv
import os
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch import nn

import train
from mask import write_manifest
from project_config import BASE_DIR


class NiftiPortabilityTests(unittest.TestCase):
    def test_external_and_space_containing_paths_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bme external ") as tmp:
            resolved = train.resolve_runtime_path(tmp)
            self.assertEqual(resolved, Path(tmp).resolve())
            self.assertFalse(str(resolved).startswith(str(Path(BASE_DIR).resolve())))

    def test_environment_variable_expands_in_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("BME_TEST_DATA")
            os.environ["BME_TEST_DATA"] = tmp
            try:
                resolved = train.resolve_runtime_path("${BME_TEST_DATA}")
                self.assertEqual(resolved, Path(tmp).resolve())
            finally:
                if old is None:
                    os.environ.pop("BME_TEST_DATA", None)
                else:
                    os.environ["BME_TEST_DATA"] = old

    def test_manifest_discovery_and_patient_split_do_not_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            affine = np.eye(4)
            for patient in ("LIDC-IDRI-0001", "LIDC-IDRI-0002", "LIDC-IDRI-0003"):
                for reader in (1, 2):
                    case_id = f"{patient}_reader{reader}"
                    image = root / f"{case_id}_img.nii.gz"
                    label = root / f"{case_id}_mask.nii.gz"
                    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), affine), image)
                    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.uint8), affine), label)

            records = train.discover_nifti_records(root)
            self.assertEqual(len(records), 6)
            train_rows, val_rows, test_rows = train.split_rows_by_group(
                records,
                (1 / 3, 1 / 3, 1 / 3),
                seed=42,
            )
            split_patients = [
                {row["patient_id"] for row in rows}
                for rows in (train_rows, val_rows, test_rows)
            ]
            self.assertTrue(split_patients[0].isdisjoint(split_patients[1]))
            self.assertTrue(split_patients[0].isdisjoint(split_patients[2]))
            self.assertTrue(split_patients[1].isdisjoint(split_patients[2]))

            manifest = root / "all_cases.csv"
            write_manifest(records, manifest)
            with manifest.open(encoding="utf-8") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 6)

    def test_auto_split_discovers_nifti_and_groups_patients(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            split_dir = Path(tmp) / "splits"
            root.mkdir()
            affine = np.eye(4)
            for patient in range(5):
                for reader in range(2):
                    case_id = f"LIDC-IDRI-{patient:04d}_reader{reader}"
                    nib.save(
                        nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), affine),
                        root / f"{case_id}_img.nii.gz",
                    )
                    nib.save(
                        nib.Nifti1Image(np.ones((4, 4, 4), np.uint8), affine),
                        root / f"{case_id}_mask.nii.gz",
                    )
            config = {
                "data_root": str(root),
                "train_csv": "train.csv",
                "val_csv": "val.csv",
                "test_csv": "test.csv",
                "auto_split": True,
                "split_dir": str(split_dir),
                "split_ratios": [0.6, 0.2, 0.2],
            }
            _, train_csv, val_csv = train.resolve_split_csv_paths(config)
            test_csv = Path(config["test_csv"])
            split_sets = []
            for path in (train_csv, val_csv, test_csv):
                with Path(path).open(encoding="utf-8") as f:
                    split_sets.append(
                        {row["patient_id"] for row in csv.DictReader(f)}
                    )
            self.assertTrue(split_sets[0].isdisjoint(split_sets[1]))
            self.assertTrue(split_sets[0].isdisjoint(split_sets[2]))
            self.assertTrue(split_sets[1].isdisjoint(split_sets[2]))

    def test_legacy_checkpoint_is_rejected(self) -> None:
        class TinyVersionedModel(nn.Linear):
            ARCHITECTURE_VERSION = "new_v1"

        model = TinyVersionedModel(2, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture_version": "legacy_v0",
                },
                path,
            )
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                train.load_checkpoint(path, model)

    def test_versioned_checkpoint_round_trip(self) -> None:
        class TinyVersionedModel(nn.Linear):
            ARCHITECTURE_VERSION = "tiny_v1"

        model = TinyVersionedModel(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "current.pt"
            train.save_checkpoint(
                path,
                model,
                optimizer,
                scheduler=None,
                epoch=3,
                best_score=0.75,
                cfg={"model": {"name": "tiny"}},
            )
            restored = TinyVersionedModel(2, 1)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            epoch, score = train.load_checkpoint(
                path,
                restored,
                optimizer=restored_optimizer,
            )
            self.assertEqual(epoch, 3)
            self.assertEqual(score, 0.75)
            for expected, actual in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(expected, actual)


if __name__ == "__main__":
    unittest.main()
