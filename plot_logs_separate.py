from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


LOGS = [
    Path(r"C:\Users\A\Downloads\history_stage2_c3.json"),
    Path(r"C:\Users\A\Downloads\history_stage2_refine.json"),
]
OUT_DIR = Path("log_plots")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def is_numeric_series(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(is_number(item) for item in value)


def slug(name: str) -> str:
    allowed = []
    for char in name.lower():
        allowed.append(char if char.isalnum() else "_")
    return "_".join("".join(allowed).split("_"))


def plot_series(x_values: list[float], y_values: list[float], title: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(8.5, 5.0))
    plt.plot(x_values, y_values, color="#2563eb", linewidth=2.0)
    plt.scatter(x_values, y_values, color="#1d4ed8", s=18, zorder=3)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.28, linewidth=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def plot_top_level_series(log_name: str, data: dict[str, Any], run_dir: Path) -> list[Path]:
    created: list[Path] = []
    for key, value in data.items():
        if not is_numeric_series(value):
            continue
        epochs = list(range(1, len(value) + 1))
        out_path = run_dir / f"{slug(key)}.png"
        plot_series(epochs, value, f"{log_name}: {key}", key, out_path)
        created.append(out_path)
    return created


def plot_validation_rows(log_name: str, data: dict[str, Any], run_dir: Path) -> list[Path]:
    validation = data.get("validation")
    if not isinstance(validation, list) or not validation:
        return []

    rows: list[dict[str, Any]] = []
    epochs: list[float] = []
    for item in validation:
        if not isinstance(item, dict):
            continue
        row = item.get("row")
        epoch = item.get("epoch")
        if isinstance(row, dict) and is_number(epoch):
            rows.append(row)
            epochs.append(epoch)

    if not rows:
        return []

    keys = sorted({key for row in rows for key, value in row.items() if is_number(value)})
    created: list[Path] = []
    for key in keys:
        values = [row.get(key) for row in rows]
        if not all(is_number(value) for value in values):
            continue
        if len(set(values)) <= 1:
            continue
        out_path = run_dir / f"validation_{slug(key)}.png"
        plot_series(epochs, values, f"{log_name}: validation {key}", key, out_path)
        created.append(out_path)
    return created


def write_index(created_by_run: dict[str, list[Path]]) -> None:
    lines = ["# Log plots", ""]
    for run_name, paths in created_by_run.items():
        lines.append(f"## {run_name}")
        lines.append("")
        for path in paths:
            rel = path.as_posix()
            lines.append(f"- [{path.stem}]({rel})")
        lines.append("")
    (OUT_DIR / "index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    created_by_run: dict[str, list[Path]] = {}

    for log_path in LOGS:
        data = json.loads(log_path.read_text(encoding="utf-8"))
        run_name = log_path.stem
        run_dir = OUT_DIR / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        created = []
        created.extend(plot_top_level_series(run_name, data, run_dir))
        created.extend(plot_validation_rows(run_name, data, run_dir))
        created_by_run[run_name] = created

    write_index(created_by_run)

    for run_name, paths in created_by_run.items():
        print(f"{run_name}: {len(paths)} plots")


if __name__ == "__main__":
    main()
