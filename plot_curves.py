# -*- coding: utf-8 -*-
"""
plot_curves.py

Plot training curves from output_dir/logs/train_log.csv.
This version matches the dynamic-loss train.py log columns.

Rules:
- one figure per metric
- no subplots
- no chart titles
- matplotlib only
- no fixed colors
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

try:
    from train import DEFAULT_CONFIG_PATH, DEFAULT_OUTPUT_DIR, resolve_existing_path, resolve_runtime_path
except Exception:
    DEFAULT_CONFIG_PATH = "configs/task_adaptive.yaml"
    DEFAULT_OUTPUT_DIR = "outputs"

    def resolve_existing_path(value):
        return Path(value).expanduser().resolve()

    def resolve_runtime_path(value):
        return Path(value).expanduser().resolve()


DEFAULT_METRICS = [
    "train_total",
    "train_dice",
    "train_ce",
    "train_tversky",
    "train_boundary",
    "train_sdf",
    "train_core",
    "w_dice",
    "w_ce",
    "w_tversky",
    "w_boundary",
    "w_sdf",
    "w_core",
    "val_dice",
    "val_iou",
    "val_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot dynamic-loss training curves.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--log-csv", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--metrics", type=str, default="", help="Comma-separated metrics; empty = default available metrics")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_metric_filename(metric_name: str) -> str:
    return metric_name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def read_log(log_csv: Path) -> pd.DataFrame:
    if not log_csv.exists():
        raise FileNotFoundError(f"找不到训练日志：{log_csv}")
    df = pd.read_csv(log_csv)
    if df.empty:
        raise ValueError(f"训练日志为空：{log_csv}")
    if "epoch" not in df.columns:
        raise KeyError(f"训练日志缺少 epoch 字段，当前字段：{list(df.columns)}")
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.dropna(subset=["epoch"]).sort_values("epoch").reset_index(drop=True)
    return df


def infer_output_dir(config_path: str) -> Path:
    path = resolve_existing_path(config_path)
    if not path.exists():
        return resolve_runtime_path(DEFAULT_OUTPUT_DIR)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        return resolve_runtime_path(DEFAULT_OUTPUT_DIR)
    return resolve_runtime_path(cfg.get("output", {}).get("output_dir", DEFAULT_OUTPUT_DIR))


def choose_metrics(df: pd.DataFrame, metrics_arg: str) -> List[str]:
    if metrics_arg.strip():
        candidates = [m.strip() for m in metrics_arg.split(",") if m.strip()]
    else:
        candidates = DEFAULT_METRICS
    return [m for m in candidates if m in df.columns]


def plot_metric(df: pd.DataFrame, metric: str, output_dir: Path) -> Path:
    y = pd.to_numeric(df[metric], errors="coerce")
    x = df["epoch"]
    fig = plt.figure(figsize=(7, 4.5))
    plt.plot(x, y, marker="o", linewidth=1.5, markersize=3)
    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    out = output_dir / f"{safe_metric_filename(metric)}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    train_output_dir = infer_output_dir(args.config)
    log_csv = (
        Path(args.log_csv).expanduser().resolve()
        if args.log_csv is not None
        else train_output_dir / "logs" / "train_log.csv"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else train_output_dir / "figures"
    )
    ensure_dir(output_dir)
    df = read_log(log_csv)
    metrics = choose_metrics(df, args.metrics)
    if not metrics:
        raise RuntimeError(f"没有可绘制的指标。日志字段：{list(df.columns)}")
    for metric in metrics:
        out = plot_metric(df, metric, output_dir)
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
