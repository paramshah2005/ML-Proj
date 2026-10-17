"""Backfill motion-aware metrics from saved held-out prediction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    evaluate_predictions,
    update_comparison_from_saved_metrics,
)


def refresh_model_metrics(model_dir: Path) -> None:
    metrics_path = model_dir / "metrics.json"
    predictions_path = model_dir / "test_predictions.csv"
    if not metrics_path.exists() or not predictions_path.exists():
        return

    prediction_frame = pd.read_csv(predictions_path)
    predicted_columns = [f"{column}_pred" for column in TARGET_COLUMNS]
    required_columns = FEATURE_COLUMNS + TARGET_COLUMNS + predicted_columns
    missing = [column for column in required_columns if column not in prediction_frame.columns]
    if missing:
        raise ValueError(f"{predictions_path} is missing columns: {missing}")

    per_joint, aggregate = evaluate_predictions(
        prediction_frame[TARGET_COLUMNS],
        prediction_frame[predicted_columns].to_numpy(dtype=float),
        prediction_frame[FEATURE_COLUMNS],
    )
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    metrics.update(aggregate)

    per_joint.to_csv(model_dir / "per_joint_metrics.csv", index=False)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recalculate saved joint, correction, baseline, and task-space metrics."
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts",
    )
    args = parser.parse_args()

    for model_dir in sorted(path for path in args.artifacts.iterdir() if path.is_dir()):
        refresh_model_metrics(model_dir)
    update_comparison_from_saved_metrics(args.artifacts)


if __name__ == "__main__":
    main()
