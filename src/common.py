"""Shared data loading, splitting, evaluation, and artifact utilities."""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit


# Select columns by name because the CSV stores yaw before pitch and roll.
POSE_COLUMNS = ["x", "y", "z", "yaw", "pitch", "roll"]
CURRENT_JOINT_COLUMNS = [f"q{i}_in" for i in range(1, 7)]
FEATURE_COLUMNS = POSE_COLUMNS + CURRENT_JOINT_COLUMNS
TARGET_COLUMNS = [f"q{i}_out" for i in range(1, 7)]
IGNORED_COLUMNS: list[str] = []
RANDOM_STATE = 42
CSV_CHUNK_SIZE = 50_000
LOGGER = logging.getLogger("ik_training")


def format_duration(seconds: float) -> str:
    """Format a duration as HH:MM:SS for progress messages."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def configure_logging(output_root: Path, log_filename: str = "training.log") -> Path:
    """Log training progress to both the terminal and an artifact file."""
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / log_filename

    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.info("Logging to %s", log_path)
    return log_path


class TrainingHeartbeat:
    """Emit periodic elapsed-time updates while an estimator is fitting."""

    def __init__(self, model_name: str, interval_seconds: float = 30.0) -> None:
        self.model_name = model_name
        self.interval_seconds = interval_seconds
        self.started_at = 0.0
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self) -> "TrainingHeartbeat":
        self.started_at = time.perf_counter()
        self.thread = threading.Thread(target=self._report, daemon=True)
        self.thread.start()
        return self

    def _report(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            elapsed = time.perf_counter() - self.started_at
            LOGGER.info(
                "%s is still training | elapsed=%s",
                self.model_name,
                format_duration(elapsed),
            )

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)


@dataclass(frozen=True)
class DatasetInfo:
    source_rows: int
    retained_samples: int
    unique_poses: int
    ambiguous_feature_count: int


@dataclass(frozen=True)
class DatasetSplit:
    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.DataFrame
    y_validation: pd.DataFrame
    y_test: pd.DataFrame
    info: DatasetInfo


def load_dataset(
    csv_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, DatasetInfo]:
    """Load all pose and current-joint inputs and validate their targets."""
    LOGGER.info("Counting dataset rows in %s", csv_path)
    load_start = time.perf_counter()
    with csv_path.open("rb") as handle:
        total_rows = max(0, sum(1 for _ in handle) - 1)

    chunks: list[pd.DataFrame] = []
    processed_rows = 0
    for chunk in pd.read_csv(csv_path, chunksize=CSV_CHUNK_SIZE):
        chunks.append(chunk)
        processed_rows += len(chunk)
        LOGGER.info(
            "Dataset rows loaded: %s/%s (%.1f%%)",
            f"{processed_rows:,}",
            f"{total_rows:,}",
            processed_rows / total_rows * 100 if total_rows else 100.0,
        )
    frame = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    LOGGER.info(
        "Loaded %s rows and %s columns in %s",
        f"{len(frame):,}",
        f"{len(frame.columns):,}",
        format_duration(time.perf_counter() - load_start),
    )
    required = FEATURE_COLUMNS + TARGET_COLUMNS
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    selected = frame[FEATURE_COLUMNS + TARGET_COLUMNS]
    if selected.isna().any().any():
        missing_counts = selected.isna().sum()
        raise ValueError(f"Dataset contains missing values:\n{missing_counts[missing_counts > 0]}")
    if not np.isfinite(selected.to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains non-finite numeric values.")

    target_variants = frame.groupby(FEATURE_COLUMNS, sort=False)[TARGET_COLUMNS].nunique()
    ambiguous_mask = target_variants.gt(1).any(axis=1)
    ambiguous_feature_count = int(ambiguous_mask.sum())
    if ambiguous_feature_count:
        raise ValueError(
            f"Found {ambiguous_feature_count} identical input vectors with multiple q_out targets."
        )

    pose_groups, unique_poses = pd.factorize(
        pd.MultiIndex.from_frame(frame[POSE_COLUMNS]),
        sort=False,
    )
    info = DatasetInfo(
        source_rows=len(frame),
        retained_samples=len(frame),
        unique_poses=len(unique_poses),
        ambiguous_feature_count=ambiguous_feature_count,
    )
    LOGGER.info(
        "Retained %s samples across %s unique poses; all q_in columns are inputs",
        f"{info.retained_samples:,}",
        f"{info.unique_poses:,}",
    )
    return selected[FEATURE_COLUMNS], selected[TARGET_COLUMNS], pose_groups, info


def make_split(
    csv_path: Path,
    random_state: int = RANDOM_STATE,
    test_fraction: float = 0.15,
    validation_fraction: float = 0.15,
) -> DatasetSplit:
    """Create deterministic 70/15/15 splits with each pose in only one set."""
    if test_fraction <= 0 or validation_fraction <= 0 or test_fraction + validation_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than 1.")

    features, targets, pose_groups, info = load_dataset(csv_path)
    holdout_fraction = test_fraction + validation_fraction
    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=holdout_fraction,
        random_state=random_state,
    )
    train_indices, holdout_indices = next(first_split.split(features, targets, groups=pose_groups))

    x_train = features.iloc[train_indices].reset_index(drop=True)
    y_train = targets.iloc[train_indices].reset_index(drop=True)
    x_holdout = features.iloc[holdout_indices].reset_index(drop=True)
    y_holdout = targets.iloc[holdout_indices].reset_index(drop=True)
    holdout_groups = pose_groups[holdout_indices]

    relative_test_fraction = test_fraction / holdout_fraction
    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_test_fraction,
        random_state=random_state,
    )
    validation_indices, test_indices = next(
        second_split.split(x_holdout, y_holdout, groups=holdout_groups)
    )
    x_validation = x_holdout.iloc[validation_indices].reset_index(drop=True)
    y_validation = y_holdout.iloc[validation_indices].reset_index(drop=True)
    x_test = x_holdout.iloc[test_indices].reset_index(drop=True)
    y_test = y_holdout.iloc[test_indices].reset_index(drop=True)

    LOGGER.info(
        "Pose-group split complete | train=%s | validation=%s | test=%s | seed=%d",
        f"{len(x_train):,}",
        f"{len(x_validation):,}",
        f"{len(x_test):,}",
        random_state,
    )
    return DatasetSplit(
        x_train=x_train,
        x_validation=x_validation,
        x_test=x_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        info=info,
    )


def evaluate_predictions(y_true: pd.DataFrame, y_pred: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return per-joint metrics and equally weighted aggregate metrics."""
    predicted = np.asarray(y_pred, dtype=float)
    actual = y_true.to_numpy(dtype=float)
    if predicted.shape != actual.shape:
        raise ValueError(f"Prediction shape {predicted.shape} does not match targets {actual.shape}.")

    mae = mean_absolute_error(actual, predicted, multioutput="raw_values")
    rmse = np.sqrt(mean_squared_error(actual, predicted, multioutput="raw_values"))
    r2 = r2_score(actual, predicted, multioutput="raw_values")
    per_joint = pd.DataFrame(
        {
            "joint": TARGET_COLUMNS,
            "mae_rad": mae,
            "rmse_rad": rmse,
            "r2": r2,
        }
    )
    aggregate = {
        "mean_joint_mae_rad": float(np.mean(mae)),
        "mean_joint_rmse_rad": float(np.mean(rmse)),
        "mean_joint_r2": float(np.mean(r2)),
        "global_mae_rad": float(np.mean(np.abs(actual - predicted))),
        "global_rmse_rad": float(np.sqrt(np.mean((actual - predicted) ** 2))),
    }
    return per_joint, aggregate


def train_and_save(
    *,
    model_name: str,
    model: RegressorMixin,
    split: DatasetSplit,
    output_root: Path,
    model_parameters: dict[str, Any],
    refit_with_validation: bool = True,
    max_fit_samples: int | None = None,
) -> dict[str, Any]:
    """Fit one model, evaluate the untouched test set, and save artifacts."""
    model_dir = output_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    if refit_with_validation:
        x_fit = pd.concat([split.x_train, split.x_validation], ignore_index=True)
        y_fit = pd.concat([split.y_train, split.y_validation], ignore_index=True)
    else:
        x_fit = split.x_train
        y_fit = split.y_train

    available_fit_samples = len(x_fit)
    if max_fit_samples is not None and available_fit_samples > max_fit_samples:
        random_generator = np.random.default_rng(RANDOM_STATE)
        selected_positions = np.sort(
            random_generator.choice(available_fit_samples, size=max_fit_samples, replace=False)
        )
        x_fit = x_fit.iloc[selected_positions].reset_index(drop=True)
        y_fit = y_fit.iloc[selected_positions].reset_index(drop=True)
        LOGGER.info(
            "%s uses a deterministic subset of %s/%s available fit samples",
            model_name,
            f"{len(x_fit):,}",
            f"{available_fit_samples:,}",
        )

    LOGGER.info("%s training samples completed: 0/%s", model_name, f"{len(x_fit):,}")
    fit_start = time.perf_counter()
    with TrainingHeartbeat(model_name):
        model.fit(x_fit, y_fit)
    fit_seconds = time.perf_counter() - fit_start
    LOGGER.info(
        "%s training samples completed: %s/%s",
        model_name,
        f"{len(x_fit):,}",
        f"{len(x_fit):,}",
    )
    LOGGER.info("Finished %s fit in %s", model_name, format_duration(fit_seconds))

    LOGGER.info("Evaluating %s on %s untouched test samples", model_name, f"{len(split.x_test):,}")
    prediction_start = time.perf_counter()
    predictions = model.predict(split.x_test)
    prediction_seconds = time.perf_counter() - prediction_start

    per_joint, aggregate = evaluate_predictions(split.y_test, predictions)
    metrics: dict[str, Any] = {
        "model": model_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "ignored_columns": IGNORED_COLUMNS,
        "source_rows": split.info.source_rows,
        "retained_samples": split.info.retained_samples,
        "unique_poses": split.info.unique_poses,
        "ambiguous_feature_count": split.info.ambiguous_feature_count,
        "available_fit_samples": available_fit_samples,
        "fit_samples": len(x_fit),
        "test_samples": len(split.x_test),
        "random_state": RANDOM_STATE,
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "prediction_microseconds_per_sample": prediction_seconds / len(split.x_test) * 1_000_000,
        "parameters": model_parameters,
        **aggregate,
    }

    joblib.dump(model, model_dir / "model.joblib")
    per_joint.to_csv(model_dir / "per_joint_metrics.csv", index=False)
    with (model_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    prediction_frame = split.x_test.reset_index(drop=True).copy()
    actual = split.y_test.reset_index(drop=True)
    predicted = pd.DataFrame(predictions, columns=[f"{column}_pred" for column in TARGET_COLUMNS])
    prediction_frame = pd.concat([prediction_frame, actual, predicted], axis=1)
    prediction_frame.to_csv(model_dir / "test_predictions.csv", index=False)
    LOGGER.info(
        "%s test metrics | mean MAE=%.6f rad | mean RMSE=%.6f rad | mean R2=%.6f",
        model_name,
        metrics["mean_joint_mae_rad"],
        metrics["mean_joint_rmse_rad"],
        metrics["mean_joint_r2"],
    )
    LOGGER.info("Saved %s artifacts to %s", model_name, model_dir)
    return metrics


def write_comparison(metrics: list[dict[str, Any]], output_root: Path) -> pd.DataFrame:
    """Save a compact model comparison ordered by test RMSE."""
    rows = [
        {
            "model": item["model"],
            "mean_joint_mae_rad": item["mean_joint_mae_rad"],
            "mean_joint_rmse_rad": item["mean_joint_rmse_rad"],
            "mean_joint_r2": item["mean_joint_r2"],
            "global_rmse_rad": item["global_rmse_rad"],
            "fit_seconds": item["fit_seconds"],
            "prediction_microseconds_per_sample": item["prediction_microseconds_per_sample"],
        }
        for item in metrics
    ]
    comparison = pd.DataFrame(rows).sort_values("mean_joint_rmse_rad").reset_index(drop=True)
    output_root.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_root / "model_comparison.csv", index=False)
    LOGGER.info("Saved model comparison to %s", output_root / "model_comparison.csv")
    return comparison


def update_comparison_from_saved_metrics(output_root: Path) -> pd.DataFrame:
    """Rebuild the comparison using every completed model metrics file."""
    saved_metrics: list[dict[str, Any]] = []
    for metrics_path in sorted(output_root.glob("*/metrics.json")):
        try:
            with metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            required = {
                "model",
                "mean_joint_mae_rad",
                "mean_joint_rmse_rad",
                "mean_joint_r2",
                "global_rmse_rad",
                "fit_seconds",
                "prediction_microseconds_per_sample",
            }
            missing = required.difference(metrics)
            if missing:
                LOGGER.warning("Skipping %s; missing metrics: %s", metrics_path, sorted(missing))
                continue
            if metrics.get("feature_columns") != FEATURE_COLUMNS:
                LOGGER.warning(
                    "Skipping stale metrics from %s; feature schema does not match the current 12 inputs",
                    metrics_path,
                )
                continue
            saved_metrics.append(metrics)
        except (OSError, json.JSONDecodeError) as error:
            LOGGER.warning("Skipping unreadable metrics file %s: %s", metrics_path, error)

    if not saved_metrics:
        raise FileNotFoundError(f"No completed model metrics found beneath {output_root}.")

    LOGGER.info("Updating comparison from %d completed model(s)", len(saved_metrics))
    return write_comparison(saved_metrics, output_root)

