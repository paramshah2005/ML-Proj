"""Train six independent RBF Support Vector Regression models."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils.validation import check_is_fitted

from common import (
    LOGGER,
    TARGET_COLUMNS,
    configure_logging,
    format_duration,
    make_split,
    train_and_save,
    update_comparison_from_saved_metrics,
)


MODEL_NAME = "rbf_svr"
C = 30.0
EPSILON = 0.02
GAMMA = "scale"
MAX_FIT_SAMPLES = 15_000


class ProgressMultiOutputRegressor(RegressorMixin, BaseEstimator):
    """Fit one estimator per target and report joint-level progress and ETA."""

    def __init__(self, estimator: SVR) -> None:
        self.estimator = estimator

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ProgressMultiOutputRegressor":
        target_array = np.asarray(y)
        if target_array.ndim != 2:
            raise ValueError("Multi-output SVR expects a 2D target matrix.")

        self.estimators_: list[SVR] = []
        self.n_outputs_ = target_array.shape[1]
        samples_per_output = len(target_array)
        total_work_units = samples_per_output * self.n_outputs_
        overall_start = time.perf_counter()
        LOGGER.info(
            "RBF-SVR sample-target work units completed: 0/%s",
            f"{total_work_units:,}",
        )

        for index in range(self.n_outputs_):
            joint_name = TARGET_COLUMNS[index] if index < len(TARGET_COLUMNS) else f"target_{index + 1}"
            joint_start = time.perf_counter()
            LOGGER.info(
                "Starting SVR joint %d/%d (%s) | work units=%s/%s (%.1f%%)",
                index + 1,
                self.n_outputs_,
                joint_name,
                f"{index * samples_per_output:,}",
                f"{total_work_units:,}",
                index / self.n_outputs_ * 100,
            )
            fitted_estimator = clone(self.estimator)
            fitted_estimator.fit(x, target_array[:, index])
            self.estimators_.append(fitted_estimator)

            elapsed = time.perf_counter() - overall_start
            joint_seconds = time.perf_counter() - joint_start
            completed = index + 1
            remaining = self.n_outputs_ - completed
            estimated_remaining = elapsed / completed * remaining
            completed_work_units = completed * samples_per_output
            LOGGER.info(
                "Completed SVR joint %d/%d (%s) | work units=%s/%s (%.1f%%) | "
                "joint_time=%s | elapsed=%s | estimated_remaining=%s",
                completed,
                self.n_outputs_,
                joint_name,
                f"{completed_work_units:,}",
                f"{total_work_units:,}",
                completed / self.n_outputs_ * 100,
                format_duration(joint_seconds),
                format_duration(elapsed),
                format_duration(estimated_remaining),
            )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "estimators_")
        return np.column_stack([estimator.predict(x) for estimator in self.estimators_])


def build_model() -> TransformedTargetRegressor:
    """Create the explicit RBF-SVR implementation, one estimator per joint."""
    joint_regressor = SVR(kernel="rbf", C=C, epsilon=EPSILON, gamma=GAMMA)
    input_pipeline = Pipeline(
        [
            ("scale_pose", StandardScaler()),
            ("regressor", ProgressMultiOutputRegressor(joint_regressor)),
        ]
    )
    return TransformedTargetRegressor(
        regressor=input_pipeline,
        transformer=StandardScaler(),
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / "artifacts"
    configure_logging(output_root, f"{MODEL_NAME}_training.log")
    split = make_split(project_root / "datasetIRB2400.csv")
    metrics = train_and_save(
        model_name=MODEL_NAME,
        model=build_model(),
        split=split,
        output_root=output_root,
        model_parameters={
            "kernel": "rbf",
            "C": C,
            "epsilon": EPSILON,
            "gamma": GAMMA,
            "multi_output_strategy": "one_svr_per_joint",
            "input_scaling": "standard",
            "target_scaling": "standard",
            "max_fit_samples": MAX_FIT_SAMPLES,
        },
        max_fit_samples=MAX_FIT_SAMPLES,
    )
    comparison = update_comparison_from_saved_metrics(output_root)
    LOGGER.info("Completed standalone %s run", MODEL_NAME)
    LOGGER.info("Metrics: %s", metrics)
    LOGGER.info("Current model comparison:\n%s", comparison.to_string(index=False))


if __name__ == "__main__":
    main()

