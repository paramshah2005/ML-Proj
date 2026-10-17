"""Train all separately implemented models on one identical data split."""

from __future__ import annotations

import time
from pathlib import Path

from common import LOGGER, configure_logging, format_duration, make_split, train_and_save, write_comparison
from train_decision_tree import MODEL_NAME as TREE_NAME
from train_decision_tree import MAX_DEPTH, MIN_SAMPLES_LEAF
from train_decision_tree import build_model as build_tree
from train_linear_regression import MODEL_NAME as LINEAR_NAME
from train_linear_regression import build_model as build_linear
from train_nystroem_sgd import (
    ALPHA as NYSTROEM_ALPHA,
    EPSILON as NYSTROEM_EPSILON,
    GAMMA as NYSTROEM_GAMMA,
    LOSS as NYSTROEM_LOSS,
    MAX_ITER as NYSTROEM_MAX_ITER,
    N_COMPONENTS,
    TOLERANCE as NYSTROEM_TOLERANCE,
)
from train_nystroem_sgd import MODEL_NAME as NYSTROEM_NAME
from train_nystroem_sgd import build_model as build_nystroem
from train_svr import C, EPSILON, GAMMA, MAX_FIT_SAMPLES
from train_svr import MODEL_NAME as SVR_NAME
from train_svr import build_model as build_svr


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / "artifacts"
    configure_logging(output_root, "training.log")
    split = make_split(project_root / "datasetIRB2400.csv")
    LOGGER.info(
        "Ready to train four models from %s source rows and %s unique poses",
        f"{split.info.source_rows:,}",
        f"{split.info.unique_poses:,}",
    )

    specifications = [
        (
            LINEAR_NAME,
            build_linear,
            {"fit_intercept": True, "input_scaling": "standard", "target_scaling": "standard"},
            None,
        ),
        (
            TREE_NAME,
            build_tree,
            {
                "max_depth": MAX_DEPTH,
                "min_samples_leaf": MIN_SAMPLES_LEAF,
                "input_scaling": "standard",
                "target_scaling": "standard",
            },
            None,
        ),
        (
            SVR_NAME,
            build_svr,
            {
                "kernel": "rbf",
                "C": C,
                "epsilon": EPSILON,
                "gamma": GAMMA,
                "multi_output_strategy": "one_svr_per_joint",
                "input_scaling": "standard",
                "target_scaling": "standard",
                "max_fit_samples": MAX_FIT_SAMPLES,
            },
            MAX_FIT_SAMPLES,
        ),
        (
            NYSTROEM_NAME,
            build_nystroem,
            {
                "kernel_approximation": "Nystroem",
                "kernel": "rbf",
                "gamma": NYSTROEM_GAMMA,
                "n_components": N_COMPONENTS,
                "regressor": "SGDRegressor",
                "loss": NYSTROEM_LOSS,
                "epsilon": NYSTROEM_EPSILON,
                "alpha": NYSTROEM_ALPHA,
                "penalty": "l2",
                "max_iter": NYSTROEM_MAX_ITER,
                "tolerance": NYSTROEM_TOLERANCE,
                "average": True,
                "multi_output_strategy": "one_sgd_regressor_per_joint",
                "input_scaling": "standard",
                "target_scaling": "standard",
                "max_fit_samples": None,
            },
            None,
        ),
    ]

    all_metrics = []
    run_start = time.perf_counter()
    total_models = len(specifications)
    for model_index, (model_name, builder, parameters, max_fit_samples) in enumerate(
        specifications,
        start=1,
    ):
        LOGGER.info(
            "Models completed: %d/%d | starting %s | overall progress=%.1f%%",
            model_index - 1,
            total_models,
            model_name,
            (model_index - 1) / total_models * 100,
        )
        metrics = train_and_save(
            model_name=model_name,
            model=builder(),
            split=split,
            output_root=output_root,
            model_parameters=parameters,
            max_fit_samples=max_fit_samples,
        )
        all_metrics.append(metrics)
        LOGGER.info(
            "Models completed: %d/%d | latest=%s | overall progress=%.1f%%",
            model_index,
            total_models,
            model_name,
            model_index / total_models * 100,
        )

    comparison = write_comparison(all_metrics, output_root)
    LOGGER.info("All training finished in %s", format_duration(time.perf_counter() - run_start))
    LOGGER.info("Test-set comparison (best RMSE first):\n%s", comparison.to_string(index=False))


if __name__ == "__main__":
    main()

