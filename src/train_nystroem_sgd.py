"""Train an approximate RBF model with Nystroem features and SGD regression."""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    FEATURE_COLUMNS,
    LOGGER,
    RANDOM_STATE,
    ProgressMultiOutputRegressor,
    ProgressNystroem,
    configure_logging,
    make_split,
    train_and_save,
    update_comparison_from_saved_metrics,
)


MODEL_NAME = "nystroem_sgd"
N_COMPONENTS = 500
GAMMA = 1.0 / len(FEATURE_COLUMNS)
LOSS = "epsilon_insensitive"
EPSILON = 0.02
ALPHA = 0.0001
MAX_ITER = 1_000
TOLERANCE = 0.001


def build_model(n_components: int = N_COMPONENTS) -> TransformedTargetRegressor:
    """Create one shared approximate-RBF map feeding six SGD regressors."""
    joint_regressor = SGDRegressor(
        loss=LOSS,
        epsilon=EPSILON,
        alpha=ALPHA,
        penalty="l2",
        max_iter=MAX_ITER,
        tol=TOLERANCE,
        shuffle=True,
        random_state=RANDOM_STATE,
        average=True,
    )
    input_pipeline = Pipeline(
        [
            ("scale_inputs", StandardScaler()),
            (
                "approximate_rbf",
                ProgressNystroem(
                    kernel="rbf",
                    gamma=GAMMA,
                    n_components=n_components,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
            (
                "regressor",
                ProgressMultiOutputRegressor(
                    estimator=joint_regressor,
                    model_label="Nystroem + SGD",
                ),
            ),
        ]
    )
    return TransformedTargetRegressor(
        regressor=input_pipeline,
        transformer=StandardScaler(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-components",
        type=int,
        default=N_COMPONENTS,
        help=f"Number of Nystroem basis components (default: {N_COMPONENTS}).",
    )
    parser.add_argument(
        "--max-fit-samples",
        type=int,
        default=None,
        help="Optional deterministic sample limit for a trial run; default uses all fit rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_components <= 0:
        raise ValueError("--n-components must be positive.")
    if args.max_fit_samples is not None and args.max_fit_samples <= 0:
        raise ValueError("--max-fit-samples must be positive when provided.")

    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / "artifacts"
    configure_logging(output_root, f"{MODEL_NAME}_training.log")
    LOGGER.info(
        "Nystroem configuration | components=%s | gamma=%.8f | fit_limit=%s",
        f"{args.n_components:,}",
        GAMMA,
        "all available rows" if args.max_fit_samples is None else f"{args.max_fit_samples:,}",
    )
    split = make_split(project_root / "datasetIRB2400.csv")
    metrics = train_and_save(
        model_name=MODEL_NAME,
        model=build_model(args.n_components),
        split=split,
        output_root=output_root,
        model_parameters={
            "kernel_approximation": "Nystroem",
            "kernel": "rbf",
            "gamma": GAMMA,
            "n_components": args.n_components,
            "regressor": "SGDRegressor",
            "loss": LOSS,
            "epsilon": EPSILON,
            "alpha": ALPHA,
            "penalty": "l2",
            "max_iter": MAX_ITER,
            "tolerance": TOLERANCE,
            "average": True,
            "multi_output_strategy": "one_sgd_regressor_per_joint",
            "input_scaling": "standard",
            "target_scaling": "standard",
            "max_fit_samples": args.max_fit_samples,
        },
        max_fit_samples=args.max_fit_samples,
    )
    comparison = update_comparison_from_saved_metrics(output_root)
    LOGGER.info("Completed standalone %s run", MODEL_NAME)
    LOGGER.info("Metrics: %s", metrics)
    LOGGER.info("Current model comparison:\n%s", comparison.to_string(index=False))


if __name__ == "__main__":
    main()
