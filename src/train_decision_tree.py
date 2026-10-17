"""Train the Decision Tree inverse-kinematics regressor."""

from __future__ import annotations

from pathlib import Path

from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from common import (
    LOGGER,
    RANDOM_STATE,
    configure_logging,
    make_split,
    train_and_save,
    update_comparison_from_saved_metrics,
)


MODEL_NAME = "decision_tree"
MAX_DEPTH = 30
MIN_SAMPLES_LEAF = 2


def build_model() -> TransformedTargetRegressor:
    """Create the explicit normalized multi-output decision tree."""
    input_pipeline = Pipeline(
        [
            ("scale_pose", StandardScaler()),
            (
                "regressor",
                DecisionTreeRegressor(
                    max_depth=MAX_DEPTH,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    random_state=RANDOM_STATE,
                ),
            ),
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
            "max_depth": MAX_DEPTH,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "random_state": RANDOM_STATE,
            "input_scaling": "standard",
            "target_scaling": "standard",
        },
    )
    comparison = update_comparison_from_saved_metrics(output_root)
    LOGGER.info("Metrics: %s", metrics)
    LOGGER.info("Current model comparison:\n%s", comparison.to_string(index=False))


if __name__ == "__main__":
    main()

