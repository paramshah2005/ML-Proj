from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common import CURRENT_JOINT_COLUMNS, POSE_COLUMNS, TARGET_COLUMNS, evaluate_predictions  # noqa: E402
from robot_visualization import tool_pose_from_joints  # noqa: E402


class EvaluationMetricTests(unittest.TestCase):
    def test_exact_predictions_beat_zero_movement_baseline(self) -> None:
        actual = np.array(
            [
                [0.35, -0.42, 0.18, 0.71, -0.54, 1.20],
                [-1.10, 0.65, -0.32, -1.20, 0.45, -2.10],
                [1.70, -1.00, 0.72, 2.20, -1.30, 3.40],
            ]
        )
        current = actual + np.array(
            [
                [0.10, -0.08, 0.04, 0.12, -0.06, 0.09],
                [-0.12, 0.06, -0.07, 0.05, 0.08, -0.11],
                [0.07, 0.11, -0.09, -0.10, 0.05, 0.13],
            ]
        )
        poses = np.vstack([tool_pose_from_joints(joints) for joints in actual])
        features = pd.DataFrame(
            np.column_stack([poses, current]),
            columns=POSE_COLUMNS + CURRENT_JOINT_COLUMNS,
        )
        targets = pd.DataFrame(actual, columns=TARGET_COLUMNS)

        per_joint, aggregate = evaluate_predictions(targets, actual, features)

        self.assertAlmostEqual(aggregate["global_rmse_rad"], 0.0)
        self.assertAlmostEqual(aggregate["mean_joint_delta_r2"], 1.0)
        self.assertGreater(aggregate["baseline_global_rmse_rad"], 0.0)
        self.assertAlmostEqual(aggregate["mean_cartesian_position_error_mm"], 0.0)
        self.assertLess(aggregate["mean_orientation_error_deg"], 1e-6)
        np.testing.assert_allclose(per_joint["delta_rmse_rad"], per_joint["rmse_rad"])


if __name__ == "__main__":
    unittest.main()
