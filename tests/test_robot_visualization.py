from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robot_visualization import (  # noqa: E402
    FRAME_COUNT,
    build_motion_figure,
    calculate_pose_error,
    forward_kinematics,
)


FIXTURES = [
    (
        [-80.9148, -121.7499, 1896.4867, -0.6941, 0.9229, -1.4411],
        [-2.2347, -0.8949, 0.6113, -3.2841, 1.2117, 5.6792],
    ),
    (
        [-336.4239, 951.0774, 366.4408, 3.0142, 0.0953, 1.0877],
        [1.9433, 1.1670, -0.0939, 1.5055, -0.3968, -6.5397],
    ),
    (
        [-2.5569, 302.5573, 1315.9100, 1.9844, -0.4109, -2.7893],
        [-1.7002, -0.9060, -0.3702, 0.5148, -1.6843, -0.5102],
    ),
]


class RobotVisualizationTests(unittest.TestCase):
    def test_dataset_fixtures_match_calibrated_chain(self) -> None:
        for target_pose, joints in FIXTURES:
            with self.subTest(target_pose=target_pose):
                error = calculate_pose_error(joints, target_pose)
                self.assertLessEqual(error["position_error_mm"], 0.15)
                self.assertLessEqual(error["orientation_error_deg"], 0.02)

    def test_self_generated_target_has_zero_error(self) -> None:
        joints = np.array([0.35, -0.42, 0.18, 0.71, -0.54, 1.2])
        tool = forward_kinematics(joints)[-1]
        rotation = tool[:3, :3]
        pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        target = [*tool[:3, 3], yaw, pitch, roll]
        error = calculate_pose_error(joints, target)
        self.assertLess(error["position_error_mm"], 1e-9)
        self.assertLess(error["orientation_error_deg"], 1e-6)

    def test_animation_endpoints_and_finite_coordinates(self) -> None:
        target_pose, predicted = FIXTURES[0]
        current = np.array(predicted) + np.array([0.02, -0.03, 0.01, 0.04, -0.02, 0.03])
        figure = build_motion_figure(current, predicted, target_pose)

        self.assertEqual(len(figure.frames), FRAME_COUNT)
        first_positions = forward_kinematics(current)[:, :3, 3]
        last_positions = forward_kinematics(predicted)[:, :3, 3]
        np.testing.assert_allclose(
            np.column_stack(
                [figure.frames[0].data[0].x, figure.frames[0].data[0].y, figure.frames[0].data[0].z]
            ),
            first_positions,
        )
        np.testing.assert_allclose(
            np.column_stack(
                [figure.frames[-1].data[0].x, figure.frames[-1].data[0].y, figure.frames[-1].data[0].z]
            ),
            last_positions,
        )
        for frame in figure.frames:
            coordinates = np.column_stack(
                [frame.data[0].x, frame.data[0].y, frame.data[0].z]
            )
            self.assertTrue(np.isfinite(coordinates).all())

    def test_extreme_and_out_of_workspace_values_remain_renderable(self) -> None:
        current = [-3.1416, -1.9199, -1.1345, -3.49, -2.0944, -6.9813]
        predicted = [3.1416, 1.7453, 1.0472, 3.49, 2.0944, 6.9813]
        target = [5000.0, -4500.0, 3500.0, 2.8, -1.4, 3.0]
        figure = build_motion_figure(current, predicted, target)
        self.assertEqual(len(figure.frames), FRAME_COUNT)
        self.assertTrue(np.isfinite(figure.layout.scene.xaxis.range).all())
        self.assertTrue(np.isfinite(figure.layout.scene.yaxis.range).all())
        self.assertTrue(np.isfinite(figure.layout.scene.zaxis.range).all())


if __name__ == "__main__":
    unittest.main()
