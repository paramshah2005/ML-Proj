"""Dataset-calibrated forward kinematics and 3D motion visualization."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import plotly.graph_objects as go


DH_A_MM = np.array([0.0, 705.0, 0.0, 0.0, 0.0, 0.0])
DH_D_MM = np.array([615.0, 0.0, 0.0, 755.0, 0.0, 85.0])
DH_ALPHA_RAD = np.array(
    [-np.pi / 2.0, 0.0, -np.pi / 2.0, np.pi / 2.0, -np.pi / 2.0, 0.0]
)
JOINT_OFFSETS_RAD = np.array([0.0, 0.0, -np.pi, 0.0, 0.0, np.pi])

FRAME_COUNT = 41
FRAME_DURATION_MS = 50

SURFACE = "#FFFFFF"
TEXT = "#151518"
MUTED = "#8A8A91"
MUTED_LIGHT = "#68686F"
GRID = "#DADADD"
ACCENT = "#E4002B"
ARM_BODY = "#C9C9CF"
ARM_EDGE = "#4F4F56"
JOINT_BODY = "#5F5F66"
LINK_RADII_MM = np.array([145.0, 128.0, 108.0, 90.0, 70.0, 52.0])
JOINT_RADII_MM = np.array([185.0, 168.0, 148.0, 122.0, 94.0, 72.0])


def _as_finite_vector(values: Iterable[float], length: int, label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size != length:
        raise ValueError(f"{label} must contain exactly {length} values.")
    if not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain only finite values.")
    return vector


def _dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
            [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
            [0.0, sin_alpha, cos_alpha, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def forward_kinematics(joints: Iterable[float]) -> np.ndarray:
    """Return base plus six cumulative transforms for joint angles in radians."""
    joint_vector = _as_finite_vector(joints, 6, "joints")
    theta = joint_vector + JOINT_OFFSETS_RAD
    transforms = [np.eye(4)]
    current = np.eye(4)
    for index in range(6):
        current = current @ _dh_transform(
            theta[index],
            DH_D_MM[index],
            DH_A_MM[index],
            DH_ALPHA_RAD[index],
        )
        transforms.append(current.copy())
    return np.stack(transforms)


def tool_pose_from_joints(joints: Iterable[float]) -> np.ndarray:
    """Return x, y, z, yaw, pitch, roll using the dashboard's Rz-Ry-Rx convention."""
    tool = forward_kinematics(joints)[-1]
    rotation = tool[:3, :3]
    pitch = float(np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(np.cos(pitch)) > 1e-8:
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
    else:
        yaw = float(np.arctan2(-rotation[0, 1], rotation[1, 1]))
        roll = 0.0
    return np.array([*tool[:3, 3], yaw, pitch, roll], dtype=float)


def _target_transform(target_pose: Iterable[float]) -> np.ndarray:
    pose = _as_finite_vector(target_pose, 6, "target_pose")
    x, y, z, yaw, pitch, roll = pose

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    rotation_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rotation_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])

    transform = np.eye(4)
    transform[:3, :3] = rotation_z @ rotation_y @ rotation_x
    transform[:3, 3] = [x, y, z]
    return transform


def calculate_pose_error(
    joints: Iterable[float], target_pose: Iterable[float]
) -> dict[str, float]:
    """Return Cartesian and orientation error for a predicted joint vector."""
    predicted = forward_kinematics(joints)[-1]
    target = _target_transform(target_pose)
    position_error = float(np.linalg.norm(predicted[:3, 3] - target[:3, 3]))
    relative_rotation = target[:3, :3].T @ predicted[:3, :3]
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
    orientation_error = float(np.degrees(np.arccos(cosine)))
    return {
        "position_error_mm": position_error,
        "orientation_error_deg": orientation_error,
    }


def calculate_pose_errors_batch(
    joints: np.ndarray, target_poses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-row Cartesian and orientation errors for joint predictions."""
    joint_matrix = np.asarray(joints, dtype=float)
    pose_matrix = np.asarray(target_poses, dtype=float)
    if joint_matrix.ndim != 2 or joint_matrix.shape[1] != 6:
        raise ValueError("joints must have shape (rows, 6).")
    if pose_matrix.shape != joint_matrix.shape:
        raise ValueError("target_poses must have the same (rows, 6) shape as joints.")
    if not np.isfinite(joint_matrix).all() or not np.isfinite(pose_matrix).all():
        raise ValueError("Batch pose-error inputs must contain only finite values.")

    row_count = len(joint_matrix)
    transforms = np.repeat(np.eye(4)[np.newaxis, :, :], row_count, axis=0)
    theta = joint_matrix + JOINT_OFFSETS_RAD
    for index in range(6):
        cos_theta = np.cos(theta[:, index])
        sin_theta = np.sin(theta[:, index])
        cos_alpha = np.cos(DH_ALPHA_RAD[index])
        sin_alpha = np.sin(DH_ALPHA_RAD[index])
        step = np.zeros((row_count, 4, 4), dtype=float)
        step[:, 0, 0] = cos_theta
        step[:, 0, 1] = -sin_theta * cos_alpha
        step[:, 0, 2] = sin_theta * sin_alpha
        step[:, 0, 3] = DH_A_MM[index] * cos_theta
        step[:, 1, 0] = sin_theta
        step[:, 1, 1] = cos_theta * cos_alpha
        step[:, 1, 2] = -cos_theta * sin_alpha
        step[:, 1, 3] = DH_A_MM[index] * sin_theta
        step[:, 2, 1] = sin_alpha
        step[:, 2, 2] = cos_alpha
        step[:, 2, 3] = DH_D_MM[index]
        step[:, 3, 3] = 1.0
        transforms = transforms @ step

    position_errors = np.linalg.norm(
        transforms[:, :3, 3] - pose_matrix[:, :3],
        axis=1,
    )

    yaw = pose_matrix[:, 3]
    pitch = pose_matrix[:, 4]
    roll = pose_matrix[:, 5]
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    target_rotations = np.empty((row_count, 3, 3), dtype=float)
    target_rotations[:, 0, 0] = cy * cp
    target_rotations[:, 0, 1] = cy * sp * sr - sy * cr
    target_rotations[:, 0, 2] = cy * sp * cr + sy * sr
    target_rotations[:, 1, 0] = sy * cp
    target_rotations[:, 1, 1] = sy * sp * sr + cy * cr
    target_rotations[:, 1, 2] = sy * sp * cr - cy * sr
    target_rotations[:, 2, 0] = -sp
    target_rotations[:, 2, 1] = cp * sr
    target_rotations[:, 2, 2] = cp * cr

    rotation_products = np.sum(target_rotations * transforms[:, :3, :3], axis=(1, 2))
    cosine = np.clip((rotation_products - 1.0) / 2.0, -1.0, 1.0)
    orientation_errors = np.degrees(np.arccos(cosine))
    return position_errors, orientation_errors


def _joint_geometry(transforms: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = transforms[:, :3, 3]
    joint_origins = transforms[:-1, :3, 3]
    joint_axes = transforms[:-1, :3, 2]
    return positions, joint_origins, joint_axes


def _segmented_lines(starts: np.ndarray, ends: np.ndarray) -> tuple[list, list, list]:
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    for start, end in zip(starts, ends, strict=True):
        x.extend([float(start[0]), float(end[0]), None])
        y.extend([float(start[1]), float(end[1]), None])
        z.extend([float(start[2]), float(end[2]), None])
    return x, y, z


def _axis_trace_coordinates(
    origins: np.ndarray, axes: np.ndarray, length: float
) -> tuple[list, list, list]:
    starts = origins - axes * (length / 2.0)
    ends = origins + axes * (length / 2.0)
    return _segmented_lines(starts, ends)


def _orientation_trace_coordinates(
    transform: np.ndarray, length: float
) -> tuple[list, list, list]:
    origin = transform[:3, 3]
    ends = origin + transform[:3, :3].T * length
    starts = np.repeat(origin[np.newaxis, :], 3, axis=0)
    return _segmented_lines(starts, ends)


def _ground_coordinates(radius: float) -> tuple[list, list, list]:
    ticks = np.linspace(-radius, radius, 9)
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    for tick in ticks:
        x.extend([-radius, radius, None, tick, tick, None])
        y.extend([tick, tick, None, -radius, radius, None])
        z.extend([0.0, 0.0, None, 0.0, 0.0, None])
    return x, y, z


def _base_coordinates(radius: float = 175.0, height: float = 90.0) -> tuple[list, list, list]:
    angles = np.linspace(0.0, 2.0 * np.pi, 33)
    lower = np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles)]
    )
    upper = lower.copy()
    upper[:, 2] = height
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    for ring in (lower, upper):
        x.extend(ring[:, 0].tolist() + [None])
        y.extend(ring[:, 1].tolist() + [None])
        z.extend(ring[:, 2].tolist() + [None])
    for index in (0, 8, 16, 24):
        x.extend([float(lower[index, 0]), float(upper[index, 0]), None])
        y.extend([float(lower[index, 1]), float(upper[index, 1]), None])
        z.extend([0.0, height, None])
    return x, y, z


def _arm_mesh_coordinates(
    positions: np.ndarray,
    joint_origins: np.ndarray,
    joint_axes: np.ndarray,
    sides: int = 14,
) -> dict[str, list[float] | list[int]]:
    """Build one articulated, tapered mesh from the seven FK node positions."""
    vertices: list[np.ndarray] = []
    vertex_colors: list[str] = []
    faces_i: list[int] = []
    faces_j: list[int] = []
    faces_k: list[int] = []
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)

    for index, (start, end) in enumerate(zip(positions[:-1], positions[1:], strict=True)):
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < 1e-9:
            continue
        direction /= length
        helper = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(direction, helper))) > 0.9:
            helper = np.array([1.0, 0.0, 0.0])
        basis_u = np.cross(direction, helper)
        basis_u /= np.linalg.norm(basis_u)
        basis_v = np.cross(direction, basis_u)

        start_radius = LINK_RADII_MM[index]
        end_radius = start_radius * 0.82
        start_ring = [
            start + start_radius * (np.cos(angle) * basis_u + np.sin(angle) * basis_v)
            for angle in angles
        ]
        end_ring = [
            end + end_radius * (np.cos(angle) * basis_u + np.sin(angle) * basis_v)
            for angle in angles
        ]
        offset = len(vertices)
        vertices.extend(start_ring)
        vertices.extend(end_ring)
        vertex_colors.extend([ARM_BODY] * (sides * 2))
        start_center = len(vertices)
        vertices.append(start)
        vertex_colors.append(ARM_BODY)
        end_center = len(vertices)
        vertices.append(end)
        vertex_colors.append(ARM_BODY)

        for side in range(sides):
            next_side = (side + 1) % sides
            start_a = offset + side
            start_b = offset + next_side
            end_a = offset + sides + side
            end_b = offset + sides + next_side
            faces_i.extend([start_a, start_a, start_center, end_center])
            faces_j.extend([start_b, end_b, start_b, end_a])
            faces_k.extend([end_b, end_a, start_a, end_b])

    latitude_steps = 8
    for center, axis, radius in zip(
        joint_origins, joint_axes, JOINT_RADII_MM, strict=True
    ):
        helper = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(axis, helper))) > 0.9:
            helper = np.array([1.0, 0.0, 0.0])
        basis_u = np.cross(axis, helper)
        basis_u /= np.linalg.norm(basis_u)
        basis_v = np.cross(axis, basis_u)
        half_length = radius * 0.55
        top_index = len(vertices)
        vertices.append(center + axis * half_length)
        vertex_colors.append(JOINT_BODY)
        ring_offsets: list[int] = []
        for latitude in range(1, latitude_steps):
            phi = np.pi * latitude / latitude_steps
            ring_offsets.append(len(vertices))
            for angle in angles:
                vertices.append(
                    center
                    + radius
                    * np.sin(phi)
                    * (np.cos(angle) * basis_u + np.sin(angle) * basis_v)
                    + half_length * np.cos(phi) * axis
                )
                vertex_colors.append(JOINT_BODY)
        bottom_index = len(vertices)
        vertices.append(center - axis * half_length)
        vertex_colors.append(JOINT_BODY)

        first_ring = ring_offsets[0]
        last_ring = ring_offsets[-1]
        for side in range(sides):
            next_side = (side + 1) % sides
            faces_i.append(top_index)
            faces_j.append(first_ring + next_side)
            faces_k.append(first_ring + side)
            faces_i.append(bottom_index)
            faces_j.append(last_ring + side)
            faces_k.append(last_ring + next_side)

        for ring_index in range(len(ring_offsets) - 1):
            upper = ring_offsets[ring_index]
            lower = ring_offsets[ring_index + 1]
            for side in range(sides):
                next_side = (side + 1) % sides
                faces_i.extend([upper + side, upper + side])
                faces_j.extend([upper + next_side, lower + next_side])
                faces_k.extend([lower + next_side, lower + side])

    vertex_array = np.asarray(vertices, dtype=float)
    return {
        "x": vertex_array[:, 0].tolist(),
        "y": vertex_array[:, 1].tolist(),
        "z": vertex_array[:, 2].tolist(),
        "i": faces_i,
        "j": faces_j,
        "k": faces_k,
        "vertexcolor": vertex_colors,
    }


def _motion_coordinates(joints: np.ndarray) -> dict[str, object]:
    transforms = forward_kinematics(joints)
    positions, joint_origins, joint_axes = _joint_geometry(transforms)
    axis_coordinates = _axis_trace_coordinates(joint_origins, joint_axes, 115.0)
    tool_coordinates = _orientation_trace_coordinates(transforms[-1], 145.0)
    return {
        "transforms": transforms,
        "positions": positions,
        "arm_mesh": _arm_mesh_coordinates(positions, joint_origins, joint_axes),
        "joint_origins": joint_origins,
        "joint_axes": axis_coordinates,
        "tool_axes": tool_coordinates,
    }


def build_motion_figure(
    current: Iterable[float], predicted: Iterable[float], target_pose: Iterable[float]
) -> go.Figure:
    """Build an interactive current-to-predicted 3D robot motion figure."""
    current_vector = _as_finite_vector(current, 6, "current")
    predicted_vector = _as_finite_vector(predicted, 6, "predicted")
    target_vector = _as_finite_vector(target_pose, 6, "target_pose")
    target_transform = _target_transform(target_vector)

    fractions = np.linspace(0.0, 1.0, FRAME_COUNT)
    motion = [
        _motion_coordinates(current_vector + fraction * (predicted_vector - current_vector))
        for fraction in fractions
    ]
    current_geometry = motion[0]

    all_positions = np.concatenate(
        [item["positions"] for item in motion]
        + [target_transform[np.newaxis, :3, 3]],
        axis=0,
    )
    horizontal_radius = max(900.0, float(np.max(np.abs(all_positions[:, :2]))) + 260.0)
    z_min = min(-150.0, float(np.min(all_positions[:, 2])) - 220.0)
    z_max = max(900.0, float(np.max(all_positions[:, 2])) + 220.0)

    ground_x, ground_y, ground_z = _ground_coordinates(horizontal_radius)
    base_x, base_y, base_z = _base_coordinates()
    target_axis_x, target_axis_y, target_axis_z = _orientation_trace_coordinates(
        target_transform, 180.0
    )

    current_positions = current_geometry["positions"]
    initial_positions = motion[0]["positions"]
    initial_joint_origins = motion[0]["joint_origins"]
    initial_axis_x, initial_axis_y, initial_axis_z = motion[0]["joint_axes"]
    initial_tool_x, initial_tool_y, initial_tool_z = motion[0]["tool_axes"]
    initial_mesh = motion[0]["arm_mesh"]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=ground_x,
            y=ground_y,
            z=ground_z,
            mode="lines",
            line={"color": GRID, "width": 2},
            hoverinfo="skip",
            name="Ground",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=base_x,
            y=base_y,
            z=base_z,
            mode="lines",
            line={"color": MUTED, "width": 4},
            hoverinfo="skip",
            name="Base",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=current_positions[:, 0],
            y=current_positions[:, 1],
            z=current_positions[:, 2],
            mode="lines+markers",
            line={"color": MUTED, "width": 7, "dash": "dash"},
            marker={"color": MUTED, "size": 4},
            hovertemplate="Current pose<extra></extra>",
            name="Current pose",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[target_vector[0]],
            y=[target_vector[1]],
            z=[target_vector[2]],
            mode="markers",
            marker={
                "color": TEXT,
                "line": {"color": SURFACE, "width": 1},
                "size": 10,
                "symbol": "cross",
            },
            hovertemplate=(
                "Desired target<br>x=%{x:.2f} mm<br>y=%{y:.2f} mm<br>z=%{z:.2f} mm"
                "<extra></extra>"
            ),
            name="Desired target",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=target_axis_x,
            y=target_axis_y,
            z=target_axis_z,
            mode="lines+text",
            line={"color": TEXT, "width": 3, "dash": "dot"},
            text=[None, "X", None, None, "Y", None, None, "Z", None],
            textfont={"color": TEXT, "family": "Helvetica Neue, Arial, sans-serif", "size": 12},
            textposition="top center",
            hoverinfo="skip",
            name="Desired orientation",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=initial_positions[:, 0],
            y=initial_positions[:, 1],
            z=initial_positions[:, 2],
            mode="lines",
            line={"color": ACCENT, "width": 10},
            hovertemplate="Moving arm<extra></extra>",
            name="Moving arm",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=initial_joint_origins[:, 0],
            y=initial_joint_origins[:, 1],
            z=initial_joint_origins[:, 2],
            mode="markers",
            marker={"color": ACCENT, "size": 7, "line": {"color": SURFACE, "width": 1}},
            text=[f"J{index}" for index in range(1, 7)],
            hovertemplate="%{text}<extra></extra>",
            name="Joints",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=initial_axis_x,
            y=initial_axis_y,
            z=initial_axis_z,
            mode="lines",
            line={"color": ACCENT, "width": 4},
            hoverinfo="skip",
            name="Joint axes",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=initial_tool_x,
            y=initial_tool_y,
            z=initial_tool_z,
            mode="lines+text",
            line={"color": MUTED_LIGHT, "width": 5},
            text=[None, "X", None, None, "Y", None, None, "Z", None],
            textfont={"color": MUTED_LIGHT, "family": "Helvetica Neue, Arial, sans-serif", "size": 12},
            textposition="top center",
            hoverinfo="skip",
            name="Predicted tool axes",
        )
    )
    figure.add_trace(
        go.Mesh3d(
            x=initial_mesh["x"],
            y=initial_mesh["y"],
            z=initial_mesh["z"],
            i=initial_mesh["i"],
            j=initial_mesh["j"],
            k=initial_mesh["k"],
            vertexcolor=initial_mesh["vertexcolor"],
            flatshading=False,
            lighting={
                "ambient": 0.58,
                "diffuse": 0.86,
                "specular": 0.18,
                "roughness": 0.72,
                "fresnel": 0.08,
            },
            lightposition={"x": 900, "y": -1200, "z": 2400},
            hoverinfo="skip",
            name="Robot body",
            showlegend=False,
        )
    )

    frames: list[go.Frame] = []
    for index, (fraction, geometry) in enumerate(zip(fractions, motion, strict=True)):
        positions = geometry["positions"]
        joint_origins = geometry["joint_origins"]
        axis_x, axis_y, axis_z = geometry["joint_axes"]
        tool_x, tool_y, tool_z = geometry["tool_axes"]
        arm_mesh = geometry["arm_mesh"]
        frames.append(
            go.Frame(
                name=str(index),
                data=[
                    go.Scatter3d(x=positions[:, 0], y=positions[:, 1], z=positions[:, 2]),
                    go.Scatter3d(
                        x=joint_origins[:, 0],
                        y=joint_origins[:, 1],
                        z=joint_origins[:, 2],
                    ),
                    go.Scatter3d(x=axis_x, y=axis_y, z=axis_z),
                    go.Scatter3d(x=tool_x, y=tool_y, z=tool_z),
                    go.Mesh3d(
                        x=arm_mesh["x"],
                        y=arm_mesh["y"],
                        z=arm_mesh["z"],
                        i=arm_mesh["i"],
                        j=arm_mesh["j"],
                        k=arm_mesh["k"],
                        vertexcolor=arm_mesh["vertexcolor"],
                    ),
                ],
                traces=[5, 6, 7, 8, 9],
                layout=go.Layout(title_text=f"Motion progress: {fraction:.0%}"),
            )
        )
    figure.frames = frames

    slider_steps = [
        {
            "args": [
                [str(index)],
                {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            "label": f"{fraction:.0%}",
            "method": "animate",
        }
        for index, fraction in enumerate(fractions)
    ]

    figure.update_layout(
        title={"text": "Motion progress: 0%", "font": {"size": 14, "color": TEXT}},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": "Helvetica Neue, Arial, sans-serif", "color": TEXT},
        height=650,
        margin={"l": 0, "r": 0, "t": 45, "b": 80},
        legend={
            "orientation": "h",
            "x": 0.0,
            "y": 1.03,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": MUTED_LIGHT},
        },
        scene={
            "bgcolor": SURFACE,
            "xaxis": {
                "title": "X (mm)",
                "range": [-horizontal_radius, horizontal_radius],
                "gridcolor": GRID,
                "zerolinecolor": MUTED,
                "color": MUTED_LIGHT,
            },
            "yaxis": {
                "title": "Y (mm)",
                "range": [-horizontal_radius, horizontal_radius],
                "gridcolor": GRID,
                "zerolinecolor": MUTED,
                "color": MUTED_LIGHT,
            },
            "zaxis": {
                "title": "Z (mm)",
                "range": [z_min, z_max],
                "gridcolor": GRID,
                "zerolinecolor": MUTED,
                "color": MUTED_LIGHT,
            },
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.12, "y": -1.32, "z": 0.9}},
            "dragmode": "orbit",
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": -0.08,
                "showactive": False,
                "bgcolor": SURFACE,
                "bordercolor": GRID,
                "font": {"color": TEXT},
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {
                                    "duration": FRAME_DURATION_MS,
                                    "redraw": True,
                                },
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.17,
                "y": -0.06,
                "len": 0.81,
                "currentvalue": {"prefix": "Frame ", "font": {"color": TEXT}},
                "font": {"color": MUTED_LIGHT},
                "tickcolor": GRID,
                "steps": slider_steps,
            }
        ],
    )
    return figure

