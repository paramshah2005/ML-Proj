from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
ARTIFACTS = ROOT / "artifacts"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# The saved SVR pipeline contains this custom estimator class. Importing the
# module before joblib.load makes the class available during deserialization.
import train_svr  # noqa: F401, E402
from robot_visualization import (  # noqa: E402
    build_motion_figure,
    calculate_pose_error,
    forward_kinematics,
    tool_pose_from_joints,
)


FEATURES = [
    "x",
    "y",
    "z",
    "yaw",
    "pitch",
    "roll",
    "q1_in",
    "q2_in",
    "q3_in",
    "q4_in",
    "q5_in",
    "q6_in",
]
TARGETS = [f"q{i}_out" for i in range(1, 7)]
MODEL_CATALOG = {
    "Linear regression": "linear_regression",
    "RBF-SVR": "rbf_svr",
    "Decision tree": "decision_tree",
    "Nystroem + SGD": "nystroem_sgd",
}
MODEL_LABELS = MODEL_CATALOG.copy()
MODEL_DISPLAY = {value: key for key, value in MODEL_CATALOG.items()}
REQUIRED_MODEL_KEYS = ("linear_regression", "rbf_svr", "decision_tree")
MODEL_ARTIFACT_FILES = ("model.joblib", "metrics.json", "per_joint_metrics.csv", "test_predictions.csv")
ACCENT = "#E4002B"
MUTED = "#8A8A91"
GRID = "#DADADD"
JOINT_LIMITS_RAD = np.array(
    [
        [-np.pi, np.pi],
        [-1.9199, 1.7453],
        [-1.1345, 1.0472],
        [-3.49, 3.49],
        [-2.0944, 2.0944],
        [-6.9813, 6.9813],
    ]
)
RANDOM_CANDIDATE_COUNT = 96


st.set_page_config(
    page_title="IRB2400 Joint Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    :root {
        --surface: #F7F7F8;
        --panel: #FFFFFF;
        --panel-2: #F0F0F2;
        --line: #DADADD;
        --line-strong: #AFAFB5;
        --text: #151518;
        --muted: #68686F;
        --signal: #E4002B;
        --signal-hover: #B60022;
        --signal-ink: #FFFFFF;
        --focus: #002FA7;
    }

    html, body {
        overflow-x: clip;
    }

    html, body, [class*="st-"], [class*="css"] {
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        font-variant-numeric: tabular-nums;
    }

    [data-testid="stAppViewContainer"] {
        background: var(--surface);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: var(--surface);
        border-bottom: 1px solid var(--line);
    }

    [data-testid="stToolbar"] { right: 1rem; }
    .block-container { max-width: 1440px; padding-top: 1.5rem; padding-bottom: 4rem; }

    h1, h2, h3, p, label, div { letter-spacing: 0; }
    h1, h2, h3 { color: var(--text); font-weight: 650; }
    h1 { font-size: 2rem; line-height: 1.1; margin: 0; }
    h2 { font-size: 1.55rem; border-top: 1px solid var(--line); padding-top: 1.25rem; }
    h3 { font-size: 1.05rem; color: var(--text); }

    .hero {
        display: flex;
        justify-content: space-between;
        gap: 2rem;
        align-items: center;
        border-bottom: 1px solid var(--line);
        padding: 0.75rem 0 1.25rem;
        margin-bottom: 0.75rem;
    }

    .status-line {
        display: flex;
        gap: 1rem;
        align-items: center;
        margin: 0;
        color: var(--muted);
        font-size: 0.875rem;
        white-space: nowrap;
    }

    [data-testid="stMetric"] {
        border-top: 1px solid var(--line);
        border-radius: 0;
        background: transparent;
        padding: 0.8rem 0;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] {
        color: var(--text);
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
        font-size: 1.35rem;
    }

    .stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
        border: 1px solid var(--signal);
        border-radius: 0.375rem;
        background: var(--signal);
        color: var(--signal-ink);
        font-weight: 650;
        box-shadow: none;
        min-height: 2.5rem;
        transition: background-color 120ms ease-out, border-color 120ms ease-out;
    }
    .stButton > button:hover, .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        border-color: var(--signal-hover);
        background: var(--signal-hover);
        color: var(--signal-ink);
    }
    .stButton > button:focus-visible, .stDownloadButton > button:focus-visible,
    [data-testid="stFormSubmitButton"] > button:focus-visible,
    button[data-baseweb="tab"]:focus-visible {
        outline: 2px solid var(--focus);
        outline-offset: 2px;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    [data-testid="stNumberInput"] input,
    textarea {
        border-radius: 0.375rem !important;
        background: var(--panel) !important;
        border-color: var(--line-strong) !important;
        color: var(--text) !important;
    }

    [data-testid="stNumberInput"] input,
    div[data-baseweb="select"] span {
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    }

    [data-testid="stForm"] {
        border: 1px solid var(--line);
        border-radius: 0.5rem;
        background: var(--panel);
        padding: 0.25rem;
    }

    button[data-baseweb="tab"] {
        border-radius: 0;
        color: var(--muted);
        font-weight: 500;
    }
    button[data-baseweb="tab"][aria-selected="true"] { color: var(--signal); }
    [data-baseweb="tab-highlight"] { background-color: var(--signal); }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--line);
        border-radius: 0.375rem;
    }

    [data-testid="stAlert"] { border-radius: 0.375rem; border: 1px solid var(--line); }
    hr { border-color: var(--line); }
    a { color: var(--signal) !important; }

    .section-note { color: var(--muted); font-size: 0.88rem; margin-top: -0.35rem; }
    .readout {
        border: 1px solid var(--line);
        background: var(--panel-2);
        color: var(--muted);
        padding: 0.85rem 1rem;
        font-size: 0.82rem;
    }
    .readout strong { color: var(--signal); }

    @media (max-width: 60rem) {
        .hero { align-items: flex-start; flex-direction: column; gap: 0.5rem; }
        .status-line { white-space: normal; }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_comparison() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / "model_comparison.csv")


@st.cache_data(show_spinner=False)
def load_metrics(model_key: str) -> dict:
    with (ARTIFACTS / model_key / "metrics.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_per_joint(model_key: str) -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / model_key / "per_joint_metrics.csv")


@st.cache_data(show_spinner=False)
def load_predictions(model_key: str) -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / model_key / "test_predictions.csv")


@st.cache_resource(show_spinner="Loading trained model...")
def load_model(model_key: str):
    return joblib.load(ARTIFACTS / model_key / "model.joblib")


def display_name(model_key: str) -> str:
    return MODEL_DISPLAY.get(model_key, model_key.replace("_", " ").title())


def chart_style(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_view(stroke=GRID)
        .configure_axis(
            gridColor=GRID,
            domainColor=GRID,
            tickColor=GRID,
            labelColor="#C7CCBF",
            titleColor="#F1F4E9",
            labelFont="monospace",
            titleFont="monospace",
        )
        .configure_legend(
            labelColor="#C7CCBF",
            titleColor="#F1F4E9",
            labelFont="monospace",
            titleFont="monospace",
        )
    )


def validate_artifacts() -> list[str]:
    missing: list[str] = []
    if not (ARTIFACTS / "model_comparison.csv").exists():
        missing.append("artifacts/model_comparison.csv")
    for key in REQUIRED_MODEL_KEYS:
        for name in MODEL_ARTIFACT_FILES:
            path = ARTIFACTS / key / name
            if not path.exists():
                missing.append(str(path.relative_to(ROOT)).replace("\\", "/"))
    return missing


def model_artifacts_complete(model_key: str) -> bool:
    return all((ARTIFACTS / model_key / name).exists() for name in MODEL_ARTIFACT_FILES)


def metric_pair(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    errors = predicted - actual
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    return mae, rmse


st.markdown(
    """
    <section class="hero">
        <h1>IRB2400 Joint Predictor</h1>
        <p class="status-line">Trained artifacts ready&nbsp;&nbsp; 12 inputs&nbsp;&nbsp; 6 outputs</p>
    </section>
    """,
    unsafe_allow_html=True,
)

missing_files = validate_artifacts()
if missing_files:
    st.error("The dashboard cannot start because required trained artifacts are missing: " + ", ".join(missing_files))
    st.stop()

MODEL_LABELS = {
    label: key for label, key in MODEL_CATALOG.items() if model_artifacts_complete(key)
}
comparison = load_comparison()
comparison = comparison[comparison["model"].isin(MODEL_LABELS.values())].reset_index(drop=True)
comparison["display_model"] = comparison["model"].map(display_name)

predict_tab, compare_tab, analysis_tab, method_tab = st.tabs(
    ["Predict", "Model comparison", "Test-set analysis", "Method"]
)


with predict_tab:
    st.subheader("Command input")
    st.markdown(
        '<p class="section-note">Start from a real held-out test record, then edit any of the 12 inputs before prediction.</p>',
        unsafe_allow_html=True,
    )

    select_col, sample_col = st.columns([2, 1])
    with select_col:
        selected_label = st.selectbox("Model", list(MODEL_LABELS), index=0)
    selected_key = MODEL_LABELS[selected_label]
    source_predictions = load_predictions(selected_key)
    with sample_col:
        sample_index = st.number_input(
            "Held-out test row",
            min_value=0,
            max_value=len(source_predictions) - 1,
            value=0,
            step=1,
        )

    sample = source_predictions.iloc[int(sample_index)]

    def populate_random_input() -> None:
        rng = np.random.default_rng()
        lower_limits = JOINT_LIMITS_RAD[:, 0]
        upper_limits = JOINT_LIMITS_RAD[:, 1]

        target_poses: list[np.ndarray] = []
        while len(target_poses) < RANDOM_CANDIDATE_COUNT:
            target_joints = rng.uniform(lower_limits, upper_limits)
            target_pose = tool_pose_from_joints(target_joints)
            if target_pose[2] >= 0.0:
                target_poses.append(target_pose)

        pose_candidates = np.vstack(target_poses)
        current_candidates = rng.uniform(
            lower_limits,
            upper_limits,
            size=(RANDOM_CANDIDATE_COUNT, 6),
        )
        candidate_inputs = pd.DataFrame(
            np.column_stack([pose_candidates, current_candidates]),
            columns=FEATURES,
        )
        candidate_predictions = np.asarray(
            load_model(selected_key).predict(candidate_inputs),
            dtype=float,
        )

        motion_scores = np.empty(RANDOM_CANDIDATE_COUNT, dtype=float)
        for index, (current_joints, predicted_joints) in enumerate(
            zip(current_candidates, candidate_predictions, strict=True)
        ):
            current_nodes = forward_kinematics(current_joints)[:, :3, 3]
            predicted_nodes = forward_kinematics(predicted_joints)[:, :3, 3]
            motion_scores[index] = np.mean(
                np.linalg.norm(predicted_nodes - current_nodes, axis=1)
            )

        top_count = max(1, RANDOM_CANDIDATE_COUNT // 10)
        top_indices = np.argpartition(motion_scores, -top_count)[-top_count:]
        selected_index = int(rng.choice(top_indices))
        random_input = candidate_inputs.iloc[selected_index]

        for feature in FEATURES[:6]:
            st.session_state[
                f"pose_{feature}_{sample_index}_{selected_key}"
            ] = float(random_input[feature])
        for feature in FEATURES[6:]:
            st.session_state[
                f"joint_{feature}_{sample_index}_{selected_key}"
            ] = float(random_input[feature])

    with st.form("prediction_form"):
        st.markdown("#### Desired end-effector pose")
        pose_columns = st.columns(3)
        values: dict[str, float] = {}
        for index, feature in enumerate(FEATURES[:6]):
            with pose_columns[index % 3]:
                values[feature] = st.number_input(
                    feature,
                    value=float(sample[feature]),
                    format="%.6f",
                    key=f"pose_{feature}_{sample_index}_{selected_key}",
                )

        st.markdown("#### Current joint state")
        joint_columns = st.columns(3)
        for index, feature in enumerate(FEATURES[6:]):
            with joint_columns[index % 3]:
                values[feature] = st.number_input(
                    feature,
                    value=float(sample[feature]),
                    format="%.6f",
                    key=f"joint_{feature}_{sample_index}_{selected_key}",
                )

        run_col, random_col = st.columns(2)
        with run_col:
            submitted = st.form_submit_button(
                "Run prediction",
                use_container_width=True,
            )
        with random_col:
            st.form_submit_button(
                "Random input",
                use_container_width=True,
                on_click=populate_random_input,
            )
        st.caption(
            "Random input explores the full joint-limit workspace, keeps the desired Z at or "
            "above zero, and favors visible motion. Wide-range combinations may be outside the "
            "training distribution."
        )

    if submitted:
        input_frame = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)
        model = load_model(selected_key)
        start = time.perf_counter()
        predicted = np.asarray(model.predict(input_frame)).reshape(-1)
        latency_ms = (time.perf_counter() - start) * 1_000

        current = np.array([values[f"q{i}_in"] for i in range(1, 7)])
        actual = sample[TARGETS].to_numpy(dtype=float)
        unchanged_test_record = np.allclose(
            input_frame.to_numpy(dtype=float).reshape(-1),
            sample[FEATURES].to_numpy(dtype=float),
        )

        result = pd.DataFrame(
            {
                "joint": [f"J{i}" for i in range(1, 7)],
                "current_rad": current,
                "predicted_rad": predicted,
                "delta_rad": predicted - current,
                "current_deg": np.degrees(current),
                "predicted_deg": np.degrees(predicted),
            }
        )
        if unchanged_test_record:
            result["actual_rad"] = actual
            result["abs_error_rad"] = np.abs(predicted - actual)
            result["actual_delta_rad"] = actual - current
            result["delta_error_rad"] = np.abs(
                (predicted - current) - (actual - current)
            )

        status_col, latency_col, model_col = st.columns(3)
        status_col.metric("Output channels", "6 / 6")
        latency_col.metric("Measured latency", f"{latency_ms:.3f} ms")
        model_col.metric("Active model", selected_label)

        target_pose = np.array([values[name] for name in FEATURES[:6]], dtype=float)
        pose_error = calculate_pose_error(predicted, target_pose)
        motion_figure = build_motion_figure(current, predicted, target_pose)

        st.markdown("#### 3D motion preview")
        position_col, orientation_col = st.columns(2)
        position_col.metric(
            "Predicted position difference",
            f'{pose_error["position_error_mm"]:.2f} mm',
        )
        orientation_col.metric(
            "Predicted orientation difference",
            f'{pose_error["orientation_error_deg"]:.2f}°',
        )
        motion_html = motion_figure.to_html(
            full_html=False,
            include_plotlyjs=True,
            auto_play=False,
            config={
                "displaylogo": False,
                "responsive": True,
                "scrollZoom": True,
            },
            default_width="100%",
            default_height="650px",
        )
        components.html(
            '<style>html,body{margin:0;background:#FFFFFF;overflow:hidden;}</style>'
            + motion_html,
            height=700,
            scrolling=False,
        )
        st.caption("Dataset-calibrated kinematic visualization; motion is not collision checked.")

        chart_frame = result[["joint", "current_rad", "predicted_rad"]].melt(
            "joint", var_name="signal", value_name="angle_rad"
        )
        readout_chart = (
            alt.Chart(chart_frame)
            .mark_bar()
            .encode(
                x=alt.X("joint:N", title="Joint", sort=[f"J{i}" for i in range(1, 7)]),
                xOffset="signal:N",
                y=alt.Y("angle_rad:Q", title="Angle (rad)"),
                color=alt.Color(
                    "signal:N",
                    scale=alt.Scale(
                        domain=["current_rad", "predicted_rad"],
                        range=[MUTED, ACCENT],
                    ),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                tooltip=["joint:N", "signal:N", alt.Tooltip("angle_rad:Q", format=".5f")],
            )
            .properties(height=320)
        )
        st.altair_chart(chart_style(readout_chart), use_container_width=True)
        st.dataframe(
            result.style.format(precision=6),
            use_container_width=True,
            hide_index=True,
        )
        if unchanged_test_record:
            st.caption("Actual output and absolute error are shown because the inputs still match the selected held-out record.")
        else:
            st.caption("Actual output is hidden because at least one held-out input was edited.")

        st.download_button(
            "Download output CSV",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_key}_joint_prediction.csv",
            mime="text/csv",
        )


with compare_tab:
    st.subheader("Model comparison")
    st.markdown(
        '<p class="section-note">All reported metrics come from the saved comparison artifact and the same 45,000-row held-out test split.</p>',
        unsafe_allow_html=True,
    )

    winner = comparison.loc[comparison["global_rmse_rad"].idxmin()]
    fastest_fit = comparison.loc[comparison["fit_seconds"].idxmin()]
    fastest_inference = comparison.loc[comparison["prediction_microseconds_per_sample"].idxmin()]
    metric_columns = st.columns(3)
    metric_columns[0].metric("Lowest global RMSE", display_name(winner["model"]), f'{winner["global_rmse_rad"]:.5f} rad')
    metric_columns[1].metric("Fastest training", display_name(fastest_fit["model"]), f'{fastest_fit["fit_seconds"]:.3f} s')
    metric_columns[2].metric(
        "Fastest inference",
        display_name(fastest_inference["model"]),
        f'{fastest_inference["prediction_microseconds_per_sample"]:.3f} µs/sample',
    )

    error_long = comparison.melt(
        id_vars=["display_model"],
        value_vars=["mean_joint_mae_rad", "global_rmse_rad"],
        var_name="metric",
        value_name="radians",
    )
    error_long["metric"] = error_long["metric"].map(
        {"mean_joint_mae_rad": "Mean joint MAE", "global_rmse_rad": "Global RMSE"}
    )
    error_chart = (
        alt.Chart(error_long)
        .mark_bar()
        .encode(
            y=alt.Y("display_model:N", title=None, sort="-x"),
            x=alt.X("radians:Q", title="Error (rad)"),
            color=alt.Color("metric:N", scale=alt.Scale(range=[ACCENT, MUTED]), legend=alt.Legend(title=None)),
            yOffset="metric:N",
            tooltip=["display_model:N", "metric:N", alt.Tooltip("radians:Q", format=".6f")],
        )
        .properties(height=250, title="Held-out prediction error")
    )

    cost_comparison = comparison.copy()
    cost_comparison["fit_label"] = cost_comparison.apply(
        lambda row: f'{row["display_model"]} | {row["fit_seconds"]:.3f} s',
        axis=1,
    )
    cost_comparison["inference_label"] = cost_comparison.apply(
        lambda row: (
            f'{row["display_model"]} | '
            f'{row["prediction_microseconds_per_sample"]:.3f} µs/sample'
        ),
        axis=1,
    )

    fit_domain = [
        float(cost_comparison["fit_seconds"].min()) / 2,
        float(cost_comparison["fit_seconds"].max()) * 20,
    ]
    fit_base = alt.Chart(cost_comparison).encode(
        y=alt.Y(
            "display_model:N",
            title=None,
            sort="x",
            axis=None,
        ),
        x=alt.X(
            "fit_seconds:Q",
            title="Training time (seconds, logarithmic scale)",
            scale=alt.Scale(type="log", domain=fit_domain),
        ),
        tooltip=[
            alt.Tooltip("display_model:N", title="Model"),
            alt.Tooltip("fit_seconds:Q", title="Training seconds", format=".4f"),
        ],
    )
    fit_cost_chart = (
        fit_base.mark_circle(size=180, color=ACCENT)
        + fit_base.mark_text(align="left", dx=12, color="#F1F4E9").encode(
            text=alt.Text("fit_label:N")
        )
    ).properties(height=170, title="Training cost")

    inference_domain = [
        float(cost_comparison["prediction_microseconds_per_sample"].min()) / 2,
        float(cost_comparison["prediction_microseconds_per_sample"].max()) * 20,
    ]
    inference_base = alt.Chart(cost_comparison).encode(
        y=alt.Y(
            "display_model:N",
            title=None,
            sort="x",
            axis=None,
        ),
        x=alt.X(
            "prediction_microseconds_per_sample:Q",
            title="Inference time (µs/sample, logarithmic scale)",
            scale=alt.Scale(type="log", domain=inference_domain),
        ),
        tooltip=[
            alt.Tooltip("display_model:N", title="Model"),
            alt.Tooltip(
                "prediction_microseconds_per_sample:Q",
                title="µs/sample",
                format=".4f",
            ),
        ],
    )
    inference_cost_chart = (
        inference_base.mark_circle(size=180, color=ACCENT)
        + inference_base.mark_text(align="left", dx=12, color="#F1F4E9").encode(
            text=alt.Text("inference_label:N")
        )
    ).properties(height=170, title="Inference cost")

    st.altair_chart(chart_style(error_chart), use_container_width=True)

    st.markdown("#### Motion-aware evaluation")
    st.caption(
        "Correction R² evaluates q_out − q_in directly. Correction MAE and RMSE equal the "
        "absolute-angle errors algebraically, so R² and the q_out = q_in baseline expose whether "
        "a model learned the requested movement."
    )
    best_delta = comparison.loc[comparison["mean_joint_delta_r2"].idxmax()]
    best_position = comparison.loc[
        comparison["mean_cartesian_position_error_mm"].idxmin()
    ]
    best_orientation = comparison.loc[
        comparison["mean_orientation_error_deg"].idxmin()
    ]
    motion_metric_columns = st.columns(3)
    motion_metric_columns[0].metric(
        "Highest correction R²",
        display_name(best_delta["model"]),
        f'{best_delta["mean_joint_delta_r2"]:.5f}',
    )
    motion_metric_columns[1].metric(
        "Lowest Cartesian error",
        display_name(best_position["model"]),
        f'{best_position["mean_cartesian_position_error_mm"]:.2f} mm',
    )
    motion_metric_columns[2].metric(
        "Lowest orientation error",
        display_name(best_orientation["model"]),
        f'{best_orientation["mean_orientation_error_deg"]:.2f}°',
    )

    baseline_long = comparison.melt(
        id_vars=["display_model"],
        value_vars=["global_rmse_rad", "baseline_global_rmse_rad"],
        var_name="prediction",
        value_name="rmse_rad",
    )
    baseline_long["prediction"] = baseline_long["prediction"].map(
        {
            "global_rmse_rad": "Trained model",
            "baseline_global_rmse_rad": "q_out = q_in baseline",
        }
    )
    baseline_chart = (
        alt.Chart(baseline_long)
        .mark_bar()
        .encode(
            y=alt.Y("display_model:N", title=None),
            x=alt.X("rmse_rad:Q", title="Global RMSE (rad)"),
            yOffset="prediction:N",
            color=alt.Color(
                "prediction:N",
                scale=alt.Scale(range=[ACCENT, MUTED]),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                alt.Tooltip("display_model:N", title="Model"),
                "prediction:N",
                alt.Tooltip("rmse_rad:Q", title="RMSE", format=".6f"),
            ],
        )
        .properties(height=250, title="Model error versus zero-movement baseline")
    )

    correction_chart = (
        alt.Chart(comparison)
        .mark_bar(color=ACCENT)
        .encode(
            y=alt.Y("display_model:N", title=None, sort="-x"),
            x=alt.X(
                "mean_joint_delta_r2:Q",
                title="Mean correction R² (symmetric log scale)",
                scale=alt.Scale(type="symlog", constant=1),
            ),
            tooltip=[
                alt.Tooltip("display_model:N", title="Model"),
                alt.Tooltip("mean_joint_delta_r2:Q", title="Correction R²", format=".6f"),
            ],
        )
        .properties(height=250, title="Joint-correction fit")
    )

    position_long = comparison.melt(
        id_vars=["display_model"],
        value_vars=[
            "mean_cartesian_position_error_mm",
            "baseline_mean_cartesian_position_error_mm",
        ],
        var_name="prediction",
        value_name="position_error_mm",
    )
    position_long["prediction"] = position_long["prediction"].map(
        {
            "mean_cartesian_position_error_mm": "Trained model",
            "baseline_mean_cartesian_position_error_mm": "q_out = q_in baseline",
        }
    )
    position_chart = (
        alt.Chart(position_long)
        .mark_bar()
        .encode(
            y=alt.Y("display_model:N", title=None, sort="x"),
            x=alt.X(
                "position_error_mm:Q",
                title="Mean Cartesian error (mm)",
            ),
            yOffset="prediction:N",
            color=alt.Color(
                "prediction:N",
                scale=alt.Scale(range=[ACCENT, MUTED]),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                alt.Tooltip("display_model:N", title="Model"),
                "prediction:N",
                alt.Tooltip(
                    "position_error_mm:Q",
                    title="Position error (mm)",
                    format=".3f",
                ),
            ],
        )
        .properties(height=250, title="Forward-kinematics position error")
    )

    orientation_long = comparison.melt(
        id_vars=["display_model"],
        value_vars=[
            "mean_orientation_error_deg",
            "baseline_mean_orientation_error_deg",
        ],
        var_name="prediction",
        value_name="orientation_error_deg",
    )
    orientation_long["prediction"] = orientation_long["prediction"].map(
        {
            "mean_orientation_error_deg": "Trained model",
            "baseline_mean_orientation_error_deg": "q_out = q_in baseline",
        }
    )
    orientation_chart = (
        alt.Chart(orientation_long)
        .mark_bar()
        .encode(
            y=alt.Y("display_model:N", title=None, sort="x"),
            x=alt.X(
                "orientation_error_deg:Q",
                title="Mean orientation error (degrees)",
            ),
            yOffset="prediction:N",
            color=alt.Color(
                "prediction:N",
                scale=alt.Scale(range=[ACCENT, MUTED]),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                alt.Tooltip("display_model:N", title="Model"),
                "prediction:N",
                alt.Tooltip(
                    "orientation_error_deg:Q",
                    title="Orientation error (degrees)",
                    format=".3f",
                ),
            ],
        )
        .properties(height=250, title="Forward-kinematics orientation error")
    )

    baseline_col, correction_col = st.columns(2)
    with baseline_col:
        st.altair_chart(chart_style(baseline_chart), width="stretch")
    with correction_col:
        st.altair_chart(chart_style(correction_chart), width="stretch")
    position_col, orientation_col = st.columns(2)
    with position_col:
        st.altair_chart(chart_style(position_chart), width="stretch")
    with orientation_col:
        st.altair_chart(chart_style(orientation_chart), width="stretch")

    st.markdown("#### Computational cost")
    st.caption("The logarithmic scales keep all three models visible despite their large timing differences.")
    st.altair_chart(chart_style(fit_cost_chart), use_container_width=True)
    st.altair_chart(chart_style(inference_cost_chart), use_container_width=True)

    comparison_table = comparison[
        [
            "display_model",
            "mean_joint_mae_rad",
            "global_rmse_rad",
            "mean_joint_r2",
            "mean_joint_delta_r2",
            "baseline_global_rmse_rad",
            "rmse_improvement_over_baseline_percent",
            "mean_cartesian_position_error_mm",
            "mean_orientation_error_deg",
            "baseline_mean_cartesian_position_error_mm",
            "baseline_mean_orientation_error_deg",
            "fit_seconds",
            "prediction_microseconds_per_sample",
        ]
    ].rename(
        columns={
            "display_model": "model",
            "mean_joint_mae_rad": "mean_joint_mae_rad",
            "global_rmse_rad": "global_rmse_rad",
            "mean_joint_r2": "mean_joint_r2",
            "fit_seconds": "fit_seconds",
            "prediction_microseconds_per_sample": "inference_us_per_sample",
        }
    )
    st.dataframe(comparison_table.style.format(precision=6), use_container_width=True, hide_index=True)
    st.markdown(
        '<div class="readout"><strong>RESULT:</strong> '
        f'{display_name(winner["model"])} currently has the lowest global RMSE. '
        f'{display_name(fastest_inference["model"])} has the fastest measured inference. '
        "Use both error and timing when selecting the deployment model.</div>",
        unsafe_allow_html=True,
    )


with analysis_tab:
    st.subheader("Held-out signal inspection")
    st.markdown(
        '<p class="section-note">Inspect actual-versus-predicted behavior and compare each model with the direct baseline q_out = q_in.</p>',
        unsafe_allow_html=True,
    )

    analysis_model_col, analysis_joint_col = st.columns(2)
    with analysis_model_col:
        analysis_label = st.selectbox("Analysis model", list(MODEL_LABELS), index=0, key="analysis_model")
    with analysis_joint_col:
        analysis_joint = st.selectbox("Joint", [f"q{i}" for i in range(1, 7)], index=0)

    analysis_key = MODEL_LABELS[analysis_label]
    prediction_frame = load_predictions(analysis_key)
    actual_column = f"{analysis_joint}_out"
    predicted_column = f"{analysis_joint}_out_pred"
    baseline_column = f"{analysis_joint}_in"

    actual_values = prediction_frame[actual_column].to_numpy(dtype=float)
    model_values = prediction_frame[predicted_column].to_numpy(dtype=float)
    baseline_values = prediction_frame[baseline_column].to_numpy(dtype=float)
    model_mae, model_rmse = metric_pair(actual_values, model_values)
    baseline_mae, baseline_rmse = metric_pair(actual_values, baseline_values)
    improvement = 100.0 * (baseline_rmse - model_rmse) / baseline_rmse if baseline_rmse else 0.0
    analysis_metrics = load_metrics(analysis_key)
    per_joint = load_per_joint(analysis_key).copy()
    selected_joint_metrics = per_joint.loc[per_joint["joint"] == actual_column].iloc[0]

    score_columns = st.columns(4)
    score_columns[0].metric("Model MAE", f"{model_mae:.5f} rad")
    score_columns[1].metric("Model RMSE", f"{model_rmse:.5f} rad")
    score_columns[2].metric("q_out = q_in RMSE", f"{baseline_rmse:.5f} rad")
    score_columns[3].metric("RMSE improvement", f"{improvement:.1f}%")

    motion_score_columns = st.columns(3)
    motion_score_columns[0].metric(
        "Joint-correction R²",
        f'{selected_joint_metrics["delta_r2"]:.5f}',
    )
    motion_score_columns[1].metric(
        "Mean Cartesian error",
        f'{analysis_metrics["mean_cartesian_position_error_mm"]:.2f} mm',
    )
    motion_score_columns[2].metric(
        "Mean orientation error",
        f'{analysis_metrics["mean_orientation_error_deg"]:.2f}°',
    )

    scatter_frame = prediction_frame[[actual_column, predicted_column]].sample(
        n=min(5_000, len(prediction_frame)), random_state=42
    )
    scatter_frame = scatter_frame.rename(columns={actual_column: "actual", predicted_column: "predicted"})
    domain_min = float(min(scatter_frame["actual"].min(), scatter_frame["predicted"].min()))
    domain_max = float(max(scatter_frame["actual"].max(), scatter_frame["predicted"].max()))
    ideal_line = pd.DataFrame({"actual": [domain_min, domain_max], "predicted": [domain_min, domain_max]})

    points = (
        alt.Chart(scatter_frame)
        .mark_circle(size=18, opacity=0.28, color=ACCENT)
        .encode(
            x=alt.X("actual:Q", title="Actual angle (rad)"),
            y=alt.Y("predicted:Q", title="Predicted angle (rad)"),
            tooltip=[alt.Tooltip("actual:Q", format=".5f"), alt.Tooltip("predicted:Q", format=".5f")],
        )
    )
    diagonal = alt.Chart(ideal_line).mark_line(color=MUTED, strokeDash=[5, 5]).encode(x="actual:Q", y="predicted:Q")
    scatter_chart = (points + diagonal).properties(height=390, title=f"{analysis_joint.upper()} actual vs predicted — 5,000-row sample")

    error_values = model_values - actual_values
    histogram_counts, histogram_edges = np.histogram(error_values, bins=60)
    error_frame = pd.DataFrame(
        {
            "error_start": histogram_edges[:-1],
            "error_end": histogram_edges[1:],
            "rows": histogram_counts,
        }
    )
    histogram = (
        alt.Chart(error_frame)
        .mark_bar(color=ACCENT)
        .encode(
            x=alt.X("error_start:Q", title="Prediction error (rad)"),
            x2="error_end:Q",
            y=alt.Y("rows:Q", title="Test rows"),
            tooltip=[
                alt.Tooltip("error_start:Q", title="From", format=".5f"),
                alt.Tooltip("error_end:Q", title="To", format=".5f"),
                alt.Tooltip("rows:Q", title="Rows"),
            ],
        )
        .properties(height=390, title=f"{analysis_joint.upper()} error distribution — all test rows")
    )

    scatter_col, histogram_col = st.columns(2)
    with scatter_col:
        st.altair_chart(chart_style(scatter_chart), use_container_width=True)
    with histogram_col:
        st.altair_chart(chart_style(histogram), use_container_width=True)

    correction_frame = prediction_frame[
        [actual_column, predicted_column, baseline_column]
    ].sample(n=min(5_000, len(prediction_frame)), random_state=42)
    correction_frame = pd.DataFrame(
        {
            "actual_correction": (
                correction_frame[actual_column] - correction_frame[baseline_column]
            ),
            "predicted_correction": (
                correction_frame[predicted_column] - correction_frame[baseline_column]
            ),
        }
    )
    correction_min = float(correction_frame.min().min())
    correction_max = float(correction_frame.max().max())
    correction_ideal = pd.DataFrame(
        {
            "actual_correction": [correction_min, correction_max],
            "predicted_correction": [correction_min, correction_max],
        }
    )
    correction_points = (
        alt.Chart(correction_frame)
        .mark_circle(size=18, opacity=0.28, color=ACCENT)
        .encode(
            x=alt.X("actual_correction:Q", title="Actual q_out − q_in (rad)"),
            y=alt.Y("predicted_correction:Q", title="Predicted q_out − q_in (rad)"),
            tooltip=[
                alt.Tooltip("actual_correction:Q", format=".5f"),
                alt.Tooltip("predicted_correction:Q", format=".5f"),
            ],
        )
    )
    correction_diagonal = (
        alt.Chart(correction_ideal)
        .mark_line(color=MUTED, strokeDash=[5, 5])
        .encode(x="actual_correction:Q", y="predicted_correction:Q")
    )
    correction_scatter = (correction_points + correction_diagonal).properties(
        height=390,
        title=f"{analysis_joint.upper()} requested versus predicted correction",
    )
    st.altair_chart(chart_style(correction_scatter), use_container_width=True)

    joint_name_column = "target" if "target" in per_joint.columns else per_joint.columns[0]
    per_joint_chart = (
        alt.Chart(per_joint)
        .mark_bar(color=ACCENT)
        .encode(
            x=alt.X(f"{joint_name_column}:N", title="Output joint", sort=TARGETS),
            y=alt.Y("rmse_rad:Q", title="RMSE (rad)"),
            tooltip=[f"{joint_name_column}:N", alt.Tooltip("rmse_rad:Q", format=".6f")],
        )
        .properties(height=260, title=f"{analysis_label} error by output joint")
    )
    st.altair_chart(chart_style(per_joint_chart), use_container_width=True)


with method_tab:
    st.subheader("Training method")
    st.markdown(
        '<p class="section-note">The dashboard reads the trained models and evaluation artifacts; it does not retrain on page load.</p>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Signal map")
    signal_map = pd.DataFrame(
        {
            "group": ["Desired position", "Desired orientation", "Current joint state", "Target joint state"],
            "columns": [
                "x, y, z",
                "yaw, pitch, roll",
                "q1_in, q2_in, q3_in, q4_in, q5_in, q6_in",
                "q1_out, q2_out, q3_out, q4_out, q5_out, q6_out",
            ],
            "role": ["Input", "Input", "Input", "Output"],
        }
    )
    st.dataframe(signal_map, use_container_width=True, hide_index=True)

    method_rows = []
    for model_key in MODEL_LABELS.values():
        metrics = load_metrics(model_key)
        method_rows.append(
            {
                "model": display_name(model_key),
                "available_fit_rows": metrics.get("available_fit_samples"),
                "rows_used_for_fit": metrics.get("fit_samples"),
                "held_out_test_rows": metrics.get("test_samples"),
                "input_features": len(metrics.get("feature_columns", FEATURES)),
                "outputs": len(metrics.get("target_columns", TARGETS)),
                "seed": metrics.get("random_state"),
            }
        )
    st.markdown("#### Evaluated training runs")
    st.dataframe(pd.DataFrame(method_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Evaluation controls")
    st.markdown(
        "- Rows are split by pose group, so repeated measurements of the same pose do not cross "
        "train, validation, and test partitions.\n"
        "- Linear regression and the decision tree use all 255,000 available train-plus-validation rows.\n"
        "- Exact RBF-SVR uses a deterministic 15,000-row subset because its kernel training cost "
        "grows much faster with sample count."
    )
    if "nystroem_sgd" in MODEL_LABELS.values():
        nystroem_metrics = load_metrics("nystroem_sgd")
        nystroem_parameters = nystroem_metrics.get("parameters", {})
        st.markdown(
            f'- Nystroem + SGD uses {nystroem_metrics.get("fit_samples", 0):,} fitting rows '
            f'with a {nystroem_parameters.get("n_components", "saved")}-component approximate '
            "RBF feature map."
        )
    st.markdown(
        "- Every model is evaluated on the same 45,000 held-out rows.\n"
        "- Motion-aware evaluation reports joint-correction R², compares each model against the "
        "zero-movement baseline q_out = q_in, and converts predictions back into Cartesian and "
        "orientation errors through the calibrated forward-kinematics chain.\n"
        "- The 3D motion preview uses a dataset-calibrated six-axis kinematic chain to map joint "
        "states back to end-effector position and orientation. It is a kinematic visualization, "
        "not a collision, dynamics, controller-safety, or physical-calibration system."
    )

