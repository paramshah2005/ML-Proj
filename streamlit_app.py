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
ACCENT = "#C6FF4A"
MUTED = "#788071"
GRID = "#30352C"


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
        --surface: #0B0C0A;
        --panel: #111310;
        --panel-2: #151813;
        --line: #30352C;
        --text: #F1F4E9;
        --muted: #92998A;
        --signal: #C6FF4A;
    }

    html, body, [class*="st-"], [class*="css"] {
        font-family: "IBM Plex Mono", "JetBrains Mono", "Cascadia Mono", Consolas, monospace;
        font-variant-numeric: tabular-nums;
    }

    [data-testid="stAppViewContainer"] {
        background: var(--surface);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(11, 12, 10, 0.92);
        border-bottom: 1px solid var(--line);
    }

    [data-testid="stToolbar"] { right: 1rem; }
    .block-container { max-width: 1440px; padding-top: 2.1rem; padding-bottom: 4rem; }

    h1, h2, h3, p, label, div { letter-spacing: 0; }
    h1, h2, h3 { color: var(--text); font-weight: 600; }
    h1 { font-size: clamp(2rem, 5vw, 4.7rem); line-height: 0.96; margin: 0.35rem 0 1rem; }
    h2 { font-size: 1.45rem; border-top: 1px solid var(--line); padding-top: 1rem; }
    h3 { font-size: 1rem; color: var(--signal); }

    .hero {
        border: 1px solid var(--line);
        border-left: 4px solid var(--signal);
        background: var(--panel);
        padding: clamp(1.1rem, 3vw, 2.2rem);
        margin-bottom: 1.2rem;
    }

    .kicker {
        color: var(--signal);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .hero-copy { max-width: 850px; color: var(--muted); margin: 0; }
    .status-line {
        display: flex;
        gap: 0.8rem;
        align-items: center;
        margin-top: 1.2rem;
        color: var(--muted);
        font-size: 0.82rem;
    }
    .status-dot { width: 9px; height: 9px; background: var(--signal); display: inline-block; }

    [data-testid="stMetric"] {
        border: 1px solid var(--line);
        border-radius: 0;
        background: var(--panel);
        padding: 0.9rem 1rem;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--text); font-size: 1.45rem; }

    .stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
        border: 1px solid var(--signal);
        border-radius: 0;
        background: var(--signal);
        color: #0B0C0A;
        font-weight: 800;
        box-shadow: none;
    }
    .stButton > button:hover, .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        border-color: #F1F4E9;
        background: #F1F4E9;
        color: #0B0C0A;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    [data-testid="stNumberInput"] input,
    textarea {
        border-radius: 0 !important;
        background: var(--panel) !important;
        border-color: var(--line) !important;
        color: var(--text) !important;
    }

    [data-testid="stForm"] {
        border: 1px solid var(--line);
        border-radius: 0;
        background: var(--panel);
    }

    button[data-baseweb="tab"] {
        border-radius: 0;
        color: var(--muted);
        font-weight: 700;
    }
    button[data-baseweb="tab"][aria-selected="true"] { color: var(--signal); }
    [data-baseweb="tab-highlight"] { background-color: var(--signal); }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--line);
        border-radius: 0;
    }

    [data-testid="stAlert"] { border-radius: 0; border: 1px solid var(--line); }
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
        <div class="kicker">ABB IRB2400 / six-axis regression</div>
        <h1>Joint target<br>control console</h1>
        <p class="hero-copy">
            Predict six desired joint angles from Cartesian pose, end-effector orientation,
            and the robot's current six-joint state. Compare the available trained regressors
            against the same pose-safe test split.
        </p>
        <div class="status-line"><span class="status-dot"></span>TRAINED ARTIFACTS READY · 12 INPUTS · 6 OUTPUTS</div>
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
    ["PREDICT", "MODEL COMPARISON", "TEST-SET ANALYSIS", "METHOD"]
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

        submitted = st.form_submit_button("RUN PREDICTION", use_container_width=True)

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
        st.plotly_chart(
            motion_figure,
            width="stretch",
            config={
                "displaylogo": False,
                "responsive": True,
                "scrollZoom": True,
            },
            key=f"robot_motion_{selected_key}_{int(sample_index)}",
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
            "DOWNLOAD OUTPUT CSV",
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

    score_columns = st.columns(4)
    score_columns[0].metric("Model MAE", f"{model_mae:.5f} rad")
    score_columns[1].metric("Model RMSE", f"{model_rmse:.5f} rad")
    score_columns[2].metric("q_out = q_in RMSE", f"{baseline_rmse:.5f} rad")
    score_columns[3].metric("RMSE improvement", f"{improvement:.1f}%")

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

    per_joint = load_per_joint(analysis_key).copy()
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
        "- The 3D motion preview uses a dataset-calibrated six-axis kinematic chain to map joint "
        "states back to end-effector position and orientation. It is a kinematic visualization, "
        "not a collision, dynamics, controller-safety, or physical-calibration system."
    )

