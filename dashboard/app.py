"""Streamlit dashboard for the final model results recorded in SDP.ipynb.

pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "model_results.json"


@st.cache_data
def _read_results(path: Path, mtime: float) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_results(path: Path = RESULTS_PATH) -> dict:
    return _read_results(path, path.stat().st_mtime)


def cv_frame(results: dict) -> pd.DataFrame:
    frame = pd.DataFrame(results["cv"]["models"], columns=["model", "estimator", "cv_mae"])
    frame = frame.sort_values("cv_mae").reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    return frame


def cv_chart(plottable: pd.DataFrame, metric: str) -> alt.Chart:
    baseline = plottable["cv_mae"].min() / 2
    return (
        alt.Chart(plottable.assign(baseline=baseline))
        .mark_bar()
        .encode(
            x=alt.X("cv_mae:Q", title=f"CV {metric} (log scale)", scale=alt.Scale(type="log")),
            x2="baseline:Q",
            y=alt.Y("model:N", sort=plottable["model"].tolist(), title=None, axis=alt.Axis(labelLimit=250)),
            color=alt.condition(alt.datum.rank == 1, alt.value("#2e7d32"), alt.value("#90a4ae")),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("estimator:N", title="Estimator"),
                alt.Tooltip("cv_mae:Q", title=f"CV {metric}", format=".4g"),
            ],
        )
        .properties(height=40 * len(plottable))
    )


def render_cv(results: dict) -> None:
    cv = results["cv"]
    frame = cv_frame(results)
    st.subheader(f"Cross-validated {cv['metric']} ({cv['folds']}-fold)")
    if frame.empty:
        st.info("No cross-validation results recorded in the notebook outputs.")
        return

    # A log axis keeps Linear Regression's ~1e10 MAE from flattening the rest; it can only plot positive values.
    plottable = frame[frame["cv_mae"].between(0, float("inf"), inclusive="neither")]
    excluded = frame.loc[~frame.index.isin(plottable.index), "model"].tolist()
    caption = f"{cv['description']} Lower is better; the axis is log-scaled."
    if excluded:
        caption += f" Not plotted (zero or non-finite MAE, see table): {', '.join(excluded)}."
    st.caption(caption)
    if plottable.empty:
        st.info("No positive, finite CV scores to plot.")
    else:
        st.altair_chart(cv_chart(plottable, cv["metric"]), width="stretch")
    st.dataframe(
        frame.rename(
            columns={
                "rank": "Rank",
                "model": "Model",
                "estimator": "Estimator",
                "cv_mae": f"CV {cv['metric']}",
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={f"CV {cv['metric']}": st.column_config.NumberColumn(format="%.4g")},
    )


def render_tuned(results: dict) -> None:
    tuned = results["tuned"]
    st.subheader("Tuned models — held-out R²")
    if not tuned["models"]:
        st.info(
            "Tuned R² scores are not saved in the committed notebook outputs. "
            "Re-run `SDP.ipynb` with the workbook, then `python tools/extract_results.py`."
        )
        return
    frame = pd.DataFrame(tuned["models"], columns=["estimator", "r2"])
    st.bar_chart(frame, x="estimator", y="r2", horizontal=True)


def render_mlp(results: dict) -> None:
    mlp = results["mlp"]
    st.subheader(f"Keras MLP — cross-validated {mlp['metric']}")
    if not mlp["variants"]:
        st.info(
            "Keras MLP scores are not saved in the committed notebook outputs. "
            "Re-run `SDP.ipynb` with the workbook, then `python tools/extract_results.py`."
        )
        return
    st.dataframe(pd.DataFrame(mlp["variants"], columns=["variant", "mse", "std"]), hide_index=True)


def main() -> None:
    st.set_page_config(page_title="CricketAnalytics — Model Results", layout="wide")
    st.title("CricketAnalytics — Model Results")
    st.caption(
        "Predicting shot effectiveness from ball type, shot type, and batsman context. "
        f"Results extracted from `{load_results()['source']}`."
    )

    results = load_results()
    frame = cv_frame(results)

    if not frame.empty:
        best = frame.iloc[0]
        runner_up = frame.iloc[1] if len(frame) > 1 else None
        col1, col2, col3 = st.columns(3)
        col1.metric("Best model", best["model"])
        col2.metric(
            f"Best CV {results['cv']['metric']}",
            f"{best['cv_mae']:.4f}",
            delta=(
                f"{best['cv_mae'] - runner_up['cv_mae']:+.4f} vs {runner_up['model']}"
                if runner_up is not None
                else None
            ),
            delta_color="inverse",
        )
        col3.metric("Models compared", len(frame))

    st.warning(
        "These numbers are preliminary: the notebook pipeline has known target leakage in "
        "the per-batsman aggregate features and scales before the train/test split. "
        "Treat them as an upper bound until the leak-free modelling work lands.",
        icon="⚠️",
    )

    render_cv(results)
    left, right = st.columns(2)
    with left:
        render_tuned(results)
    with right:
        render_mlp(results)


if __name__ == "__main__":
    main()
