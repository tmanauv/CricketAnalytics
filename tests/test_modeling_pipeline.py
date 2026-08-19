from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from cricket_analytics.modeling.aggregates import SCORE_COLUMNS
from cricket_analytics.modeling.ingest import matches_frame_from_workbook
from cricket_analytics.modeling.pipeline import (
    BallTypeMeanRegressor,
    build_pipeline,
    cross_validate_model,
    evaluate,
    grouped_train_test_split,
    split_features_target,
    tune,
)
from cricket_analytics.modeling.study import run_study
from make_synthetic_dataset import build_workbook, write_workbook


@pytest.fixture(scope="module")
def matches(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    path = write_workbook(
        build_workbook(balls_per_match=180, seed=11),
        tmp_path_factory.mktemp("data") / "Cricket Data Set.xlsx",
    )
    return matches_frame_from_workbook(path)


def test_ingest_produces_a_modelable_frame(matches: pd.DataFrame) -> None:
    assert {"Batsman Name", "Bowler Name", "Ball type", "Effectiveness"} <= set(
        matches.columns
    )
    assert "Sno" not in matches.columns
    assert set(matches["Wicket"]) <= {0.0, 1.0}
    assert matches["Effectiveness"].between(-1.0, 1.0).all()
    assert matches["Batsman Name"].nunique() >= 4


def test_ingest_drops_invalid_rows_with_a_warning(tmp_path: Path) -> None:
    path = write_workbook(
        build_workbook(balls_per_match=60, seed=3, invalid=True),
        tmp_path / "Cricket Data Set.xlsx",
    )
    with pytest.warns(UserWarning, match="value-domain validation"):
        frame = matches_frame_from_workbook(path)
    assert "Zz" not in set(frame["Ball type"])
    assert "Qq" not in set(frame["Type of Shot"])
    assert set(frame["Control"]) <= {0.0, 1.0}


def test_target_columns_that_determine_the_target_are_removed(
    matches: pd.DataFrame,
) -> None:
    features, target = split_features_target(matches)
    # Effectiveness is -1.0 exactly on a wicket, so these are the answer sheet.
    assert "Wicket" not in features.columns
    assert "Dismissal kind" not in features.columns
    assert target.name == "Effectiveness"
    assert (target == -1.0).any()


def test_grouped_split_shares_no_batsman(matches: pd.DataFrame) -> None:
    features, target = split_features_target(matches)
    X_train, X_test, y_train, _ = grouped_train_test_split(features, target)

    assert not set(X_train["Batsman Name"]) & set(X_test["Batsman Name"])
    assert len(X_train) + len(X_test) == len(features)
    assert len(y_train) == len(X_train)


def test_preprocessor_is_refit_per_fit_and_emits_score_features(
    matches: pd.DataFrame,
) -> None:
    features, target = split_features_target(matches)
    pipeline = build_pipeline(matches, DecisionTreeRegressor(random_state=0))
    pipeline.fit(features, target)

    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    assert set(SCORE_COLUMNS) <= set(names)
    assert not any(name.startswith("Wicket") for name in names)

    # Refitting on a subset must change the learned aggregates, i.e. they are
    # not computed once over the whole dataset.
    first = pipeline.named_steps["preprocess"].named_transformers_["batsman_scores"]
    learned_all = first.named_steps["scores"].scores_.copy()
    subset = features["Batsman Name"] == features["Batsman Name"].iloc[0]
    pipeline.fit(features[subset], target[subset])
    learned_subset = (
        pipeline.named_steps["preprocess"]
        .named_transformers_["batsman_scores"]
        .named_steps["scores"]
        .scores_
    )
    assert len(learned_subset) == 1 < len(learned_all)


def test_unseen_categories_do_not_break_prediction(matches: pd.DataFrame) -> None:
    features, target = split_features_target(matches)
    X_train, X_test, y_train, _ = grouped_train_test_split(features, target)
    pipeline = build_pipeline(matches, DecisionTreeRegressor(random_state=0))
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    assert np.isfinite(predictions).all()
    assert len(predictions) == len(X_test)


def test_ball_type_mean_baseline_predicts_training_means(
    matches: pd.DataFrame,
) -> None:
    features, target = split_features_target(matches)
    baseline = BallTypeMeanRegressor().fit(features, target)
    expected = target.groupby(features["Ball type"]).mean()

    predictions = baseline.predict(features)
    for ball_type, mean in expected.items():
        mask = (features["Ball type"] == ball_type).to_numpy()
        assert predictions[mask] == pytest.approx(mean)

    unseen = features.head(1).assign(**{"Ball type": "ZZ"})
    assert baseline.predict(unseen)[0] == pytest.approx(float(target.mean()))


def test_cross_validation_uses_grouped_folds(matches: pd.DataFrame) -> None:
    features, target = split_features_target(matches)
    scores = cross_validate_model(
        matches, features, target, DecisionTreeRegressor(random_state=0), n_splits=3
    )
    assert len(scores) == 3
    assert np.isfinite(scores).all()


def test_tuning_never_sees_the_test_split(matches: pd.DataFrame) -> None:
    features, target = split_features_target(matches)
    X_train, X_test, y_train, y_test = grouped_train_test_split(features, target)

    search = tune(
        matches,
        X_train,
        y_train,
        DecisionTreeRegressor(random_state=0),
        {"max_depth": [2, 4]},
        n_splits=3,
    )
    assert search.best_params_["estimator__max_depth"] in (2, 4)
    # best_score_ is an inner-CV score on training batsmen only; the honest
    # test-set number is computed separately and is allowed to be worse.
    test_metrics = evaluate(search.best_estimator_, X_test, y_test)
    assert np.isfinite(test_metrics.r2)
    assert test_metrics.rmse == pytest.approx(np.sqrt(test_metrics.mse))


def test_run_study_reports_models_against_baselines(matches: pd.DataFrame) -> None:
    results = run_study(
        matches,
        estimators={"decision_tree": DecisionTreeRegressor(random_state=0)},
        n_splits=3,
    )
    assert set(results["model"]) == {"mean", "ball_type_mean", "decision_tree"}
    assert set(results.columns) >= {"cv_r2_mean", "cv_r2_std", "r2", "mae", "rmse"}
    assert results["cv_r2_mean"].is_monotonic_decreasing
    assert np.isfinite(results[["cv_r2_mean", "r2", "mae", "rmse"]].to_numpy()).all()
