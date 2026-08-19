"""Checks that the synthetic workbook matches the schema in ``data/README.md``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from make_synthetic_dataset import (
    BALL_TYPES,
    DISMISSAL_KINDS,
    EFFECTIVENESS_VALUES,
    MATCH_COLUMNS,
    MATCHES,
    SHOT_TYPES,
    WICKET_EFFECTIVENESS,
    build_workbook,
    write_workbook,
)


@pytest.fixture(scope="module")
def workbook_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    sheets = build_workbook(balls_per_match=60, seed=7)
    return write_workbook(
        sheets, tmp_path_factory.mktemp("data") / "Cricket Data Set.xlsx"
    )


@pytest.fixture(scope="module")
def sheets(workbook_path: Path) -> dict[str, pd.DataFrame]:
    excel = pd.ExcelFile(workbook_path)
    return {name: excel.parse(name) for name in excel.sheet_names}


def test_sheet_names(sheets: dict[str, pd.DataFrame]) -> None:
    assert set(sheets) == {
        "control_sheet",
        "parameters",
        "analysis_parameters",
        *MATCHES,
    }


def test_parameters_sheet_carries_all_three_legends(
    sheets: dict[str, pd.DataFrame],
) -> None:
    parameters = sheets["parameters"]
    assert parameters["ball symbol"].dropna().tolist() == list(BALL_TYPES)
    assert parameters["shot symbol"].dropna().tolist() == list(SHOT_TYPES)
    assert parameters["dismisal symbol"].dropna().tolist() == list(
        DISMISSAL_KINDS.values()
    )
    assert parameters["Dismissal Kind"].dropna().tolist() == list(DISMISSAL_KINDS)


def test_control_sheet_is_three_column_blocks_per_ball_type(
    sheets: dict[str, pd.DataFrame],
) -> None:
    control = sheets["control_sheet"]
    assert control.columns[:2].tolist() == ["Unnamed: 0", "Unnamed: 1"]

    blocks = control.drop(columns=["Unnamed: 0", "Unnamed: 1"])
    assert blocks.shape[1] == 3 * len(BALL_TYPES)
    # The pipeline reads each block positionally and keys it off the first
    # column's header, which must be the ball symbol.
    assert blocks.columns[::3].tolist() == list(BALL_TYPES)


def test_control_sheet_shot_types_forward_fill_to_the_full_legend(
    sheets: dict[str, pd.DataFrame],
) -> None:
    control = sheets["control_sheet"].dropna(how="all")
    shot_type = control["Unnamed: 0"].ffill()
    assert isinstance(control["Unnamed: 0"].iloc[0], str), (
        "first cell must not be blank"
    )
    assert set(shot_type) == set(SHOT_TYPES)


def test_control_sheet_effectiveness_and_control_domains(
    sheets: dict[str, pd.DataFrame],
) -> None:
    control = sheets["control_sheet"].dropna(how="all")
    allowed_effectiveness = set(EFFECTIVENESS_VALUES) | {WICKET_EFFECTIVENESS}
    for position in range(len(BALL_TYPES)):
        block = control.iloc[:, 2 + position * 3 : 5 + position * 3]
        runs, correct, effectiveness = (block.iloc[:, i] for i in range(3))
        assert runs.dropna().between(0, 6).all()
        assert set(correct.dropna()) <= {0.0, 1.0}
        assert set(effectiveness.dropna().round(1)) <= allowed_effectiveness


def test_match_sheets_have_the_documented_columns(
    sheets: dict[str, pd.DataFrame],
) -> None:
    for name in MATCHES:
        assert sheets[name].columns.tolist() == MATCH_COLUMNS


def test_match_sheet_value_domains(sheets: dict[str, pd.DataFrame]) -> None:
    allowed_effectiveness = set(EFFECTIVENESS_VALUES) | {WICKET_EFFECTIVENESS}
    for name in MATCHES:
        match = sheets[name]
        assert set(match["Ball type"]) <= set(BALL_TYPES)
        assert set(match["Type of Shot"]) <= set(SHOT_TYPES)
        assert set(match["Control"]) <= {0.0, 1.0}
        assert set(match["X"].round(1)) <= allowed_effectiveness
        assert set(match["Dismissal kind"].dropna()) <= set(DISMISSAL_KINDS.values())
        assert match["Sno"].is_unique


def test_wickets_are_marked_consistently(sheets: dict[str, pd.DataFrame]) -> None:
    for name in MATCHES:
        match = sheets[name]
        wickets = match["Wicket"].notna()
        assert (match.loc[wickets, "Dismissal kind"].notna()).all()
        assert (match.loc[wickets, "X"] == WICKET_EFFECTIVENESS).all()
        assert (match.loc[~wickets, "Dismissal kind"].isna()).all()
        assert (match.loc[~wickets, "X"] != WICKET_EFFECTIVENESS).all()


def test_batsmen_appear_across_enough_rows_to_group_by(
    sheets: dict[str, pd.DataFrame],
) -> None:
    # Phase 2 splits by batsman, so every match must contribute several.
    for name, (batsmen, _) in MATCHES.items():
        assert set(sheets[name]["Player Name"]) == set(batsmen)
        assert sheets[name].groupby("Player Name").size().min() >= 2


def test_generation_is_deterministic_for_a_seed() -> None:
    first = build_workbook(balls_per_match=30, seed=99)
    second = build_workbook(balls_per_match=30, seed=99)
    for name, frame in first.items():
        pd.testing.assert_frame_equal(frame, second[name])


def test_invalid_mode_injects_rows_the_cleaning_code_must_reject() -> None:
    clean = build_workbook(balls_per_match=30, seed=5)
    dirty = build_workbook(balls_per_match=30, seed=5, invalid=True)

    for name in MATCHES:
        clean_match, dirty_match = clean[name], dirty[name]
        assert len(dirty_match) == len(clean_match) + 6

        injected = dirty_match.iloc[len(clean_match) :]
        assert "Zz" in set(injected["Ball type"])
        assert "Qq" in set(injected["Type of Shot"])
        assert 7.5 in set(injected["X"])
        assert 5.0 in set(injected["Control"])
        assert injected.iloc[-1].isna().all(), "expected a fully blank row"

        dismissal_without_flag = injected[
            injected["Dismissal kind"].notna() & injected["Wicket"].isna()
        ]
        assert len(dismissal_without_flag) == 1
