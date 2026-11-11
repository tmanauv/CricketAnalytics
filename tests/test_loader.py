import pandas as pd
import pytest

from cricket_analytics.data.control_sheet import BLOCK_COLUMNS, split_by_ball_type
from cricket_analytics.data.loader import (
    is_match_sheet,
    load_workbook,
    match_sheets,
    normalize_sheet_name,
)


def test_normalize_sheet_name():
    assert normalize_sheet_name("Control Sheet") == "control_sheet"
    assert normalize_sheet_name("AUS vs  ENG") == "aus_vs_eng"


def test_is_match_sheet():
    assert is_match_sheet("IND_vs_AFG")
    assert not is_match_sheet("Analysis Parameters")


def test_load_workbook_keys(sheets):
    assert {
        "control_sheet",
        "parameters",
        "analysis_parameters",
        "ind_vs_afg",
        "aus_vs_eng",
    } <= set(sheets)
    assert set(match_sheets(sheets)) == {"ind_vs_afg", "aus_vs_eng"}


def test_load_workbook_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_synthetic_workbook"):
        load_workbook(tmp_path / "missing.xlsx")


def test_match_sheets_requires_one():
    with pytest.raises(ValueError):
        match_sheets({"parameters": pd.DataFrame()})


def test_parameters_legend(legend):
    assert legend.ball_types["LB"] == "Leg Break"
    assert legend.shot_types["DR"] == "Drive"
    assert legend.dismissal_name_to_symbol()["CAUGHT"] == "CT"


def test_split_by_ball_type(sheets, legend):
    frames = split_by_ball_type(sheets["control_sheet"], legend)
    assert set(frames) == set(legend.ball_types.values())
    frame = frames["Slower Ball"]
    assert list(frame.columns) == ["Ball Type", "Shot Type", *BLOCK_COLUMNS]
    assert frame["Ball Type"].eq("SL").all()
    assert frame["Shot Type"].notna().all()
    assert frame["Shot Type"].iloc[1] == frame["Shot Type"].iloc[0]
