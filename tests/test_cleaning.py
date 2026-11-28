import numpy as np
import pandas as pd

from cricket_analytics.data import columns as C
from cricket_analytics.data.cleaning import build_matches_df, clean_match_df
from cricket_analytics.data.loader import match_sheets


def _sheet(**overrides):
    row = {
        "Sno": 1,
        "Player Name": "A",
        "Player ID": 1,
        "Bowler": "B",
        "Player ID.1": 2,
        "Ball type": "LB",
        "Type of Shot": "DR",
        "Pitch Type": np.nan,
        "Match ID": 7,
        "Wide": np.nan,
        "No ball": np.nan,
        "Wicket": np.nan,
        "Hit the Bat": "Yes",
        "Dismissal kind": np.nan,
        "Control": 1.0,
        "X": 0.5,
    }
    rows = [dict(row)]
    for i, changes in enumerate(overrides.get("rows", []), start=2):
        rows.append({**row, "Sno": i, **changes})
    return pd.DataFrame(rows)


def test_clean_match_df_renames_and_coerces(legend, cfg):
    df = clean_match_df(_sheet(), legend, cfg["cleaning"])
    assert {C.BATSMAN, C.BOWLER, C.TARGET, C.BATSMAN_ID, C.BOWLER_ID} <= set(df.columns)
    assert "Sno" not in df.columns and "Pitch Type" not in df.columns
    assert df.loc[0, "Hit the Bat"] == 1.0 and df.loc[0, "Wide"] == 0.0
    assert df.loc[0, C.DISMISSAL] == C.NO_DISMISSAL


def test_clean_match_df_drops_invalid_codes(legend, cfg):
    sheet = _sheet(
        rows=[
            {"Type of Shot": "ZZ"},
            {"Ball type": "QQ"},
            {"X": 0.55},
            {"Control": 3.0},
            {"Dismissal kind": "Hit wicket"},
        ]
    )
    assert len(clean_match_df(sheet, legend, cfg["cleaning"])) == 1


def test_clean_match_df_normalizes_codes(legend, cfg):
    sheet = _sheet(rows=[{"Type of Shot": "dr ", "Ball type": "lb", "Control": -1.0}])
    df = clean_match_df(sheet, legend, cfg["cleaning"])
    assert df[C.SHOT].tolist() == ["DR", "DR"]
    assert df[C.BALL_TYPE].tolist() == ["LB", "LB"]
    assert df[C.CONTROL].tolist() == [1.0, 0.0]


def test_clean_match_df_dismissals(legend, cfg):
    sheet = _sheet(
        rows=[{"Dismissal kind": d, "X": -1.0} for d in ["C", "Leg Stump", "Stump Out", "Caught"]]
    )
    df = clean_match_df(sheet, legend, cfg["cleaning"])
    assert df[C.DISMISSAL].tolist() == [C.NO_DISMISSAL, "CT", "B", "ST", "CT"]
    assert df[C.WICKET].tolist() == [0.0, 1.0, 1.0, 1.0, 1.0]


def test_build_matches_df(sheets, legend, cfg):
    matches = build_matches_df(match_sheets(sheets), legend, cfg["cleaning"])
    assert len(matches) == 598
    assert "Kedar Jadhav" not in set(matches[C.BATSMAN])
    assert set(matches[C.SHOT]) <= legend.shot_symbols
    assert set(matches[C.TARGET].round(1)) <= set(C.EFFECTIVENESS_VALUES)
    assert not matches.isna().any().any()
