"""Shared test fixtures."""

import pandas as pd
import pytest


@pytest.fixture()
def sample_ball_df() -> pd.DataFrame:
    """Minimal ball-by-ball DataFrame for unit tests."""
    return pd.DataFrame(
        {
            "match_id": ["m1"] * 10,
            "innings": [1] * 10,
            "over": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "ball": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "batsman": ["Player A"] * 5 + ["Player B"] * 5,
            "bowler": ["Bowler X"] * 10,
            "ball_type": ["OC", "SL", "LB", "Y", "OC", "GY", "SL", "OC", "LB", "Y"],
            "shot_type": ["D", "C", "P", "DR", "L", "DR", "D", "C", "P", "RS"],
            "runs_scored": [0, 4, 1, 2, 0, 4, 0, 4, 1, 6],
            "is_wide": [False] * 10,
            "is_no_ball": [False] * 10,
            "is_wicket": [False] * 9 + [True],
            "dismissal_kind": [None] * 9 + ["Ct"],
            "control": [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        }
    )
