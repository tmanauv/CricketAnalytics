import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def parameters_df() -> pd.DataFrame:
    """Miniature version of the workbook's parameters legend sheet."""
    return pd.DataFrame(
        {
            "Ball Type": ["Full Toss", "Yorker", "Googly", "Leg Break", "Short"],
            "ball symbol": ["FT", "Y", "GY", "LB", "SL"],
            "Shot Type": ["Drive", "Cut", "Pull", "Flick", "Defense"],
            "shot symbol": ["DR", "C", "P", "F", "D"],
            "Dismissal Kind": ["Bowled", "Caught", "Stumped", np.nan, np.nan],
            "dismisal symbol": ["B", "Ct", "St", np.nan, np.nan],
        }
    )


@pytest.fixture
def match_df() -> pd.DataFrame:
    """Miniature raw ``<team>_vs_<team>`` sheet before cleaning."""
    return pd.DataFrame(
        {
            "Sno": [1, 2, 3, 4],
            "Player Name": ["Rohit", "Rohit", "Virat", "Virat"],
            "Player ID": [101, 101, 102, 102],
            "Bowler": ["Rashid", "Rashid", "Mujeeb", "Mujeeb"],
            "Player ID.1": [201, 201, 202, 202],
            "Ball type": ["FT", "Y", "GY", "LB"],
            "Type of Shot": ["DR", "C", "P", "F"],
            "Runs": [4.0, np.nan, 1.0, 6.0],
            "Control": [1.0, 0.0, 1.0, -1.0],
            "X": [0.8, 0.2, 0.5, 1.0],
            "Wicket": [np.nan, "Wicket", np.nan, np.nan],
            "Dismissal kind": [np.nan, "Ct", np.nan, np.nan],
            "Pitch Type": ["Dry", np.nan, "Dry", "Dry"],
        }
    )


@pytest.fixture
def matches_df() -> pd.DataFrame:
    """Miniature cleaned, stacked ball-by-ball frame for feature engineering."""
    return pd.DataFrame(
        {
            "Batsman Name": [
                "Rohit",
                "Rohit",
                "Rohit",
                "Rohit",
                "Virat",
                "Virat",
                "Virat",
                "Virat",
            ],
            "Ball type": ["FT", "GY", "LB", "Y", "FT", "GY", "SL", "Y"],
            "Type of Shot": ["DR", "C", "P", "D", "RS", "F", "LG", "L"],
            "Control": [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            "Effectiveness": [0.8, 0.6, 0.4, 0.2, 1.0, 0.5, 0.7, 0.3],
        }
    )
