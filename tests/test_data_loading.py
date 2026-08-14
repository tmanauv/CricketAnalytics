import numpy as np
import pandas as pd
import pytest

from cricket_analytics.data_loading import (
    build_ball_type_dict,
    forward_fill_shot_type,
    generate_ballwise_dfs,
    load_workbook,
    prepare_control_sheet,
    sheet_to_df_name,
)


class TestSheetToDfName:
    def test_match_sheet(self):
        assert sheet_to_df_name("IND vs AFG") == "ind_vs_afg_df"

    def test_single_word(self):
        assert sheet_to_df_name("Parameters") == "parameters_df"

    def test_collapses_extra_whitespace(self):
        assert sheet_to_df_name("Control   Sheet") == "control_sheet_df"


class TestLoadWorkbook:
    def test_loads_all_sheets(self, tmp_path):
        path = tmp_path / "book.xlsx"
        with pd.ExcelWriter(path) as writer:
            pd.DataFrame({"a": [1]}).to_excel(
                writer, sheet_name="Control Sheet", index=False
            )
            pd.DataFrame({"b": [2]}).to_excel(
                writer, sheet_name="IND vs AFG", index=False
            )

        result = load_workbook(str(path))

        assert set(result.keys()) == {"control_sheet_df", "ind_vs_afg_df"}
        assert result["control_sheet_df"]["a"].tolist() == [1]
        assert result["ind_vs_afg_df"]["b"].tolist() == [2]


class TestPrepareControlSheet:
    def test_renames_and_drops_columns(self):
        df = pd.DataFrame(
            {
                "Unnamed: 0": ["Drive", np.nan],
                "Unnamed: 1": [1, 2],
                "Sl": [3, 4],
            }
        )

        result = prepare_control_sheet(df)

        assert "Shot Type" in result.columns
        assert "SL" in result.columns
        assert "Unnamed: 1" not in result.columns

    def test_drops_all_nan_rows(self):
        df = pd.DataFrame(
            {"Unnamed: 0": ["Drive", np.nan], "FT": [1.0, np.nan]}
        )

        result = prepare_control_sheet(df)

        assert len(result) == 1

    def test_keeps_partial_rows(self):
        df = pd.DataFrame({"Unnamed: 0": [np.nan, "Cut"], "FT": [1.0, np.nan]})

        result = prepare_control_sheet(df)

        assert len(result) == 2

    def test_no_unnamed_1_column(self):
        df = pd.DataFrame({"Unnamed: 0": ["Drive"], "FT": [1.0]})

        result = prepare_control_sheet(df)

        assert list(result.columns) == ["Shot Type", "FT"]


class TestForwardFillShotType:
    def test_fills_nan_with_previous_string(self):
        shot_type = pd.Series(["Drive", np.nan, np.nan, "Cut", np.nan])

        result = forward_fill_shot_type(shot_type)

        assert result.tolist() == ["Drive", "Drive", "Drive", "Cut", "Cut"]

    def test_all_strings_unchanged(self):
        shot_type = pd.Series(["Drive", "Cut", "Pull"])

        result = forward_fill_shot_type(shot_type)

        assert result.tolist() == ["Drive", "Cut", "Pull"]

    def test_does_not_mutate_input(self):
        shot_type = pd.Series(["Drive", np.nan])

        forward_fill_shot_type(shot_type)

        assert pd.isna(shot_type[1])


class TestBuildBallTypeDict:
    def test_maps_symbols_to_names(self, parameters_df):
        result = build_ball_type_dict(parameters_df)

        assert result == {
            "FT": "Full Toss",
            "Y": "Yorker",
            "GY": "Googly",
            "LB": "Leg Break",
            "SL": "Short",
        }

    def test_drops_nan_rows(self):
        parameters_df = pd.DataFrame(
            {
                "Ball Type": ["Full Toss", np.nan],
                "ball symbol": ["FT", "Y"],
            }
        )

        result = build_ball_type_dict(parameters_df)

        assert result == {"FT": "Full Toss"}


class TestGenerateBallwiseDfs:
    @pytest.fixture
    def control_df(self):
        return pd.DataFrame(
            {
                "FT": [4, 1],
                "FT.1": [1.0, 0.0],
                "FT.2": [0.8, 0.2],
                "Y": [0, 2],
                "Y.1": [0.0, 1.0],
                "Y.2": [0.1, 0.6],
            }
        )

    def test_one_frame_per_ball_type(self, control_df):
        shot_type = pd.Series(["Drive", "Cut"])
        ball_type_dict = {"FT": "Full Toss", "Y": "Yorker"}

        result = generate_ballwise_dfs(control_df, shot_type, ball_type_dict)

        assert set(result.keys()) == {"Full Toss_shots_df", "Yorker_shots_df"}

    def test_frame_structure(self, control_df):
        shot_type = pd.Series(["Drive", "Cut"])
        ball_type_dict = {"FT": "Full Toss", "Y": "Yorker"}

        result = generate_ballwise_dfs(control_df, shot_type, ball_type_dict)
        ft = result["Full Toss_shots_df"]

        assert list(ft.columns) == [
            "Ball Type",
            "Shot Type",
            "Runs Scored",
            "Correct execution of shot according to the ball",
            "Effectiveness of the shot execution",
        ]
        assert ft["Ball Type"].tolist() == ["FT", "FT"]
        assert ft["Shot Type"].tolist() == ["Drive", "Cut"]
        assert ft["Runs Scored"].tolist() == [4, 1]

    def test_does_not_mutate_input(self, control_df):
        original = control_df.copy()
        shot_type = pd.Series(["Drive", "Cut"])

        generate_ballwise_dfs(
            control_df, shot_type, {"FT": "Full Toss", "Y": "Yorker"}
        )

        pd.testing.assert_frame_equal(control_df, original)
