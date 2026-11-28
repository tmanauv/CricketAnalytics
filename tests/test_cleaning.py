import numpy as np
import pandas as pd

from cricket_analytics.cleaning import (
    build_matches_df,
    clean_match_df,
    clean_missing_values,
    drop_incorrect_data,
    normalize_batsman_names,
    unique_values,
    validate_column_values,
)


class TestUniqueValues:
    def test_excludes_sno(self):
        df = pd.DataFrame({"Sno": [1, 2], "a": ["x", "x"]})

        result = unique_values(df)

        assert result == [("a", ["x"])]

    def test_lists_unique_values_per_column(self):
        df = pd.DataFrame({"a": ["x", "y", "x"], "b": [1, 1, 2]})

        result = dict(unique_values(df))

        assert result["a"] == ["x", "y"]
        assert result["b"] == [1, 2]


class TestCleanMissingValues:
    def test_drops_unary_cols_with_nans(self):
        df = pd.DataFrame({"Pitch Type": ["Dry", np.nan], "Runs": [1.0, 2.0]})

        result = clean_missing_values(df)

        assert "Pitch Type" not in result.columns

    def test_keeps_unary_cols_without_nans(self):
        df = pd.DataFrame({"Pitch Type": ["Dry", "Dry"], "Runs": [1.0, 2.0]})

        result = clean_missing_values(df)

        assert "Pitch Type" in result.columns

    def test_binarises_presence_cols(self):
        df = pd.DataFrame({"Wicket": ["Wicket", np.nan], "Runs": [1.0, 2.0]})

        result = clean_missing_values(df)

        assert result["Wicket"].tolist() == [1.0, 0.0]

    def test_zero_fills_other_cols(self):
        df = pd.DataFrame({"Runs": [1.0, np.nan]})

        result = clean_missing_values(df)

        assert result["Runs"].tolist() == [1.0, 0.0]


class TestDropIncorrectData:
    def test_drops_rows_with_invalid_values(self):
        df = pd.DataFrame({"Ball type": ["FT", "ZZ", "Y"]})

        drop_incorrect_data(df, "Ball type", ["FT", "ZZ", "Y"], ["FT", "Y"])

        assert df["Ball type"].tolist() == ["FT", "Y"]

    def test_no_op_when_all_valid(self):
        df = pd.DataFrame({"Ball type": ["FT", "Y"]})

        drop_incorrect_data(df, "Ball type", ["FT", "Y"], ["FT", "Y"])

        assert len(df) == 2


class TestValidateColumnValues:
    def test_drops_invalid_shot_rows(self, parameters_df):
        df = pd.DataFrame(
            {"Type of Shot": ["DR", "ZZ"], "Runs": [1.0, 2.0]}
        )

        result = validate_column_values(df, parameters_df)

        assert result["Type of Shot"].tolist() == ["DR"]

    def test_drops_invalid_ball_type_rows(self, parameters_df):
        df = pd.DataFrame({"Ball type": ["FT", "QQ"], "Runs": [1.0, 2.0]})

        result = validate_column_values(df, parameters_df)

        assert result["Ball type"].tolist() == ["FT"]

    def test_control_minus_one_becomes_zero(self, parameters_df):
        df = pd.DataFrame({"Control": [1.0, -1.0, 0.0]})

        result = validate_column_values(df, parameters_df)

        assert result["Control"].tolist() == [1.0, 0.0, 0.0]

    def test_drops_invalid_control_values(self, parameters_df):
        df = pd.DataFrame({"Control": [1.0, 5.0]})

        result = validate_column_values(df, parameters_df)

        assert result["Control"].tolist() == [1.0]

    def test_drops_invalid_effectiveness_values(self, parameters_df):
        df = pd.DataFrame({"Effectiveness": [0.5, 2.5, -1.0]})

        result = validate_column_values(df, parameters_df)

        assert result["Effectiveness"].tolist() == [0.5, -1.0]

    def test_dismissal_c_normalised_to_ct(self, parameters_df):
        df = pd.DataFrame({"Dismissal kind": ["C", 0, "B"]})

        result = validate_column_values(df, parameters_df)

        assert result["Dismissal kind"].tolist() == ["Ct", 0, "B"]

    def test_long_dismissal_names_mapped_to_symbols(self, parameters_df):
        df = pd.DataFrame({"Dismissal kind": ["Bowled", 0]})

        result = validate_column_values(df, parameters_df)

        assert result["Dismissal kind"].tolist() == ["B", 0]


class TestCleanMatchDf:
    def test_full_pipeline(self, match_df, parameters_df):
        result = clean_match_df(match_df, parameters_df)

        assert "Sno" not in result.columns
        assert "Batsman Name" in result.columns
        assert "Bowler Name" in result.columns
        assert "Effectiveness" in result.columns
        assert "Pitch Type" not in result.columns
        assert result["Wicket"].tolist() == [0.0, 1.0, 0.0, 0.0]
        assert result["Runs"].tolist() == [4.0, 0.0, 1.0, 6.0]
        assert result["Control"].tolist() == [1.0, 0.0, 1.0, 0.0]


class TestNormalizeBatsmanNames:
    def test_fixes_known_misspellings(self):
        df = pd.DataFrame(
            {"Batsman Name": ["Kedar Jadhav", "Shikar Dhawan", "Rohit"]}
        )

        result = normalize_batsman_names(df)

        assert result["Batsman Name"].tolist() == [
            "Kedhar Jadhav",
            "Shikhar Dhawan",
            "Rohit",
        ]


class TestBuildMatchesDf:
    def test_stacks_only_vs_frames(self, match_df, parameters_df):
        df_dict = {
            "ind_vs_afg_df": match_df.copy(),
            "ind_vs_ban_df": match_df.copy(),
            "parameters_df": parameters_df,
        }

        result = build_matches_df(df_dict, parameters_df)

        assert len(result) == 8
        assert "Sno" not in result.columns
