import pandas as pd

from cricket_analytics.dataframes import (
    dataframe_names,
    filter_names,
    print_unique_values,
    symbol_list,
    unique_values,
)


def test_dataframe_names_snake_cases_sheet_names():
    assert dataframe_names(["control sheet", "IND_vs_AFG", "analysis parameters"]) == [
        "control_sheet_df",
        "ind_vs_afg_df",
        "analysis_parameters_df",
    ]


def test_filter_names_selects_match_and_non_match_sheets():
    names = ["control_sheet_df", "ind_vs_afg_df", "parameters_df", "sa_vs_eng_df"]

    assert list(filter_names(names, contains="vs")) == [
        "ind_vs_afg_df",
        "sa_vs_eng_df",
    ]
    assert list(filter_names(names, excludes="vs")) == [
        "control_sheet_df",
        "parameters_df",
    ]


def test_unique_values_skips_sno():
    df = pd.DataFrame({"Sno": [1, 2, 3], "Control": [1.0, 0.0, 1.0]})

    assert unique_values(df) == [("Control", [1.0, 0.0])]


def test_symbol_list_upper_cases_and_drops_blanks():
    parameters_df = pd.DataFrame({"ball symbol": ["oc", None, "Y", "lb"]})

    assert symbol_list(parameters_df, "ball symbol") == ["OC", "Y", "LB"]


def test_print_unique_values_only_prints_selected_frames(capsys):
    frames = {
        "ind_vs_afg_df": pd.DataFrame({"Control": [1.0]}),
        "parameters_df": pd.DataFrame({"ball symbol": ["OC"]}),
    }

    print_unique_values(frames, contains="vs")

    out = capsys.readouterr().out
    assert "ind_vs_afg_df" in out
    assert "Control : " in out
    assert "parameters_df" not in out
