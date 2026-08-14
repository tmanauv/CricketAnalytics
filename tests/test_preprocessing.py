import numpy as np
import pandas as pd

from cricket_analytics.preprocessing import (
    chi_square_filter,
    label_encode,
    scale_and_split,
    split_variable_types,
)


class TestSplitVariableTypes:
    def test_splits_by_dtype(self):
        df = pd.DataFrame(
            {"name": ["a", "b"], "runs": [1.0, 2.0], "count": [1, 2]}
        )

        categorical, continuous = split_variable_types(df)

        assert categorical == ["name"]
        assert continuous == ["runs", "count"]

    def test_all_numeric(self):
        df = pd.DataFrame({"a": [1], "b": [2.0]})

        categorical, continuous = split_variable_types(df)

        assert categorical == []
        assert continuous == ["a", "b"]


class TestChiSquareFilter:
    def test_drops_unrelated_column(self):
        rng = np.random.default_rng(0)
        n = 200
        target = rng.choice([0.0, 1.0], size=n)
        related = np.where(target == 1.0, "hi", "lo")
        unrelated = rng.choice(["x", "y"], size=n)
        df = pd.DataFrame(
            {
                "related": related,
                "unrelated": unrelated,
                "Effectiveness": target,
            }
        )

        result = chi_square_filter(df, ["related", "unrelated"])

        assert "related" in result.columns
        assert "unrelated" not in result.columns

    def test_skips_empty_contingency_table(self):
        df = pd.DataFrame({"col": pd.Series(dtype=str), "Effectiveness": pd.Series(dtype=float)})

        result = chi_square_filter(df, ["col"])

        assert "col" in result.columns

    def test_does_not_mutate_input(self):
        df = pd.DataFrame(
            {
                "col": ["x", "y"] * 50,
                "Effectiveness": [0.0, 1.0] * 50,
            }
        )
        original = df.copy()

        chi_square_filter(df, ["col"])

        pd.testing.assert_frame_equal(df, original)


class TestLabelEncode:
    def test_encodes_categorical_columns(self):
        df = pd.DataFrame({"shot": ["DR", "C", "DR"], "runs": [1.0, 2.0, 3.0]})

        result = label_encode(df, ["shot"])

        assert result["shot"].dtype != "object"
        assert result["shot"].tolist() == [1, 0, 1]

    def test_leaves_non_categorical_untouched(self):
        df = pd.DataFrame({"shot": ["DR", "C"], "runs": [1.0, 2.0]})

        result = label_encode(df, ["shot"])

        assert result["runs"].tolist() == [1.0, 2.0]


class TestScaleAndSplit:
    def test_split_shapes(self):
        df = pd.DataFrame(
            {
                "f1": np.arange(10, dtype=float),
                "f2": np.arange(10, 20, dtype=float),
                "Effectiveness": np.linspace(0, 1, 10),
            }
        )

        X_train, X_test, y_train, y_test = scale_and_split(df)

        assert X_train.shape == (8, 2)
        assert X_test.shape == (2, 2)
        assert len(y_train) == 8
        assert len(y_test) == 2

    def test_features_are_scaled(self):
        df = pd.DataFrame(
            {
                "f1": np.arange(100, dtype=float),
                "Effectiveness": np.linspace(0, 1, 100),
            }
        )

        X_train, X_test, _, _ = scale_and_split(df)
        all_X = np.concatenate([X_train, X_test])

        assert abs(all_X.mean()) < 1e-9
        assert abs(all_X.std() - 1.0) < 1e-9

    def test_reproducible_with_random_state(self):
        df = pd.DataFrame(
            {
                "f1": np.arange(10, dtype=float),
                "Effectiveness": np.linspace(0, 1, 10),
            }
        )

        _, _, y_train_a, _ = scale_and_split(df, random_state=42)
        _, _, y_train_b, _ = scale_and_split(df, random_state=42)

        assert y_train_a.tolist() == y_train_b.tolist()
