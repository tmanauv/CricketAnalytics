"""Fit-on-train / apply-to-test feature construction.

The notebook computed batsman scores, chi-square selection and label encoding
on the full dataset before splitting. ``FeatureBuilder`` learns all of these
from the training rows only, so test metrics are not contaminated by the
test targets.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from cricket_analytics.data import columns as C
from cricket_analytics.features.batsman_scores import compute_batsman_scores
from cricket_analytics.features.selection import chi2_select, split_feature_types


class FeatureBuilder:
    def __init__(self, feature_cfg: Mapping[str, Any]):
        self.target: str = feature_cfg.get("target", C.TARGET)
        self.score_groups: Mapping[str, Mapping[str, Any]] = feature_cfg.get("score_groups", {})
        self.chi2_alpha: float = float(feature_cfg.get("chi2_alpha", 0.05))
        self.drop_columns: list[str] = list(feature_cfg.get("drop_columns", []))

    def fit(self, df: pd.DataFrame) -> FeatureBuilder:
        self.batsman_scores_ = compute_batsman_scores(df, self.score_groups)
        enriched = self._add_scores(df)
        enriched = enriched.drop(columns=self.drop_columns, errors="ignore")

        categorical, continuous = split_feature_types(enriched, self.target)
        self.categorical_, self.chi2_p_values_ = chi2_select(
            enriched, categorical, self.target, self.chi2_alpha
        )
        self.continuous_ = continuous
        self.feature_names_ = self.categorical_ + self.continuous_

        self.encoder_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        if self.categorical_:
            self.encoder_.fit(enriched[self.categorical_].astype(str))
        return self

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        enriched = self._add_scores(df)
        X = enriched[self.feature_names_].copy()
        if self.categorical_:
            X[self.categorical_] = self.encoder_.transform(X[self.categorical_].astype(str))
        return X.astype(float), enriched[self.target].astype(float)

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return self.fit(df).transform(df)

    def _add_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        score_cols = [c for c in self.batsman_scores_.columns if c != C.BATSMAN]
        merged = df.merge(self.batsman_scores_, on=C.BATSMAN, how="left")
        merged[score_cols] = merged[score_cols].fillna(0.0)
        return merged
