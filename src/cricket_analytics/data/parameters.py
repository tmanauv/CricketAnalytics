"""Symbol legends from the ``parameters`` sheet."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SymbolLegend:
    ball_types: dict[str, str]
    shot_types: dict[str, str]
    dismissals: dict[str, str]

    @property
    def ball_symbols(self) -> set[str]:
        return set(self.ball_types)

    @property
    def shot_symbols(self) -> set[str]:
        return set(self.shot_types)

    @property
    def dismissal_symbols(self) -> set[str]:
        return set(self.dismissals)

    def dismissal_name_to_symbol(self) -> dict[str, str]:
        return {name.upper(): symbol for symbol, name in self.dismissals.items()}


def _pair(df: pd.DataFrame, symbol_col: str, name_col: str) -> dict[str, str]:
    pairs = df[[symbol_col, name_col]].dropna()
    return {
        str(symbol).strip().upper(): str(name).strip()
        for symbol, name in zip(pairs[symbol_col], pairs[name_col], strict=True)
    }


def parse_parameters(parameters_df: pd.DataFrame) -> SymbolLegend:
    """Build symbol -> name maps for balls, shots and dismissals."""
    return SymbolLegend(
        ball_types=_pair(parameters_df, "ball symbol", "Ball Type"),
        shot_types=_pair(parameters_df, "shot symbol", "Shot Type"),
        dismissals=_pair(parameters_df, "dismisal symbol", "Dismissal Kind"),
    )
