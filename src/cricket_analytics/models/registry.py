"""Named regressors used by the baseline comparison and tuning stages.

Every model is wrapped in a ``Pipeline`` with a ``StandardScaler`` so scaling
is fit inside each CV fold instead of on the full dataset.
"""

from __future__ import annotations

from collections.abc import Callable

from catboost import CatBoostRegressor
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

MODEL_STEP = "model"

_FACTORIES: dict[str, Callable[[int | None], RegressorMixin]] = {
    "svr": lambda seed: SVR(kernel="rbf"),
    "linear": lambda seed: LinearRegression(),
    "random_forest": lambda seed: RandomForestRegressor(random_state=seed, n_jobs=-1),
    "xgboost": lambda seed: XGBRegressor(random_state=seed, n_jobs=-1, verbosity=0),
    "gradient_boosting": lambda seed: GradientBoostingRegressor(random_state=seed),
    "decision_tree": lambda seed: DecisionTreeRegressor(random_state=seed),
    "catboost": lambda seed: CatBoostRegressor(
        random_state=seed, verbose=0, allow_writing_files=False, thread_count=-1
    ),
}

AVAILABLE_MODELS = tuple(_FACTORIES)


def make_model(name: str, random_state: int | None = None) -> Pipeline:
    if name not in _FACTORIES:
        raise KeyError(f"Unknown model {name!r}; choose from {AVAILABLE_MODELS}")
    return Pipeline([("scaler", StandardScaler()), (MODEL_STEP, _FACTORIES[name](random_state))])


def prefixed_grid(grid: dict[str, list]) -> dict[str, list]:
    """Prefix param names so they address the model step of the pipeline."""
    return {f"{MODEL_STEP}__{key}": list(values) for key, values in grid.items()}
