"""Pluggable models for expected games played.

Everything predicts an *availability rate* in [0, 1]; callers multiply by 17 to get
expected games. Models share one interface -- fit(X_df, y) / predict(X_df) on a pandas
DataFrame with `history.FEATURE_COLS` -- so you can drop new estimators into
`default_models()` (or pass your own dict) and compare them with `evaluate_models`.

`evaluate_models` uses forward-chaining CV by season (train only on seasons strictly
before each evaluation season) so the comparison never leaks future information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .history import FEATURE_COLS

NUMERIC = ["age", "experience", "draft_round", "prior_rate", "prior2_rate", "career_avg_rate", "prior_games"]
CATEGORICAL = ["position"]
GAMES_PER_SEASON = 17


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERIC),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="UNK")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )


def _pipe(estimator) -> Pipeline:
    return Pipeline([("prep", _preprocessor()), ("model", estimator)])


class ColumnBaseline(BaseEstimator, RegressorMixin):
    """Predict a single feature column (e.g. prior_rate), falling back to train mean.

    A deliberately dumb reference point: if a fancy model can't beat "assume this
    player repeats last year's availability", it isn't earning its complexity.
    """

    def __init__(self, column: str = "prior_rate"):
        self.column = column

    def fit(self, X: pd.DataFrame, y):
        self.fallback_ = float(np.nanmean(y))
        return self

    def predict(self, X: pd.DataFrame):
        vals = X[self.column].to_numpy(dtype=float, na_value=np.nan)
        return np.where(np.isnan(vals), self.fallback_, vals)


def default_models() -> dict[str, object]:
    """Registry of candidate models. Extend or replace freely."""
    return {
        "baseline_prior": ColumnBaseline("prior_rate"),
        "baseline_career": ColumnBaseline("career_avg_rate"),
        "ridge": _pipe(Ridge(alpha=1.0)),
        "random_forest": _pipe(
            RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1)
        ),
        "grad_boost": _pipe(
            GradientBoostingRegressor(random_state=0)
        ),
    }


def _Xy(df: pd.DataFrame):
    return df[FEATURE_COLS].copy(), df["target_rate"].to_numpy(dtype=float)


def evaluate_models(
    train_df: pd.DataFrame,
    models: dict[str, object] | None = None,
    eval_seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Forward-chaining CV. Returns MAE (in games) and bias per model, sorted best-first.

    For each eval season s, train on rows with season < s and score on season s.
    MAE is reported in games (rate error * 17) so it's directly interpretable.
    """
    models = models or default_models()
    seasons = sorted(train_df["season"].unique())
    if eval_seasons is None:
        # evaluate on the most recent seasons that still have >=1 prior training season
        eval_seasons = [s for s in seasons if s > seasons[0]][-4:]

    rows = []
    for name, proto in models.items():
        abs_errs, signed = [], []
        for s in eval_seasons:
            tr = train_df[train_df["season"] < s]
            te = train_df[train_df["season"] == s]
            if len(tr) == 0 or len(te) == 0:
                continue
            from sklearn.base import clone

            model = clone(proto) if not isinstance(proto, ColumnBaseline) else ColumnBaseline(proto.column)
            Xtr, ytr = _Xy(tr)
            Xte, yte = _Xy(te)
            model.fit(Xtr, ytr)
            pred = np.clip(model.predict(Xte), 0.0, 1.0)
            err_games = (pred - yte) * GAMES_PER_SEASON
            abs_errs.append(np.abs(err_games))
            signed.append(err_games)
        all_abs = np.concatenate(abs_errs)
        rows.append(
            {
                "model": name,
                "mae_games": float(all_abs.mean()),
                "bias_games": float(np.concatenate(signed).mean()),
                "n_eval": int(all_abs.size),
                "eval_seasons": ",".join(map(str, eval_seasons)),
            }
        )
    return pd.DataFrame(rows).sort_values("mae_games").reset_index(drop=True)


def fit_predict_games(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    model_name: str = "grad_boost",
    models: dict[str, object] | None = None,
) -> pd.Series:
    """Fit `model_name` on all training rows, predict expected games for target rows.

    Returns a Series (index aligned to target_df) of expected games in [0, 17].
    """
    models = models or default_models()
    if model_name not in models:
        raise KeyError(f"{model_name!r} not in models: {list(models)}")
    model = models[model_name]
    Xtr, ytr = _Xy(train_df)
    model.fit(Xtr, ytr)
    rate = np.clip(model.predict(target_df[FEATURE_COLS]), 0.0, 1.0)
    return pd.Series(rate * GAMES_PER_SEASON, index=target_df.index, name="expected_games")
