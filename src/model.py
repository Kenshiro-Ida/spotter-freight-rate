"""Hybrid freight rate model: a linear model in log space for the parts that must
extrapolate in time (market level, trend, quarter-end ramp), plus a gradient
boosted model on its residuals for nonlinear lane / equipment / weight effects."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .features import CATEGORICAL

LINEAR_NUMERIC = [
    "log_distance", "log_distance_sq", "weight_k", "weight_k_sq",
    "quote_signal", "quote_dev_sq", "log_mi", "log_mi_day",
    "t_years", "quarter_end_month", "quarter_end_ramp",
]
# The boosted stage gets no calendar or market features on purpose: tree models
# can't extrapolate, so anything time-dependent is left to the linear stage.
GBM_FEATURES = [
    "pickup", "delivery", "equipment", "log_distance", "weight", "quote_signal",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "weight_missing",
]
GBM_PARAMS = dict(
    n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=42, verbose=-1,
)


def _linear_matrix(d: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(d[CATEGORICAL], drop_first=True, dtype=float)
    return pd.concat([d[LINEAR_NUMERIC].astype(float), dummies], axis=1)


class HybridRateModel:
    def __init__(self, use_gbm: bool = True, ridge_alpha: float = 1.0):
        self.use_gbm = use_gbm
        self.ridge_alpha = ridge_alpha

    def fit(self, d: pd.DataFrame, y_log: pd.Series) -> "HybridRateModel":
        X = _linear_matrix(d)
        self.columns = X.columns
        self.linear = Ridge(alpha=self.ridge_alpha).fit(X, y_log)
        if self.use_gbm:
            residual = y_log - self.linear.predict(X)
            self.gbm = lgb.LGBMRegressor(**GBM_PARAMS).fit(d[GBM_FEATURES], residual)
        return self

    def predict_log(self, d: pd.DataFrame) -> np.ndarray:
        X = _linear_matrix(d).reindex(columns=self.columns, fill_value=0.0)
        pred = self.linear.predict(X)
        if self.use_gbm:
            pred = pred + self.gbm.predict(d[GBM_FEATURES])
        return pred

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(d))

    def linear_coefficients(self) -> pd.Series:
        return pd.Series(self.linear.coef_, index=self.columns)


def flag_rate_outliers(d: pd.DataFrame, y_log: pd.Series, threshold: float = 0.35) -> pd.Series:
    """Flag rows whose posted_rate is wildly off what similar loads pay.

    Two passes of the hybrid model: fit on everything, flag big residuals, refit on
    the rest and flag again. Residuals are clearly bimodal: ~98.6% of loads sit within
    +/-0.25 log units, while ~1.4% sit beyond +/-0.6 (about 2x to 6x too high or too
    low), with nothing in between. Those look like entry errors and are dropped from
    training. Any threshold inside the empty gap (0.25-0.6) gives the same result.
    """
    keep = pd.Series(True, index=d.index)
    for _ in range(2):
        model = HybridRateModel().fit(d[keep], y_log[keep])
        residual = (y_log - model.predict_log(d)).abs()
        keep = residual <= threshold
    return ~keep
