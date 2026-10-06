"""Data cleaning and feature engineering for the freight rate model."""
from __future__ import annotations

import numpy as np
import pandas as pd

START_DATE = pd.Timestamp("2025-01-01")
QUARTER_END_MONTHS = (3, 6, 9, 12)
CATEGORICAL = ["pickup", "delivery", "equipment"]


def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def daily_market_index(*frames: pd.DataFrame) -> pd.Series:
    """Average market_index per date.

    market_index is a daily market level plus row-level noise (std ~0.025 within a day),
    so the daily mean is a cleaner signal. It only uses features, never labels, so
    validation.csv can be included to get the Nov/Dec market levels.
    """
    rows = pd.concat([f[["date", "market_index"]] for f in frames])
    return rows.groupby("date")["market_index"].mean()


def city_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    """One (lat, lon) per city. Coordinates are consistent per city in the data."""
    pick = frame[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1)
    drop = frame[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1)
    return pd.concat([pick, drop]).groupby("city")[["lat", "lon"]].first()


class FeatureBuilder:
    """Cleans raw rows and produces model features. Fitted on training data only."""

    def fit(self, train: pd.DataFrame, market_by_day: pd.Series) -> "FeatureBuilder":
        self.market_by_day = market_by_day
        weights = train["weight"].abs()
        self.weight_median = weights.groupby(train["equipment"]).median()
        self.coords = city_coordinates(train)
        self.categories = {c: sorted(train[c].unique()) for c in CATEGORICAL}
        return self

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        d = raw.copy()

        # Lat/lon are missing in the December chart inputs, so fill them from the city lookup.
        for side in ("pickup", "delivery"):
            for axis in ("lat", "lon"):
                col = f"{side}_{axis}"
                lookup = d[side].map(self.coords[axis])
                d[col] = d[col].fillna(lookup) if col in d else lookup

        # Data quality: ~0.6% of weights are negative (sign errors) and ~0.6% are missing.
        d["weight_missing"] = d["weight"].isna().astype(int)
        d["weight"] = d["weight"].abs()
        d["weight"] = d["weight"].fillna(d["equipment"].map(self.weight_median))

        # Market level: daily mean (denoised) and the row value (missing -> daily mean).
        d["mi_day"] = d["date"].map(self.market_by_day)
        if "market_index" not in d:
            d["market_index"] = np.nan
        d["mi_missing"] = d["market_index"].isna().astype(int)
        d["mi"] = d["market_index"].fillna(d["mi_day"])
        d["log_mi_day"] = np.log(d["mi_day"])
        d["log_mi"] = np.log(d["mi"])

        # Distance terms: rate has a fixed component plus a per-mile component.
        d["log_distance"] = np.log(d["distance"])
        d["log_distance_sq"] = d["log_distance"] ** 2
        d["weight_k"] = d["weight"] / 10_000
        d["weight_k_sq"] = d["weight_k"] ** 2

        # quote_signal has a U-shaped relation with the residual rate.
        if "quote_signal" in d:
            d["quote_dev_sq"] = (d["quote_signal"] - 2.05) ** 2

        # Calendar: a slow upward trend, plus a ramp through each quarter-end month
        # (rates climb ~4% from the 1st to the last day of Mar/Jun/Sep, then reset).
        d["t_years"] = (d["date"] - START_DATE).dt.days / 365.0
        is_qe = d["date"].dt.month.isin(QUARTER_END_MONTHS).astype(float)
        frac = (d["date"].dt.day - 1) / (d["date"].dt.days_in_month - 1)
        d["quarter_end_month"] = is_qe
        d["quarter_end_ramp"] = is_qe * frac

        for c in CATEGORICAL:
            d[c] = pd.Categorical(d[c], categories=self.categories[c])
        return d
