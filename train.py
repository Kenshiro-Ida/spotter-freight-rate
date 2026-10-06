"""Train, validate and generate submission files.

    python train.py

Writes to outputs/:
    validation_predictions.csv     load_id,predicted_rate for the 12,000 validation loads
    december_chart_inputs.csv      the December inputs with predicted_rate filled
    cv_results.csv / metrics.json  validation numbers used in the report
    figures/*.png                  charts used in the report
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import FeatureBuilder, daily_market_index, load
from src.model import GBM_PARAMS, HybridRateModel, flag_rate_outliers

DATA = Path("data")
OUT = Path("outputs")
FIG = OUT / "figures"

# Forward-chaining time splits: always train on the past, test on the next block.
# Each test block mirrors the real task (forecast 1-2 months ahead) and two of
# them contain a quarter-end month, like December.
TIME_FOLDS = [
    ("2025-06-01", "2025-07-01"),  # train Jan-May  -> test Jun
    ("2025-07-01", "2025-09-01"),  # train Jan-Jun  -> test Jul-Aug
    ("2025-09-01", "2025-11-01"),  # train Jan-Aug  -> test Sep-Oct
]


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - actual
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_%": float(100 * np.mean(np.abs(err) / actual)),
        "bias_%": float(100 * np.mean(err / actual)),
    }


# ---------------------------------------------------------------- candidate models
class LaneMedianBaseline:
    """Median rate-per-mile for lane x equipment (falls back to equipment) x distance."""

    def fit(self, d, y_log):
        rpm = np.exp(y_log) / d["distance"]
        key = d["pickup"].astype(str) + ">" + d["delivery"].astype(str) + "|" + d["equipment"].astype(str)
        self.lane = rpm.groupby(key).median()
        self.equip = rpm.groupby(d["equipment"].astype(str)).median()
        return self

    def predict(self, d):
        key = d["pickup"].astype(str) + ">" + d["delivery"].astype(str) + "|" + d["equipment"].astype(str)
        rpm = key.map(self.lane).fillna(d["equipment"].astype(str).map(self.equip))
        return (rpm * d["distance"]).to_numpy()


class PlainGBM:
    """LightGBM on every feature including market index and date ordinal."""

    feats = ["pickup", "delivery", "equipment", "log_distance", "weight", "quote_signal",
             "mi", "mi_day", "t_years", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]

    def fit(self, d, y_log):
        self.m = lgb.LGBMRegressor(**{**GBM_PARAMS, "n_estimators": 1500}).fit(d[self.feats], y_log)
        return self

    def predict(self, d):
        return np.exp(self.m.predict(d[self.feats]))


CANDIDATES = {
    "Lane median rate/mile": LaneMedianBaseline,
    "LightGBM (all features)": PlainGBM,
    "Linear only (log space)": lambda: HybridRateModel(use_gbm=False),
    "Hybrid: linear + GBM residual": lambda: HybridRateModel(use_gbm=True),
}


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    train_raw = load(DATA / "train_test.csv")
    valid_raw = load(DATA / "validation.csv")
    december_raw = load(DATA / "december_chart_inputs.csv")
    market_by_day = daily_market_index(train_raw, valid_raw)

    fb = FeatureBuilder().fit(train_raw, market_by_day)
    train = fb.transform(train_raw)
    y_log = np.log(train["posted_rate"])

    # ---------------------------------------------------------- data quality
    outlier = flag_rate_outliers(train, y_log)
    quality = {
        "rows": len(train_raw),
        "negative_weight": int((train_raw["weight"] < 0).sum()),
        "missing_weight": int(train_raw["weight"].isna().sum()),
        "missing_market_index": int(train_raw["market_index"].isna().sum()),
        "rate_outliers_removed": int(outlier.sum()),
        "duplicate_load_ids": int(train_raw["load_id"].duplicated().sum()),
    }
    print("Data quality:", quality)
    clean = ~outlier

    # ---------------------------------------------------------- validation
    rows = []
    for name, make in CANDIDATES.items():
        for start, end in TIME_FOLDS:
            tr = clean & (train["date"] < start)
            te = (train["date"] >= start) & (train["date"] < end)
            model = make().fit(train[tr], y_log[tr])
            pred = model.predict(train[te])
            m_all = metrics(train.loc[te, "posted_rate"].to_numpy(), pred)
            te_clean = te & clean
            m_clean = metrics(train.loc[te_clean, "posted_rate"].to_numpy(), model.predict(train[te_clean]))
            rows.append({"model": name, "test_start": start, "test_end": end,
                         **{f"{k} (clean)": v for k, v in m_clean.items()},
                         "MAPE_% (all rows)": m_all["MAPE_%"]})
    cv = pd.DataFrame(rows)
    cv.to_csv(OUT / "cv_results.csv", index=False)
    summary = cv.groupby("model", sort=False)[["MAE (clean)", "RMSE (clean)", "MAPE_% (clean)",
                                               "bias_% (clean)", "MAPE_% (all rows)"]].mean()
    print("\nForward-chaining time validation (mean of 3 folds):\n", summary.round(2))

    # Random K-fold, only to show how optimistic a shuffled split is here.
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 5, len(train))
    rand = []
    for k in range(5):
        tr, te = clean & (fold != k), clean & (fold == k)
        m = HybridRateModel().fit(train[tr], y_log[tr])
        rand.append(metrics(train.loc[te, "posted_rate"].to_numpy(), m.predict(train[te]))["MAPE_%"])
    print(f"Random 5-fold MAPE (hybrid, for comparison only): {np.mean(rand):.2f}%")

    # ---------------------------------------------------------- final model
    final = HybridRateModel().fit(train[clean], y_log[clean])
    coefs = final.linear_coefficients()

    valid = fb.transform(valid_raw)
    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    pred_valid = pd.Series(final.predict(valid), index=valid["load_id"])
    submission = template[["load_id"]].copy()
    submission["predicted_rate"] = submission["load_id"].map(pred_valid).round(2)
    assert submission["predicted_rate"].notna().all() and len(submission) == 12_000
    submission.to_csv(OUT / "validation_predictions.csv", index=False)

    # December chart: fixed lane/equipment/weight, only the date changes. These rows
    # have no quote_signal, so we average predictions over the quote_signal values
    # actually seen in December (marginalising instead of plugging in one value).
    dec = fb.transform(december_raw)
    dec_quotes = valid_raw.loc[valid_raw["date"].dt.month == 12, "quote_signal"].to_numpy()
    preds = []
    for q in dec_quotes:
        dq = dec.copy()
        dq["quote_signal"] = q
        dq["quote_dev_sq"] = (q - 2.05) ** 2
        preds.append(final.predict(dq))
    december = december_raw.copy()
    december["date"] = december["date"].dt.strftime("%Y-%m-%d")
    december["predicted_rate"] = np.mean(preds, axis=0).round(2)
    december.to_csv(OUT / "december_chart_inputs.csv", index=False)

    # ---------------------------------------------------------- report figures
    make_figures(train, y_log, outlier, market_by_day, final, fb)
    with open(OUT / "metrics.json", "w") as f:
        json.dump({
            "data_quality": quality,
            "time_cv_summary": summary.round(3).reset_index().to_dict(orient="records"),
            "random_kfold_mape_%": round(float(np.mean(rand)), 3),
            "linear_coefficients": {k: round(float(coefs[k]), 4) for k in
                                    ["log_mi", "log_mi_day", "t_years", "quarter_end_month",
                                     "quarter_end_ramp", "quote_signal", "quote_dev_sq"]},
            "validation_prediction_mean": float(submission["predicted_rate"].mean()),
            "december_min_max": [float(december["predicted_rate"].min()),
                                 float(december["predicted_rate"].max())],
        }, f, indent=2)
    print("\nWrote outputs/validation_predictions.csv and outputs/december_chart_inputs.csv")


def make_figures(train, y_log, outlier, market_by_day, model, fb) -> None:
    # 1. Rate per mile: shows the two outlier clusters.
    rpm = train["posted_rate"] / train["distance"]
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    ax.hist(np.log10(rpm[~outlier]), bins=80, color="#064A56", label="kept")
    ax.hist(np.log10(rpm[outlier]), bins=80, color="#D1495B", label="flagged as rate error")
    ax.set_yscale("log")
    ax.set_xlabel("log10(posted_rate / distance)")
    ax.set_ylabel("loads (log scale)")
    ax.set_title("Rate per mile: ~1.4% of loads are 2-6x off", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "rate_outliers.png")
    plt.close(fig)

    # 2. Daily rate level after removing lane/equipment/distance/weight/market effects.
    clean = ~outlier
    d = train[clean].copy()
    static = model.linear_coefficients()
    time_cols = ["t_years", "quarter_end_month", "quarter_end_ramp"]
    X_time = d[time_cols].astype(float)
    full_log = model.predict_log(d)
    time_part = X_time.to_numpy() @ static[time_cols].to_numpy()
    resid_time = (y_log[clean] - full_log + time_part).groupby(d["date"]).mean()
    fitted = pd.Series(time_part, index=d.index).groupby(d["date"]).mean()
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150)
    ax.plot(resid_time.index, resid_time - resid_time.mean(), color="#9DAFB3", lw=1, label="observed daily level")
    ax.plot(fitted.index, fitted - resid_time.mean(), color="#064A56", lw=2, label="model: trend + quarter-end ramp")
    ax.set_ylabel("log rate effect")
    ax.set_title("Rates climb through each quarter-end month (Mar, Jun, Sep), then reset", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "quarter_end_ramp.png")
    plt.close(fig)

    # 3. Daily market index across train + validation period.
    fig, ax = plt.subplots(figsize=(9, 3.0), dpi=150)
    ax.plot(market_by_day.index, market_by_day.values, color="#064A56", lw=1.2)
    ax.axvline(pd.Timestamp("2025-11-01"), color="#D1495B", ls="--", lw=1)
    ax.text(pd.Timestamp("2025-11-03"), market_by_day.max() * 0.98, "validation period", color="#D1495B", fontsize=8)
    ax.set_title("Daily mean market_index", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "market_index.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
