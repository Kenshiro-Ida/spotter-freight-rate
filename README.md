# Freight Rate Prediction: Spotter ML Assessment

Predicts the posted rate for a truckload from lane, distance, equipment, weight, date and market signals.

**Result.** Forward-chaining time validation (3 folds, always train on the past and test on the next 1–2 months):

| Model | MAE ($) | MAPE |
|---|---|---|
| Lane median rate per mile | 114 | 4.84% |
| LightGBM (all features) | 88 | 4.58% |
| Linear model, log space | 43 | 1.79% |
| **Hybrid: linear + LightGBM on residuals (final)** | **37** | **1.53%** |

Notes on the table:

- Metrics are on clean rows. Including the ~1.4% of rows with rate entry errors, the final model's MAPE is 3.8%.
- A shuffled 5-fold split gives 1.28% MAPE for the same model. That number is optimistic because the real task is forecasting future months.

## Quick start

```bash
python -m pip install -r requirements.txt
python train.py   # ~1-2 min: validation, final fit, writes outputs/
python score.py --predictions outputs/validation_predictions.csv \
                --december-predictions outputs/december_chart_inputs.csv
```

`score.py` is Spotter's scorer (unchanged). It validates both files and writes `scorer_results/candidate_december.png`.

## Layout

```
data/                      provided CSVs (train_test, validation, template, december inputs)
src/features.py            cleaning + feature engineering (FeatureBuilder)
src/model.py               HybridRateModel + outlier flagging
train.py                   validation, final training, submission files, report figures
score.py                   Spotter's scorer
outputs/
  validation_predictions.csv      load_id,predicted_rate (12,000 rows)  <- submission
  december_chart_inputs.csv       December inputs with predicted_rate filled
  cv_results.csv, metrics.json    validation numbers
  figures/                        charts used in the report
report/Freight_Rate_Report.pdf    write-up
```

## Approach in short

1. **Data quality.**
   - 292 negative weights (sign errors) are converted to absolute values.
   - 300 missing weights are filled with the median for that equipment type.
   - 374 missing `market_index` values are filled with that day's mean.
   - 677 rows (1.4%) have a `posted_rate` 2–6x off what similar loads pay. The residuals are clearly bimodal with an empty gap, so these are dropped from training as entry errors.
2. **Key findings.**
   - Rate scales with distance (log-log slope ~0.87, meaning a fixed cost plus a per-mile cost). Reefer is ~13% and flatbed ~8% above dry van.
   - `market_index` is a daily market level plus row noise, so the daily mean is a cleaner signal.
   - Rates trend up ~7% per year.
   - Rates **ramp up ~4% through every quarter-end month (Mar, Jun, Sep)** and reset on the 1st of the next quarter. December is a quarter-end month.
3. **Model.**
   - A ridge regression in log space handles everything that must extrapolate in time: market index, trend and quarter-end ramp, plus lane and equipment effects.
   - A LightGBM model on the residuals learns nonlinear lane, weight and quote interactions. It gets no time features, because trees can't extrapolate to Nov/Dec.
4. **Validation.** Forward-chaining time splits (train → Jun, Jul–Aug, Sep–Oct). Two of the three test blocks contain a quarter-end month, like December.
5. **December chart.** The fixed inputs have no `quote_signal`. Predictions are averaged over the December `quote_signal` values seen in `validation.csv` rather than plugging in a single value. Daily market level comes from the validation rows for those dates (features only, no labels).
