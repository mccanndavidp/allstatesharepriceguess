# Allstate Share Price Guess

A Python app that pulls 5-year historic closing prices for **Allstate (ALL)**
and the **S&P 500 (^GSPC)**, builds a simple polynomial regression model for
each, and predicts the share price at any date you choose.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Predict price one year from today (default)
python app.py

# 3. Predict price on a specific date
python app.py --date 2026-12-31

# 4. Tune the model (higher polynomial degree = more curvature)
python app.py --date 2026-12-31 --degree 3

# 5. Save the chart to a custom file
python app.py --date 2026-12-31 --output allstate_forecast.png
```

## Output

For each ticker the app prints:

| Field | Description |
|-------|-------------|
| Data range | First and last trading day in the 5-year window |
| Last close | Most recent closing price |
| Model MAE | Mean Absolute Error of the regression fit |
| Model R² | Coefficient of determination of the fit |
| Predicted | Predicted closing price on the target date |

A chart (`price_prediction.png` by default) is also saved showing the
historical price series, the regression trend line, and the prediction point.

## How it works

1. **Data** — `yfinance` downloads 5 years of adjusted daily closing prices.
2. **Feature engineering** — dates are converted to *days since the first
   trading day* so the model only needs one numeric feature.
3. **Model** — a `PolynomialFeatures` transformer (degree 2 by default)
   followed by `sklearn.LinearRegression` is fitted on each ticker
   independently.
4. **Prediction** — the fitted model is evaluated at the ordinal value for
   the requested date.

> **Disclaimer** — This is a toy/educational model. Polynomial regression on
> time is not a reliable price forecasting method for real investment decisions.

## Running tests

```bash
python -m pytest test_app.py -v
```
