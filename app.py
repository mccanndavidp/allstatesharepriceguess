#!/usr/bin/env python3
"""
Allstate Share Price Predictor
================================
Fetches 5-year historical closing prices for Allstate (ALL) and the
S&P 500 (^GSPC), trains a prediction model of the chosen type, and
predicts the ALL price at a user-supplied date.

Models
------
  poly  Polynomial regression of closing price vs days-since-start
        (one model per ticker, independent).
  var   Vector AutoRegression (statsmodels VAR) on the log-returns of
        *both* ALL and ^GSPC jointly.  Lag order is selected by AIC.
        This is a multivariate time-series model that captures the
        co-movement between Allstate and the broader market.

Usage
-----
    python app.py                               # poly, 1 year from today
    python app.py --model var                   # VAR model
    python app.py --date 2026-12-31             # specific date
    python app.py --date 2026-12-31 --degree 3  # poly degree 3
    python app.py --model var --lags 5          # VAR with fixed lag order
    python app.py --date 2026-12-31 --output my_chart.png
"""

import argparse
import sys
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def fetch_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download 5-year adjusted closing prices for *ticker*.

    Returns a single-column DataFrame with the ticker symbol as the column
    name and a DatetimeIndex.
    """
    print(f"  Downloading {ticker} …", end=" ", flush=True)
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    # yfinance may return a MultiIndex (Ticker × Field) when group_by is used;
    # normalise to a plain single-level index.
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs("Close", axis=1, level=0)
        close = raw[ticker] if ticker in raw.columns else raw.iloc[:, 0]
    else:
        close = raw["Close"]

    df = pd.DataFrame({ticker: close})
    df.index = pd.to_datetime(df.index)
    df = df.dropna()
    print(f"{len(df)} trading days")
    return df


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _days_since_start(index: pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    """Convert a DatetimeIndex to 'days since *origin*' as a float column."""
    return ((index - origin) / pd.Timedelta(days=1)).values.reshape(-1, 1)


def build_model(df: pd.DataFrame, ticker: str, degree: int = 2):
    """Fit a polynomial regression on days-since-start → closing price.

    Returns
    -------
    model   : fitted LinearRegression
    poly    : fitted PolynomialFeatures transformer
    origin  : pd.Timestamp used as day-0 for the feature
    metrics : dict with keys 'mae' and 'r2'
    """
    origin = df.index[0]
    X = _days_since_start(df.index, origin)
    y = df[ticker].values

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    y_pred = model.predict(X_poly)
    metrics = {
        "mae": mean_absolute_error(y, y_pred),
        "r2": r2_score(y, y_pred),
    }
    return model, poly, origin, metrics


def predict_price(
    model: LinearRegression,
    poly: PolynomialFeatures,
    origin: pd.Timestamp,
    target_date: datetime,
) -> float:
    """Return the predicted closing price for *target_date*."""
    days = (pd.Timestamp(target_date) - origin) / pd.Timedelta(days=1)
    X = poly.transform(np.array([[days]]))
    return float(model.predict(X)[0])


# ---------------------------------------------------------------------------
# VAR model
# ---------------------------------------------------------------------------

def build_var_model(
    df_all: pd.DataFrame,
    df_sp500: pd.DataFrame,
    maxlags: int = 10,
):
    """Fit a Vector AutoRegression (VAR) model on ALL and ^GSPC log returns.

    Both series are used as endogenous variables so that the model captures
    cross-series dynamics (e.g. S&P 500 movements informing Allstate price).
    Log returns are used to satisfy the stationarity assumption; AIC-based
    lag-order selection is applied automatically when *maxlags* > 1.

    Parameters
    ----------
    df_all:
        Single-column DataFrame with column ``'ALL'`` and a DatetimeIndex.
    df_sp500:
        Single-column DataFrame with column ``'^GSPC'`` and a DatetimeIndex.
    maxlags:
        Upper bound on the lag order searched during AIC selection.
        Pass ``1`` to skip selection and fit a VAR(1).

    Returns
    -------
    var_result : VARResultsWrapper
    combined   : pd.DataFrame with columns ``['ALL', 'GSPC']`` (price levels,
                 inner-joined on common dates).
    metrics    : dict with keys ``aic``, ``bic``, ``hqic``, ``lags``,
                 ``mae`` (price-level one-step-ahead MAE on ALL), and
                 ``r2`` (price-level R² on ALL).
    """
    # Align on shared trading days
    combined = pd.concat(
        [df_all["ALL"].rename("ALL"), df_sp500["^GSPC"].rename("GSPC")],
        axis=1,
    ).dropna()

    log_ret = np.log(combined).diff().dropna()

    # Stationarity check — warn only, do not abort
    for col in log_ret.columns:
        adf_stat, adf_p = adfuller(log_ret[col])[:2]
        if adf_p > 0.05:
            print(
                f"  Warning: {col} log returns may not be stationary "
                f"(ADF p={adf_p:.3f}); results should be interpreted with care.",
                file=sys.stderr,
            )

    # Lag-order selection
    var_model = VAR(log_ret)
    safe_maxlags = min(maxlags, max(1, len(log_ret) // 10 - 1))
    try:
        lag_sel = var_model.select_order(maxlags=safe_maxlags)
        optimal_lags = max(int(lag_sel.aic), 1)
    except Exception:
        optimal_lags = 1

    var_result = var_model.fit(optimal_lags)

    # In-sample one-step-ahead price-level metrics for ALL
    k = var_result.k_ar
    n_fit = len(var_result.fittedvalues)
    prev_prices = combined["ALL"].values[k : k + n_fit]
    actual_prices = combined["ALL"].values[k + 1 : k + 1 + n_fit]
    fitted_prices = prev_prices * np.exp(var_result.fittedvalues["ALL"].values)

    metrics = {
        "aic": float(var_result.aic),
        "bic": float(var_result.bic),
        "hqic": float(var_result.hqic),
        "lags": int(var_result.k_ar),
        "mae": float(mean_absolute_error(actual_prices, fitted_prices)),
        "r2": float(r2_score(actual_prices, fitted_prices)),
    }
    return var_result, combined, metrics


def predict_var_price(
    var_result,
    combined: pd.DataFrame,
    target_date: datetime,
) -> float:
    """Forecast the ALL closing price at *target_date* using a fitted VAR.

    Log-return steps are projected forward from the last known date, then
    converted back to a price level via cumulative exponentiation.

    Parameters
    ----------
    var_result:
        Fitted ``VARResultsWrapper`` returned by :func:`build_var_model`.
    combined:
        Price-level DataFrame (columns ``['ALL', 'GSPC']``) used when fitting;
        must end on the last known trading day.
    target_date:
        The date for which the ALL price should be predicted.

    Returns
    -------
    float
        Predicted ALL closing price.
    """
    last_date = combined.index[-1]
    target_ts = pd.Timestamp(target_date)
    if target_ts <= last_date:
        return float(combined["ALL"].iloc[-1])

    steps = len(pd.bdate_range(last_date, target_ts)) - 1
    if steps <= 0:
        return float(combined["ALL"].iloc[-1])

    log_ret = np.log(combined).diff().dropna()
    k = var_result.k_ar
    last_obs = log_ret.values[-k:]  # shape (k, 2); ALL is column 0

    forecast = var_result.forecast(last_obs, steps=steps)  # shape (steps, 2)
    cumulative_log_ret_all = float(np.sum(forecast[:, 0]))
    last_price = float(combined["ALL"].iloc[-1])
    return float(last_price * np.exp(cumulative_log_ret_all))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def plot_results(
    results: dict,
    target_date: datetime,
    output_file: str = "price_prediction.png",
) -> str:
    """Plot historical prices and the predicted point for each ticker.

    *results* maps ticker → {df, prediction, name}.  When ``'model'`` and
    ``'poly'`` keys are present (polynomial model) a regression trend line is
    drawn; for VAR results only the scatter prediction point is shown.
    """
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(13, 5 * n), squeeze=False)

    palette = ["steelblue", "darkorange", "seagreen", "purple"]

    for ax, (ticker, info), color in zip(axes[:, 0], results.items(), palette):
        df = info["df"]
        pred_price = info["prediction"]
        name = info["name"]

        ax.plot(
            df.index,
            df[ticker],
            label="Historical close",
            color=color,
            linewidth=1.2,
        )

        # Trend line — only available for the polynomial model
        if "model" in info and "poly" in info:
            trend_x = pd.date_range(df.index[0], target_date, periods=300)
            trend_y = [
                predict_price(info["model"], info["poly"], info["origin"], d)
                for d in trend_x
            ]
            ax.plot(
                trend_x,
                trend_y,
                label="Regression trend",
                color=color,
                linewidth=1,
                linestyle="--",
                alpha=0.6,
            )

        ax.axvline(
            target_date,
            color="red",
            linestyle=":",
            linewidth=1.5,
            alpha=0.8,
            label="Prediction date",
        )
        ax.scatter(
            [target_date],
            [pred_price],
            color="red",
            zorder=6,
            s=100,
            label=f"Predicted: ${pred_price:,.2f}",
        )

        ax.set_title(f"{name} ({ticker})", fontsize=13, fontweight="bold")
        ax.set_ylabel("Price (USD)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"Share Price History & Prediction  ·  target date: {target_date.strftime('%Y-%m-%d')}",
        fontsize=14,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved → {output_file}")
    return output_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

TICKERS = {
    "ALL": "Allstate Corporation",
    "^GSPC": "S&P 500 Index",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Pull 5-year historic share prices for Allstate (ALL) and the "
            "S&P 500 (^GSPC), build a prediction model, and predict the "
            "closing price at a given date.\n\n"
            "Models:\n"
            "  poly  Polynomial regression (one model per ticker)\n"
            "  var   Vector AutoRegression using both ALL and ^GSPC jointly\n"
            "        (lag order selected by AIC; ALL price predicted from\n"
            "        the multivariate model)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        default=None,
        help=(
            "Target date for the price prediction "
            "(default: one year from today)"
        ),
    )
    parser.add_argument(
        "--model",
        choices=["poly", "var"],
        default="poly",
        help=(
            "Forecasting model to use: 'poly' (polynomial regression, default) "
            "or 'var' (Vector AutoRegression with AIC lag selection)"
        ),
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=2,
        metavar="N",
        help="Polynomial degree for the 'poly' model (default: 2)",
    )
    parser.add_argument(
        "--lags",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum lag order to consider for AIC selection in the 'var' model "
            "(default: 10).  Pass the exact desired lag count to skip selection."
        ),
    )
    parser.add_argument(
        "--output",
        default="price_prediction.png",
        metavar="FILE",
        help="Output chart filename (default: price_prediction.png)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # ── target date ────────────────────────────────────────────────────────
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            print(
                f"Error: '{args.date}' is not a valid date. Use YYYY-MM-DD.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        target_date = datetime.today() + timedelta(days=365)
        print(f"No --date supplied; defaulting to {target_date.strftime('%Y-%m-%d')}.")

    print(f"\nPrediction target : {target_date.strftime('%Y-%m-%d')}")
    print(f"Model             : {args.model}")
    if args.model == "poly":
        print(f"Polynomial degree : {args.degree}\n")
    else:
        maxlags = args.lags if args.lags is not None else 10
        print(f"VAR max lags      : {maxlags} (AIC selection)\n")

    # ── fetch data ─────────────────────────────────────────────────────────
    results = {}

    if args.model == "var":
        # VAR: fetch both tickers, build one joint model, predict ALL only
        maxlags = args.lags if args.lags is not None else 10
        try:
            df_all = fetch_data("ALL")
            df_sp500 = fetch_data("^GSPC")
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        try:
            var_result, combined, metrics = build_var_model(
                df_all, df_sp500, maxlags=maxlags
            )
        except Exception as exc:
            print(f"Error building VAR model: {exc}", file=sys.stderr)
            sys.exit(1)

        pred = predict_var_price(var_result, combined, target_date)

        results["ALL"] = {
            "df": df_all,
            "name": TICKERS["ALL"],
            "var_result": var_result,
            "combined": combined,
            "metrics": metrics,
            "prediction": pred,
        }

    else:
        # Polynomial: independent model per ticker (existing behaviour)
        for ticker, name in TICKERS.items():
            try:
                df = fetch_data(ticker)
            except ValueError as exc:
                print(f"  Warning: {exc}", file=sys.stderr)
                continue

            model, poly, origin, metrics = build_model(df, ticker, degree=args.degree)
            pred = predict_price(model, poly, origin, target_date)

            results[ticker] = {
                "df": df,
                "name": name,
                "model": model,
                "poly": poly,
                "origin": origin,
                "metrics": metrics,
                "prediction": pred,
            }

    if not results:
        print("Error: No data could be retrieved. Check your internet connection.", file=sys.stderr)
        sys.exit(1)

    # ── print summary ──────────────────────────────────────────────────────
    print()
    for ticker, info in results.items():
        df = info["df"]
        m = info["metrics"]
        print("=" * 52)
        print(f"  {info['name']} ({ticker})")
        print("=" * 52)
        print(f"  Data range  : {df.index[0].date()} → {df.index[-1].date()}")
        print(f"  Last close  : ${float(df[ticker].iloc[-1]):>10,.2f}")
        if args.model == "var":
            print(f"  VAR lags    : {m['lags']}")
            print(f"  AIC         : {m['aic']:>14.2f}")
            print(f"  BIC         : {m['bic']:>14.2f}")
        print(f"  Model MAE   : ${m['mae']:>10,.2f}")
        print(f"  Model R²    : {m['r2']:>11.4f}")
        print(f"  Predicted   : ${info['prediction']:>10,.2f}  "
              f"(on {target_date.strftime('%Y-%m-%d')})")

    # ── chart ──────────────────────────────────────────────────────────────
    if results:
        plot_results(results, target_date, output_file=args.output)

    return results


if __name__ == "__main__":
    main()
