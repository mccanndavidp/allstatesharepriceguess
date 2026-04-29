#!/usr/bin/env python3
"""
Allstate Share Price Predictor
================================
Fetches 5-year historical closing prices for Allstate (ALL) and the
S&P 500 (^GSPC), trains a polynomial regression model for each, and
predicts the price at a user-supplied date.

Usage
-----
    python app.py                        # predict 1 year from today
    python app.py --date 2026-12-31      # predict on a specific date
    python app.py --date 2026-12-31 --degree 3
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
# Output
# ---------------------------------------------------------------------------

def plot_results(
    results: dict,
    target_date: datetime,
    output_file: str = "price_prediction.png",
) -> str:
    """Plot historical prices and the predicted point for each ticker.

    *results* maps ticker → {df, prediction, name}.
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

        # Trend line from the model
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
            "S&P 500 (^GSPC), build a polynomial regression model, and "
            "predict the closing price at a given date."
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
        "--degree",
        type=int,
        default=2,
        metavar="N",
        help="Polynomial degree for the regression model (default: 2)",
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
    print(f"Polynomial degree : {args.degree}\n")

    # ── fetch & model ──────────────────────────────────────────────────────
    results = {}
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
