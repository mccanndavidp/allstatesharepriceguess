#!/usr/bin/env python3
"""
Prediction Accuracy Script
===========================
Walks forward through historical closing-price data to measure how well the
polynomial regression model predicts prices *N* trading days in the future.

For each sample point the model is trained on all data up to that point,
then used to predict the price ``--days-ahead`` trading days later.  The
predicted value is compared to the actual closing price recorded in the
dataset, and per-sample errors plus aggregate metrics are reported.

Usage
-----
    python predict_accuracy.py --days-ahead 30
    python predict_accuracy.py --days-ahead 90 --ticker ALL --degree 3
    python predict_accuracy.py --days-ahead 60 --samples 50
"""

import argparse
import sys

import numpy as np
import pandas as pd

from app import build_model, fetch_data, predict_price

# ---------------------------------------------------------------------------
# Tickers (mirrors app.py)
# ---------------------------------------------------------------------------

TICKERS = {
    "ALL": "Allstate Corporation",
    "^GSPC": "S&P 500 Index",
}

# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------


def walk_forward_accuracy(
    df: pd.DataFrame,
    ticker: str,
    days_ahead: int,
    degree: int = 2,
    n_samples: int = 50,
) -> pd.DataFrame:
    """Perform a walk-forward backtest and return per-sample results.

    For each of ``n_samples`` evenly-spaced sample points in the historical
    data (each leaving at least ``days_ahead`` rows after it), the model is
    trained on all data up to that point and used to predict the closing price
    ``days_ahead`` trading days later.  The prediction is compared to the
    actual closing price in the dataset.

    Parameters
    ----------
    df:
        Single-column closing-price DataFrame with a DatetimeIndex.
    ticker:
        Column name in *df* (also the ticker symbol).
    days_ahead:
        Number of trading days ahead to predict.
    degree:
        Polynomial degree passed to :func:`app.build_model`.
    n_samples:
        Maximum number of walk-forward sample points to evaluate.

    Returns
    -------
    pd.DataFrame
        Columns: ``sample_date``, ``predict_date``, ``predicted``, ``actual``,
        ``error``, ``error_pct``.

    Raises
    ------
    ValueError
        If *df* does not contain enough rows for the requested evaluation.
    """
    n = len(df)
    min_train = max(degree + 2, 30)  # need at least this many rows to fit

    max_sample_idx = n - days_ahead - 1
    if max_sample_idx < min_train:
        raise ValueError(
            f"Not enough data for a {days_ahead}-day-ahead backtest with "
            f"degree={degree}: need at least {min_train + days_ahead + 1} rows, "
            f"got {n}."
        )

    # Evenly-spaced candidate indices in [min_train, max_sample_idx]
    raw_indices = np.linspace(min_train, max_sample_idx, n_samples, dtype=int)
    # De-duplicate while preserving order
    _, unique_pos = np.unique(raw_indices, return_index=True)
    sample_indices = raw_indices[np.sort(unique_pos)]

    rows = []
    for idx in sample_indices:
        train_df = df.iloc[: idx + 1]
        model, poly, origin, _ = build_model(train_df, ticker, degree=degree)

        target_idx = idx + days_ahead
        target_date = df.index[target_idx].to_pydatetime()

        predicted = predict_price(model, poly, origin, target_date)
        actual = float(df[ticker].iloc[target_idx])

        error = predicted - actual
        error_pct = (error / actual * 100.0) if actual != 0 else float("nan")

        rows.append(
            {
                "sample_date": df.index[idx].date(),
                "predict_date": df.index[target_idx].date(),
                "predicted": round(predicted, 2),
                "actual": round(actual, 2),
                "error": round(error, 2),
                "error_pct": round(error_pct, 4),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def compute_metrics(df_results: pd.DataFrame) -> dict:
    """Return a dict of aggregate accuracy metrics from walk-forward results."""
    errors = df_results["error"]
    abs_errors = errors.abs()
    pcts = df_results["error_pct"]
    abs_pcts = pcts.abs()

    return {
        "n_samples": len(df_results),
        "mean_error": float(errors.mean()),
        "mae": float(abs_errors.mean()),
        "rmse": float(np.sqrt((errors**2).mean())),
        "mean_error_pct": float(pcts.mean()),
        "mape": float(abs_pcts.mean()),
        "max_abs_error": float(abs_errors.max()),
        "max_abs_error_pct": float(abs_pcts.max()),
    }


def print_report(
    df_results: pd.DataFrame,
    ticker: str,
    name: str,
    days_ahead: int,
) -> dict:
    """Print the per-sample table and aggregate metrics; return the metrics dict."""
    metrics = compute_metrics(df_results)

    print()
    print("=" * 76)
    print(f"  {name} ({ticker})  —  {days_ahead}-day-ahead prediction accuracy")
    print("=" * 76)
    header = (
        f"  {'Sample Date':<13}  {'Predict Date':<13}  "
        f"{'Predicted':>11}  {'Actual':>11}  "
        f"{'Error ($)':>11}  {'Error (%)':>10}"
    )
    print(header)
    print("  " + "-" * 72)
    for _, row in df_results.iterrows():
        print(
            f"  {str(row['sample_date']):<13}  {str(row['predict_date']):<13}  "
            f"${row['predicted']:>10,.2f}  ${row['actual']:>10,.2f}  "
            f"${row['error']:>+10,.2f}  {row['error_pct']:>+9.2f}%"
        )

    print()
    print("  Aggregate metrics:")
    print(f"    Samples evaluated  : {metrics['n_samples']}")
    print(f"    Mean Error ($)     : ${metrics['mean_error']:>+12,.2f}")
    print(f"    MAE ($)            : ${metrics['mae']:>12,.2f}")
    print(f"    RMSE ($)           : ${metrics['rmse']:>12,.2f}")
    print(f"    Mean Error (%)     : {metrics['mean_error_pct']:>+12.2f}%")
    print(f"    MAPE (%)           : {metrics['mape']:>12.2f}%")
    print(f"    Max Abs Error ($)  : ${metrics['max_abs_error']:>12,.2f}")
    print(f"    Max Abs Error (%)  : {metrics['max_abs_error_pct']:>12.2f}%")
    print()

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Walk-forward backtest: train on data up to each sample point "
            "and predict --days-ahead trading days forward, then compare "
            "to the actual closing price and report accuracy metrics."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--days-ahead",
        type=int,
        required=True,
        metavar="N",
        help="Number of trading days ahead to predict (required).",
    )
    parser.add_argument(
        "--ticker",
        default=None,
        metavar="SYMBOL",
        help=(
            "Restrict evaluation to a single ticker symbol "
            "(default: all — ALL and ^GSPC).  Example: --ticker ALL"
        ),
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=2,
        metavar="N",
        help="Polynomial degree for the regression model (default: 2).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        metavar="N",
        help="Number of walk-forward sample points to evaluate (default: 50).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.days_ahead < 1:
        print("Error: --days-ahead must be at least 1.", file=sys.stderr)
        sys.exit(1)
    if args.samples < 1:
        print("Error: --samples must be at least 1.", file=sys.stderr)
        sys.exit(1)

    tickers = TICKERS
    if args.ticker:
        if args.ticker not in TICKERS:
            print(
                f"Error: unknown ticker '{args.ticker}'. "
                f"Choose from: {', '.join(TICKERS)}.",
                file=sys.stderr,
            )
            sys.exit(1)
        tickers = {args.ticker: TICKERS[args.ticker]}

    print(f"\nDays ahead  : {args.days_ahead}")
    print(f"Poly degree : {args.degree}")
    print(f"Samples     : {args.samples}")
    print()

    all_results = {}
    for ticker, name in tickers.items():
        try:
            print(f"Fetching data for {ticker}…")
            df = fetch_data(ticker)
        except ValueError as exc:
            print(f"  Warning: {exc}", file=sys.stderr)
            continue

        try:
            df_results = walk_forward_accuracy(
                df,
                ticker,
                days_ahead=args.days_ahead,
                degree=args.degree,
                n_samples=args.samples,
            )
        except ValueError as exc:
            print(f"  Warning: {exc}", file=sys.stderr)
            continue

        metrics = print_report(df_results, ticker, name, args.days_ahead)
        all_results[ticker] = {"samples": df_results, "metrics": metrics}

    if not all_results:
        print("Error: no results could be computed.", file=sys.stderr)
        sys.exit(1)

    return all_results


if __name__ == "__main__":
    main()
