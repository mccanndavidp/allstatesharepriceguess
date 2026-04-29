"""
Unit tests for app.py (no network required — yfinance is mocked).
"""

import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers to generate synthetic price data
# ---------------------------------------------------------------------------

def _make_price_df(ticker: str, n_days: int = 252 * 5, start: str = "2020-01-02") -> pd.DataFrame:
    """Return a synthetic closing-price DataFrame similar to yfinance output."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start=start, periods=n_days)
    # Simple random-walk price starting at 100
    returns = rng.normal(0.0003, 0.015, size=n_days)
    prices = 100 * np.cumprod(1 + returns)
    return pd.DataFrame({ticker: prices}, index=dates)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFetchData(unittest.TestCase):
    """fetch_data should return a clean single-column DataFrame."""

    def _mock_download_simple(self, ticker, **_kwargs):
        df = _make_price_df(ticker)
        df.columns = pd.Index(["Close"])          # plain index, as older yfinance
        return df

    def _mock_download_multiindex(self, ticker, **_kwargs):
        df = _make_price_df(ticker)
        df.columns = pd.MultiIndex.from_tuples([("Close", ticker)])
        return df

    def test_simple_index(self):
        from app import fetch_data
        with patch("app.yf.download", side_effect=self._mock_download_simple):
            df = fetch_data("ALL")
        self.assertIn("ALL", df.columns)
        self.assertFalse(df.empty)

    def test_multiindex(self):
        from app import fetch_data
        with patch("app.yf.download", side_effect=self._mock_download_multiindex):
            df = fetch_data("ALL")
        self.assertIn("ALL", df.columns)
        self.assertFalse(df.empty)

    def test_empty_raises(self):
        from app import fetch_data
        with patch("app.yf.download", return_value=pd.DataFrame()):
            with self.assertRaises(ValueError):
                fetch_data("FAKE")


class TestBuildModel(unittest.TestCase):
    """build_model should return a trained model with sane metrics."""

    def setUp(self):
        self.ticker = "ALL"
        self.df = _make_price_df(self.ticker)

    def test_returns_four_items(self):
        from app import build_model
        result = build_model(self.df, self.ticker, degree=2)
        self.assertEqual(len(result), 4)  # model, poly, origin, metrics

    def test_metrics_keys(self):
        from app import build_model
        _, _, _, metrics = build_model(self.df, self.ticker, degree=2)
        self.assertIn("mae", metrics)
        self.assertIn("r2", metrics)

    def test_r2_reasonable(self):
        from app import build_model
        _, _, _, metrics = build_model(self.df, self.ticker, degree=2)
        self.assertGreater(metrics["r2"], -1.0)   # not wildly wrong

    def test_mae_positive(self):
        from app import build_model
        _, _, _, metrics = build_model(self.df, self.ticker, degree=2)
        self.assertGreater(metrics["mae"], 0)

    def test_degree_1(self):
        from app import build_model
        model, poly, origin, metrics = build_model(self.df, self.ticker, degree=1)
        self.assertIsNotNone(model)


class TestPredictPrice(unittest.TestCase):
    """predict_price should return a positive float."""

    def setUp(self):
        from app import build_model
        self.ticker = "ALL"
        self.df = _make_price_df(self.ticker)
        self.model, self.poly, self.origin, _ = build_model(
            self.df, self.ticker, degree=2
        )

    def test_returns_float(self):
        from app import predict_price
        target = self.df.index[-1].to_pydatetime() + timedelta(days=365)
        result = predict_price(self.model, self.poly, self.origin, target)
        self.assertIsInstance(result, float)

    def test_prediction_within_same_range_is_ballpark(self):
        """Predicting a training date should be close to the actual price."""
        from app import predict_price
        mid_date = self.df.index[len(self.df) // 2].to_pydatetime()
        pred = predict_price(self.model, self.poly, self.origin, mid_date)
        actual = float(self.df[self.ticker].iloc[len(self.df) // 2])
        # Within 50 % — loose check; model is intentionally simple
        self.assertLess(abs(pred - actual) / actual, 0.5)


class TestParseArgs(unittest.TestCase):
    """parse_args should handle valid and invalid arguments."""

    def test_default_date_is_none(self):
        from app import parse_args
        args = parse_args([])
        self.assertIsNone(args.date)

    def test_explicit_date(self):
        from app import parse_args
        args = parse_args(["--date", "2027-06-30"])
        self.assertEqual(args.date, "2027-06-30")

    def test_degree_default(self):
        from app import parse_args
        args = parse_args([])
        self.assertEqual(args.degree, 2)

    def test_degree_override(self):
        from app import parse_args
        args = parse_args(["--degree", "3"])
        self.assertEqual(args.degree, 3)

    def test_output_default(self):
        from app import parse_args
        args = parse_args([])
        self.assertEqual(args.output, "price_prediction.png")


class TestMainWithMockedData(unittest.TestCase):
    """End-to-end test of main() with mocked yfinance and matplotlib."""

    def _make_raw(self, ticker, **_kwargs):
        df = _make_price_df(ticker)
        df.columns = pd.Index(["Close"])
        return df

    @patch("app.plt.savefig")
    @patch("app.plt.tight_layout")
    @patch("app.yf.download")
    def test_main_runs_without_error(self, mock_dl, _mock_tight, _mock_save):
        mock_dl.side_effect = self._make_raw
        from app import main
        results = main(["--date", "2026-12-31"])
        self.assertIn("ALL", results)
        self.assertIn("^GSPC", results)

    @patch("app.plt.savefig")
    @patch("app.plt.tight_layout")
    @patch("app.yf.download")
    def test_main_predictions_are_positive(self, mock_dl, _mock_tight, _mock_save):
        mock_dl.side_effect = self._make_raw
        from app import main
        results = main(["--date", "2026-12-31"])
        for ticker, info in results.items():
            self.assertGreater(info["prediction"], 0, msg=f"{ticker} prediction ≤ 0")

    def test_main_bad_date_exits(self):
        from app import main
        with self.assertRaises(SystemExit) as ctx:
            main(["--date", "not-a-date"])
        self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
