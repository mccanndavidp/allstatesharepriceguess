"""
Unit tests for predict_accuracy.py (no network required — yfinance is mocked).
"""

import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper: synthetic price data
# ---------------------------------------------------------------------------

def _make_price_df(ticker: str, n_days: int = 252 * 5, start: str = "2020-01-02") -> pd.DataFrame:
    """Return a synthetic closing-price DataFrame similar to yfinance output."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range(start=start, periods=n_days)
    returns = rng.normal(0.0003, 0.015, size=n_days)
    prices = 100 * np.cumprod(1 + returns)
    return pd.DataFrame({ticker: prices}, index=dates)


# ---------------------------------------------------------------------------
# Tests for walk_forward_accuracy
# ---------------------------------------------------------------------------

class TestWalkForwardAccuracy(unittest.TestCase):

    def setUp(self):
        self.ticker = "ALL"
        self.df = _make_price_df(self.ticker)

    def test_returns_dataframe(self):
        from predict_accuracy import walk_forward_accuracy
        result = walk_forward_accuracy(self.df, self.ticker, days_ahead=30, n_samples=10)
        self.assertIsInstance(result, pd.DataFrame)

    def test_expected_columns(self):
        from predict_accuracy import walk_forward_accuracy
        result = walk_forward_accuracy(self.df, self.ticker, days_ahead=30, n_samples=10)
        for col in ("sample_date", "predict_date", "predicted", "actual", "error", "error_pct"):
            self.assertIn(col, result.columns, msg=f"missing column: {col}")

    def test_sample_count_at_most_n_samples(self):
        from predict_accuracy import walk_forward_accuracy
        result = walk_forward_accuracy(self.df, self.ticker, days_ahead=30, n_samples=20)
        self.assertLessEqual(len(result), 20)

    def test_sample_count_positive(self):
        from predict_accuracy import walk_forward_accuracy
        result = walk_forward_accuracy(self.df, self.ticker, days_ahead=30, n_samples=5)
        self.assertGreater(len(result), 0)

    def test_error_equals_predicted_minus_actual(self):
        from predict_accuracy import walk_forward_accuracy
        result = walk_forward_accuracy(self.df, self.ticker, days_ahead=30, n_samples=5)
        for _, row in result.iterrows():
            self.assertAlmostEqual(row["error"], round(row["predicted"] - row["actual"], 2), places=1)

    def test_error_pct_sign_consistent_with_error(self):
        from predict_accuracy import walk_forward_accuracy
        result = walk_forward_accuracy(self.df, self.ticker, days_ahead=30, n_samples=10)
        for _, row in result.iterrows():
            if row["error"] > 0:
                self.assertGreater(row["error_pct"], 0)
            elif row["error"] < 0:
                self.assertLess(row["error_pct"], 0)

    def test_predict_date_after_sample_date(self):
        from predict_accuracy import walk_forward_accuracy
        result = walk_forward_accuracy(self.df, self.ticker, days_ahead=30, n_samples=10)
        for _, row in result.iterrows():
            self.assertGreater(row["predict_date"], row["sample_date"])

    def test_insufficient_data_raises(self):
        from predict_accuracy import walk_forward_accuracy
        tiny_df = self.df.iloc[:10]  # way too few rows
        with self.assertRaises(ValueError):
            walk_forward_accuracy(tiny_df, self.ticker, days_ahead=30, n_samples=5)

    def test_degree_1_works(self):
        from predict_accuracy import walk_forward_accuracy
        result = walk_forward_accuracy(
            self.df, self.ticker, days_ahead=30, degree=1, n_samples=5
        )
        self.assertGreater(len(result), 0)

    def test_days_ahead_1(self):
        from predict_accuracy import walk_forward_accuracy
        result = walk_forward_accuracy(self.df, self.ticker, days_ahead=1, n_samples=5)
        self.assertGreater(len(result), 0)

    def test_large_days_ahead(self):
        from predict_accuracy import walk_forward_accuracy
        # 252 * 3 = 3 years ahead — still within the 5-year data window
        result = walk_forward_accuracy(
            self.df, self.ticker, days_ahead=252 * 3, n_samples=5
        )
        self.assertGreater(len(result), 0)


# ---------------------------------------------------------------------------
# Tests for compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics(unittest.TestCase):

    def _make_results(self, errors):
        """Build a minimal df_results with the given list of errors."""
        import datetime
        base = datetime.date(2023, 1, 2)
        rows = []
        for i, e in enumerate(errors):
            actual = 100.0
            predicted = actual + e
            rows.append(
                {
                    "sample_date": base,
                    "predict_date": base,
                    "predicted": predicted,
                    "actual": actual,
                    "error": e,
                    "error_pct": e / actual * 100.0,
                }
            )
        return pd.DataFrame(rows)

    def test_returns_dict_with_expected_keys(self):
        from predict_accuracy import compute_metrics
        df = self._make_results([1.0, -2.0, 3.0])
        m = compute_metrics(df)
        for key in ("n_samples", "mean_error", "mae", "rmse", "mean_error_pct", "mape",
                    "max_abs_error", "max_abs_error_pct"):
            self.assertIn(key, m)

    def test_mae_is_mean_of_abs_errors(self):
        from predict_accuracy import compute_metrics
        errors = [1.0, -2.0, 3.0]
        df = self._make_results(errors)
        m = compute_metrics(df)
        expected_mae = np.mean(np.abs(errors))
        self.assertAlmostEqual(m["mae"], expected_mae, places=6)

    def test_rmse_correct(self):
        from predict_accuracy import compute_metrics
        errors = [1.0, -2.0, 3.0]
        df = self._make_results(errors)
        m = compute_metrics(df)
        expected_rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        self.assertAlmostEqual(m["rmse"], expected_rmse, places=6)

    def test_mape_non_negative(self):
        from predict_accuracy import compute_metrics
        df = self._make_results([1.0, -2.0, 3.0])
        m = compute_metrics(df)
        self.assertGreaterEqual(m["mape"], 0)

    def test_zero_error_gives_zero_mae(self):
        from predict_accuracy import compute_metrics
        df = self._make_results([0.0, 0.0, 0.0])
        m = compute_metrics(df)
        self.assertAlmostEqual(m["mae"], 0.0, places=6)

    def test_n_samples_count(self):
        from predict_accuracy import compute_metrics
        df = self._make_results([1.0, 2.0, 3.0, 4.0])
        m = compute_metrics(df)
        self.assertEqual(m["n_samples"], 4)


# ---------------------------------------------------------------------------
# Tests for parse_args
# ---------------------------------------------------------------------------

class TestParseArgs(unittest.TestCase):

    def test_days_ahead_required(self):
        from predict_accuracy import parse_args
        with self.assertRaises(SystemExit):
            parse_args([])

    def test_days_ahead_parsed(self):
        from predict_accuracy import parse_args
        args = parse_args(["--days-ahead", "30"])
        self.assertEqual(args.days_ahead, 30)

    def test_ticker_default_none(self):
        from predict_accuracy import parse_args
        args = parse_args(["--days-ahead", "10"])
        self.assertIsNone(args.ticker)

    def test_ticker_explicit(self):
        from predict_accuracy import parse_args
        args = parse_args(["--days-ahead", "10", "--ticker", "ALL"])
        self.assertEqual(args.ticker, "ALL")

    def test_degree_default(self):
        from predict_accuracy import parse_args
        args = parse_args(["--days-ahead", "10"])
        self.assertEqual(args.degree, 2)

    def test_samples_default(self):
        from predict_accuracy import parse_args
        args = parse_args(["--days-ahead", "10"])
        self.assertEqual(args.samples, 50)

    def test_samples_override(self):
        from predict_accuracy import parse_args
        args = parse_args(["--days-ahead", "10", "--samples", "20"])
        self.assertEqual(args.samples, 20)

    def test_model_default_is_poly(self):
        from predict_accuracy import parse_args
        args = parse_args(["--days-ahead", "10"])
        self.assertEqual(args.model, "poly")

    def test_model_var(self):
        from predict_accuracy import parse_args
        args = parse_args(["--days-ahead", "10", "--model", "var"])
        self.assertEqual(args.model, "var")

    def test_lags_default_is_none(self):
        from predict_accuracy import parse_args
        args = parse_args(["--days-ahead", "10"])
        self.assertIsNone(args.lags)

    def test_lags_override(self):
        from predict_accuracy import parse_args
        args = parse_args(["--days-ahead", "10", "--model", "var", "--lags", "3"])
        self.assertEqual(args.lags, 3)

    def test_invalid_model_exits(self):
        from predict_accuracy import parse_args
        with self.assertRaises(SystemExit):
            parse_args(["--days-ahead", "10", "--model", "bad"])


# ---------------------------------------------------------------------------
# Tests for walk_forward_accuracy_var
# ---------------------------------------------------------------------------

class TestWalkForwardAccuracyVar(unittest.TestCase):

    def setUp(self):
        self.df_all = _make_price_df("ALL")
        # Use a distinct seed so the two series are not perfectly collinear
        rng = np.random.default_rng(99)
        dates = pd.bdate_range(start="2020-01-02", periods=252 * 5)
        prices = 3000 * np.cumprod(1 + rng.normal(0.0002, 0.012, size=252 * 5))
        self.df_sp500 = pd.DataFrame({"^GSPC": prices}, index=dates)

    def test_returns_dataframe(self):
        from predict_accuracy import walk_forward_accuracy_var
        result = walk_forward_accuracy_var(
            self.df_all, self.df_sp500, days_ahead=30, maxlags=2, n_samples=5
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_expected_columns(self):
        from predict_accuracy import walk_forward_accuracy_var
        result = walk_forward_accuracy_var(
            self.df_all, self.df_sp500, days_ahead=30, maxlags=2, n_samples=5
        )
        for col in ("sample_date", "predict_date", "predicted", "actual", "error", "error_pct"):
            self.assertIn(col, result.columns, msg=f"missing column: {col}")

    def test_sample_count_positive(self):
        from predict_accuracy import walk_forward_accuracy_var
        result = walk_forward_accuracy_var(
            self.df_all, self.df_sp500, days_ahead=30, maxlags=2, n_samples=5
        )
        self.assertGreater(len(result), 0)

    def test_sample_count_at_most_n_samples(self):
        from predict_accuracy import walk_forward_accuracy_var
        result = walk_forward_accuracy_var(
            self.df_all, self.df_sp500, days_ahead=30, maxlags=2, n_samples=10
        )
        self.assertLessEqual(len(result), 10)

    def test_predict_date_after_sample_date(self):
        from predict_accuracy import walk_forward_accuracy_var
        result = walk_forward_accuracy_var(
            self.df_all, self.df_sp500, days_ahead=30, maxlags=2, n_samples=5
        )
        for _, row in result.iterrows():
            self.assertGreater(row["predict_date"], row["sample_date"])

    def test_error_equals_predicted_minus_actual(self):
        from predict_accuracy import walk_forward_accuracy_var
        result = walk_forward_accuracy_var(
            self.df_all, self.df_sp500, days_ahead=30, maxlags=2, n_samples=5
        )
        for _, row in result.iterrows():
            self.assertAlmostEqual(
                row["error"], round(row["predicted"] - row["actual"], 2), places=1
            )

    def test_insufficient_data_raises(self):
        from predict_accuracy import walk_forward_accuracy_var
        tiny_all = self.df_all.iloc[:10]
        tiny_sp500 = self.df_sp500.iloc[:10]
        with self.assertRaises(ValueError):
            walk_forward_accuracy_var(
                tiny_all, tiny_sp500, days_ahead=30, maxlags=2, n_samples=5
            )


# ---------------------------------------------------------------------------
# Tests for main()
# ---------------------------------------------------------------------------

class TestMain(unittest.TestCase):

    def _make_raw(self, ticker, **_kwargs):
        seed = 42 if "ALL" in str(ticker) else 99
        rng = np.random.default_rng(seed)
        n_days = 252 * 5
        dates = pd.bdate_range(start="2020-01-02", periods=n_days)
        start_price = 100 if "ALL" in str(ticker) else 3000
        prices = start_price * np.cumprod(1 + rng.normal(0.0003, 0.015, size=n_days))
        return pd.DataFrame({"Close": prices}, index=dates)

    @patch("app.yf.download")
    def test_main_returns_results_for_all_tickers(self, mock_dl):
        mock_dl.side_effect = self._make_raw
        from predict_accuracy import main
        results = main(["--days-ahead", "30", "--samples", "5"])
        self.assertIn("ALL", results)
        self.assertIn("^GSPC", results)

    @patch("app.yf.download")
    def test_main_single_ticker(self, mock_dl):
        mock_dl.side_effect = self._make_raw
        from predict_accuracy import main
        results = main(["--days-ahead", "30", "--ticker", "ALL", "--samples", "5"])
        self.assertIn("ALL", results)
        self.assertNotIn("^GSPC", results)

    @patch("app.yf.download")
    def test_main_result_has_samples_and_metrics(self, mock_dl):
        mock_dl.side_effect = self._make_raw
        from predict_accuracy import main
        results = main(["--days-ahead", "30", "--ticker", "ALL", "--samples", "5"])
        self.assertIn("samples", results["ALL"])
        self.assertIn("metrics", results["ALL"])

    def test_main_invalid_days_ahead_exits(self):
        from predict_accuracy import main
        with self.assertRaises(SystemExit) as ctx:
            main(["--days-ahead", "0"])
        self.assertEqual(ctx.exception.code, 1)

    def test_main_invalid_samples_exits(self):
        from predict_accuracy import main
        with self.assertRaises(SystemExit) as ctx:
            main(["--days-ahead", "10", "--samples", "0"])
        self.assertEqual(ctx.exception.code, 1)

    def test_main_unknown_ticker_exits(self):
        from predict_accuracy import main
        with self.assertRaises(SystemExit) as ctx:
            main(["--days-ahead", "10", "--ticker", "FAKE"])
        self.assertEqual(ctx.exception.code, 1)

    @patch("app.yf.download", return_value=pd.DataFrame())
    def test_main_no_data_exits(self, _mock_dl):
        from predict_accuracy import main
        with self.assertRaises(SystemExit) as ctx:
            main(["--days-ahead", "30", "--samples", "5"])
        self.assertEqual(ctx.exception.code, 1)

    @patch("app.yf.download")
    def test_main_var_returns_all(self, mock_dl):
        mock_dl.side_effect = self._make_raw
        from predict_accuracy import main
        results = main(["--days-ahead", "30", "--model", "var", "--lags", "2", "--samples", "5"])
        self.assertIn("ALL", results)
        self.assertNotIn("^GSPC", results)

    @patch("app.yf.download")
    def test_main_var_result_has_samples_and_metrics(self, mock_dl):
        mock_dl.side_effect = self._make_raw
        from predict_accuracy import main
        results = main(["--days-ahead", "30", "--model", "var", "--lags", "2", "--samples", "5"])
        self.assertIn("samples", results["ALL"])
        self.assertIn("metrics", results["ALL"])


if __name__ == "__main__":
    unittest.main()
