"""Tests for weighted booking/flight month comparisons."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import correlations


def sample() -> pd.DataFrame:
    return pd.DataFrame({
        "BookingDate": ["2024-01-01", "2024-01-02", "2025-02-01", "2024-12-01", "2024-01-01"],
        "FlightDate": ["2024-01-02", "2024-02-01", "2025-02-02", "2025-01-01", "2024-01-01"],
        "FlightMarketCluster": ["City", "City", "City", "City", "Other"],
        "Tix": [1, 1, 2, 100, 100],
        "Revenue": [2, 6, 8, 1000, 1000],
        "BookingMonth": [12, 12, 12, 12, 12],
        "FlightMonth": [12, 12, 12, 12, 12],
    })


class CorrelationTests(unittest.TestCase):
    def test_monthly_totals_pool_years_exclude_cross_year_and_other_cluster(self) -> None:
        totals = correlations.monthly_totals(sample(), "City")
        self.assertEqual(totals["Tix"].shape, (12, 12))
        self.assertEqual(totals["Tix"].loc[1, 1], 1)
        self.assertEqual(totals["Tix"].loc[2, 1], 1)
        self.assertEqual(totals["Tix"].loc[2, 2], 2)
        self.assertEqual(totals["Tix"].to_numpy().sum(), 4)
        self.assertEqual(totals["Revenue"].loc[2, 1], 6)
        self.assertEqual(totals["Tix"].loc[1, 12], 0)

    def test_weighted_correlations_and_both_metrics(self) -> None:
        data = sample()
        summary, matrices = correlations.analyze_cluster(data, data, "City")
        tix = matrices["Tix"]["actuals"]
        revenue = matrices["Revenue"]["actuals"]
        self.assertAlmostEqual(tix.loc[1, 1], 1 / np.sqrt(3))
        self.assertAlmostEqual(tix.loc[2, 1], -1 / np.sqrt(3))
        self.assertAlmostEqual(tix.loc[2, 2], 1 / np.sqrt(3))
        self.assertAlmostEqual(revenue.loc[1, 1], 1 / np.sqrt(7))
        self.assertTrue(np.isnan(tix.loc[3, 1]))
        self.assertTrue(np.isnan(tix.loc[1, 3]))
        self.assertAlmostEqual(summary.loc[0, "actuals"], 1 / np.sqrt(3))
        self.assertEqual(summary["forecast_minus_actuals"].tolist(), [0, 0])
        self.assertEqual(summary["metric"].tolist(), ["Tix", "Revenue"])

    def test_constant_marginals_are_undefined(self) -> None:
        data = sample().iloc[:2]
        totals = correlations.monthly_totals(data, "City")["Tix"]
        self.assertTrue(correlations.indicator_correlations(totals).isna().all().all())
        self.assertTrue(np.isnan(correlations.month_number_correlation(totals)))

    def test_comparison_uses_forecast_weights(self) -> None:
        actuals = sample()
        forecast = actuals.copy()
        forecast.loc[1, "Tix"] = 0
        summary, matrices = correlations.analyze_cluster(actuals, forecast, "City")
        self.assertAlmostEqual(matrices["Tix"]["forecast"].loc[1, 1], 1.0)
        self.assertAlmostEqual(summary.loc[0, "forecast"], 1.0)
        self.assertAlmostEqual(summary.loc[0, "forecast_minus_actuals"], 1 - 1 / np.sqrt(3))
        self.assertAlmostEqual(summary.loc[1, "forecast_minus_actuals"], 0)

    def test_invalid_inputs_are_rejected(self) -> None:
        data = sample()
        with self.assertRaisesRegex(ValueError, "Cluster not found"):
            correlations.monthly_totals(data, "missing")
        with self.assertRaisesRegex(ValueError, "Missing required"):
            correlations.monthly_totals(data.drop(columns="Revenue"), "City")
        for column, value, message in [("Tix", -1, "non-negative"),
                                      ("Revenue", float("nan"), "non-negative"),
                                      ("BookingDate", "invalid", "invalid dates"),
                                      ("FlightDate", "2023-01-01", "earlier")]:
            with self.subTest(column=column):
                changed = data.copy()
                changed.loc[0, column] = value
                with self.assertRaisesRegex(ValueError, message):
                    correlations.monthly_totals(changed, "City")
        cross_year_only = data.iloc[[3]]
        with self.assertRaisesRegex(ValueError, "No same-year"):
            correlations.monthly_totals(cross_year_only, "City")
        zero = data.copy()
        zero["Revenue"] = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            correlations.analyze_cluster(zero, data, "City")

    def test_cli_writes_selected_metric_plot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actuals, forecast = root / "actuals.csv", root / "ipf_filtered.csv"
            sample().to_csv(actuals, sep=";", index=False)
            sample().iloc[:4].to_csv(forecast, sep=";", index=False)
            argv = ["correlations.py", "--actuals", str(actuals), "--ipf-output", str(forecast),
                    "--clusters", "City", "--metric", "Revenue", "--plot-dir", str(root)]
            with patch("sys.argv", argv):
                correlations.main()
            path = root / "month_correlations_City_Revenue.png"
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()