"""Focused checks for the historical EDA plots."""

import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import matplotlib.axes
import pandas as pd

from src import plots


class PlotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "actuals.csv"
        booking = date(2024, 1, 1)
        lead_times = [0, 7, 8, 30, 31, 90, 91, 180, 181]
        self.original = pd.DataFrame({
            "BookingDate": [booking.isoformat()] * len(lead_times),
            "FlightDate": [(booking + timedelta(days=days)).isoformat() for days in lead_times],
            "FlightMarketCluster": ["City"] * len(lead_times),
            "Tix": ["2"] * len(lead_times),
            "Revenue": [str(value) for value in range(1, 10)],
            "AOV": ["123.1234567890123456789"] * len(lead_times),
        })
        self.original.to_csv(self.path, sep=";", index=False)

    def test_months_preserve_values_and_are_idempotent(self) -> None:
        plots.add_month_columns(self.path)
        plots.add_month_columns(self.path)
        result = pd.read_csv(self.path, sep=";", dtype=str)
        pd.testing.assert_frame_equal(result[self.original.columns], self.original)
        self.assertEqual(result["BookingMonth"].tolist(), ["1"] * len(result))
        self.assertEqual(result["FlightMonth"].tolist(), ["1", "1", "1", "1", "2", "3", "4", "6", "6"])

    def test_bucket_boundaries_and_images(self) -> None:
        data = plots.add_month_columns(self.path)
        data["lead_time_days"] = (
            pd.to_datetime(data["FlightDate"]) - pd.to_datetime(data["BookingDate"])
        ).dt.days
        data["Revenue"] = pd.to_numeric(data["Revenue"])
        bar_values = []
        original_bar = matplotlib.axes.Axes.bar

        def capture_bar(axis, x, height, *args, **kwargs):
            bar_values.extend(height)
            return original_bar(axis, x, height, *args, **kwargs)

        output = Path(self.directory.name) / "plots"
        with patch.object(matplotlib.axes.Axes, "bar", autospec=True, side_effect=capture_bar):
            distribution = plots.plot_lead_time_distribution(data, "City", output)
        self.assertEqual(bar_values, [3, 7, 11, 15, 9])
        bubble = plots.plot_month_relationship(data, "City", "Revenue", output)
        heatmap = plots.plot_median_lead_time(data, output)
        for path in (distribution, bubble, heatmap):
            self.assertTrue(path.is_file())
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_invalid_date_does_not_replace_original(self) -> None:
        self.original.loc[0, "BookingDate"] = "invalid"
        self.original.to_csv(self.path, sep=";", index=False)
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            plots.add_month_columns(self.path)
        self.assertEqual(self.path.read_bytes(), before)

    def test_rejects_negative_lead_time(self) -> None:
        self.original.loc[0, "FlightDate"] = "2023-12-31"
        self.original.to_csv(self.path, sep=";", index=False)
        with self.assertRaisesRegex(ValueError, "earlier"):
            plots.month_relationship_totals(plots.add_month_columns(self.path), "City", "Tix")

    def test_same_and_later_year_totals_are_separate(self) -> None:
        data = pd.DataFrame({
            "BookingDate": ["2023-01-03", "2024-01-03", "2024-01-03",
                            "2025-01-03", "2024-12-31", "2024-01-03"],
            "FlightDate": ["2023-02-01", "2024-02-01", "2025-02-01",
                           "2027-03-01", "2025-01-01", "2024-02-01"],
            "FlightMarketCluster": ["City"] * 5 + ["Other"],
            "BookingMonth": ["12"] * 6,
            "FlightMonth": ["12"] * 6,
            "Tix": ["5", "7", "3", "4", "2", "100"],
            "Revenue": ["50", "70", "30", "40", "20", "1000"],
        })
        for metric, expected in (("Tix", [12, 7, 2]), ("Revenue", [120, 70, 20])):
            totals = plots.month_relationship_totals(data, "City", metric)
            self.assertEqual(totals["BookingMonth"].tolist(), [1, 1, 12])
            self.assertEqual(totals["FlightMonthGroup"].tolist(), [2, 13, 13])
            self.assertEqual(totals[metric].tolist(), expected)

        output = Path(self.directory.name) / "plots"
        captured_ticks = []
        original_set_yticks = matplotlib.axes.Axes.set_yticks

        def capture_yticks(axis, ticks, *args, **kwargs):
            captured_ticks.extend(ticks)
            return original_set_yticks(axis, ticks, *args, **kwargs)

        with patch.object(matplotlib.axes.Axes, "set_yticks", autospec=True,
                          side_effect=capture_yticks):
            plots.plot_month_relationship(data, "City", "Tix", output)
        self.assertIn(13, captured_ticks)

    def test_rejects_invalid_date_and_metric(self) -> None:
        data = self.original.copy()
        data.loc[0, "FlightDate"] = "invalid"
        with self.assertRaisesRegex(ValueError, "valid dates"):
            plots.month_relationship_totals(data, "City", "Tix")
        data = self.original.copy()
        data.loc[0, "Tix"] = "invalid"
        with self.assertRaisesRegex(ValueError, "Tix"):
            plots.month_relationship_totals(data, "City", "Tix")


if __name__ == "__main__":
    unittest.main()