"""Checks for row-level and monthly AOV comparisons."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import matplotlib.axes

from src import post_processing as post


class AovTests(unittest.TestCase):
    def test_row_aov_and_weighted_monthly_margins(self) -> None:
        joint = pd.DataFrame({
            "BookingDate": pd.to_datetime(["2026-01-01", "2026-01-02", "2025-12-31", "2026-02-01"]),
            "FlightDate": pd.to_datetime(["2026-02-01", "2027-01-01", "2026-01-01", "2026-02-01"]),
            "Tix": [1.0, 3.0, 2.0, 0.0],
            "Revenue": [10.0, 90.0, 100.0, 5.0],
        })
        rows = post.add_aov(joint)
        self.assertEqual(rows["AOV"].iloc[:3].tolist(), [10.0, 30.0, 50.0])
        self.assertTrue(pd.isna(rows["AOV"].iloc[3]))
        # Carry-out contributes to booking January; carry-in to flight January.
        self.assertEqual(post.monthly_aov(rows, "BookingDate")[1], 25.0)
        self.assertEqual(post.monthly_aov(rows, "FlightDate")[1], 50.0)
        self.assertEqual(post.monthly_aov(rows, "FlightDate")[2], 15.0)
        self.assertTrue(pd.isna(post.monthly_aov(rows, "BookingDate")[2]))
        self.assertTrue(pd.isna(post.monthly_aov(rows, "BookingDate")[3]))

    def test_main_writes_row_and_monthly_results_and_plot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            joint_path = root / "joint.csv"
            booking_path = root / "booking.csv"
            flight_path = root / "flight.csv"
            output = root / "comparison"
            pd.DataFrame({
                "BookingDate": ["2026-01-01", "2025-12-31"],
                "FlightDate": ["2026-02-01", "2026-01-01"],
                "Tix": [1.0, 0.0], "Revenue": [10.0, 5.0],
            }).to_csv(joint_path, sep=";", index=False)
            forecast = pd.DataFrame({"Date": ["2026-01-01"], "Tix": [2.0], "Revenue": [40.0]})
            forecast.to_csv(booking_path, sep=";", index=False)
            forecast.to_csv(flight_path, sep=";", index=False)
            output.mkdir()
            for period in ("booking", "flight"):
                date_column = f"{period.title()}Date"
                pd.DataFrame({date_column: ["2026-01-01"], "Tix_error": [1.0],
                              "Revenue_error": [-5.0]}).to_csv(
                    output / f"joint_{period}_comparison.csv", sep=";", index=False,
                )
            with (patch.object(post, "JOINT_FILE", joint_path),
                  patch.object(post, "BOOKING_FILE", booking_path),
                  patch.object(post, "FLIGHT_FILE", flight_path),
                  patch.object(post, "COMPARISON_DIR", output)):
                post.main()
            rows = pd.read_csv(output / "joint_forecast_2026_unfiltered_with_aov.csv", sep=";")
            self.assertEqual(rows["AOV"].iloc[0], 10.0)
            self.assertTrue(pd.isna(rows["AOV"].iloc[1]))
            monthly = pd.read_csv(output / "monthly_aov_comparison.csv", sep=";")
            self.assertEqual(len(monthly), 12)
            self.assertEqual(monthly.loc[0, "Booking forecast"], 20.0)
            self.assertEqual(monthly.loc[0, "Joint (booking month)"], 10.0)
            self.assertTrue((output / "aov_comparison.png").stat().st_size > 0)
            self.assertTrue((output / "forecast_error_comparison.png").stat().st_size > 0)


class ErrorPlotTests(unittest.TestCase):
    def test_monthly_errors_sum_signed_values_without_mixing_years(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "booking.csv"
            pd.DataFrame({
                "BookingDate": ["2026-01-01", "2026-01-03", "2026-03-01", "2027-01-01"],
                "Tix_error": [3.0, -1.0, 0.0, 10.0],
                "Revenue_error": [-5.0, 2.0, 0.0, 20.0],
            }).to_csv(path, sep=";", index=False)
            errors = post.monthly_errors(path, "BookingDate")
            self.assertEqual(errors.loc["2026-01-01", "Tix_error"], 2.0)
            self.assertEqual(errors.loc["2026-01-01", "Revenue_error"], -3.0)
            self.assertTrue(pd.isna(errors.loc["2026-02-01", "Tix_error"]))
            self.assertEqual(errors.loc["2027-01-01", "Tix_error"], 10.0)


class BookingRevenueDifferenceTests(unittest.TestCase):
    def test_daily_totals_align_dates_and_preserve_missing_values(self) -> None:
        actuals = pd.DataFrame({
            "BookingDate": pd.to_datetime([
                "2025-01-01", "2025-01-01", "2025-01-02", "2025-12-31", "2024-01-01",
            ]),
            "Revenue": [10, 20, 50, 40, 1000],
        })
        forecast = pd.DataFrame({
            "Date": pd.to_datetime([
                "2026-01-01", "2026-01-01", "2026-01-02", "2026-01-03", "2025-01-01",
            ]),
            "Revenue": [15, 25, 30, 200, 1000],
        })
        difference = post.booking_revenue_difference(actuals, forecast)
        self.assertEqual(len(difference), 365)
        self.assertEqual(difference.loc["2026-01-01"], 10)
        self.assertEqual(difference.loc["2026-01-02"], -20)
        self.assertTrue(pd.isna(difference.loc["2026-01-03"]))
        self.assertTrue(pd.isna(difference.loc["2026-12-31"]))

    def test_standalone_cli_only_writes_requested_plot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual_path, forecast_path = root / "actuals.csv", root / "forecast.csv"
            pd.DataFrame({"BookingDate": ["2025-01-01", "2025-01-02"], "Revenue": [100, 50]}).to_csv(
                actual_path, sep=";", index=False,
            )
            pd.DataFrame({"Date": ["2026-01-01", "2026-01-03"], "Revenue": [120, 80]}).to_csv(
                forecast_path, sep=";", index=False,
            )
            output = root / "comparison"
            original_scatter = matplotlib.axes.Axes.scatter
            captured = []

            def capture_scatter(axis, x, y, *args, **kwargs):
                captured.append((x, y))
                return original_scatter(axis, x, y, *args, **kwargs)

            with (patch.object(post, "ACTUALS_FILE", actual_path),
                  patch.object(post, "BOOKING_FILE", forecast_path),
                  patch.object(post, "COMPARISON_DIR", output),
                  patch.object(post, "main") as main,
                  patch("sys.argv", ["post_processing.py", "--booking-revenue-difference-only"]),
                  patch.object(matplotlib.axes.Axes, "scatter", autospec=True,
                               side_effect=capture_scatter)):
                post.cli()
            main.assert_not_called()
            self.assertEqual(len(captured), 1)
            self.assertEqual(captured[0][0].tolist(), [pd.Timestamp("2026-01-01")])
            self.assertEqual(captured[0][1].tolist(), [20])
            paths = list(output.iterdir())
            self.assertEqual([path.name for path in paths], ["booking_forecast_revenue_difference.png"])
            self.assertEqual(paths[0].read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()