"""Focused checks for the historical prior and booking-to-flight seed."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import ipf_joint_table as joint


class PriorAndGridTests(unittest.TestCase):
    def test_prior_uses_ticket_weights_and_excludes_censored_bookings(self) -> None:
        actuals = pd.DataFrame({
            "BookingDate": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-10"]),
            "lead_time_days": [0, 2, 2],
            "BookingMonth": [1, 1, 1],
            "Tix": [10, 2, 1],
            "Revenue": [100, 40, 10000],
        })
        prior = joint.build_prior(actuals, pd.Timestamp("2024-01-10"))

        self.assertEqual(prior.L, 2)
        self.assertEqual(prior.lead.shape, (12, 3))
        np.testing.assert_allclose(prior.lead.sum(axis=1), 1.0)
        self.assertTrue((prior.lead > 0).all())
        # All observed leads lie in the same bucket, so shrinkage preserves its AOV.
        self.assertAlmostEqual(prior.aov[0, 0], 140 / 12)

    def test_grid_and_seed_keep_carry_in_and_out_in_band(self) -> None:
        grid = joint.build_grid(2)
        prior = joint.Prior(lead=np.full((12, 3), 1 / 3),
                            aov=np.ones((12, 5)), L=2)
        bookings = pd.Series([9.0, 6.0], index=pd.to_datetime([
            "2026-01-01", "2026-12-31",
        ]))
        seed = joint.build_seed(grid, prior, bookings)
        row = lambda date: grid.bookings.get_loc(pd.Timestamp(date))
        col = lambda date: grid.flights.get_loc(pd.Timestamp(date))

        self.assertTrue(grid.mask[row("2025-12-31"), col("2026-01-01")])
        self.assertTrue(grid.mask[row("2026-12-31"), col("2027-01-01")])
        self.assertFalse(grid.mask[row("2026-01-01"), col("2026-01-04")])
        self.assertEqual(seed[row("2026-01-01"), col("2026-01-01")], 3.0)
        self.assertGreater(seed[row("2025-12-31"), col("2026-01-01")], 2.0)
        self.assertEqual(seed[row("2026-01-02"), col("2026-01-02")], 0.0)
        self.assertTrue((seed[~grid.mask] == 0).all())


class OutputTests(unittest.TestCase):
    def test_output_paths_use_cluster_names_or_initials(self) -> None:
        base = Path("output") / "forecast.csv"
        self.assertEqual(joint.output_paths(base, ["North Africa"]),
                         (base.with_name("forecast_North_Africa.csv"),
                          base.with_name("forecast_North_Africa_filtered.csv")))
        self.assertEqual(joint.output_paths(base, ["North Africa", "City"]),
                         (base.with_name("forecast_C_NA.csv"),
                          base.with_name("forecast_C_NA_filtered.csv")))

    def test_filter_requires_both_dates_for_the_same_cluster(self) -> None:
        out = pd.DataFrame({
            "FlightMarketCluster": ["A", "A", "A", "A", "B"],
            "BookingDate": pd.to_datetime([
                "2026-01-01", "2025-12-31", "2026-01-01", "2026-01-02", "2026-01-02",
            ]),
            "FlightDate": pd.to_datetime([
                "2026-01-02", "2026-01-02", "2027-01-01", "2026-01-02", "2026-01-02",
            ]),
        })
        booking = pd.DataFrame({"FlightMarketCluster": ["A", "B"],
                                "Date": pd.to_datetime(["2026-01-01", "2026-01-02"])})
        flight = pd.DataFrame({"FlightMarketCluster": ["A", "B"],
                               "Date": pd.to_datetime(["2026-01-02", "2026-01-03"])})
        filtered = joint.filter_to_forecasts(out, booking, flight)
        self.assertEqual(filtered.index.tolist(), [0])

    def test_booking_comparison_aggregates_clusters_and_includes_missing_dates(self) -> None:
        out = pd.DataFrame({
            "BookingDate": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-03"]),
            "Tix": [1.5, 2.0, 4.0], "Revenue": [15.0, 40.0, 80.0],
        })
        forecast = pd.DataFrame({
            "Date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02"]),
            "Tix": [2.0, 3.0, 6.0], "Revenue": [20.0, 60.0, 90.0],
        })
        comparison = joint.compare_booking_forecast(out, forecast)
        self.assertEqual(comparison.columns.tolist(), [
            "BookingDate", "Tix_aggregated", "Tix_forecast", "Revenue_aggregated",
            "Revenue_forecast", "Tix_error", "Revenue_error",
        ])
        np.testing.assert_allclose(comparison.drop(columns="BookingDate").to_numpy(), [
            [3.5, 5, 55, 80, -1.5, -25],
            [0, 6, 0, 90, -6, -90],
            [4, 0, 80, 0, 4, 80],
        ])

    def test_post_process_only_uses_selected_clusters_without_fitting(self) -> None:
        forecast = pd.DataFrame({
            "FlightMarketCluster": ["City", "Greece", "City"],
            "Date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02"]),
            "Tix": [3.0, 50.0, 2.0], "Revenue": [30.0, 500.0, 20.0],
        })
        with tempfile.TemporaryDirectory() as directory:
            filtered_path = Path(directory) / "filtered.csv"
            pd.DataFrame({
                "FlightMarketCluster": ["City", "Greece"],
                "BookingDate": ["2026-01-01", "2026-01-01"],
                "FlightDate": ["2026-01-01", "2026-01-01"],
                "Tix": [1.25, 20.0], "Revenue": [12.5, 200.0],
            }).to_csv(filtered_path, sep=";", index=False)
            with (patch.object(joint, "load_actuals") as load_actuals,
                  patch.object(joint, "fit_cluster") as fit_cluster,
                  patch.object(joint, "load_forecast", return_value=forecast) as load_forecast,
                  patch("sys.argv", ["ipf_joint_table.py", "--post-process-only",
                                     str(filtered_path), "--clusters", "City"])):
                joint.main()
            load_actuals.assert_not_called()
            fit_cluster.assert_not_called()
            load_forecast.assert_called_once_with(joint.BOOKING_FC_FILE)
            comparison = pd.read_csv(Path(directory) / "comparison" /
                                     "filtered_booking_comparison.csv", sep=";")
            self.assertEqual(comparison["Tix_aggregated"].tolist(), [1.25, 0.0])
            self.assertEqual(comparison["Tix_forecast"].tolist(), [3.0, 2.0])
            self.assertEqual(comparison["Revenue_error"].tolist(), [-17.5, -20.0])

    def test_main_writes_outputs_with_six_decimals(self) -> None:
        actuals = pd.DataFrame({
            "FlightMarketCluster": ["City", "Greece"],
            "BookingDate": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "FlightDate": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "BookingMonth": [1, 1], "lead_time_days": [1, 1],
            "Tix": [10.0, 10.0], "Revenue": [100.0, 100.0],
        })
        forecast = pd.DataFrame({
            "FlightMarketCluster": ["City", "City", "Greece", "Greece"],
            "Date": pd.to_datetime(["2026-01-01", "2026-12-31"] * 2),
            "Tix": [1.234567] * 4, "Revenue": [12.345678] * 4,
        })
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "results" / "forecast.csv"
            with (patch.object(joint, "load_actuals", return_value=actuals),
                  patch.object(joint, "load_forecast", side_effect=[forecast, forecast, forecast]),
                patch.object(joint, "MAX_ITER", 20),
                  patch("sys.argv", ["ipf_joint_table.py", "--output", str(base)])):
                joint.main()
            full_path, filtered_path = joint.output_paths(base, ["City", "Greece"])
            with full_path.open(newline="") as file:
                full_rows = list(csv.DictReader(file, delimiter=";"))
            with filtered_path.open(newline="") as file:
                filtered_rows = list(csv.DictReader(file, delimiter=";"))

            self.assertGreater(len(full_rows), len(filtered_rows))
            self.assertTrue(filtered_rows)
            for row in full_rows + filtered_rows:
                self.assertRegex(row["Tix"], r"^\d+\.\d{6}$")
                self.assertRegex(row["Revenue"], r"^\d+\.\d{6}$")
            year_end = next(row for row in filtered_rows if row["BookingDate"] == "2026-12-31")
            self.assertEqual(year_end["Tix"], "1.234567")
            self.assertEqual(year_end["Revenue"], "12.345678")
            self.assertTrue((base.parent / "comparison" /
                             f"{filtered_path.stem}_booking_comparison.csv").exists())
            self.assertTrue(any(row["BookingDate"].startswith("2025") or
                                row["FlightDate"].startswith("2027") for row in full_rows))
            self.assertTrue(all(row["BookingDate"] in {"2026-01-01", "2026-12-31"} and
                                row["FlightDate"] in {"2026-01-01", "2026-12-31"}
                                for row in filtered_rows))


if __name__ == "__main__":
    unittest.main()