"""Tests for the booking-to-flight allocation matrices."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import bf_allocation_matrix as allocation


class AllocationMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.actuals = pd.DataFrame({
            "BookingDate": ["2024-01-03", "2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01"],
            "FlightDate": ["2024-01-05", "2024-01-05", "2024-01-03", "2024-01-03", "2024-01-05"],
            "FlightMarketCluster": ["City", "City", "City", "City", "Other"],
            "Tix": [4, 3, 1, 2, 100],
            "Revenue": [40, 30, 10, 25, 1000],
        })
        aggregated = allocation.aggregate_actuals(self.actuals)
        self.city = aggregated[aggregated["FlightMarketCluster"] == "City"]

    def test_matrix_order_zero_fill_and_both_metrics(self) -> None:
        for metric, expected in (("Tix", [[3, 3], [0, 4]]),
                                 ("Revenue", [[35, 30], [0, 40]])):
            path = allocation.save_matrix(self.city, "City", metric, self.output)
            matrix = pd.read_csv(path, sep=";", index_col="BookingDate")
            self.assertEqual(matrix.index.tolist(), ["2024-01-01", "2024-01-03"])
            self.assertEqual(matrix.columns.tolist(), ["2024-01-03", "2024-01-05"])
            self.assertEqual(matrix.values.tolist(), expected)
            self.assertEqual(path.name, "actuals_booking_to_flight_matrix_City.csv")

    def test_ticket_heatmap_preserves_totals_and_calendar_gaps(self) -> None:
        values, _, booking_width, flight_width = allocation.heatmap_values(self.city)
        self.assertEqual((booking_width, flight_width), (1, 1))
        self.assertEqual(values.shape, (3, 3))
        np.testing.assert_array_equal(values, [[3, 0, 3], [0, 0, 0], [0, 0, 4]])
        path = allocation.plot_heatmap(self.city, "City", self.output)
        self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_large_date_range_uses_bounded_summed_bins(self) -> None:
        data = pd.DataFrame({
            "BookingDate": ["2020-01-01", "2025-01-01"],
            "FlightDate": ["2020-01-02", "2027-01-01"],
            "Tix": [2, 7],
        })
        values, _, booking_width, flight_width = allocation.heatmap_values(data)
        self.assertLessEqual(max(values.shape), allocation.MAX_HEATMAP_BINS)
        self.assertGreater(booking_width, 1)
        self.assertGreater(flight_width, 1)
        self.assertEqual(values.sum(), 9)
        self.assertEqual(values[0, 0], 2)
        self.assertEqual(values[-1, -1], 7)

    def test_cli_selects_cluster_and_heatmap_always_uses_tix(self) -> None:
        actuals_path = self.output / "actuals.csv"
        self.actuals.to_csv(actuals_path, sep=";", index=False)
        with patch.object(allocation, "ACTUALS_FILE", actuals_path), \
             patch.object(allocation, "MATRIX_DIR", self.output / "matrices"), \
             patch.object(allocation, "PLOT_DIR", self.output / "plots"), \
             patch("sys.argv", ["allocation", "--clusters", "City", "--metric", "Revenue"]), \
             patch.object(allocation, "plot_heatmap", wraps=allocation.plot_heatmap) as heatmap:
            allocation.main()
        self.assertEqual(heatmap.call_count, 1)
        self.assertEqual(heatmap.call_args.args[0]["Tix"].sum(), 10)
        self.assertEqual(len(list((self.output / "matrices").glob("*.csv"))), 1)
        self.assertEqual(len(list((self.output / "plots").glob("*.png"))), 1)

    def test_unknown_cluster_is_reported(self) -> None:
        actuals_path = self.output / "actuals.csv"
        self.actuals.to_csv(actuals_path, sep=";", index=False)
        with patch.object(allocation, "ACTUALS_FILE", actuals_path), \
             patch("sys.argv", ["allocation", "--clusters", "Missing"]), \
             patch("sys.stderr") as stderr:
            with self.assertRaises(SystemExit):
                allocation.main()
        self.assertIn("Unknown clusters: Missing", stderr.write.call_args_list[-1].args[0])


if __name__ == "__main__":
    unittest.main()