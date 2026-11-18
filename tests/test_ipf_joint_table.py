"""Focused checks for the historical prior and booking-to-flight seed."""

import unittest

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


if __name__ == "__main__":
    unittest.main()