"""Compare weighted booking/flight month correlations in actuals and IPF output.

Each heatmap cell is the weighted Pearson correlation of the indicators
``booking month == b`` and ``flight month == f``. Tix or Revenue supplies
the weights. Only same-calendar-year booking/flight pairs are compared.
"""

from __future__ import annotations

import argparse
import calendar
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ACTUALS_FILE = ROOT / "data" / "EWH_DummyActuals_sorted_mod_clean.csv"
PLOT_DIR = ROOT / "plots"
METRICS = ("Tix", "Revenue")
MONTHS = pd.Index(range(1, 13), name="FlightMonth")
BOOKING_MONTHS = pd.Index(range(1, 13), name="BookingMonth")
MONTH_LABELS = list(calendar.month_abbr)[1:]
REQUIRED_COLUMNS = ("BookingDate", "FlightDate", "FlightMarketCluster", *METRICS)


def load_table(path: Path) -> pd.DataFrame:
	"""Load a semicolon-delimited actuals or filtered IPF table."""
	return pd.read_csv(path, sep=";", usecols=list(REQUIRED_COLUMNS))


def monthly_totals(data: pd.DataFrame, cluster: str) -> dict[str, pd.DataFrame]:
	"""Sum each metric by flight month (rows) and booking month (columns).

	Pool all years, excluding cross-year pairs so the historical scope matches
	a same-year filtered IPF table. Input month columns are not trusted.
	"""

	frame = data.loc[data["FlightMarketCluster"] == cluster, list(REQUIRED_COLUMNS)].copy()
	if frame.empty:
		raise ValueError(f"Cluster not found: {cluster}")

	for column in ("BookingDate", "FlightDate"):
		frame[column] = pd.to_datetime(frame[column], format="%Y-%m-%d", errors="coerce")
		if frame[column].isna().any():
			raise ValueError(f"{column} contains missing or invalid dates")
	
	for metric in METRICS:
		frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
		values = frame[metric].to_numpy(dtype=float)

	frame = frame.loc[frame["BookingDate"].dt.year == frame["FlightDate"].dt.year].copy()
	if frame.empty:
		raise ValueError(f"No same-year booking/flight pairs for cluster: {cluster}")
	frame["BookingMonth"] = frame["BookingDate"].dt.month
	frame["FlightMonth"] = frame["FlightDate"].dt.month
	grouped = frame.groupby(["FlightMonth", "BookingMonth"])[list(METRICS)].sum()
	return {
		metric: grouped[metric].unstack("BookingMonth").reindex(
			index=MONTHS, columns=BOOKING_MONTHS, fill_value=0.0
		).fillna(0.0)
		for metric in METRICS
	}

def monthly_totals2(
    data: pd.DataFrame,
    cluster: str,
    *,
    booking_year: int,
    same_year_only: bool = False,
) -> dict[str, pd.DataFrame]:
    """Aggregate one booking-year cohort, optionally excluding carry-out."""

    frame = data.loc[
        data["FlightMarketCluster"] == cluster,
        list(REQUIRED_COLUMNS),
    ].copy()

    for column in ("BookingDate", "FlightDate"):
        frame[column] = pd.to_datetime(
            frame[column], format="%Y-%m-%d", errors="raise"
        )

    keep = frame["BookingDate"].dt.year.eq(booking_year)

    if same_year_only:
        keep &= frame["FlightDate"].dt.year.eq(booking_year)

    frame = frame.loc[keep].copy()
    if frame.empty:
        raise ValueError(
            f"No selected observations for {cluster}, {booking_year}"
        )

    for metric in METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors="raise")
        values = frame[metric].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{metric} must be finite and nonnegative")

    frame["BookingMonth"] = frame["BookingDate"].dt.month
    frame["FlightMonth"] = frame["FlightDate"].dt.month

    grouped = frame.groupby(
        ["FlightMonth", "BookingMonth"]
    )[list(METRICS)].sum()

    return {
        metric: grouped[metric]
        .unstack("BookingMonth")
        .reindex(index=MONTHS, columns=BOOKING_MONTHS)
        .fillna(0.0)
        for metric in METRICS
    }


def indicator_correlations(totals: pd.DataFrame) -> pd.DataFrame:
	"""Return weighted Pearson correlations for every pair of month indicators.

	If a month has zero or all weight, its indicator is constant and the
	corresponding correlations are undefined (NaN), not zero.
	"""
	weights = totals.to_numpy(dtype=float)
	total = weights.sum()
	if not np.isfinite(total) or total <= 0:
		raise ValueError("Metric total must be positive and finite")
	joint = weights / total
	flight = np.clip(joint.sum(axis=1), 0, 1)
	booking = np.clip(joint.sum(axis=0), 0, 1)
	numerator = joint - np.outer(flight, booking)
	denominator = np.sqrt(np.outer(flight * (1 - flight), booking * (1 - booking)))
	if np.count_nonzero(weights.sum(axis=1)) == 1:
		denominator[:] = 0
	if np.count_nonzero(weights.sum(axis=0)) == 1:
		denominator[:] = 0
	correlation = np.full_like(joint, np.nan)
	np.divide(numerator, denominator, out=correlation, where=denominator > 0)
	return pd.DataFrame(np.clip(correlation, -1, 1), index=totals.index, columns=totals.columns)


def month_number_correlation(totals: pd.DataFrame) -> float:
	"""Return weighted Pearson correlation of ordinal month numbers 1–12.

	Months are treated as linear numbers, not circular dates; a constant
	booking or flight month makes the result undefined (NaN).
	"""
	weights = totals.to_numpy(dtype=float)
	total = weights.sum()
	if not np.isfinite(total) or total <= 0:
		raise ValueError("Metric total must be positive and finite")
	if (np.count_nonzero(weights.sum(axis=1)) < 2 or
			np.count_nonzero(weights.sum(axis=0)) < 2):
		return float("nan")
	joint = weights / total
	flight = np.arange(1, 13, dtype=float)
	booking = np.arange(1, 13, dtype=float)
	flight_mean = joint.sum(axis=1) @ flight
	booking_mean = joint.sum(axis=0) @ booking
	flight_delta = flight - flight_mean
	booking_delta = booking - booking_mean
	covariance = flight_delta @ joint @ booking_delta
	flight_var = joint.sum(axis=1) @ (flight_delta ** 2)
	booking_var = joint.sum(axis=0) @ (booking_delta ** 2)
	if flight_var <= 0 or booking_var <= 0:
		return float("nan")
	return float(np.clip(covariance / np.sqrt(flight_var * booking_var), -1, 1))


def analyze_cluster(
	actuals: pd.DataFrame, forecast: pd.DataFrame, cluster: str
) -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
	"""Calculate both metrics; return scalar comparison and indicator matrices.

	This function accepts DataFrames directly.
	Matrices are indexed first by metric, then by ``actuals`` or ``forecast``.
	"""
	actual_totals = monthly_totals(actuals, cluster)
	forecast_totals = monthly_totals(forecast, cluster)
	matrices = {}
	rows = []
	for metric in METRICS:
		matrices[metric] = {
			"actuals": indicator_correlations(actual_totals[metric]),
			"forecast": indicator_correlations(forecast_totals[metric]),
		}
		actual = month_number_correlation(actual_totals[metric])
		predicted = month_number_correlation(forecast_totals[metric])
		rows.append({"FlightMarketCluster": cluster, "metric": metric,
					 "actuals": actual, "forecast": predicted,
					 "forecast_minus_actuals": predicted - actual})
	return pd.DataFrame(rows), matrices


def cluster_filename(cluster: str) -> str:
	"""Make a safe filename component from a cluster name."""
	return re.sub(r"[^\w-]+", "_", cluster, flags=re.UNICODE).strip("_") or "cluster"


def plot_correlations(
	matrices: dict[str, pd.DataFrame], cluster: str, metric: str, output_dir: Path
) -> Path:
	"""Save side-by-side indicator-correlation heatmaps on a common scale."""
	if metric not in METRICS:
		raise ValueError("metric must be Tix or Revenue")
	fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
	cmap = plt.get_cmap("coolwarm").with_extremes(bad="lightgrey")
	for ax, source in zip(axes, ("actuals", "forecast")):
		matrix = matrices[source]
		image = ax.imshow(matrix.to_numpy(), cmap=cmap, vmin=-1, vmax=1, origin="upper")
		ax.set_xticks(range(12), MONTH_LABELS, rotation=45)
		ax.set_yticks(range(12), MONTH_LABELS)
		ax.set(xlabel="Booking month", ylabel="Flight month", title=source.capitalize())
		for row in range(12):
			for col in range(12):
				value = matrix.iat[row, col]
				if np.isfinite(value):
					ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=6)
	fig.suptitle(f"{cluster}: {metric}-weighted month indicator correlations")
	fig.colorbar(image, ax=axes, label="Weighted Pearson correlation", shrink=0.8)
	output_dir.mkdir(parents=True, exist_ok=True)
	path = output_dir / f"month_correlations_{cluster_filename(cluster)}_{metric}.png"
	try:
		fig.savefig(path, dpi=150)
	finally:
		plt.close(fig)
	return path


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--actuals", type=Path, default=ACTUALS_FILE)
	parser.add_argument("--ipf-output", type=Path, required=True,
						help="Path to a filtered IPF output CSV")
	parser.add_argument("--clusters", nargs="+", help="Clusters to compare (default: all in IPF output)")
	parser.add_argument("--metric", choices=METRICS, default="Tix", help="Heatmap metric (default: Tix)")
	parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR)
	args = parser.parse_args()

	try:
		actuals, forecast = load_table(args.actuals), load_table(args.ipf_output)
		clusters = args.clusters or sorted(forecast["FlightMarketCluster"].dropna().unique())
		if not clusters:
			raise ValueError("No clusters in IPF output")
		results = []
		for cluster in dict.fromkeys(clusters):
			summary, matrices = analyze_cluster(actuals, forecast, cluster)
			results.append(summary)
			print(f"Saved {plot_correlations(matrices[args.metric], cluster, args.metric, args.plot_dir)}")
		print("Weighted Pearson correlation of month numbers (1–12):")
		print(pd.concat(results, ignore_index=True).to_string(index=False))
	except (ValueError, OSError) as exc:
		parser.error(str(exc))


if __name__ == "__main__":
	main()