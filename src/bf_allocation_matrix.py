"""Build booking-to-flight matrices and ticket heatmaps from clean actuals."""

import argparse
import csv
import hashlib
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ACTUALS_FILE = ROOT / "data" / "EWH_DummyActuals_sorted_mod_clean.csv"
MATRIX_DIR = ROOT / "data" / "matrices"
PLOT_DIR = ROOT / "plots"
DATE_COLUMNS = ["BookingDate", "FlightDate"]
MAX_HEATMAP_BINS = 512


def cluster_filename(cluster: str) -> str:
	"""Preserve normal cluster names, making unsafe filenames unambiguous."""
	safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", cluster).rstrip(" .")
	if safe != cluster or not safe or len(safe) > 100:
		digest = hashlib.sha256(cluster.encode("utf-8")).hexdigest()[:12]
		safe = f"{safe[:80] or 'cluster'}_{digest}"
	return safe


def aggregate_actuals(data: pd.DataFrame) -> pd.DataFrame:
	"""Sum both measures for each observed cluster and date pair."""
	return data.groupby(
		["FlightMarketCluster", *DATE_COLUMNS], as_index=False, sort=True
	)[["Tix", "Revenue"]].sum()


def save_matrix(data: pd.DataFrame, cluster: str, metric: str, output_dir: Path) -> Path:
	"""Stream a date-by-date CSV without building a dense matrix in memory."""
	booking_dates = sorted(data["BookingDate"].unique())
	flight_dates = sorted(data["FlightDate"].unique())
	flight_positions = {flight_date: i for i, flight_date in enumerate(flight_dates)}
	output_dir.mkdir(parents=True, exist_ok=True)
	path = output_dir / f"actuals_booking_to_flight_matrix_{cluster_filename(cluster)}.csv"

	# Rows are grouped once; only one zero-filled row is materialized at a time.
	grouped = data.groupby("BookingDate", sort=True)
	with path.open("w", encoding="utf-8", newline="") as matrix_file:
		writer = csv.writer(matrix_file, delimiter=";")
		writer.writerow(["BookingDate", *flight_dates])
		for booking_date in booking_dates:
			row = [0] * len(flight_dates)
			observations = grouped.get_group(booking_date)
			for flight_date, value in observations[["FlightDate", metric]].itertuples(
				index=False, name=None
			):
				row[flight_positions[flight_date]] = value
			writer.writerow([booking_date, *row])
	return path


def heatmap_values(data: pd.DataFrame, metric: str = "Tix") -> tuple[np.ndarray, tuple[float, ...], int, int]:
	"""Sum tickets into bounded, uniform calendar-day bins for visualization."""
	booking = mdates.date2num(pd.to_datetime(data["BookingDate"]).to_numpy())
	flight = mdates.date2num(pd.to_datetime(data["FlightDate"]).to_numpy())
	booking_start, flight_start = int(booking.min()), int(flight.min())
	booking_days = int(booking.max()) - booking_start + 1
	flight_days = int(flight.max()) - flight_start + 1
	booking_width = max(1, math.ceil(booking_days / MAX_HEATMAP_BINS))
	flight_width = max(1, math.ceil(flight_days / MAX_HEATMAP_BINS))
	rows = math.ceil(booking_days / booking_width)
	columns = math.ceil(flight_days / flight_width)
	values = np.zeros((rows, columns), dtype=float)
	np.add.at(
		values,
		((booking.astype(int) - booking_start) // booking_width,
		 (flight.astype(int) - flight_start) // flight_width),
		data[metric].to_numpy(),
	)
	extent = (flight_start, flight_start + columns * flight_width,
			  booking_start, booking_start + rows * booking_width)
	return values, extent, booking_width, flight_width


def plot_heatmap(data: pd.DataFrame, cluster: str, output_dir: Path, metric: str = "Tix") -> Path:
	"""Save a ticket heatmap; aggregate days only when the date range is large."""
	values, extent, booking_width, flight_width = heatmap_values(data, metric)
	masked = np.ma.masked_less_equal(values, 0)          # zeros -> white
	cmap = plt.get_cmap("YlOrRd").copy()
	cmap.set_bad("white")
	vmin, vmax = (masked.min(), masked.max()) if masked.count() else (1, 1)
	norm = LogNorm(vmin=vmin, vmax=vmax) if vmax > vmin else None
    
	fig, ax = plt.subplots(figsize=(12, 8), facecolor="white")
	ax.set_facecolor("white")
	image = ax.imshow(masked, origin="lower", aspect="auto", extent=extent,
					  interpolation="nearest", cmap=cmap, norm=norm)
	ax.xaxis_date()
	ax.yaxis_date()
	ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
	ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
	ax.tick_params(axis="x", labelrotation=45)
	ax.set(xlabel="Flight date", ylabel="Booking date",
		   title=f"{cluster}: tickets by booking and flight date\n"
				 f"Summed per {booking_width}-day booking × {flight_width}-day flight bin")
	fig.colorbar(image, ax=ax, label="Tix (sum per bin)")
	fig.tight_layout()
	output_dir.mkdir(parents=True, exist_ok=True)
	path = output_dir / f"actuals_booking_to_flight_heatmap_{cluster_filename(cluster)}.png"
	fig.savefig(path, dpi=150)
	plt.close(fig)
	return path


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--clusters", nargs="+", help="Clusters to include (default: all)")
	parser.add_argument("--metric", choices=("Tix", "Revenue"), default="Tix",
						help="Matrix values (default: Tix); heatmaps always show Tix")
	args = parser.parse_args()

	# Cleaned actuals already contain the necessary dates and measures.
	data = pd.read_csv(
		ACTUALS_FILE, sep=";", usecols=["FlightMarketCluster", *DATE_COLUMNS, "Tix", "Revenue"],
		parse_dates=DATE_COLUMNS
	)
	available = set(data["FlightMarketCluster"].unique())
	clusters = list(dict.fromkeys(args.clusters)) if args.clusters else sorted(available)
	unknown = set(clusters) - available
	if unknown:
		parser.error(f"Unknown clusters: {', '.join(sorted(unknown))}")
	data = aggregate_actuals(data[data["FlightMarketCluster"].isin(clusters)])
	for cluster in clusters:
		cluster_data = data[data["FlightMarketCluster"] == cluster]
		matrix = save_matrix(cluster_data, cluster, args.metric, MATRIX_DIR)
		heatmap = plot_heatmap(cluster_data, cluster, PLOT_DIR)
		print(f"Saved {args.metric} matrix: {matrix}")
		print(f"Saved Tix heatmap: {heatmap}")


if __name__ == "__main__":
	main()