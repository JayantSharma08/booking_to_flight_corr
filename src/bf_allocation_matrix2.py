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

# --- Matrix: full calendar, both metrics, with checks ---------------------
def build_matrix(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    b = pd.date_range(data["BookingDate"].min(), data["BookingDate"].max(), freq="D")
    f = pd.date_range(data["FlightDate"].min(), data["FlightDate"].max(), freq="D")
    m = (data.pivot_table(index="BookingDate", columns="FlightDate", values=metric,
                          aggfunc="sum", fill_value=0)
             .reindex(index=b, columns=f, fill_value=0))
    validate_matrix(m, data, metric)
    return m


def validate_matrix(m: pd.DataFrame, data: pd.DataFrame, metric: str) -> None:
    assert np.isclose(m.to_numpy().sum(), data[metric].sum()), "Total mismatch"
    row = data.groupby("BookingDate")[metric].sum()
    col = data.groupby("FlightDate")[metric].sum()
    assert np.allclose(m.sum(axis=1).loc[row.index], row), "Row sums mismatch"
    assert np.allclose(m.sum(axis=0).loc[col.index], col), "Column sums mismatch"
    assert (data["FlightDate"] >= data["BookingDate"]).all(), "Flight before booking"


def save_matrices(data: pd.DataFrame, cluster: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("Tix", "Revenue"):
        m = build_matrix(data, metric)
        m.index = m.index.strftime("%Y-%m-%d")
        m.columns = m.columns.strftime("%Y-%m-%d")
        m.to_csv(output_dir / f"actuals_b_to_f_matrix_{metric.lower()}_{cluster_filename(cluster)}.csv",
                 sep=";", index_label="BookingDate")


# --- Heatmap: white background, log scale, reference lines ----------------
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


def plot_heatmap(data: pd.DataFrame, cluster: str, output_dir: Path,
                 metric: str = "Tix") -> Path:
    values, extent, bw, fw = heatmap_values(data, metric)
    masked = np.ma.masked_less_equal(values, 0)          # zeros -> white

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("white")
    vmin, vmax = (masked.min(), masked.max()) if masked.count() else (1, 1)
    norm = LogNorm(vmin=vmin, vmax=vmax) if vmax > vmin else None

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="white")
    ax.set_facecolor("white")
    img = ax.imshow(masked, origin="lower", aspect="auto", extent=extent,
                    interpolation="nearest", cmap=cmap, norm=norm)

    # Reference: diagonal Booking = Flight (valid data lies on/below it)
    lo, hi = min(extent[0], extent[2]), max(extent[1], extent[3])
    ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8, ls="--", label="Booking = Flight")

    # Reference: year boundaries to spot cross-year bookings
    for year in range(mdates.num2date(lo).year + 1, mdates.num2date(hi).year + 1):
        x = mdates.date2num(pd.Timestamp(f"{year}-01-01"))
        ax.axvline(x, color="lightgrey", lw=0.6, ls=":")
        ax.axhline(x, color="lightgrey", lw=0.6, ls=":")

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.xaxis_date(); ax.yaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(False)
    ax.legend(loc="upper left", frameon=False)
    ax.set(xlabel="Flight date", ylabel="Booking date",
           title=f"{cluster}: {metric} by booking × flight date "
                 f"({bw}-day × {fw}-day bins)")
    fig.colorbar(img, ax=ax, label=f"{metric} per bin (log scale)")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"actuals_bf_heatmap_{metric.lower()}_{cluster_filename(cluster)}.png"
    fig.savefig(path, dpi=150, facecolor="white")
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
	
	for cluster in clusters:
		cluster_data = data[data["FlightMarketCluster"] == cluster]
		save_matrices(cluster_data, cluster, MATRIX_DIR)
		heatmap = plot_heatmap(cluster_data, cluster, PLOT_DIR)
		print(f"Saved matrix to {MATRIX_DIR}")
		print(f"Saved Tix heatmap: {heatmap}")


if __name__ == "__main__":
	main()