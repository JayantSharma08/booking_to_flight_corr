"""Add month features to clean actuals and create historical EDA plots."""

import argparse
import calendar
import hashlib
import math
import os
import re
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save plots without requiring a graphical display.
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ACTUALS_FILE = ROOT / "data" / "EWH_DummyActuals_sorted_mod_clean.csv"
PLOT_DIR = ROOT / "plots"
MONTHS = list(range(1, 13))
MONTH_LABELS = list(calendar.month_abbr)[1:]
LEAD_TIME_LABELS = ["0-7", "8-30", "31-90", "91-180", "181+"]


def add_month_columns(path: Path) -> pd.DataFrame:
    """Update the CSV with month numbers, preserving existing column values."""
    data = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False)
    dates = {}
    for column in ("BookingDate", "FlightDate"):
        if column not in data:
            raise ValueError(f"Missing required column: {column}")
        dates[column] = pd.to_datetime(data[column], format="%Y-%m-%d", errors="coerce")
        if dates[column].isna().any():
            raise ValueError(f"{column} contains missing or invalid dates")

    data["BookingMonth"] = dates["BookingDate"].dt.month
    data["FlightMonth"] = dates["FlightDate"].dt.month

    # Replace the original only after a complete CSV has been written.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            data.to_csv(temporary_file, sep=";", index=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return data


def cluster_filename(cluster: str) -> str:
    """Create a safe, collision-resistant filename component for a cluster."""
    name = re.sub(r"[^\w-]+", "_", cluster, flags=re.UNICODE).strip("_")[:80]
    return f"{name or 'cluster'}"


def month_relationship_totals(
    data: pd.DataFrame, cluster: str, size_variable: str
) -> pd.DataFrame:
    """Sum same-year months separately from all later-year flights."""
    if size_variable not in ("Tix", "Revenue"):
        raise ValueError("size_variable must be Tix or Revenue")
    cluster_data = data.loc[data["FlightMarketCluster"] == cluster].copy()
    if cluster_data.empty:
        raise ValueError(f"Unknown cluster: {cluster}")

    booking_dates = pd.to_datetime(
        cluster_data["BookingDate"], format="%Y-%m-%d", errors="coerce"
    )
    flight_dates = pd.to_datetime(
        cluster_data["FlightDate"], format="%Y-%m-%d", errors="coerce"
    )
    if booking_dates.isna().any() or flight_dates.isna().any():
        raise ValueError("BookingDate and FlightDate must contain valid dates")
    if (flight_dates < booking_dates).any():
        raise ValueError("FlightDate cannot be earlier than BookingDate")

    values = pd.to_numeric(cluster_data[size_variable], errors="coerce")
    if values.isna().any() or not values.map(math.isfinite).all() or (values < 0).any():
        raise ValueError(f"{size_variable} must contain finite, non-negative numbers")
    cluster_data[size_variable] = values
    cluster_data["BookingMonth"] = booking_dates.dt.month
    cluster_data["FlightMonthGroup"] = flight_dates.dt.month.where(
        flight_dates.dt.year == booking_dates.dt.year, 13
    )
    return cluster_data.groupby(
        ["BookingMonth", "FlightMonthGroup"], as_index=False
    )[size_variable].sum()


def plot_month_relationship(
    data: pd.DataFrame, cluster: str, size_variable: str, output_dir: Path
) -> Path:
    """Plot same-year flight months and a separate later-year flight bucket."""
    totals = month_relationship_totals(data, cluster, size_variable)
    totals = totals.loc[totals[size_variable] > 0]

    fig, ax = plt.subplots(figsize=(10, 7))
    if not totals.empty:
        maximum = totals[size_variable].max()
        areas = 800 * totals[size_variable] / maximum
        ax.scatter(
            totals["BookingMonth"], totals["FlightMonthGroup"], s=areas,
            color="steelblue", alpha=0.65, edgecolors="white",
        )
        for fraction in (0.25, 0.5, 1.0):
            ax.scatter([], [], s=800 * fraction, color="steelblue", alpha=0.65,
                       label=f"{maximum * fraction:,.2f}")
            # this above code is a hack to create a legend with the correct sizes without plotting extra points
        ax.legend(title=f"Total {size_variable}", loc="upper left",
              bbox_to_anchor=(1.02, 1), labelspacing=2, handleheight=2)
    else:
        ax.text(0.5, 0.5, f"No positive {size_variable} totals", ha="center",
                transform=ax.transAxes)
    ax.set_xticks(MONTHS, MONTH_LABELS)
    ax.set_yticks([*MONTHS, 13], [*MONTH_LABELS, "Later year"])
    ax.set(xlim=(0.5, 12.5), ylim=(0.5, 13.5), xlabel="Booking month",
           ylabel="Flight month relative to booking year",
           title=f"{cluster}: booking vs flight month (all booking years)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"booking_flight_month_{cluster_filename(cluster)}_{size_variable}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_lead_time_distribution(
    data: pd.DataFrame, cluster: str, output_dir: Path
) -> Path:
    """Plot total booking revenue in mutually exclusive lead-time buckets."""
    cluster_data = data.loc[data["FlightMarketCluster"] == cluster].copy()
    cluster_data["bucket"] = pd.cut(
        cluster_data["lead_time_days"], bins=[-1, 7, 30, 90, 180, float("inf")],
        labels=LEAD_TIME_LABELS,
    )
    revenue = cluster_data.groupby("bucket", observed=False)["Revenue"].sum()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(LEAD_TIME_LABELS, revenue.reindex(LEAD_TIME_LABELS), color="steelblue")
    ax.set(xlabel="Lead time (days)", ylabel="Total booking revenue (EUR)",
           title=f"{cluster}: revenue by lead time (all years)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"lead_time_revenue_{cluster_filename(cluster)}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_median_lead_time(data: pd.DataFrame, output_dir: Path) -> Path:
    """Plot the unweighted median lead time per row, month and cluster."""
    medians = data.pivot_table(
        index="FlightMarketCluster", columns="BookingMonth",
        values="lead_time_days", aggfunc="median",
    ).reindex(columns=MONTHS).sort_index()
    fig, ax = plt.subplots(figsize=(12, max(4, len(medians) * 0.45)))
    # A missing month/cluster is not a zero.
    cmap = plt.get_cmap("viridis").with_extremes(bad="lightgrey")
    image = ax.imshow(medians, aspect="auto", cmap=cmap)
    ax.set_xticks(range(12), MONTH_LABELS)
    ax.set_yticks(range(len(medians)), medians.index)
    ax.set(xlabel="Booking month", ylabel="Flight market cluster",
           title="Median lead time by booking month (all years, unweighted)")
    fig.colorbar(image, ax=ax, label="Median lead time (days)")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "median_lead_time_booking_month_cluster.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", nargs="+", help="Clusters to plot (default: all)")
    parser.add_argument("--size-variable", choices=("Tix", "Revenue"), default="Tix",
                        help="Bubble size measure (default: Tix)")
    args = parser.parse_args()
    # First check whether the actuals file already contains the month columns, and if not, only then add them. This avoids unnecessary writes and ensures idempotency.
    data = pd.read_csv(ACTUALS_FILE, sep=";", dtype=str, keep_default_na=False)
    if "BookingMonth" not in data.columns or "FlightMonth" not in data.columns:
        data = add_month_columns(ACTUALS_FILE)
    for column in ("BookingMonth", "FlightMonth", "lead_time_days", "Tix", "Revenue"):
        data[column] = pd.to_numeric(data[column], errors="raise")
    available_clusters = sorted(data["FlightMarketCluster"].unique())
    clusters = args.clusters if args.clusters is not None else available_clusters
    unknown = sorted(set(clusters) - set(available_clusters))
    if unknown:
        parser.error(f"Unknown clusters: {', '.join(unknown)}")
    for cluster in dict.fromkeys(clusters):
        print(f"Saved {plot_month_relationship(data, cluster, args.size_variable, PLOT_DIR)}")
        print(f"Saved {plot_lead_time_distribution(data, cluster, PLOT_DIR)}")
    print(f"Saved {plot_median_lead_time(data, PLOT_DIR)}")


if __name__ == "__main__":
    main()