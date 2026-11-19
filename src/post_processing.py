"""Compare ticket-weighted monthly AOVs in the joint and input forecasts."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
JOINT_FILE = DATA / "output" / "joint_forecast_2026_unfiltered.csv"
BOOKING_FILE = DATA / "EWH_DummyBookingPeriodFc_sorted_mod_clean.csv"
FLIGHT_FILE = DATA / "EWH_DummyFlightPeriodFc_sorted_mod_clean.csv"
COMPARISON_DIR = DATA / "output" / "comparison"
YEAR = 2026


def add_aov(data: pd.DataFrame) -> pd.DataFrame:
	"""Calculate revenue per ticket; AOV is undefined when tickets are zero."""
	return data.assign(AOV=data["Revenue"] / data["Tix"].where(data["Tix"] != 0))


def monthly_aov(data: pd.DataFrame, date_column: str) -> pd.Series:
	"""Divide monthly total revenue by total tickets, including carry rows."""
	dates = data[date_column]
	in_year = dates.dt.year == YEAR
	totals = data.loc[in_year].groupby(dates.loc[in_year].dt.month)[["Tix", "Revenue"]].sum()
	return add_aov(totals)["AOV"].reindex(range(1, 13))


def plot_comparison(comparison: pd.DataFrame, path: Path) -> None:
	"""Compare both joint date margins with the two forecast series."""
	fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True, sharey=True)
	for ax, joint_column, title in zip(
		axes,
		["Joint (booking month)", "Joint (flight month)"],
		["Joint grouped by booking month", "Joint grouped by flight month"],
	):
		ax.plot(comparison.index, comparison[joint_column], color="tab:blue",
		        linewidth=4, label=joint_column)
		for column, color in [("Booking forecast", "tab:orange"),
		                      ("Flight forecast", "tab:green")]:
			ax.plot(comparison.index, comparison[column], color=color, linestyle="--",
			        marker="o", markersize=4, label=column)
		ax.set_title(title)
		ax.set_ylabel("Revenue per ticket (AOV)")
		ax.grid(True, alpha=0.25)
		ax.legend()
	axes[-1].set_xlabel(f"Month in {YEAR}")
	axes[-1].set_xticks(range(1, 13))
	fig.tight_layout()
	fig.savefig(path, dpi=150)
	plt.close(fig)


def monthly_errors(path: Path, date_column: str) -> pd.DataFrame:
	"""Sum signed daily forecast errors by calendar month."""
	comparison = pd.read_csv(path, sep=";", parse_dates=[date_column])
	return comparison.groupby(pd.Grouper(key=date_column, freq="MS"))[
		["Tix_error", "Revenue_error"]
	].sum(min_count=1)


def plot_forecast_errors(booking_path: Path, flight_path: Path, output_path: Path) -> None:
	"""Plot monthly joint-minus-forecast errors for both forecast periods."""
	booking = monthly_errors(booking_path, "BookingDate")
	flight = monthly_errors(flight_path, "FlightDate")
	fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
	for ax, metric, label in zip(axes, ["Tix_error", "Revenue_error"], ["Tickets", "Revenue"]):
		ax.plot(booking.index, booking[metric], marker="o", label="Booking period")
		ax.plot(flight.index, flight[metric], marker="s", label="Flight period")
		ax.axhline(0, color="black", linewidth=0.8)
		ax.set_ylabel(f"{label} error")
		ax.grid(True, alpha=0.25)
		ax.legend()
	locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
	# the x-axis should only show the month names
	axes[-1].xaxis.set_major_locator(locator)
	axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
	axes[-1].set_xlabel("Month")
	axes[-1].set_xlim([pd.Timestamp(f"{YEAR}-01-01"), pd.Timestamp(f"{YEAR}-12-31")])
	fig.suptitle(f"Monthly forecast errors (joint minus forecast) for year {YEAR}")
	fig.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def main() -> None:
	joint = pd.read_csv(JOINT_FILE, sep=";", parse_dates=["BookingDate", "FlightDate"])
	booking = pd.read_csv(BOOKING_FILE, sep=";", parse_dates=["Date"])
	flight = pd.read_csv(FLIGHT_FILE, sep=";", parse_dates=["Date"])

	joint = add_aov(joint)
	comparison = pd.DataFrame({
		"Joint (booking month)": monthly_aov(joint, "BookingDate"),
		"Joint (flight month)": monthly_aov(joint, "FlightDate"),
		"Booking forecast": monthly_aov(booking, "Date"),
		"Flight forecast": monthly_aov(flight, "Date"),
	})
	comparison.index.name = "Month"

	COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
	joint.to_csv(COMPARISON_DIR / "joint_forecast_2026_unfiltered_with_aov.csv", sep=";", index=False)
	comparison.to_csv(COMPARISON_DIR / "monthly_aov_comparison.csv", sep=";")
	plot_comparison(comparison, COMPARISON_DIR / "aov_comparison.png")
	plot_forecast_errors(
		COMPARISON_DIR / f"{JOINT_FILE.stem}_booking_comparison.csv",
		COMPARISON_DIR / f"{JOINT_FILE.stem}_flight_comparison.csv",
		COMPARISON_DIR / "forecast_error_comparison.png",
	)


if __name__ == "__main__":
	main()