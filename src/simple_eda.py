#  This will be implementing the simple analysis, checks and visualize the cleaned actuals data and the two forecasts.

from email import parser
from typing import List
from pathlib import Path
import argparse
import calendar

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ACTUALS_FILE = ROOT / "data" / "EWH_DummyActuals_sorted_mod_clean.csv"
RAW_BOOKING_FORECAST_FILE = ROOT / "data" / "EWH_DummyBookingPeriodFc.csv"
RAW_FLIGHT_FORECAST_FILE = ROOT / "data" / "EWH_DummyFlightPeriodFc.csv"
BOOKING_FORECAST_FILE = ROOT / "data" / "EWH_DummyBookingPeriodFc_sorted_mod_clean.csv"
FLIGHT_FORECAST_FILE = ROOT / "data" / "EWH_DummyFlightPeriodFc_sorted_mod_clean.csv"
DATE_COLUMNS = ["BookingDate", "FlightDate"]
CLUSTER_VIEW_DIR = ROOT / "data" / "cluster_view"

def find_highest_lead_times(data: pd.DataFrame, max_n: int) -> List[tuple[int, int]]:
      """Extract the n highest lead_time_days alongwith their total corresponding Tix from the input dataframe."""
      return list(data.nlargest(max_n, "lead_time_days")[["lead_time_days", "Tix"]].itertuples(index=False, name=None))

# create a function that takes as input a dataframe, a list of clusters, a list of months and a year, and returns a filtered dataframe based on the input parameters.
def cluster_view(
    data: pd.DataFrame,
    clusters: List[str],
    months: List[int],
    year: int = 2024,
    date_column: str = "BookingDate",
) -> pd.DataFrame:
    """Filter by clusters, months and year using the selected date column.

    The selected date column must contain pandas datetime values.
    """
    if date_column not in DATE_COLUMNS:
        raise ValueError(f"date_column must be one of {DATE_COLUMNS}")

    dates = data[date_column]
    mask = (
        data["FlightMarketCluster"].isin(clusters)
        & dates.dt.year.eq(year)
        & dates.dt.month.isin(months)
    )
    return data.loc[mask].copy()

# TODO: Implement a function that calculates and visualizes for each cluster, the difference between the monthly total Tix(or the monthly revenue total) from the booking forecast file versus the same quantity from the flight forecast file. Print the average difference and visualize the 12 months differences appropriately.



def main():
    parser = argparse.ArgumentParser(description="Simple EDA on cleaned actuals and forecasts data")
    # User can input a cluster name and a month and year to create and save a filtered data for verification. Default is all clusters and all months of the year 2024 (also changeable to 2025).
    parser.add_argument("--clusters", nargs="+", help="Clusters to use (default: all)")
    parser.add_argument(
        "--month", type=int, nargs="+", choices=range(1, 13), help="One or more months as numbers 1–12 (default: all)",)
    parser.add_argument("--year", type=int, default=2024, help="Year to filter (default: 2024)")
    parser.add_argument("--date-column",choices=DATE_COLUMNS,default="BookingDate",help="Date used for the month and year filter (default: BookingDate)",)
    args = parser.parse_args()
    # Load the cleaned actuals data
    actuals = pd.read_csv(ACTUALS_FILE, sep=";", parse_dates=DATE_COLUMNS)
    print(f" The 10 highest lead times with their corresponding Tix from the actuals data are:\n{find_highest_lead_times(actuals, 10)}")

    available_clusters = actuals["FlightMarketCluster"].dropna().unique().tolist()
    clusters = list(dict.fromkeys(args.clusters or available_clusters))
    months = list(dict.fromkeys(args.month or range(1, 13)))

    CLUSTER_VIEW_DIR.mkdir(parents=True, exist_ok=True)
    files_saved = 0

    for cluster in clusters:
        for month in months:
            view = cluster_view(actuals, [cluster], [month], args.year, args.date_column)
            if view.empty:
                continue

            filename = (
                f"cluster_view_{cluster}_"
                f"{calendar.month_abbr[month]}_{args.year}_{args.date_column}.csv"
            )
            output_file = CLUSTER_VIEW_DIR / filename
            view.to_csv(output_file, sep=";", index=False)
            print(f"Saved {len(view)} rows to {output_file}")
            files_saved += 1

    if files_saved == 0:
        print("No rows matched the selected filters.")

    # # Load the cleaned booking forecast data
    # booking_forecast = pd.read_csv(BOOKING_FORECAST_FILE, sep=";", parse_dates=["Date"])

    # # Load the cleaned flight forecast data
    # flight_forecast = pd.read_csv(FLIGHT_FORECAST_FILE, sep=";", parse_dates=["Date"])

    # # Perform basic checks on the data
    # print("Actuals Data Info:")
    # print(actuals.info())
    # print("\nBooking Forecast Data Info:")
    # print(booking_forecast.info())
    # print("\nFlight Forecast Data Info:")
    # print(flight_forecast.info())
    
    # print rows from the raw booking and forecast files where the revenue is zero, to check if there are any such rows in the raw data.
    # raw_booking_forecast = pd.read_csv(RAW_BOOKING_FORECAST_FILE, sep=";", parse_dates=["Date"])
    # raw_flight_forecast = pd.read_csv(RAW_FLIGHT_FORECAST_FILE, sep=";", parse_dates=["Date"])
    # print("\nRows from the raw booking forecast data where Revenue is zero:")
    # print(raw_booking_forecast[raw_booking_forecast["Revenue"] == 0])
    # print("\nRows from the raw flight forecast data where Revenue is zero:")
    # print(raw_flight_forecast[raw_flight_forecast["Revenue"] == 0])


if __name__ == "__main__":
	main()