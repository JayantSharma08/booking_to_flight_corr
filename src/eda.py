"""Load and sanity-check the actuals and the two forecast files."""

from itertools import combinations
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
METRICS = ("Tix", "Revenue")
CLUSTER = "FlightMarketCluster"


def load_and_sort_data(
	filename: str, date_columns: list[str], sort_columns: list[str]
) -> pd.DataFrame:
	"""Normalize fields, report conversion failures and return sorted data.

	Rows are retained, including duplicates and invalid values, so the checks
	describe the input rather than silently discarding potentially valid data.
	"""
	data = pd.read_csv(DATA_DIR / filename, sep=";", dtype="string")
	required_columns = set(date_columns) | {CLUSTER, *METRICS}
	missing_columns = required_columns - set(data.columns)
	if missing_columns:
		raise ValueError(f"{filename}: missing columns: {sorted(missing_columns)}")

	print(f"\n=== {filename} ===")
	for column in [*date_columns, CLUSTER, *METRICS]:
		data[column] = data[column].str.strip().replace("", pd.NA)

	# for column in date_columns:
	# 	original = data[column]
	# 	converted = pd.to_datetime(original, format="%Y-%m-%d", errors="coerce")
	# 	print(f"Invalid {column} values: {(original.notna() & converted.isna()).sum()}")
	# 	data[column] = converted

	# for column in METRICS:
	# 	original = data[column]
	# 	converted = pd.to_numeric(original, errors="coerce")
	# 	print(f"Non-numeric {column} values: {(original.notna() & converted.isna()).sum()}")
	# 	data[column] = converted

	# sort the data by the specified columns once and save it to new csv file and return the sorted data
	sorted_data = data.sort_values(sort_columns, na_position="last").reset_index(drop=True)
	# sorted_data.to_csv(sorted_filename, sep=";", index=False)
	# print(f"Sorted data saved to {sorted_filename}")

	return sorted_data


def report_data_quality(data: pd.DataFrame) -> None:
	"""Print data types, missing values, duplicates and non-positive metrics."""
	print(f"Rows: {len(data):,}")
	# print("Data types:\n", data.dtypes.to_string(), sep="")
	# print("Null values by column:\n", data.isna().sum().to_string(), sep="") #no nulls found
	# print(f"Duplicate full rows (excluding first occurrence): {data.duplicated().sum():,}")
	for column in METRICS:
		print(f"{column} zero: {data[column].eq(0).sum():,}")
		print(f"{column} negative: {data[column].lt(0).sum():,}")


def compare_clusters(datasets: dict[str, pd.DataFrame]) -> None:
	"""Compare non-null cluster names in every pair of datasets."""
	clusters = {
		name: set(data[CLUSTER].dropna()) for name, data in datasets.items()
	}
	print("\n=== Cluster comparison ===")
	for name, values in clusters.items():
		print(f"{name} clusters: {sorted(values)}")
	# for left, right in combinations(clusters, 2):
	# 	print(f"In {left}, not {right}: {sorted(clusters[left] - clusters[right])}")
	# 	print(f"In {right}, not {left}: {sorted(clusters[right] - clusters[left])}")

# Following function adds aov column and returns the dataframe
def add_aov(data: pd.DataFrame) -> pd.DataFrame:
	"""Add AOV = Revenue / Tix to the dataset."""

	data["AOV"] = data["Revenue"].astype(float) / data["Tix"].astype(float)
	# print rows where AOV is zero
	# zero_aov_rows_list = []
	# zero_aov_rows = data.loc[data["AOV"] == 0]
	# zero_aov_rows_list.append(zero_aov_rows)
	# print(f"Rows with zero AOV: {zero_aov_rows_list}")
	return data

# Using the actuals data, plot booking date on the x-axis, lead time on the y-axis, and color by cluster.
def plot_lead_time(data: pd.DataFrame, cluster_list: list[str]) -> None:
	"""Plot lead time vs booking date with appropriate units and coloured dots by market cluster with proper legend. Then save the plot to a file."""

    # plot lead time vs booking date with appropriate units and coloured dots by market cluster
	plt.figure(figsize=(12, 6))
	# Only include clusters that are in the cluster_list
	sns.scatterplot(data=data[data["FlightMarketCluster"].isin(cluster_list)], x="BookingDate", y="lead_time_days", hue="FlightMarketCluster", alpha=0.5)
	plt.title("Lead time vs Booking date by Market Cluster")
	plt.xlabel("Booking Date")
	plt.ylabel("Lead time (days)")
	# Ensure proper legend with colours for each market cluster
	plt.legend(title="Market Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
	plt.tight_layout()
	# plt.show()
	# save the plot to a file with suffix constructed from the initial letter of each cluster in the cluster_list in caps
	suffix = "_".join([cluster[0].upper() for cluster in cluster_list])
	plot_filename = PLOT_DIR / f"lead_time_vs_booking_date_{suffix}.png"
	plt.savefig(plot_filename, dpi=300)
	print(f"Plot saved to {plot_filename}")


def main() -> None:
	datasets = {}
	files = [
		(
			"Actuals",
			"EWH_DummyActuals.csv",
			["BookingDate", "FlightDate"],
			["BookingDate", "FlightDate", CLUSTER],
		),
		(
			"Booking forecast",
			"EWH_DummyBookingPeriodFc.csv",
			["Date"],
			["Date", CLUSTER],
		),
		(
			"Flight forecast",
			"EWH_DummyFlightPeriodFc.csv",
			["Date"],
			["Date", CLUSTER],
		),
	]
	for name, filename, date_columns, sort_columns in files:
		data = load_and_sort_data(filename, date_columns, sort_columns)
		# report_data_quality(data)
		if name == "Actuals":
			invalid_dates = data["FlightDate"] < data["BookingDate"]
			print(f"FlightDate before BookingDate: {invalid_dates.sum():,}")
			# remove the row where FlightDate < BookingDate
			# data = data[~invalid_dates]
			# add lead_time_days = (FlightDate - BookingDate).days to the actuals dataset and print the row where lead_time_days is negative
			# Need to convert the dates into suitable format to calculate the difference in days. Assuming the dates are in string format, we can convert them to datetime first.
			data["BookingDate"] = pd.to_datetime(data["BookingDate"], format="%Y-%m-%d", errors="coerce")
			data["FlightDate"] = pd.to_datetime(data["FlightDate"], format="%Y-%m-%d", errors="coerce")
			data["lead_time_days"] = (data["FlightDate"] - data["BookingDate"]).dt.days
			row_with_negative_lead_time = data[data["lead_time_days"] < 0]
			print(f"Row with negative lead_time_days:\n{row_with_negative_lead_time}")
			# TODO: Clean the actuals data by removing the Nordics cluster and removing the only row with negative lead_time_days
			data = data[data["FlightMarketCluster"] != "Nordics"]
			data = data[data["lead_time_days"] >= 0]
		    # plot the lead time vs booking dates for each cluster and save the plot to a file
			plot_lead_time(data, cluster_list=["Balearic Islands", "City"])
			
			
        # call add_aov function to add aov column to the dataset
		data = add_aov(data)
		# Clean the data by removing rows where Revenue is zero
		data = data[data["Revenue"].astype(float) > 0]
		# save the sorted and modified data to a new csv file with the same name as the original file but with "_sorted_modified" appended to the name
		# run the following commands only when the _sorted_modified.csv file does not exist
		if not (DATA_DIR / f"{Path(filename).stem}_sorted_mod_clean.csv").exists():
			sorted_mod_clean_filename = DATA_DIR / f"{Path(filename).stem}_sorted_mod_clean.csv"
			data.to_csv(sorted_mod_clean_filename, sep=";", index=False)
			print(f"Sorted and modified data saved to {sorted_mod_clean_filename}")
		
	# compare_clusters(datasets)


if __name__ == "__main__":
	main()
