"# booking_to_flight_corr" 
## Usage:
- requirements: `pip install -r requirements.txt`
- run `python src/eda.py` to add lead time and AOV to the actuals dataset

## Historical EDA plots

Run `python src/plots.py` from the project root to update the month columns in
the clean actuals CSV and save plots for all clusters in `plots/`. Use
`python src/plots.py --clusters "City" "Greece" --size-variable Revenue` to
select clusters and size bubbles by revenue; the default size is `Tix`.

Month plots combine booking years. Same-year flights are grouped by their flight
month; all flights in a later year are combined in the separate `Later year` row
for each booking month. This row does not distinguish individual later years or
flight months.
Lead-time bars show total revenue, with 180 days in the 91–180 bucket and 181+
in the last bucket. The heatmap uses the unweighted median per input row across
all years; grey cells indicate missing combinations.

## Booking-to-flight allocation matrices

Run `python src/bf_allocation_matrix.py` to process all clusters in the clean
actuals. Use `--clusters "City" "Greece"` to select clusters and
`--metric Revenue` to save revenue instead of the default `Tix` matrix.
Each semicolon-delimited CSV in `data/matrices/` has sorted booking dates as
rows, sorted observed flight dates as columns, and zeros for unobserved pairs.
Duplicate date pairs are summed. The required filename does not include the
metric, so rerunning for another metric overwrites the previous matrix.

The CSV is a conventional dense matrix, but it is streamed one row at a time;
the intermediate observed pairs remain sparse. The heatmap in `plots/` always
shows summed `Tix`. Its axes cover calendar dates, including dates without
observations. To bound memory and make large date ranges legible, each axis is
limited to 512 equal-width calendar-day bins; the title states the bin widths.
The CSV retains the original daily resolution regardless of the heatmap bins.

## Main IPF model

Run `python src/ipf_joint_table.py` to process all clusters. Use 
`--clusters "City" "Greece"` to select clusters and `--metric Tix` to save ticket counts 
instead of the default `Revenue` matrix. The output will be a semicolon-delimited CSV
in `data/joint_tables/`.
