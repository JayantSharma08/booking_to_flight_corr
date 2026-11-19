## EDA Tasks:
# [X] 1. Load, clean and sanity-check all three files (eda.py)
- Load all 3 CSVs with `separator=';'`; parse `BookingDate`, `FlightDate` and `Date` as datetime. The dates are in the format YYYY-MM-DD.
- Sort actuals (EWH_DummyActuals.csv) by `BookingDate`, `FlightDate`, `FlightMarketCluster`; sort forecasts (EWH_DummyBookingPeriodFc.csv and EWH_DummyFlightPeriodFc.csv) by `Date`, `FlightMarketCluster`
- Check data types, nulls, duplicates, and Tix/Revenue values that are zero or negative
- Compare the cluster lists across the 3 files to check for missing or extra clusters
- In actuals (EWH_DummyActuals.csv), count rows where `FlightDate < BookingDate` (these are invalid)

# [X] 2. Add basic features to actuals (eda.py)
- `lead_time_days = (FlightDate - BookingDate).days` to the actuals dataset
- `AOV = Revenue / Tix` to all three datasets
<!-- - `cross_year` flag: booking year ≠ flight year -->

# [X] 3. First EDA on lead time distributions (plots.py)
- Plot the relationship between booking month and flight month per cluster. For each cluster, simply plot the booking month on the x-axis and the flight month on the y-axis, with the size of the points representing Tix or Revenue. This will help visualize the seasonality of bookings and flights for each cluster.
- Analyse the lead-time distribution separately for each FlightMarketCluster. For example, a bucket 0-7 days can mean last-minute bookings, 8-30 days short term bookings, 31-90 days medium term bookings, 91-180 days long term bookings, and 180+ days very long term bookings. This can be visualised as a histogram or boxplot.
- Cluster-specific historical lead-time distributions, potentially conditioned on booking month. Plot median lead time by booking month × cluster, shown as a heatmap. So plot booking month on x-axis, cluster on y-axis, and median lead time as the color of the heatmap. This should give a historical allocation pattern.
<!-- - AOV vs lead-time bucket (0–7, 8–30, 31–90, 91–180, 180+ days) per cluster or Revenue vs lead-time bucket per cluster
- Forecast files: total Tix, total Revenue and AOV per cluster, Booking Fc vs Flight Fc side by side, plus the difference between them -->

# [X] 4. Build a booking-to-flight allocation matrix (bf_allocation_matrix.py)
- For each FlightMarketCluster (user defined), construct a matrix using only the actuals data, where:
    - Rows = Booking Dates.
    - Columns = Flight Dates.
    - Cells = Tix or Revenue
    Complete this step-by-step, iterating through each booking date, finding all the flight dates that have bookings and filling in either Tix or Revenue (user defined, Tix default) values from the actuals data. The matrix can be saved in a suitable format (like CSV) inside the `data/matrices` folder. The matrix would be sparse, with many cells being zero, as not all booking dates will have all flight dates. The name of the matrix file should be `actuals_b_to_f_matrix_<metric>_<cluster>.csv` where `<metric>` is Tix or Revenue and `<cluster>` is the FlightMarketCluster name.
- Visualize the allocation matrix as a heatmap for each cluster, where the x-axis is FlightDate, y-axis is BookingDate, and the color intensity represents Tix or Revenue.

# Now the core method- Iterative Proportional Fitting (IPF) to construct the joint bookings-flights table:
**Available Data** = The sorted, modified and cleaned actual data (`./data/EWH_DummyActuals_sorted_mod_clean.csv`), allocation matrix created per cluster using the actuals dataset (implemented in bf_allocation_matrix.py) and the sorted and cleaned forecasts (booking and flight forecasts).
**Desired Output** = A joint bookings-flights table per cluster, where the sum of Tix and Revenue across booking dates matches the booking forecast, and the sum across flight dates matches the flight forecast. The output table should be saved in a suitable format (like CSV) inside the `data/joint_tables` folder. The name of the output file should be `joint_b_to_f_table_<metric>_<cluster>.csv` where `<metric>` is Tix or Revenue and `<cluster>` is the FlightMarketCluster name.
- Seed to use - P(lead time | booking month, cluster) i.e. Probability of lead time given booking month and cluster. This can be derived from the actuals data, and can be used to initialize the IPF algorithm.
- Handling of Free Cells:
                    Flights 2026 (fixed = FlightFc)   Flights 2027+ (free)
    Bookings 2025 (free)    carry-in cells              –  (out of scope)
    Bookings 2026 (fixed)   core cells                  carry-out cells
- 2026 rows and columns are scaled to the forecasts exactly.2025 rows and 2027 columns are never scaled to a target. IPF only adjusts them through the other dimension, so they take up the gap between the two forecasts automatically. Seed for the 2025 rows: the booking forecast for the same day one year later (a simple seasonal proxy). Revenue: run a second IPF with seed = fitted Tix × historical AOV (booking month × lead bucket). Each cell's AOV then stays close to the historical AOV times bounded scaling factors, so it remains plausible.

# [X] 5. Lead-time prior (ipf_joint_forecast.py → build_prior)
- Per cluster: lead cap L = min(365, Tix-weighted P99 lead time)
- Keep only bookings with BookingDate + L <= max FlightDate in actuals (right-censoring)
- Build matrix counts[booking_month(12), lead(0..L)] of Tix
- Smooth along lead axis (Gaussian, sigma=2); shrink each month to the cluster profile:
  p_m = (counts_m + K·p_cluster) / (n_m + K), K=200
- Add floor 1e-6, row-normalise → P(lead | booking month, cluster)
- Also build AOV[booking_month, lead_bucket] (buckets 0-7, 8-30, 31-90, 91-180, 181+), shrunk to bucket AOV

# [X] 6. Grid and seed (ipf_joint_forecast.py → build_grid, build_seed)
- Booking axis: 2026-01-01 minus L … 2026-12-31 (daily); flight axis: 2026-01-01 … 2026-12-31 plus L
- Allowed cells: 0 <= FlightDate - BookingDate <= L; exclude booking <2026 AND flight >2026
- Seed (Tix units): S[b,f] = W(b) · P(f-b | month(b))
  - W(b) = BookingFc Tix for 2026 rows (missing days = 0)
  - W(b) = BookingFc Tix of same day +1 year for 2025 rows, plus a small floor

# [X] 7. IPF with partial margins (ipf_joint_forecast.py → fit_cluster)
- Rows fixed only for 2026 bookings (target = BookingFc); columns fixed only for 2026 flights (target = FlightFc)
- Carry-in rows / carry-out columns are free (scaled only through the other dimension)
- Alternate row and column scaling until max relative row error < 1e-10 (max 2000 iterations)
- Before iterating, raise an error if any fixed row/column has target > 0 but seed sum = 0
- Tix first; then Revenue IPF with seed = fitted Tix × AOV[month, lead bucket], revenue margins

# [X] 8. Output and checks (ipf_joint_forecast.py → to_long, output_paths)
- Long table: BookingDate; FlightDate; FlightMarketCluster; Tix; Revenue (sep=';', ISO dates)
- Assert: 2026 booking and flight sums match forecasts (Tix & Revenue) with acceptable tolerance, FlightDate >= BookingDate, no negatives
- Per-cluster summary: iterations, max margin error, carry-in share, carry-out share vs historical
  cross-year share, share of Tix in cells with AOV outside [1/3, 3] × forecast AOV

  # [ ] 9. Correlation Calculation (correlations.py)
- Calculate the correlation between Booking Month and Flight Month for each FlightMarketCluster based on total Tix and Revenue
- Compare the correlation values for the actuals data table and the output table generated by the IPF model (tables stored in csv files with the '_filtered' suffix)
- For each FlightMarketCluster, plot side-by-side the correlation heatmaps (either the Tix or Revenue based) from the actuals data and the IPF generated table. 
  In these heatmaps the x-axis should represent the booking month, the y-axis should represent the flight month, and the color intensity should represent the correlation value.
- Correlations on three levels: (1) between booking and flight ordinal month numbers, (2) between booking and flight month indicators, (3) between booking and flight month  but considering the carry-in and carry-out too. All of these are weighted pearson correlations by Tix or Revenue. The first is a single number per cluster; the second is a matrix of correlations for each booking month × flight month pair in the same year; the third is a matrix of correlations for each booking month × flight month pair but including the across years circulation. The first and second are already implemented in the script `correlations.py` and the third is yet to be implemented.
- Lastly, implement the correlation with the unfiltered csv output and compare it with the actuals csv