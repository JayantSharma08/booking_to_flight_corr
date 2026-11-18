## General Notes:
- Dates are in the format YYYY-MM-DD.
- Instead of commas, the separator used is a semicolon (;).
- Booking Lead Time (BLT) is the number of days between the booking date and the flight date. It can be calculated as:
```
BLT = FlightDate - BookingDate
```

## Potential Solution TODOs:
1. In the file `data/EWH_DummyActuals.csv`, sort all the rows by BookingDate and FlightDate and store this sorted file as `data/EWH_DummyActualsSorted.csv`. - Done
2. In the file `data/EWH_DummyBookingPeriodFc.csv`, sort all the rows by BookingDate and in the file `data/EWH_DummyFlightPeriodFc.csv`, sort all the rows by FlightDate and store both files with the suffix `Sorted` as `data/EWH_DummyBookingPeriodFcSorted.csv` and `data/EWH_DummyFlightPeriodFcSorted.csv`. - Done
3. Add a feature called lead_time_days to calculate the number of days between the booking date and the flight date. - Done
4. I can also split the actuals data and reserve part of it for validation or testing purposes. This will help in evaluating the model's performance on real but unseen data.
5. Should also check the total bookings and total flights timeseries across all clusters and compare them with the forecasts to check if the seasonalities are consistent and if the forecasts are reasonable.
6. Check if the Tix and Revenue per cluster per month in the forecast files match to each other. If not then the AOV will not be same across the two forecasts. 
7. Flight-year total − booking-year total = incoming bookings − outgoing bookings. Check this with both - the forecasts data as well as the atuals datas.

## Some findings from the EDA:
- No null values in any columns in any of the three files and no duplicate rows.
- The Tix column in the Booking Period Forecast file is of float type, while in the Flight Period Forecast file it is of int type. This inconsistency may lead to issues when merging or comparing the two datasets.
- One actuals row where the flight precedes the booking.
- Zero revenue in 16 actuals, 3 booking forecasts, and 2 flight forecasts.
- The Nordics cluster appears in actuals but not in either forecast.
- Rows from the raw booking forecast data where Revenue is zero:
           Date FlightMarketCluster  Tix  Revenue
516  2026-02-11              Greece  2.0        0
1120 2026-04-02            Mallorca  2.0        0
1332 2026-02-04   Iberian Peninsula  2.0        0

Rows from the raw flight forecast data where Revenue is zero:
           Date FlightMarketCluster  Tix  Revenue
811  2026-05-29            Canaries    2        0
1914 2026-11-23             Balkans    1        0