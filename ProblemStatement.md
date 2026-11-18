# Case Study Data Science:
A dummy dataset containing transaction data is provided. Each transaction has two time dimensions:
- Booking Period: when the booking was made.
- Flight Period: when the flight takes place at a future date.

Based on this data, two forecast files have been created, one for the Booking Period (`data/EWH_DummyBookingPeriodFc.csv`) and one for the Flight Period (`data/EWH_DummyFlightPeriodFc.csv`), each showing the expected number of transactions.

**Your task:** Using these two forecast files, build a model to forecast both Booking Date and Flight Date dimensions in the same dataset, enabling a bidirectional relationship between bookings and flights for both Tix and Revenue.

Target table with the forecast should look like:
```
BookingDate; FlightDate; FlightMarketCluster; Tix; Revenue
yyyy-MM-dd; yyyy-MM-dd; string; double; double
```

**Expected Outcomes**
- A valid correlation between Booking Period and Flight Period for each FlightMarketCluster.
- When aggregating Tix or Revenue by Booking Period, the totals must exactly match the values in EWH_DummyBookingPeriodFc.csv.
- When aggregating Tix or Revenue by Flight Period, the totals must exactly match the values in EWH_DummyFlightPeriodFc.csv.
- Assume that some bookings for the forecast year were made in previous years, and that some bookings made during the forecast year will be flown in subsequent years.
- The Average Order Value (AOV) must remain at the same level as in the original forecast files.

**Questions to be answered:**
1. How would you evaluate the quality and plausibility of the final table?
2. How would you calculate, validate, and demonstrate that the booking-to-flight correlations (linked across) are correct?
3. How would you present the results in a simple and business-friendly way to stakeholders?
4. Demonstrate deployment capability and accessibility from BI team.
5. Demonstrate error metrics for forecast and highlight limitations.

**Vocabulary:**
- Tix: Tickets sold; equivalent to one traveler including both flight and accommodation.
- Revenue: Revenue generated from ticket sales (€).
- AOV (Average Order Value): Revenue divided by Tix.
- FlightMarketCluster: A grouping of transactions based on the destination market of the journey.