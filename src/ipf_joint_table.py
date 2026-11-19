"""Joint BookingDate × FlightDate forecast via IPF with a lead-time prior.

Per cluster:
  1. Prior   P(lead | booking month) from actuals: censoring-safe, smoothed,
             shrunk towards the cluster-wide lead-time profile.
  2. Seed    S[b, f] = W(b) · P(f - b | month(b)) on the band 0 <= f - b <= L.
  3. IPF     Tix: 2026 rows -> BookingFc, 2026 columns -> FlightFc.
             Carry-in rows (booked < 2026) and carry-out columns (flown > 2026)
             are free: they absorb the gap between the two forecasts.
  4. IPF     Revenue: seed = fitted Tix × historical AOV(month, lead bucket).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ACTUALS_FILE = DATA / "EWH_DummyActuals_sorted_mod_clean.csv"
BOOKING_FC_FILE = DATA / "EWH_DummyBookingPeriodFc_sorted_mod_clean.csv"
FLIGHT_FC_FILE = DATA / "EWH_DummyFlightPeriodFc_sorted_mod_clean.csv"
OUTPUT_FILE = DATA / "output" / "joint_forecast_2026.csv"

YEAR = 2026
MAX_LEAD = 365            # hard cap on lead time (days)
LEAD_QUANTILE = 0.99     # per-cluster cap = min(MAX_LEAD, Tix-weighted quantile)
SMOOTH_SIGMA = 2.0        # Gaussian smoothing along the lead axis (days)
SHRINK_K = 200.0          # pseudo-Tix pulling sparse months to the cluster profile
PRIOR_FLOOR = 1e-6        # keeps all allowed cells > 0 (IPF cannot revive zeros)
BUCKET_EDGES = [7, 30, 90, 180]   # lead buckets: 0-7, 8-30, 31-90, 91-180, 181+
TOL, MAX_ITER = 1e-10, 2000


# ---------------------------------------------------------------- loading
def load_actuals(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", parse_dates=["BookingDate", "FlightDate"],
                     usecols=["BookingDate", "FlightDate", "FlightMarketCluster", "Tix", "Revenue", "lead_time_days", "BookingMonth", "FlightMonth"])
    return df


def load_forecast(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", parse_dates=["Date"])
    df = df[df["Date"].dt.year == YEAR]
    return df.groupby(["FlightMarketCluster", "Date"], as_index=False)[["Tix", "Revenue"]].sum()


# ---------------------------------------------------------------- prior
@dataclass
class Prior:
    lead: np.ndarray      # (12, L+1)  P(lead | booking month)
    aov: np.ndarray       # (12, n_buckets)  AOV by booking month × lead bucket
    L: int


def lead_cap(act: pd.DataFrame) -> int:
    """Return the Tix-weighted 99th percentile lead, capped at MAX_LEAD."""
    s = act.groupby("lead_time_days")["Tix"].sum().sort_index()
    idx = np.searchsorted(s.cumsum().to_numpy(), LEAD_QUANTILE * s.sum())
    return int(min(MAX_LEAD, s.index[min(idx, len(s) - 1)]))


def build_prior(act: pd.DataFrame, data_end: pd.Timestamp) -> Prior:
    """Estimate monthly lead probabilities and bucket AOVs for one cluster.

    ``data_end`` is assumed to be the last fully observed flight date. The
    caller must verify this cutoff: the largest date in a file is not proof
    that all transactions through that date have been observed.
    """
    L = lead_cap(act)
    print(f"Lead cap = {L} days")
    # Right-censoring: keep bookings whose full lead window [b, b+L] is observed.
    hist = act[act["BookingDate"] + pd.Timedelta(days=L) <= data_end] # for the given actuals it might not make much sense as all the transactions in actuals anyways have flight date <= data_end.
    if hist.empty:
        print(f"Warning: no bookings with BookingDate + {L} <= {data_end.date()}; using all actuals")
        hist = act
    hist = hist[hist["lead_time_days"] <= L]
    # BookingMonth column is already present in the actuals data, so utilize that
    month = hist["BookingMonth"].to_numpy() - 1
    tix, rev = hist["Tix"].to_numpy(float), hist["Revenue"].to_numpy(float)

    # Lead-time profile: smooth -> shrink to cluster profile -> floor -> normalise.
    counts = np.zeros((12, L + 1))
    np.add.at(counts, (month, hist["lead_time_days"].to_numpy()), tix) # this code adds the ticket counts to the counts array based on the booking month and lead time days. It uses numpy's advanced indexing to efficiently accumulate the ticket counts for each combination of booking month and lead time days.
    counts = gaussian_filter1d(counts, SMOOTH_SIGMA, axis=1, mode="nearest") # smoothening using a bell-shaped kernel along the lead-time axis (axis=1). The mode "nearest" means that the values at the edges are replicated to avoid boundary effects during smoothing.
    cluster_profile = counts.sum(0) / counts.sum() # total tickets per lead time across all months, normalized to sum to 1. This represents the overall lead-time distribution for the cluster.
    # The following line is basically the probability (P(lead_time_days | BookingMonth)) of lead time given the booking month, shrunk towards the overall cluster profile. The shrinkage is controlled by the parameter SHRINK_K, which acts as a pseudo-count to pull sparse months towards the overall distribution.
    lead = (counts + SHRINK_K * cluster_profile) / (counts.sum(1, keepdims=True) + SHRINK_K) # The parameter SHRINK_K controls the strength of this shrinkage: a larger value means more influence from the cluster profile.
    # a second way without shrinking: lead = counts / counts.sum(1, keepdims=True)

    lead += PRIOR_FLOOR # so that no cells are zero as IPF cannot revive zeros.
    lead /= lead.sum(1, keepdims=True) # so that the lead is still a valid probability distribution after adding the floor.

    # AOV by booking month × lead bucket, shrunk to the bucket-level AOV.
    bucket = np.digitize(hist["lead_time_days"].to_numpy(), BUCKET_EDGES, right=True) # creating lead buckets based on the lead time days. The right=True parameter means that the bins are right-inclusive, so a lead time of 7 days would fall into the first bucket (0-7 days).
    nb = len(BUCKET_EDGES) + 1
    t_mb, r_mb = np.zeros((12, nb)), np.zeros((12, nb)) #t_mb is the total tickets per booking month and lead bucket, r_mb is the total revenue per booking month and lead bucket.
    np.add.at(t_mb, (month, bucket), tix)
    np.add.at(r_mb, (month, bucket), rev)
    overall = rev.sum() / tix.sum() # overall aov
    aov_bucket = np.divide(r_mb.sum(0), t_mb.sum(0), out=np.full(nb, overall), where=t_mb.sum(0) > 0) # this is the average revenue per ticket for each lead bucket across all booking months. If there are no tickets in a bucket, it defaults to the overall AOV.
    aov = (r_mb + SHRINK_K * aov_bucket) / (t_mb + SHRINK_K)
    return Prior(lead=lead, aov=aov, L=L)


# ---------------------------------------------------------------- grid & seed
@dataclass
class Grid:
    bookings: pd.DatetimeIndex
    flights: pd.DatetimeIndex
    lead: np.ndarray      # (nb, nf) lead time in days
    mask: np.ndarray      # allowed cells


def build_grid(L: int) -> Grid:
    """Build daily axes and the allowed lead-time band, including carry cells."""
    y0, y1 = pd.Timestamp(f"{YEAR}-01-01"), pd.Timestamp(f"{YEAR}-12-31")
    bookings = pd.date_range(y0 - pd.Timedelta(days=L), y1, freq="D")
    flights = pd.date_range(y0, y1 + pd.Timedelta(days=L), freq="D")
    lead = (flights.values[None, :] - bookings.values[:, None]) // np.timedelta64(1, "D")
    out_of_scope = (bookings < y0)[:, None] & (flights > y1)[None, :]   # in neither forecast
    mask = (lead >= 0) & (lead <= L) & ~out_of_scope
    return Grid(bookings, flights, lead, mask)


def build_seed(grid: Grid, prior: Prior, booking_tix: pd.Series) -> np.ndarray:
    """Distribute booking weights over allowed leads using the monthly prior.

    For carry-in rows use next year's booking forecast plus a small floor.
    Free margins have no target, so their seed levels affect the solution.
    """
    w = booking_tix.reindex(grid.bookings, fill_value=0.0).to_numpy(float, copy=True)
    carry_in = np.asarray(grid.bookings.year < YEAR)
    proxy = grid.bookings[carry_in] + pd.DateOffset(years=1)          # proxy for the carry-in dates to move the same date one year later
    w[carry_in] = (booking_tix.reindex(proxy, fill_value=0.0).to_numpy(float)
                   + 1e-3 * booking_tix.sum() / 365)                  # keep columns reachable
    month = grid.bookings.month.to_numpy() - 1
    p = prior.lead[month[:, None], np.clip(grid.lead, 0, prior.L)]
    return np.where(grid.mask, w[:, None] * p, 0.0) # seed_tickets = booking_weights * P(lead_time_days | booking_month) for the allowed cells in the grid.


def revenue_seed(tix: np.ndarray, grid: Grid, prior: Prior) -> np.ndarray:
    """Weight the fitted ticket table by historical month/lead-bucket AOV."""
    month = grid.bookings.month.to_numpy() - 1
    bucket = np.digitize(np.clip(grid.lead, 0, None), BUCKET_EDGES, right=True)
    return tix * prior.aov[month[:, None], bucket]


# ---------------------------------------------------------------- IPF
def _factors(current: np.ndarray, target: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """ Return the multiplicative factors to scale current to target, leaving free margins at 1.0."""
    f = np.ones_like(current)
    ok = fixed & (current > 0)
    f[ok] = target[ok] / current[ok]
    return f


def ipf(seed, row_t, col_t, row_fixed, col_fixed) -> tuple[np.ndarray, int, float]:
    """Scale fixed margins alternately, leaving carry-in/out margins free.
    Inputs:
        seed: initial table with weighted lead-time prior (booking month × lead time) for the allowed cells in the grid.
        row_t: target row sums (booking forecast)
        col_t: target column sums (flight forecast)
        row_fixed: this is a boolean array indicating which rows have fixed targets (i.e., which rows correspond to the booking forecast for the current year).
        col_fixed: similarly, this is a boolean array indicating which columns have fixed targets (i.e., which columns correspond to the flight forecast for the current year).
    Outputs:
        x: the fitted table after IPF, which should have row sums close to row_t and column sums close to col_t for the fixed rows and columns.
        it: the number of iterations taken to converge
        err: this is the maximum absolute error between the fixed-row and fixed-column sums. It gives a sense of how well the fitted table matches the booking forecast.
    Error is now the maximum between the fixed-row absolute error and the fixed-column absolute error, divided by the total fixed-row and fixed-column targets respectively. 
    This is a more balanced measure of how well the fitted table matches both the booking and flight forecasts.
    Reaching MAX_ITER does not raise an error.
    Positive support for each margin alone does not establish feasibility.
    """
    dead_r = row_fixed & (row_t > 0) & (seed.sum(1) == 0)
    dead_c = col_fixed & (col_t > 0) & (seed.sum(0) == 0)
    if dead_r.any() or dead_c.any():
        raise ValueError(f"Infeasible: {dead_r.sum()} rows / {dead_c.sum()} cols lack seed support")
    x = seed.copy()
    row_scale = max(row_t[row_fixed].sum(), 1.0)
    col_scale = max(col_t[col_fixed].sum(), 1.0)
    for it in range(1, MAX_ITER + 1):
        x *= _factors(x.sum(1), row_t, row_fixed)[:, None]      # adjust the booking rows to match the booking forecast
        x *= _factors(x.sum(0), col_t, col_fixed)[None, :]      # columns adjustment
        # Check both dimensions after the factor adjustment.
        row_error = np.abs(x.sum(axis=1) - row_t)[row_fixed].max() / row_scale
        col_error = np.abs(x.sum(axis=0) - col_t)[col_fixed].max() / col_scale
        err = max(row_error, col_error)

        if np.isfinite(err) and err < TOL:
            break
    return x, it, err


def fit_cluster(act, bfc, ffc, data_end) -> tuple[Grid, Prior, np.ndarray, np.ndarray, dict]:
    prior = build_prior(act, data_end) # to obtain P(lead | booking month) and AOV for each cluster from the actuals.
    grid = build_grid(prior.L) # to build the grid of booking dates and flight dates for the given year, including the allowed lead-time band and carry-in/out cells.
    rows_fixed = np.asarray(grid.bookings.year == YEAR)
    cols_fixed = np.asarray(grid.flights.year == YEAR)

    def target(fc, dates, m):
        return fc.set_index("Date")[m].reindex(dates, fill_value=0.0).to_numpy(float)

    tix, it_t, err_t = ipf(build_seed(grid, prior, bfc.set_index("Date")["Tix"]),
                           target(bfc, grid.bookings, "Tix"), target(ffc, grid.flights, "Tix"),
                           rows_fixed, cols_fixed)
    rev, it_r, err_r = ipf(revenue_seed(tix, grid, prior),
                           target(bfc, grid.bookings, "Revenue"), target(ffc, grid.flights, "Revenue"),
                           rows_fixed, cols_fixed)
    info = dict(L=prior.L, iter_tix=it_t, err_tix=err_t, iter_rev=it_r, err_rev=err_r)
    return grid, prior, tix, rev, info


# ---------------------------------------------------------------- output & checks
def output_paths(base: Path, clusters: list[str]) -> tuple[Path, Path]:
    """Append the fitted cluster name or cluster initials to both output paths."""
    words = [re.findall(r"[A-Za-z0-9]+", cluster) for cluster in sorted(clusters)]
    if any(not parts for parts in words):
        raise ValueError("Cluster names must contain letters or digits")
    if len(words) == 1:
        label = "_".join(words[0])
    else:
        label = "_".join("".join(word[0].upper() for word in parts) for parts in words)
    full = base.with_name(f"{base.stem}_{label}{base.suffix}")
    filtered = full.with_name(f"{full.stem}_filtered{full.suffix}")
    return full, filtered


def filter_to_forecasts(out: pd.DataFrame, bfc: pd.DataFrame, ffc: pd.DataFrame) -> pd.DataFrame:
    """Keep rows whose booking and flight dates occur in their cluster's forecasts."""
    cluster = out["FlightMarketCluster"]
    bookings = pd.MultiIndex.from_arrays([cluster, out["BookingDate"]])
    flights = pd.MultiIndex.from_arrays([cluster, out["FlightDate"]])
    booking_dates = pd.MultiIndex.from_frame(bfc[["FlightMarketCluster", "Date"]])
    flight_dates = pd.MultiIndex.from_frame(ffc[["FlightMarketCluster", "Date"]])
    return out[bookings.isin(booking_dates) & flights.isin(flight_dates)] # instead of a hard 'and' we can also try to use an 'or'
    # return out[bookings.isin(booking_dates) | flights.isin(flight_dates)]


def compare_booking_forecast(out: pd.DataFrame, bfc: pd.DataFrame) -> pd.DataFrame:
    """Compare filtered joint-table totals with booking forecasts across clusters."""
    aggregated = out.groupby("BookingDate")[["Tix", "Revenue"]].sum()
    forecast = bfc.groupby("Date")[["Tix", "Revenue"]].sum()
    # only compare based on the forecast booking dates
    comparison = aggregated.rename(columns=lambda name: f"{name}_aggregated").join(forecast.rename(columns=lambda name: f"{name}_forecast"), how="right").fillna(0).sort_index()
    comparison["Tix_error"] = comparison["Tix_aggregated"] - comparison["Tix_forecast"]
    # Report both errors with a rounding off to 6 decimal places
    comparison["Tix_error"] = comparison["Tix_error"].round(6)
    comparison["Revenue_error"] = comparison["Revenue_aggregated"] - comparison["Revenue_forecast"]
    comparison["Revenue_error"] = comparison["Revenue_error"].round(6)
    comparison.index.name = "BookingDate"
    return comparison.reset_index()[["BookingDate", "Tix_aggregated", "Tix_forecast",
                                     "Revenue_aggregated", "Revenue_forecast",
                                     "Tix_error", "Revenue_error"]]

# TODO: similarly compare the filtered joint-table totals with flight forecasts across clusters.
def compare_flight_forecast(out: pd.DataFrame, ffc: pd.DataFrame) -> pd.DataFrame:
    """Compare filtered joint-table totals with flight forecasts across clusters."""
    aggregated = out.groupby("FlightDate")[["Tix", "Revenue"]].sum()
    forecast = ffc.groupby("Date")[["Tix", "Revenue"]].sum()
    # only compare based on the forecast flight dates
    comparison = aggregated.rename(columns=lambda name: f"{name}_aggregated").join(forecast.rename(columns=lambda name: f"{name}_forecast"), how="right").fillna(0).sort_index()
    comparison["Tix_error"] = comparison["Tix_aggregated"] - comparison["Tix_forecast"]
    # Report both errors with a rounding off to 6 decimal places
    comparison["Tix_error"] = comparison["Tix_error"].round(6)
    comparison["Revenue_error"] = comparison["Revenue_aggregated"] - comparison["Revenue_forecast"]
    comparison["Revenue_error"] = comparison["Revenue_error"].round(6)
    comparison.index.name = "FlightDate"
    return comparison.reset_index()[["FlightDate", "Tix_aggregated", "Tix_forecast",
                                     "Revenue_aggregated", "Revenue_forecast",
                                     "Tix_error", "Revenue_error"]]


def save_booking_comparison(filtered_path: Path, clusters: list[str] | None = None) -> Path:
    """Compare a saved filtered table with the clean forecast for its clusters."""
    out = pd.read_csv(filtered_path, sep=";", parse_dates=["BookingDate", "FlightDate"])
    if clusters is None:
        clusters = out["FlightMarketCluster"].dropna().unique().tolist()
    if not clusters:
        raise ValueError("Cannot determine clusters from an empty filtered table; specify --clusters")
    out = out[out["FlightMarketCluster"].isin(clusters)]
    bfc = load_forecast(BOOKING_FC_FILE)
    ffc = load_forecast(FLIGHT_FC_FILE)
    bfc = bfc[bfc["FlightMarketCluster"].isin(clusters)]
    ffc = ffc[ffc["FlightMarketCluster"].isin(clusters)]
    comparison = compare_booking_forecast(out, bfc)
    comparison_ffc = compare_flight_forecast(out, ffc)
    # print the ten highest absolute errors in the comparison for both Tix and Revenue
    print("Top 10 Tix errors compared to booking forecast:")
    print(comparison.reindex(comparison["Tix_error"].abs().sort_values(ascending=False).head(10).index))
    print("Top 10 Tix errors compared to flight forecast:")
    print(comparison_ffc.reindex(comparison_ffc["Tix_error"].abs().sort_values(ascending=False).head(10).index))
    # print(comparison.reindex(comparison["Revenue_error"].abs().sort_values(ascending=False).head(10).index))
    path = filtered_path.parent / "comparison" / f"{filtered_path.stem}_booking_comparison.csv"
    path_ffc = filtered_path.parent / "comparison" / f"{filtered_path.stem}_flight_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(path, sep=";", index=False, date_format="%Y-%m-%d")
    comparison_ffc.to_csv(path_ffc, sep=";", index=False, date_format="%Y-%m-%d")
    print(f"Saved {len(comparison):,} booking comparison rows -> {path}")
    print(f"Saved {len(comparison_ffc):,} flight comparison rows -> {path_ffc}")
    return path


def to_long(grid: Grid, tix: np.ndarray, rev: np.ndarray, cluster: str) -> pd.DataFrame:
    i, j = np.nonzero(tix > 1e-12)
    return pd.DataFrame({"BookingDate": grid.bookings[i], "FlightDate": grid.flights[j],
                         "FlightMarketCluster": cluster, "Tix": tix[i, j], "Revenue": rev[i, j]})


def margin_error(out: pd.DataFrame, fc: pd.DataFrame, date_col: str) -> dict[str, float]:
    """Return maximum absolute ticket and revenue errors by date."""
    got = out[out[date_col].dt.year == YEAR].groupby(date_col)[["Tix", "Revenue"]].sum()
    exp = fc.set_index("Date")[["Tix", "Revenue"]]
    # Include dates that occur in either the produced output or forecast.
    idx = got.index.union(exp.index)
    errors = (got.reindex(idx, fill_value=0) - exp.reindex(idx, fill_value=0)).abs()
    return {"Tix": float(errors["Tix"].to_numpy().max()) if not errors.empty else 0.0, 
            "Revenue": float(errors["Revenue"].to_numpy().max()) if not errors.empty else 0.0}



def summarize(out, act, bfc, ffc, data_end, prior, info, cluster) -> dict:
    assert (out["FlightDate"] >= out["BookingDate"]).all() and (out[["Tix", "Revenue"]] >= 0).all().all()
    hist = act[act["BookingDate"] + pd.Timedelta(days=prior.L) <= data_end]
    hist_out = (hist.loc[hist["FlightDate"].dt.year > hist["BookingDate"].dt.year, "Tix"].sum()
                / hist["Tix"].sum()) if not hist.empty else np.nan
    fc_aov = bfc["Revenue"].sum() / bfc["Tix"].sum()
    cells = out[out["Tix"] >= 0.1]
    cells_aov = cells["Revenue"] / cells["Tix"]
    ratio = cells_aov / fc_aov
    booking_errors = margin_error(out, bfc, "BookingDate")
    flight_errors = margin_error(out, ffc, "FlightDate")
    return dict(cluster=cluster, **info,
                max_err_booking_tix=booking_errors["Tix"], max_err_booking_rev=booking_errors["Revenue"],
                max_err_flight_tix=flight_errors["Tix"], max_err_flight_rev=flight_errors["Revenue"],
                carry_in_share=out.loc[out["BookingDate"].dt.year < YEAR, "Tix"].sum() / ffc["Tix"].sum(),
                carry_out_share=out.loc[out["FlightDate"].dt.year > YEAR, "Tix"].sum() / bfc["Tix"].sum(),
                hist_carry_out_share=hist_out, # since actuals data spans multi-year, it can also tell us how many tickets are carried out to the next year from the current year.
                aov_flag_share=cells.loc[(ratio < 1 / 3) | (ratio > 3), "Tix"].sum() / out["Tix"].sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", nargs="+", help="Clusters to fit (default: all)")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE,
                        help="Base output path; fitted cluster names are appended")
    parser.add_argument("--post-process-only", type=Path, metavar="FILTERED_CSV",
                        help="Only compare an existing filtered joint table; do not fit")
    args = parser.parse_args()

    if args.post_process_only:
        save_booking_comparison(args.post_process_only, args.clusters)
        return

    act = load_actuals(ACTUALS_FILE)
    bfc, ffc = load_forecast(BOOKING_FC_FILE), load_forecast(FLIGHT_FC_FILE)
    data_end = act["FlightDate"].max()
    clusters = args.clusters or sorted(set(bfc["FlightMarketCluster"]) | set(ffc["FlightMarketCluster"]))

    results, summaries = [], []
    for cluster in clusters:
        a, b, f = (d[d["FlightMarketCluster"] == cluster] for d in (act, bfc, ffc))
    
        print(f"\nFitting cluster {cluster} ({len(a):,} actuals, {len(b):,} booking fc, {len(f):,} flight fc rows)")
        try:
            grid, prior, tix, rev, info = fit_cluster(a, b, f, data_end)
        except ValueError as exc:
            print(f"Skip {cluster}: {exc}")
            continue
        out = to_long(grid, tix, rev, cluster)
        results.append(out)
        summaries.append(summarize(out, a, b, f, data_end, prior, info, cluster))

    result = pd.concat(results).sort_values(["FlightMarketCluster", "BookingDate", "FlightDate"])
    filtered = filter_to_forecasts(result, bfc, ffc)
    full_path, filtered_path = output_paths(args.output, [summary["cluster"] for summary in summaries])
    full_path.parent.mkdir(parents=True, exist_ok=True)
    for frame, path in ((result, full_path), (filtered, filtered_path)):
        frame.to_csv(path, sep=";", index=False, date_format="%Y-%m-%d", float_format="%.6f")
        print(f"Saved {len(frame):,} rows -> {path}")
    print(pd.DataFrame(summaries).round(6).to_string(index=False))
    save_booking_comparison(filtered_path, [summary["cluster"] for summary in summaries])


if __name__ == "__main__":
    main()