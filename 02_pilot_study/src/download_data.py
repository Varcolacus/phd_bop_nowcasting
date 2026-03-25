"""
Data Download & Preparation for BoP Nowcasting Pilot Study
===========================================================

Data sources (tried in order):
  1. ECB Statistical Data Warehouse (public, no auth required)
     - BP6: Balance of Payments (BPM6) for France
     - EXR: Exchange Rates
     - MNA: National Accounts (GDP)
     - ICP: Harmonised Index of Consumer Prices
     - FM:  Financial Markets (Euribor)
  2. Eurostat SDMX REST API (public, partial coverage)
  3. Synthetic dataset (fallback for development/testing)

Author: PhD Pilot Study
Date: March 2026
"""

import os
import json
import logging
import warnings
import urllib3
import requests
import pandas as pd
import numpy as np
from io import StringIO
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Some institutional proxies / corporate firewalls replace upstream TLS
# certificates, causing SSL verification failures against the ECB SDW API.
# Override via environment variable NOWCAST_VERIFY_SSL=false if needed.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
VERIFY_SSL = os.getenv('NOWCAST_VERIFY_SSL', 'true').lower() == 'true'

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ECB Statistical Data Warehouse
ECB_BASE = "https://data-api.ecb.europa.eu/service/data"

# Eurostat (fallback)
EUROSTAT_BASE = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data"


# ===================================================================
# PART 1: ECB STATISTICAL DATA WAREHOUSE (primary source)
# ===================================================================

# Series definitions: name -> (dataset, series_key, frequency, description)
ECB_SERIES = {
    # Balance of Payments - France quarterly (BPM6 methodology)
    "bop_ca":       ("BP6", "Q.N.FR.W1.S1.S1.T.B.CA._Z._Z._Z.EUR._T._X.N",
                     "Q", "Current Account balance"),
    "bop_goods":    ("BP6", "Q.N.FR.W1.S1.S1.T.B.G._Z._Z._Z.EUR._T._X.N",
                     "Q", "Goods balance"),
    "bop_services": ("BP6", "Q.N.FR.W1.S1.S1.T.B.S._Z._Z._Z.EUR._T._X.N",
                     "Q", "Services balance"),

    # Exchange rates - monthly (will be aggregated to quarterly)
    "eurusd":       ("EXR", "M.USD.EUR.SP00.A",
                     "M", "EUR/USD exchange rate"),

    # France GDP - quarterly
    "gdp":          ("MNA", "Q.Y.FR.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.V.N",
                     "Q", "France GDP (EUR, chain-linked volumes)"),

    # France HICP - monthly (will be aggregated to quarterly)
    "hicp":         ("ICP", "M.FR.N.000000.4.ANR",
                     "M", "France HICP (annual rate of change)"),

    # Euribor 3-month - monthly (will be aggregated to quarterly)
    "interest_rate": ("FM", "M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA",
                      "M", "Euribor 3-month rate"),
}


def ecb_download_series(dataset, series_key, start_period="1999"):
    """Download a single series from ECB SDW as CSV."""
    url = f"{ECB_BASE}/{dataset}/{series_key}"
    params = {"format": "csvdata", "startPeriod": start_period}
    try:
        resp = requests.get(url, params=params, timeout=30, verify=VERIFY_SSL)
        resp.raise_for_status()
        return resp.text
    except requests.exceptions.RequestException as e:
        logger.warning("ECB download failed for %s/%s: %s", dataset, series_key, e)
        return None


def ecb_parse_csv(csv_text, freq):
    """Parse ECB CSV response into a clean DataFrame with date and value."""
    if csv_text is None:
        return pd.DataFrame()

    df = pd.read_csv(StringIO(csv_text))

    if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
        return pd.DataFrame()

    result = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
    result.columns = ["period", "value"]
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    result = result.dropna(subset=["value"])

    # Convert period to datetime
    if freq == "Q":
        # Format: 2020-Q1
        result["date"] = pd.PeriodIndex(result["period"], freq="Q").to_timestamp()
    elif freq == "M":
        # Format: 2020-01
        result["date"] = pd.to_datetime(result["period"], format="%Y-%m")
    else:
        result["date"] = pd.to_datetime(result["period"])

    return result[["date", "value"]].sort_values("date").reset_index(drop=True)


def ecb_to_quarterly(df, freq, agg="mean"):
    """Aggregate monthly series to quarterly frequency."""
    if df.empty:
        return df
    if freq == "Q":
        return df  # Already quarterly
    if freq == "M":
        df = df.set_index("date")
        df_q = df.resample("QS").agg(agg)
        df_q = df_q.dropna().reset_index()
        return df_q
    return df


def try_ecb_sdw():
    """
    Download all needed series from ECB Statistical Data Warehouse.
    Returns a dict of quarterly DataFrames {name: DataFrame(date, value)}.
    """
    print("\n  -- ECB Statistical Data Warehouse --")
    print(f"  Base URL: {ECB_BASE}")

    results = {}
    for name, (dataset, key, freq, desc) in ECB_SERIES.items():
        print(f"  [{name}] {desc}...")
        csv_text = ecb_download_series(dataset, key)
        if csv_text is None:
            print(f"    [WARN] Download failed")
            continue

        df = ecb_parse_csv(csv_text, freq)
        if df.empty:
            print(f"    [WARN] No data parsed")
            continue

        # Aggregate to quarterly if needed
        df_q = ecb_to_quarterly(df, freq)
        obs = len(df_q)
        if obs > 0:
            d_min = df_q["date"].min()
            d_max = df_q["date"].max()
            period_min = f"{d_min.year}-Q{(d_min.month - 1) // 3 + 1}"
            period_max = f"{d_max.year}-Q{(d_max.month - 1) // 3 + 1}"
        else:
            period_min = period_max = "n/a"

        # Save raw series
        raw_path = DATA_DIR / f"{name}_raw.csv"
        df_q.to_csv(raw_path, index=False)

        results[name] = df_q
        print(f"    [OK] {obs} quarterly obs ({period_min} to {period_max})")

    return results


def merge_ecb_series(series_dict):
    """
    Merge multiple quarterly ECB series into a single wide DataFrame.
    Each series becomes a column; rows are aligned by quarterly date.
    """
    if not series_dict:
        return pd.DataFrame()

    # Start with the first series
    names = list(series_dict.keys())
    merged = series_dict[names[0]].rename(columns={"value": names[0]})

    for name in names[1:]:
        df = series_dict[name].rename(columns={"value": name})
        merged = merged.merge(df, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)

    # Trim to range where at least the BoP target variable exists
    if "bop_ca" in merged.columns:
        merged = merged.dropna(subset=["bop_ca"])

    # Forward-fill small gaps in other columns
    for col in merged.columns:
        if col != "date":
            merged[col] = merged[col].interpolate(method="linear", limit=2)

    return merged


# ===================================================================
# PART 2: EUROSTAT API (secondary source for missing variables)
# ===================================================================

def download_eurostat(dataset_code, filters, params=None):
    """Download data from Eurostat SDMX REST API."""
    url = f"{EUROSTAT_BASE}/{dataset_code}/{filters}"
    default_params = {"format": "TSV", "compressed": "false"}
    if params:
        default_params.update(params)
    print(f"  Downloading {dataset_code}...")
    try:
        resp = requests.get(url, params=default_params,
                            timeout=15, verify=VERIFY_SSL)
        resp.raise_for_status()
        return resp.text
    except requests.exceptions.RequestException as e:
        print(f"  [WARN] Failed: {type(e).__name__}")
        return None


def parse_eurostat_tsv(raw_text):
    """Parse Eurostat TSV into a tidy DataFrame."""
    if raw_text is None:
        return pd.DataFrame()
    df = pd.read_csv(StringIO(raw_text), sep="\t")
    first_col = df.columns[0]
    time_cols = [c for c in df.columns if c != first_col]
    key_parts = first_col.split(",")
    dim_df = df[first_col].str.split(",", expand=True)
    dim_df.columns = key_parts
    df_combined = pd.concat([dim_df, df[time_cols]], axis=1)
    df_long = df_combined.melt(id_vars=key_parts, var_name="time",
                                value_name="value")
    df_long["value"] = df_long["value"].astype(str).str.strip()
    df_long["value"] = df_long["value"].str.replace(r"[a-zA-Z :]", "",
                                                      regex=True)
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")
    df_long["time"] = df_long["time"].str.strip()
    return df_long


def try_eurostat_extras():
    """Try Eurostat for series not available in ECB SDW."""
    datasets = {
        "fx":    ("ert_bil_eur_q",  "Q.AVG.NAC.USD"),
        "ir":    ("irt_st_q",      "Q.IRT_M3.EA"),
    }
    results = {}
    for name, (code, filters) in datasets.items():
        raw = download_eurostat(code, filters)
        if raw:
            df = parse_eurostat_tsv(raw)
            if not df.empty:
                results[name] = df
                print(f"  [OK] {name}: {len(df)} records")
    return results


# ===================================================================
# PART 3: SYNTHETIC DATASET (fallback)
# ===================================================================

def generate_synthetic_dataset():
    """
    Generate a realistic synthetic quarterly dataset for France's BoP
    and macro indicators, calibrated to published statistics.

    WARNING: THIS IS NOT REAL DATA. For model development/testing only.
    Replace with actual data before any publication.
    """
    print("\n  Generating synthetic dataset...")
    print("  [WARN] SYNTHETIC data -- for model testing only!\n")
    warnings.warn(
        "Using SYNTHETIC data — results are for model testing only, "
        "not for publication.",
        RuntimeWarning,
        stacklevel=2,
    )

    np.random.seed(42)
    dates = pd.date_range("2000-01-01", "2025-12-31", freq="QS")
    n = len(dates)
    t = np.arange(n)

    # Business cycle
    cycle = np.sin(2 * np.pi * t / 28) * 0.5
    trend = t * 0.002

    # Crisis shocks
    gfc = np.zeros(n)
    covid = np.zeros(n)
    energy = np.zeros(n)
    for i, d in enumerate(dates):
        if pd.Timestamp("2008-07-01") <= d <= pd.Timestamp("2009-06-30"):
            gfc[i] = -1.5 * np.exp(-((d - pd.Timestamp("2009-01-01")).days / 120) ** 2)
        if pd.Timestamp("2020-01-01") <= d <= pd.Timestamp("2020-09-30"):
            covid[i] = -2.0 * np.exp(-((d - pd.Timestamp("2020-04-01")).days / 60) ** 2)
        if pd.Timestamp("2022-01-01") <= d <= pd.Timestamp("2022-12-31"):
            energy[i] = -0.8 * np.exp(-((d - pd.Timestamp("2022-07-01")).days / 120) ** 2)
    shock = gfc + covid + energy

    # Variables (realistic magnitudes for France)
    bop_ca = (-8000 + 2000 * cycle + trend * 500 + shock * 5000
              + np.cumsum(np.random.normal(0, 300, n)) * 0.3
              + np.random.normal(0, 1500, n))
    bop_goods = -18000 + 3000 * cycle + shock * 8000 + np.random.normal(0, 2000, n)
    bop_services = 7000 + 1000 * cycle + shock * 3000 + np.random.normal(0, 1000, n)
    gdp = 540000 + t * 1200 + 5000 * cycle + shock * 25000 + np.random.normal(0, 3000, n)

    ip_base = 85 + t * 0.15 + 3 * cycle + shock * 10 + np.random.normal(0, 1.5, n)
    idx_2021 = np.where(dates >= pd.Timestamp("2021-01-01"))[0]
    if len(idx_2021) > 0:
        ip_base = ip_base * (100 / ip_base[idx_2021[0]])

    hicp = 100 + t * 0.4 + np.cumsum(np.random.normal(0, 0.2, n))
    for i, d in enumerate(dates):
        if d >= pd.Timestamp("2021-07-01"):
            hicp[i] += (d - pd.Timestamp("2021-07-01")).days * 0.005

    eurusd = np.clip(1.15 + 0.1 * np.sin(2 * np.pi * t / 20)
                     + np.cumsum(np.random.normal(0, 0.01, n)) * 0.3
                     + np.random.normal(0, 0.02, n), 0.85, 1.60)

    unemp = np.clip(9.0 - t * 0.02 - 1.5 * cycle - shock * 1.5
                    + np.random.normal(0, 0.3, n), 5.0, 12.0)

    ir = 3.0 - t * 0.04 + 0.5 * cycle + np.random.normal(0, 0.1, n)
    for i, d in enumerate(dates):
        if pd.Timestamp("2014-01-01") <= d <= pd.Timestamp("2022-06-30"):
            ir[i] = max(-0.5, ir[i] * 0.1)
        if d >= pd.Timestamp("2022-07-01"):
            ir[i] = ir[i] + (d - pd.Timestamp("2022-07-01")).days * 0.003
    ir = np.clip(ir, -0.6, 5.5)

    pmi = np.clip(50 + 5 * cycle + shock * 8 + np.random.normal(0, 1.5, n), 30, 65)

    return pd.DataFrame({
        "date": dates,
        "bop_ca": bop_ca.round(0),
        "bop_goods": bop_goods.round(0),
        "bop_services": bop_services.round(0),
        "gdp": gdp.round(0),
        "ip_index": ip_base.round(1),
        "hicp": hicp.round(1),
        "eurusd": eurusd.round(4),
        "unemployment": unemp.round(1),
        "interest_rate": ir.round(2),
        "pmi": pmi.round(1),
    })


# ===================================================================
# PART 4: PREPARE MODELING DATASET
# ===================================================================

def prepare_modeling_dataset(df):
    """Add derived features for modeling."""
    print("Preparing modeling dataset...")

    for col in ["bop_ca", "bop_goods", "bop_services", "gdp"]:
        if col in df.columns:
            df[f"{col}_yoy"] = df[col].diff(4)
            df[f"{col}_qoq"] = df[col].diff(1)

    for col in ["gdp", "ip_index"]:
        if col in df.columns:
            df[f"{col}_growth"] = df[col].pct_change(4) * 100

    target = "bop_ca"
    if target in df.columns:
        df[f"{target}_lag1"] = df[target].shift(1)
        df[f"{target}_lag2"] = df[target].shift(2)
        df[f"{target}_lag4"] = df[target].shift(4)

    df = df.dropna().reset_index(drop=True)
    return df


# ===================================================================
# MAIN PIPELINE
# ===================================================================

def build_dataset():
    """
    Try data sources in priority order:
      1. ECB Statistical Data Warehouse (no auth, public)
      2. Eurostat SDMX API (supplement)
      3. Synthetic fallback
    """
    print("=" * 60)
    print("  BoP NOWCASTING PILOT -- DATA PIPELINE")
    print("=" * 60)

    # --- Source 1: ECB Statistical Data Warehouse ---
    print("\n[1/3] ECB Statistical Data Warehouse...")
    ecb_data = try_ecb_sdw()

    if len(ecb_data) >= 5:
        print(f"\n  [OK] ECB SDW: {len(ecb_data)}/{len(ECB_SERIES)} series downloaded!")
        df = merge_ecb_series(ecb_data)

        # Check if we have enough data
        if len(df) >= 20 and "bop_ca" in df.columns:
            print(f"  Merged dataset: {len(df)} quarters, {len(df.columns)} columns")
            print("  >>> Using REAL ECB data! <<<")
            use_real = True
        else:
            print(f"  [WARN] Merged data too small ({len(df)} rows), falling back.")
            use_real = False
    else:
        print(f"\n  [WARN] Only {len(ecb_data)}/{len(ECB_SERIES)} series downloaded.")
        use_real = False

    # --- Source 2: Eurostat supplement ---
    if not use_real:
        print("\n[2/3] Eurostat SDMX API (supplementary)...")
        eurostat_data = try_eurostat_extras()
        if eurostat_data:
            print(f"  Got {len(eurostat_data)} extra series from Eurostat.")

    # --- Source 3: Synthetic fallback ---
    if not use_real:
        print("\n[3/3] Using synthetic data (insufficient real data).")
        df = generate_synthetic_dataset()

    df = prepare_modeling_dataset(df)

    out_path = DATA_DIR / "modeling_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Shape: {df.shape[0]} quarters x {df.shape[1]} variables")
    print(f"Period: {df['date'].min()} to {df['date'].max()}")

    print("\n  Variables:")
    for col in df.columns:
        if col != "date":
            print(f"    - {col:<25} (min: {df[col].min():>10.1f}, "
                  f"max: {df[col].max():>10.1f})")

    return df


if __name__ == "__main__":
    df = build_dataset()
    print("\n[OK] Data pipeline complete. Preview:")
    print(df.head(5).to_string())
