"""
Alternative Data Sources for Chapter 2 — BoP Nowcasting
=========================================================

Downloads and prepares four categories of alternative data:
  A) Trade-activity proxies   — Baltic Dry Index, Container Throughput
  B) Financial-flow proxies   — EUR/USD realised volatility, sovereign CDS
  C) Text / sentiment         — Trade Policy Uncertainty, Google Trends
  D) Satellite / geospatial   — Nighttime lights proxy

Public sources are downloaded directly; unavailable sources use realistic
synthetic fallbacks (flagged in output).

Author: PhD Pilot Study
Date: March 2026
"""

import os
import warnings
import requests
import numpy as np
import pandas as pd
from io import StringIO
from pathlib import Path

from download_data import ECB_BASE, VERIFY_SSL, DATA_DIR, ecb_download_series, ecb_parse_csv

warnings.filterwarnings("ignore")

ALT_DATA_DIR = DATA_DIR / "alternative"
ALT_DATA_DIR.mkdir(exist_ok=True)


# =====================================================================
# CATEGORY A — Trade-activity proxies
# =====================================================================

def download_bdi_from_fred():
    """
    Download the Baltic Dry Index from FRED (series DBDI).
    Uses the public CSV graph endpoint (no API key required).
    Returns a monthly DataFrame(date, bdi).
    """
    print("  [BDI] Downloading from FRED...")
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    params = {"id": "DBDI", "cosd": "2000-01-01", "coed": "2024-12-31"}
    try:
        resp = requests.get(url, params=params, timeout=30, verify=VERIFY_SSL)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df.columns = ["date", "bdi"]
        df["date"] = pd.to_datetime(df["date"])
        df["bdi"] = pd.to_numeric(df["bdi"], errors="coerce")
        df = df.dropna(subset=["bdi"])
        # Aggregate daily to monthly average
        df = df.set_index("date").resample("MS").mean().reset_index()
        df.to_csv(ALT_DATA_DIR / "bdi_raw.csv", index=False)
        print(f"    [OK] {len(df)} monthly obs")
        return df
    except Exception as e:
        print(f"    [WARN] FRED download failed: {e}")
        return None


def generate_synthetic_bdi(date_range):
    """Generate synthetic BDI with realistic properties."""
    print("  [BDI] Using synthetic fallback")
    n = len(date_range)
    np.random.seed(42)
    # BDI typically 500-11000, mean ~1500, high volatility, mean-reverting
    bdi = np.zeros(n)
    bdi[0] = 1500
    for i in range(1, n):
        bdi[i] = bdi[i-1] + 0.95 * (1500 - bdi[i-1]) * 0.02 + np.random.normal(0, 80)
    bdi = np.clip(bdi, 300, 12000)
    df = pd.DataFrame({"date": date_range, "bdi": bdi})
    df.to_csv(ALT_DATA_DIR / "bdi_synthetic.csv", index=False)
    return df


def download_container_throughput():
    """
    Container Throughput Index (CTI) — RWI/ISL.
    This requires registration; we generate a realistic synthetic series.
    """
    print("  [CTI] Container Throughput Index (synthetic — requires RWI/ISL access)")
    return None


def generate_synthetic_cti(date_range):
    """Generate synthetic Container Throughput Index."""
    n = len(date_range)
    np.random.seed(43)
    # CTI indexed to 100 at 2008-01, trending upward, big COVID drop
    trend = np.linspace(85, 130, n)
    cycle = 5 * np.sin(np.linspace(0, 8 * np.pi, n))
    noise = np.random.normal(0, 3, n)
    cti = trend + cycle + noise
    # COVID shock (2020-03 to 2020-06)
    for i, d in enumerate(date_range):
        if pd.Timestamp("2020-03-01") <= d <= pd.Timestamp("2020-06-01"):
            cti[i] -= 25
        elif pd.Timestamp("2020-07-01") <= d <= pd.Timestamp("2020-12-01"):
            cti[i] -= 10  # partial recovery
    df = pd.DataFrame({"date": date_range, "cti": cti})
    df.to_csv(ALT_DATA_DIR / "cti_synthetic.csv", index=False)
    return df


# =====================================================================
# CATEGORY B — Financial-flow proxies
# =====================================================================

def compute_fx_volatility():
    """
    Compute EUR/USD realised volatility from ECB daily exchange rate data.
    Downloads daily EUR/USD rates and computes 20-day rolling std of log returns,
    aggregated to monthly.
    """
    print("  [FX_VOL] Computing EUR/USD realised volatility from ECB daily rates...")
    # Download daily EUR/USD
    csv_text = ecb_download_series("EXR", "D.USD.EUR.SP00.A", start_period="2000")
    if csv_text is None:
        print("    [WARN] Daily FX download failed")
        return None

    df = pd.read_csv(StringIO(csv_text))
    if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
        print("    [WARN] No data parsed")
        return None

    df = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
    df.columns = ["date", "rate"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)

    # Log returns
    df["log_ret"] = np.log(df["rate"]).diff()
    # 20-day rolling standard deviation (annualised)
    df["vol_20d"] = df["log_ret"].rolling(20).std() * np.sqrt(252) * 100

    # Monthly average of daily realised vol
    df = df.set_index("date")
    monthly = df["vol_20d"].resample("MS").mean().reset_index()
    monthly.columns = ["date", "fx_vol"]
    monthly = monthly.dropna()
    monthly.to_csv(ALT_DATA_DIR / "fx_vol_raw.csv", index=False)
    print(f"    [OK] {len(monthly)} monthly obs")
    return monthly


def generate_synthetic_fx_vol(date_range):
    """Generate synthetic FX volatility."""
    print("  [FX_VOL] Using synthetic fallback")
    n = len(date_range)
    np.random.seed(44)
    # Typical FX vol 5-15%, spikes during crises
    vol = np.random.lognormal(np.log(8), 0.3, n)
    for i, d in enumerate(date_range):
        if pd.Timestamp("2008-09-01") <= d <= pd.Timestamp("2009-03-01"):
            vol[i] *= 1.8
        elif pd.Timestamp("2020-03-01") <= d <= pd.Timestamp("2020-05-01"):
            vol[i] *= 1.6
    df = pd.DataFrame({"date": date_range, "fx_vol": vol})
    df.to_csv(ALT_DATA_DIR / "fx_vol_synthetic.csv", index=False)
    return df


def download_france_cds():
    """
    France CDS spread — requires Bloomberg/Refinitiv terminal.
    Returns None; synthetic fallback used.
    """
    print("  [CDS] France CDS spread (synthetic — requires terminal access)")
    return None


def generate_synthetic_cds(date_range):
    """Generate synthetic France 5Y CDS spread."""
    n = len(date_range)
    np.random.seed(45)
    # France CDS typically 10-200bp, spikes in GFC and euro crisis
    cds = np.zeros(n)
    cds[0] = 20
    for i in range(1, n):
        cds[i] = cds[i-1] + 0.9 * (30 - cds[i-1]) * 0.03 + np.random.normal(0, 5)
    cds = np.clip(cds, 5, 300)
    # Crisis spikes
    for i, d in enumerate(date_range):
        if pd.Timestamp("2008-09-01") <= d <= pd.Timestamp("2009-06-01"):
            cds[i] += 80
        elif pd.Timestamp("2011-06-01") <= d <= pd.Timestamp("2012-06-01"):
            cds[i] += 120  # Euro crisis
        elif pd.Timestamp("2020-03-01") <= d <= pd.Timestamp("2020-05-01"):
            cds[i] += 40
    df = pd.DataFrame({"date": date_range, "cds_spread": cds})
    df.to_csv(ALT_DATA_DIR / "cds_synthetic.csv", index=False)
    return df


# =====================================================================
# CATEGORY C — Text / Sentiment
# =====================================================================

def download_trade_policy_uncertainty():
    """
    Trade Policy Uncertainty Index (Caldara & Iacoviello).
    Try direct download from their website.
    """
    print("  [TPU] Trade Policy Uncertainty Index...")
    # Try the public download URL
    urls = [
        "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls",
        "https://www.policyuncertainty.com/media/Trade_Policy_Uncertainty_Index.csv",
    ]
    for url in urls:
        try:
            resp = requests.get(url, timeout=30, verify=VERIFY_SSL)
            resp.raise_for_status()
            if url.endswith(".xls"):
                df = pd.read_excel(StringIO(resp.text))
            else:
                df = pd.read_csv(StringIO(resp.text))
            print(f"    [OK] Downloaded from {url.split('/')[2]}")
            return df
        except Exception:
            continue
    print("    [WARN] TPU download failed — using synthetic")
    return None


def generate_synthetic_tpu(date_range):
    """Generate synthetic Trade Policy Uncertainty index."""
    n = len(date_range)
    np.random.seed(46)
    # TPU mean ~100, spikes during trade wars, Brexit, etc.
    tpu = np.zeros(n)
    tpu[0] = 100
    for i in range(1, n):
        tpu[i] = tpu[i-1] + 0.85 * (100 - tpu[i-1]) * 0.05 + np.random.normal(0, 15)
    tpu = np.clip(tpu, 20, 600)
    # Trade war spike
    for i, d in enumerate(date_range):
        if pd.Timestamp("2018-03-01") <= d <= pd.Timestamp("2019-12-01"):
            tpu[i] += 100  # US-China trade war
        elif pd.Timestamp("2020-03-01") <= d <= pd.Timestamp("2020-06-01"):
            tpu[i] += 80
    df = pd.DataFrame({"date": date_range, "tpu": tpu})
    df.to_csv(ALT_DATA_DIR / "tpu_synthetic.csv", index=False)
    return df


def download_google_trends():
    """
    Download Google Trends data for trade-related keywords.
    Uses pytrends (rate-limited, may fail).
    """
    print("  [GTRENDS] Google Trends for trade keywords...")
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="fr-FR", tz=60, timeout=(10, 25))
        kw_list = ["exportation France", "importation France"]
        pytrends.build_payload(kw_list, timeframe="2008-01-01 2022-12-31", geo="FR")
        df = pytrends.interest_over_time()
        if df.empty:
            print("    [WARN] Empty result from Google Trends")
            return None
        df = df.drop(columns=["isPartial"], errors="ignore")
        df = df.reset_index()
        df.columns = ["date", "gtrends_export", "gtrends_import"]
        df["date"] = pd.to_datetime(df["date"])
        # Monthly average (Trends returns weekly)
        df = df.set_index("date").resample("MS").mean().reset_index()
        df.to_csv(ALT_DATA_DIR / "gtrends_raw.csv", index=False)
        print(f"    [OK] {len(df)} monthly obs")
        return df
    except Exception as e:
        print(f"    [WARN] Google Trends failed: {e}")
        return None


def generate_synthetic_gtrends(date_range):
    """Generate synthetic Google Trends-like indices."""
    n = len(date_range)
    np.random.seed(47)
    export_idx = 50 + np.random.normal(0, 10, n).cumsum() * 0.1
    import_idx = 50 + np.random.normal(0, 10, n).cumsum() * 0.1
    export_idx = np.clip(export_idx, 10, 100)
    import_idx = np.clip(import_idx, 10, 100)
    df = pd.DataFrame({
        "date": date_range,
        "gtrends_export": export_idx,
        "gtrends_import": import_idx,
    })
    df.to_csv(ALT_DATA_DIR / "gtrends_synthetic.csv", index=False)
    return df


# =====================================================================
# CATEGORY D — Satellite / Geospatial
# =====================================================================

def download_nighttime_lights():
    """
    Nighttime lights (NTL) from VIIRS/World Bank.
    Not available via simple API; synthetic fallback.
    """
    print("  [NTL] Nighttime lights (synthetic — requires VIIRS data access)")
    return None


def generate_synthetic_ntl(date_range):
    """Generate synthetic nighttime lights index for France."""
    n = len(date_range)
    np.random.seed(48)
    # NTL index: upward trend, seasonal, COVID dip
    trend = np.linspace(100, 115, n)
    seasonal = 3 * np.sin(np.linspace(0, len(date_range) / 12 * 2 * np.pi, n))
    noise = np.random.normal(0, 1.5, n)
    ntl = trend + seasonal + noise
    for i, d in enumerate(date_range):
        if pd.Timestamp("2020-03-01") <= d <= pd.Timestamp("2020-06-01"):
            ntl[i] -= 12  # Lockdown
        elif pd.Timestamp("2020-07-01") <= d <= pd.Timestamp("2020-12-01"):
            ntl[i] -= 5
    df = pd.DataFrame({"date": date_range, "ntl": ntl})
    df.to_csv(ALT_DATA_DIR / "ntl_synthetic.csv", index=False)
    return df


# =====================================================================
# Master download function
# =====================================================================

def download_all_alternative_data(start="2008-01-01", end="2022-12-31"):
    """
    Download all alternative data sources.
    Returns dict of DataFrames and a metadata dict flagging real vs synthetic.

    Returns:
        data: dict of {name: DataFrame(date, value)}
        meta: dict of {name: "real" | "synthetic"}
    """
    date_range = pd.date_range(start, end, freq="MS")
    data = {}
    meta = {}

    print("\n" + "=" * 60)
    print("  DOWNLOADING ALTERNATIVE DATA SOURCES")
    print("=" * 60)

    # --- Category A: Trade-activity ---
    print("\n  CATEGORY A — Trade-activity proxies")
    print("  " + "-" * 40)

    bdi = download_bdi_from_fred()
    if bdi is not None and len(bdi) > 12:
        data["bdi"] = bdi
        meta["bdi"] = "real"
    else:
        data["bdi"] = generate_synthetic_bdi(date_range)
        meta["bdi"] = "synthetic"

    cti_raw = download_container_throughput()
    data["cti"] = generate_synthetic_cti(date_range) if cti_raw is None else cti_raw
    meta["cti"] = "synthetic" if cti_raw is None else "real"

    # --- Category B: Financial-flow ---
    print("\n  CATEGORY B — Financial-flow proxies")
    print("  " + "-" * 40)

    fx_vol = compute_fx_volatility()
    if fx_vol is not None and len(fx_vol) > 12:
        data["fx_vol"] = fx_vol
        meta["fx_vol"] = "real"
    else:
        data["fx_vol"] = generate_synthetic_fx_vol(date_range)
        meta["fx_vol"] = "synthetic"

    cds_raw = download_france_cds()
    data["cds_spread"] = generate_synthetic_cds(date_range) if cds_raw is None else cds_raw
    meta["cds_spread"] = "synthetic" if cds_raw is None else "real"

    # --- Category C: Text/Sentiment ---
    print("\n  CATEGORY C — Text / Sentiment")
    print("  " + "-" * 40)

    tpu_raw = download_trade_policy_uncertainty()
    if tpu_raw is None:
        data["tpu"] = generate_synthetic_tpu(date_range)
        meta["tpu"] = "synthetic"
    else:
        data["tpu"] = tpu_raw
        meta["tpu"] = "real"

    gt = download_google_trends()
    if gt is not None and len(gt) > 12:
        data["gtrends"] = gt
        meta["gtrends"] = "real"
    else:
        data["gtrends"] = generate_synthetic_gtrends(date_range)
        meta["gtrends"] = "synthetic"

    # --- Category D: Satellite ---
    print("\n  CATEGORY D — Satellite / Geospatial")
    print("  " + "-" * 40)

    ntl_raw = download_nighttime_lights()
    data["ntl"] = generate_synthetic_ntl(date_range) if ntl_raw is None else ntl_raw
    meta["ntl"] = "synthetic" if ntl_raw is None else "real"

    # --- Summary ---
    print("\n  " + "=" * 50)
    print("  DATA SOURCE SUMMARY")
    print("  " + "=" * 50)
    for name, status in meta.items():
        flag = "✓ REAL" if status == "real" else "~ SYNTH"
        print(f"    {name:<20} [{flag}]  {len(data[name])} obs")

    return data, meta


def merge_alternative_with_baseline(baseline_df, alt_data, target_date_col="date"):
    """
    Merge alternative data into the baseline monthly DataFrame.
    Returns a single wide DataFrame aligned on monthly dates.
    """
    merged = baseline_df.copy()
    for name, df in alt_data.items():
        if "date" not in df.columns:
            continue
        # Get the value column(s) — everything except date
        val_cols = [c for c in df.columns if c != "date"]
        df_merge = df[["date"] + val_cols].copy()
        merged = merged.merge(df_merge, on="date", how="left")

    # Forward-fill then back-fill short gaps
    for col in merged.columns:
        if col != "date" and merged[col].dtype in [np.float64, np.int64, float]:
            merged[col] = merged[col].interpolate(method="linear", limit=3)

    return merged
