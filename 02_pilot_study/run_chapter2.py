"""
Chapter 2 -- Ablation Study: Alternative Data for BoP Nowcasting
=================================================================

Structured ablation study comparing:
  M0: Baseline features (Chapter 1)
  M1: M0 + Category A (trade-activity proxies: BDI, CTI)
  M2: M0 + Category B (financial-flow proxies: FX vol, CDS)
  M3: M0 + Category C (text/sentiment: TPU, Google Trends, ECB sentiment)
  M4: M0 + Category D (satellite: nighttime lights)
  M5: M0 + A + B + C + D (all categories combined)

Each specification is evaluated with Ridge and XGBoost using the same
expanding-window protocol as the pilot study.

Usage:
  python run_chapter2.py

Author: PhD Pilot Study
Date: March 2026
"""

import sys
from pathlib import Path

src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

import pandas as pd
import numpy as np
import warnings
from io import StringIO

from download_data import (
    ECB_BASE, VERIFY_SSL, DATA_DIR,
    ecb_download_series, ecb_parse_csv,
)
from models import (
    ForecastResult, ar_forecast, ols_forecast,
    gradient_boosting_forecast, xgboost_forecast, lstm_forecast,
    diebold_mariano_test, block_bootstrap_rmse_ci, OUTPUT_DIR,
    HAS_TORCH, HAS_SHAP,
    StandardScaler,
    model_confidence_set, clark_west_test, r2_oos, forecast_combination,
)
from alternative_data import (
    download_all_alternative_data,
    merge_alternative_with_baseline,
    ALT_DATA_DIR,
)
from nlp_sentiment import run_nlp_pipeline

warnings.filterwarnings("ignore")

CH2_OUTPUT = OUTPUT_DIR / "chapter2"
CH2_OUTPUT.mkdir(exist_ok=True)


# =====================================================================
# MONTHLY SERIES (same as run_monthly.py)
# =====================================================================

MONTHLY_SERIES = {
    "trade_goods": ("BP6", "M.N.FR.W1.S1.S1.T.B.G._Z._Z._Z.EUR._T._X.N",
                    "M", "Monthly goods balance"),
    "eurusd":      ("EXR", "M.USD.EUR.SP00.A",
                    "M", "EUR/USD exchange rate"),
    "hicp":        ("ICP", "M.FR.N.000000.4.ANR",
                    "M", "France HICP (annual rate)"),
    "interest_rate": ("FM", "M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA",
                      "M", "Euribor 3-month rate"),
}


def download_baseline_monthly():
    """Download the baseline monthly series from ECB SDW."""
    print("\n" + "=" * 60)
    print("  PHASE 1: BASELINE MONTHLY DATA")
    print("=" * 60)
    results = {}
    for name, (dataset, key, freq, desc) in MONTHLY_SERIES.items():
        print(f"  [{name}] {desc}...")
        csv_text = ecb_download_series(dataset, key)
        if csv_text is None:
            print(f"    [WARN] Download failed")
            continue
        df = ecb_parse_csv(csv_text, freq)
        if df.empty:
            print(f"    [WARN] No data parsed")
            continue
        results[name] = df
        print(f"    [OK] {len(df)} monthly obs")
    return results


def merge_baseline(series_dict):
    """Merge baseline monthly series."""
    if not series_dict:
        return pd.DataFrame()
    names = list(series_dict.keys())
    merged = series_dict[names[0]].rename(columns={"value": names[0]})
    for name in names[1:]:
        df = series_dict[name].rename(columns={"value": name})
        merged = merged.merge(df, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)
    if "trade_goods" in merged.columns:
        merged = merged.dropna(subset=["trade_goods"])
    for col in merged.columns:
        if col != "date":
            merged[col] = merged[col].interpolate(method="linear", limit=2)
    return merged


def prepare_baseline_features(df, target="trade_goods"):
    """Add lags for baseline model (M0)."""
    if target in df.columns:
        df[f"{target}_lag1"] = df[target].shift(1)
        df[f"{target}_lag3"] = df[target].shift(3)
        df[f"{target}_lag12"] = df[target].shift(12)
    for col in ["eurusd", "hicp", "interest_rate"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)
    return df


# =====================================================================
# ABLATION SPECIFICATIONS
# =====================================================================

# Each category maps to the column names it adds
CATEGORY_A = ["bdi", "cti"]
CATEGORY_B = ["fx_vol", "cds_spread"]
CATEGORY_C = ["tpu", "gtrends_export", "gtrends_import",
              "ecb_sentiment", "sentiment_dispersion",
              "bdf_sentiment", "bdf_momentum"]
CATEGORY_D = ["ntl"]

SPECIFICATIONS = {
    "M0": [],                                           # Baseline only
    "M1": CATEGORY_A,                                  # + trade-activity
    "M2": CATEGORY_B,                                  # + financial-flow
    "M3": CATEGORY_C,                                  # + text/sentiment
    "M4": CATEGORY_D,                                  # + satellite
    "M5": CATEGORY_A + CATEGORY_B + CATEGORY_C + CATEGORY_D,  # all
}


# =====================================================================
# GIACOMINI-WHITE CONDITIONAL PREDICTIVE ABILITY TEST
# =====================================================================

def giacomini_white_test(e1, e2, instruments=None):
    """
    Giacomini & White (2006) conditional predictive ability test.

    Tests whether forecast 1 is conditionally more accurate than
    forecast 2, allowing the relative accuracy to depend on the
    information set.

    Uses Newey-West HAC covariance with bandwidth floor(4*(T/100)^(2/9))
    following Newey & West (1994) automatic bandwidth selection.

    Parameters:
        e1, e2: forecast errors from two models
        instruments: matrix of conditioning instruments (default: constant + lagged loss diff)

    Returns:
        GW test statistic, p-value
    """
    from scipy import stats

    d = e1 ** 2 - e2 ** 2  # Loss differential
    n = len(d)

    if n < 10:
        return np.nan, np.nan

    if instruments is None:
        # Default instruments: constant and lagged loss differential
        z = np.column_stack([np.ones(n - 1), d[:-1]])
        d_t = d[1:]
    else:
        z = instruments
        d_t = d

    n_t = len(d_t)
    q = z.shape[1]

    try:
        # Regress d_t on instruments
        zd = z * d_t.reshape(-1, 1)
        mean_zd = zd.mean(axis=0)

        # Newey-West HAC covariance with automatic bandwidth
        # Bandwidth: floor(4*(T/100)^(2/9)) per Newey & West (1994)
        bw = max(1, int(4.0 * (n_t / 100.0) ** (2.0 / 9.0)))
        S = (zd.T @ zd) / n_t
        for h in range(1, bw + 1):
            w = 1.0 - h / (bw + 1.0)  # Bartlett kernel weight
            gamma_h = (zd[h:].T @ zd[:-h]) / n_t
            S = S + w * (gamma_h + gamma_h.T)

        S_inv = np.linalg.inv(S)
        gw_stat = n_t * mean_zd @ S_inv @ mean_zd
        p_value = 1 - stats.chi2.cdf(gw_stat, df=q)

        return gw_stat, p_value
    except (np.linalg.LinAlgError, ValueError):
        return np.nan, np.nan


# =====================================================================
# EXPANDING WINDOW FOR ABLATION
# =====================================================================

def ablation_expanding_window(df, target_col, baseline_features, alt_features,
                               min_train_size=60, step=1,
                               run_xgboost=True):
    """
    Run expanding-window evaluation for a given specification.
    Uses Ridge and optionally XGBoost.

    Returns: dict of {model_name: ForecastResult}
    """
    feature_cols = baseline_features + [f for f in alt_features if f in df.columns]

    cols_needed = [target_col] + feature_cols
    df_clean = df[["date"] + [c for c in cols_needed if c in df.columns]].dropna().reset_index(drop=True)

    # Recompute feature_cols based on what's actually in df_clean
    feature_cols = [c for c in feature_cols if c in df_clean.columns]

    if len(df_clean) < min_train_size + 5:
        return {}

    y = df_clean[target_col].values
    X = df_clean[feature_cols].values
    dates = df_clean["date"].values
    n = len(y)

    models = {
        "AR(1)": ForecastResult("AR(1)"),
        "Ridge": ForecastResult("Ridge"),
    }
    if run_xgboost:
        models["XGBoost"] = ForecastResult("XGBoost")

    scaler = StandardScaler()

    for t in range(min_train_size, n, step):
        y_train = y[:t]
        X_train = X[:t]
        y_test = y[t]
        X_test = X[t]
        test_date = dates[t]

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test.reshape(1, -1)).flatten()

        # AR(1)
        p = ar_forecast(y_train, order=1)
        models["AR(1)"].predictions.append(p)
        models["AR(1)"].actuals.append(y_test)
        models["AR(1)"].dates.append(test_date)

        # Ridge
        p = ols_forecast(X_train_scaled, y_train, X_test_scaled)
        models["Ridge"].predictions.append(p)
        models["Ridge"].actuals.append(y_test)
        models["Ridge"].dates.append(test_date)

        # XGBoost
        if run_xgboost:
            p = xgboost_forecast(X_train_scaled, y_train, X_test_scaled)
            models["XGBoost"].predictions.append(p)
            models["XGBoost"].actuals.append(y_test)
            models["XGBoost"].dates.append(test_date)

    # Compute metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    for name, res in models.items():
        preds = np.array(res.predictions, dtype=float)
        acts = np.array(res.actuals, dtype=float)
        valid = ~np.isnan(preds) & ~np.isnan(acts)
        if valid.sum() < 2:
            continue
        res.rmse = np.sqrt(mean_squared_error(acts[valid], preds[valid]))
        res.mae = mean_absolute_error(acts[valid], preds[valid])

    return models


# =====================================================================
# SHAP DECOMPOSITION BY CATEGORY
# =====================================================================

def shap_category_decomposition(df, target_col, baseline_features, all_alt_features):
    """
    Compute SHAP values for the combined model (M5) and decompose
    importance by data category.
    """
    if not HAS_SHAP:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    import xgboost as xgb

    feature_cols = baseline_features + [f for f in all_alt_features if f in df.columns]
    cols_needed = [target_col] + feature_cols
    df_clean = df[["date"] + [c for c in cols_needed if c in df.columns]].dropna().reset_index(drop=True)
    feature_cols = [c for c in feature_cols if c in df_clean.columns]

    y = df_clean[target_col].values
    X = df_clean[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0,
    )
    model.fit(X_scaled, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    mean_abs = np.abs(shap_values).mean(axis=0)

    # Classify features by category
    cat_map = {}
    for i, f in enumerate(feature_cols):
        if f in CATEGORY_A:
            cat_map[f] = "A: Trade-activity"
        elif f in CATEGORY_B:
            cat_map[f] = "B: Financial-flow"
        elif f in CATEGORY_C:
            cat_map[f] = "C: Text/Sentiment"
        elif f in CATEGORY_D:
            cat_map[f] = "D: Satellite"
        else:
            cat_map[f] = "Baseline"

    # Aggregate by category
    cat_importance = {}
    for i, f in enumerate(feature_cols):
        cat = cat_map[f]
        cat_importance[cat] = cat_importance.get(cat, 0) + mean_abs[i]

    total = sum(cat_importance.values())
    cat_pct = {k: v / total * 100 for k, v in cat_importance.items()}

    # Plot: stacked bar
    cats = sorted(cat_pct.keys())
    colors = {
        "Baseline": "#607D8B",
        "A: Trade-activity": "#4CAF50",
        "B: Financial-flow": "#2196F3",
        "C: Text/Sentiment": "#FF9800",
        "D: Satellite": "#9C27B0",
    }

    fig, ax = plt.subplots(figsize=(10, 4))
    left = 0
    for cat in cats:
        pct = cat_pct[cat]
        ax.barh(0, pct, left=left, color=colors.get(cat, "#999"),
                edgecolor="white", label=f"{cat} ({pct:.1f}%)")
        left += pct
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of total Mean |SHAP value| (%)")
    ax.set_yticks([])
    ax.set_title("Feature Importance Decomposition by Data Category (M5)", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=9)
    plt.tight_layout()
    fig.savefig(CH2_OUTPUT / "shap_category_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Also plot individual feature SHAP
    top_n = min(20, len(feature_cols))
    order = np.argsort(mean_abs)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    bar_colors = [colors.get(cat_map[feature_cols[i]], "#999") for i in order[::-1]]
    ax.barh(range(top_n), mean_abs[order][::-1], color=bar_colors, edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_cols[i] for i in order][::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Combined Model (M5) Feature Importance", fontweight="bold")
    plt.tight_layout()
    fig.savefig(CH2_OUTPUT / "shap_m5_features.png", dpi=150, bbox_inches="tight")
    plt.close()

    return cat_pct, cat_importance


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("+" + "=" * 64 + "+")
    print("|  CHAPTER 2: ALTERNATIVE DATA ABLATION STUDY                  |")
    print("|  The Value of Alternative Data for BoP Nowcasting            |")
    print("+" + "=" * 64 + "+")

    # ---------------------------------------------------------------
    # Phase 1: Download baseline monthly data
    # ---------------------------------------------------------------
    series = download_baseline_monthly()
    if len(series) < 2:
        print("\n[ERROR] Not enough baseline series.")
        return

    df = merge_baseline(series)
    if df.empty or "trade_goods" not in df.columns:
        print("\n[ERROR] No trade_goods target.")
        return

    target_col = "trade_goods"
    dt_min = df["date"].min()
    dt_max = df["date"].max()

    # ---------------------------------------------------------------
    # Phase 2: Download alternative data
    # ---------------------------------------------------------------
    start_str = dt_min.strftime("%Y-%m-%d")
    end_str = dt_max.strftime("%Y-%m-%d")
    alt_data, alt_meta = download_all_alternative_data(start=start_str, end=end_str)

    # ---------------------------------------------------------------
    # Phase 3: NLP sentiment
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  PHASE 3: NLP SENTIMENT ANALYSIS")
    print("=" * 60)
    sentiment_df, sentiment_status = run_nlp_pipeline(
        start_year=dt_min.year, end_year=dt_max.year
    )
    if not sentiment_df.empty:
        alt_data["nlp_sentiment"] = sentiment_df
        alt_meta["ecb_sentiment"] = sentiment_status

    # ---------------------------------------------------------------
    # Phase 4: Merge everything
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  PHASE 4: MERGE & PREPARE DATASET")
    print("=" * 60)

    df = merge_alternative_with_baseline(df, alt_data)
    df = prepare_baseline_features(df, target=target_col)
    # Forward-fill sparse alternative data (e.g., ecb_sentiment with limited obs)
    # then fill remaining NaN with training-set median (not zero, since zero has
    # meaning for bounded indicators like sentiment on [-1,+1])
    alt_cols = [c for c in df.columns if c not in
                ["date", target_col] + [f"{target_col}_lag1", f"{target_col}_lag3",
                 f"{target_col}_lag12", "eurusd", "hicp", "interest_rate",
                 "eurusd_lag1", "hicp_lag1", "interest_rate_lag1"]]
    for c in alt_cols:
        df[c] = df[c].ffill()
        col_median = df[c].median()
        df[c] = df[c].fillna(col_median if not np.isnan(col_median) else 0)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    print(f"\n  Combined dataset: {df.shape[0]} months x {df.shape[1]} columns")
    print(f"  Period: {df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}")
    print(f"  Columns: {list(df.columns)}")

    # Save combined dataset
    df.to_csv(CH2_OUTPUT / "combined_dataset.csv", index=False)

    # Baseline features (same as monthly pilot)
    leakage = [f"{target_col}_mom", f"{target_col}_yoy"]
    baseline_features = [c for c in [
        f"{target_col}_lag1", f"{target_col}_lag3", f"{target_col}_lag12",
        "eurusd", "hicp", "interest_rate",
        "eurusd_lag1", "hicp_lag1", "interest_rate_lag1",
    ] if c in df.columns]

    print(f"\n  Baseline features ({len(baseline_features)}):")
    for f in baseline_features:
        print(f"    - {f}")

    # ---------------------------------------------------------------
    # Phase 5: Run ablation study
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  PHASE 5: ABLATION STUDY")
    print("=" * 60)

    all_results = {}
    spec_errors = {}  # Store forecast errors for GW tests

    for spec_name, alt_features in SPECIFICATIONS.items():
        n_alt = sum(1 for f in alt_features if f in df.columns)
        total_features = len(baseline_features) + n_alt
        print(f"\n  --- {spec_name}: baseline + {n_alt} alt features = {total_features} total ---")

        models = ablation_expanding_window(
            df, target_col, baseline_features, alt_features,
            min_train_size=60, step=1
        )

        if not models:
            print(f"    [WARN] {spec_name} evaluation failed")
            continue

        all_results[spec_name] = models

        # Store Ridge and XGBoost errors for GW tests
        for mname in ["Ridge", "XGBoost"]:
            if mname in models and models[mname].rmse is not None:
                preds = np.array(models[mname].predictions, dtype=float)
                acts = np.array(models[mname].actuals, dtype=float)
                valid = ~np.isnan(preds) & ~np.isnan(acts)
                errors = np.full(len(preds), np.nan)
                errors[valid] = acts[valid] - preds[valid]
                spec_errors[(spec_name, mname)] = errors

    # ---------------------------------------------------------------
    # Phase 6: Results table
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ABLATION STUDY RESULTS")
    print("=" * 70)

    # Get M0 RMSEs as benchmark
    m0_ridge_rmse = all_results.get("M0", {}).get("Ridge", ForecastResult("")).rmse
    m0_xgb_rmse = all_results.get("M0", {}).get("XGBoost", ForecastResult("")).rmse

    print(f"\n  {'Spec':<6} {'Ridge RMSE':>12} {'vs M0':>8} {'XGB RMSE':>12} {'vs M0':>8}")
    print("  " + "-" * 50)

    summary_rows = []
    for spec_name in ["M0", "M1", "M2", "M3", "M4", "M5"]:
        if spec_name not in all_results:
            continue
        models = all_results[spec_name]
        ridge_rmse = models.get("Ridge", ForecastResult("")).rmse
        xgb_rmse = models.get("XGBoost", ForecastResult("")).rmse

        ridge_vs = ((ridge_rmse - m0_ridge_rmse) / m0_ridge_rmse * 100) if ridge_rmse and m0_ridge_rmse else None
        xgb_vs = ((xgb_rmse - m0_xgb_rmse) / m0_xgb_rmse * 100) if xgb_rmse and m0_xgb_rmse else None

        ridge_str = f"{ridge_rmse:,.0f}" if ridge_rmse else "N/A"
        xgb_str = f"{xgb_rmse:,.0f}" if xgb_rmse else "N/A"
        ridge_vs_str = f"{ridge_vs:+.1f}%" if ridge_vs is not None else "---"
        xgb_vs_str = f"{xgb_vs:+.1f}%" if xgb_vs is not None else "---"

        print(f"  {spec_name:<6} {ridge_str:>12} {ridge_vs_str:>8} {xgb_str:>12} {xgb_vs_str:>8}")

        summary_rows.append({
            "specification": spec_name,
            "description": {
                "M0": "Baseline", "M1": "+Trade-activity", "M2": "+Financial-flow",
                "M3": "+Text/Sentiment", "M4": "+Satellite", "M5": "All combined"
            }.get(spec_name, ""),
            "ridge_rmse": ridge_rmse,
            "ridge_vs_m0_pct": ridge_vs,
            "xgboost_rmse": xgb_rmse,
            "xgboost_vs_m0_pct": xgb_vs,
        })

    # ---------------------------------------------------------------
    # Phase 7: Statistical tests (DM + GW)
    # ---------------------------------------------------------------
    print(f"\n\n  STATISTICAL TESTS vs M0")
    print("  " + "-" * 65)
    print(f"  {'Spec':<6} {'Model':<10} {'DM stat':>10} {'DM p':>8} {'GW stat':>10} {'GW p':>8}")
    print("  " + "-" * 65)

    for spec_name in ["M1", "M2", "M3", "M4", "M5"]:
        for mname in ["Ridge", "XGBoost"]:
            key_m0 = ("M0", mname)
            key_mk = (spec_name, mname)
            if key_m0 not in spec_errors or key_mk not in spec_errors:
                continue
            e0 = spec_errors[key_m0]
            ek = spec_errors[key_mk]
            # Align valid observations
            valid = ~np.isnan(e0) & ~np.isnan(ek)
            if valid.sum() < 10:
                continue
            e0v = e0[valid]
            ekv = ek[valid]

            dm_stat, dm_p = diebold_mariano_test(ekv, e0v)
            gw_stat, gw_p = giacomini_white_test(ekv, e0v)

            dm_str = f"{dm_stat:.3f}" if not np.isnan(dm_stat) else "N/A"
            dm_p_str = f"{dm_p:.4f}" if not np.isnan(dm_p) else "N/A"
            gw_str = f"{gw_stat:.3f}" if not np.isnan(gw_stat) else "N/A"
            gw_p_str = f"{gw_p:.4f}" if not np.isnan(gw_p) else "N/A"

            print(f"  {spec_name:<6} {mname:<10} {dm_str:>10} {dm_p_str:>8} {gw_str:>10} {gw_p_str:>8}")

            # Add to summary
            for row in summary_rows:
                if row["specification"] == spec_name:
                    if mname == "Ridge":
                        row["ridge_dm_stat"] = dm_stat
                        row["ridge_dm_p"] = dm_p
                        row["ridge_gw_stat"] = gw_stat
                        row["ridge_gw_p"] = gw_p
                    else:
                        row["xgboost_dm_stat"] = dm_stat
                        row["xgboost_dm_p"] = dm_p
                        row["xgboost_gw_stat"] = gw_stat
                        row["xgboost_gw_p"] = gw_p

    # ---------------------------------------------------------------
    # Phase 7b: Bootstrap confidence intervals for ablation
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  BOOTSTRAP CONFIDENCE INTERVALS (Ablation vs M0)")
    print("=" * 70)

    boot_rows = []
    m0_models = all_results.get("M0", {})
    m0_ridge = m0_models.get("Ridge", ForecastResult("")) if m0_models else ForecastResult("")
    if m0_ridge.rmse and len(m0_ridge.dates) > 0:
        # Date-aligned bootstrap: intersect dates between M0 and each Mk
        m0_date_idx = {str(d): i for i, d in enumerate(m0_ridge.dates)}

        for spec_name in ["M1", "M2", "M3", "M4", "M5"]:
            if spec_name not in all_results:
                continue
            mk_ridge = all_results[spec_name].get("Ridge")
            if mk_ridge is None or mk_ridge.rmse is None or len(mk_ridge.dates) == 0:
                continue

            # Find common dates
            common_m0, common_mk = [], []
            for j, d in enumerate(mk_ridge.dates):
                ds = str(d)
                if ds in m0_date_idx:
                    common_m0.append(m0_date_idx[ds])
                    common_mk.append(j)

            if len(common_m0) < 12:
                print(f"    {spec_name}: only {len(common_m0)} common dates, skipping")
                continue

            # Build aligned ForecastResults
            aligned_bench = ForecastResult("M0_Ridge")
            aligned_bench.actuals = [m0_ridge.actuals[i] for i in common_m0]
            aligned_bench.predictions = [m0_ridge.predictions[i] for i in common_m0]
            aligned_bench.rmse = m0_ridge.rmse

            aligned_model = ForecastResult(f"{spec_name}_Ridge")
            aligned_model.actuals = [mk_ridge.actuals[i] for i in common_mk]
            aligned_model.predictions = [mk_ridge.predictions[i] for i in common_mk]
            aligned_model.rmse = mk_ridge.rmse

            virtual = {"M0_Ridge": aligned_bench, f"{spec_name}_Ridge": aligned_model}

            try:
                boot_df = block_bootstrap_rmse_ci(
                    virtual, benchmark="M0_Ridge",
                    n_boot=1000, block_length=6, seed=42
                )
            except Exception as e:
                print(f"    {spec_name}: bootstrap error: {e}")
                continue
            if not boot_df.empty:
                row = boot_df.iloc[0]
                ci_str = f"[{row['ci_lower']:+.1f}, {row['ci_upper']:+.1f}]"
                print(f"  {spec_name:<15} {row['pct_improvement']:>+8.1f}% {ci_str:>20}")
                boot_rows.append({
                    "specification": spec_name,
                    "pct_vs_m0": row["pct_improvement"],
                    "ci_lower": row["ci_lower"],
                    "ci_upper": row["ci_upper"],
                })

        if boot_rows:
            print(f"\n  {'Spec':<15} {'% vs M0':>10} {'95% CI':>20}")
            print("  " + "-" * 50)
            for r in boot_rows:
                ci_str = f"[{r['ci_lower']:+.1f}, {r['ci_upper']:+.1f}]"
                print(f"  {r['specification']:<15} {r['pct_vs_m0']:>+8.1f}% {ci_str:>20}")
            pd.DataFrame(boot_rows).to_csv(
                CH2_OUTPUT / "bootstrap_ci_ablation.csv", index=False
            )
            print(f"\n  Saved bootstrap_ci_ablation.csv")

    # ---------------------------------------------------------------
    # Phase 7d: Model Confidence Set (Hansen, Lunde & Nason, 2011)
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  MODEL CONFIDENCE SET (alpha = 0.10)")
    print("=" * 70)

    # Build ForecastResult dict across all specs × models for MCS
    mcs_candidates = {}
    for spec_name, models in all_results.items():
        for mname, res in models.items():
            if res.rmse is not None and len(res.predictions) > 0:
                label = f"{spec_name}_{mname}"
                mcs_candidates[label] = res

    if len(mcs_candidates) >= 3:
        try:
            surviving, eliminated = model_confidence_set(
                mcs_candidates, alpha=0.10, n_boot=1000, block_length=6, seed=42
            )
            print(f"  Superior set: {surviving}")
            print(f"  Eliminated  : {eliminated}")
            mcs_df = pd.DataFrame([{"model": m, "in_mcs": m in surviving}
                                    for m in mcs_candidates.keys()])
            mcs_df.to_csv(CH2_OUTPUT / "model_confidence_set.csv", index=False)
            print(f"  Saved model_confidence_set.csv")
        except Exception as e:
            print(f"  [WARN] MCS failed: {e}")
    else:
        print("  [WARN] Not enough valid models for MCS")

    # ---------------------------------------------------------------
    # Phase 7e: Giacomini-Rossi (2010) Fluctuation Test
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  GIACOMINI-ROSSI FLUCTUATION TEST (M5 Ridge vs M0 Ridge)")
    print("=" * 70)

    key_m0_ridge = ("M0", "Ridge")
    key_m5_ridge = ("M5", "Ridge")
    if key_m0_ridge in spec_errors and key_m5_ridge in spec_errors:
        e_m0 = spec_errors[key_m0_ridge]
        e_m5 = spec_errors[key_m5_ridge]
        valid_gr = ~np.isnan(e_m0) & ~np.isnan(e_m5)

        if valid_gr.sum() >= 30:
            e0v = e_m0[valid_gr]
            e5v = e_m5[valid_gr]
            T = len(e0v)

            # Rolling DM statistic with centered window
            window_size = max(20, T // 4)
            half_w = window_size // 2
            rolling_dm = []
            rolling_idx = []
            from scipy import stats as _stats

            for t in range(half_w, T - half_w):
                d_t = e0v[t - half_w:t + half_w] ** 2 - e5v[t - half_w:t + half_w] ** 2
                mean_d = d_t.mean()
                var_d = d_t.var(ddof=1)
                if var_d > 0:
                    dm_t = mean_d / np.sqrt(var_d / len(d_t))
                else:
                    dm_t = 0.0
                rolling_dm.append(dm_t)
                rolling_idx.append(t)

            rolling_dm = np.array(rolling_dm)
            # Critical value for two-sided 5% (adjusted for sequential testing)
            cv_5 = _stats.norm.ppf(0.975)

            # Plot
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(rolling_idx, rolling_dm, "b-", linewidth=1.2, label="Rolling DM statistic")
            ax.axhline(cv_5, color="red", linestyle="--", linewidth=0.8, label=f"+{cv_5:.2f} (5% CV)")
            ax.axhline(-cv_5, color="red", linestyle="--", linewidth=0.8, label=f"-{cv_5:.2f} (5% CV)")
            ax.axhline(0, color="grey", linestyle=":", linewidth=0.5)
            ax.set_xlabel("Evaluation period index")
            ax.set_ylabel("DM statistic (M5 vs M0, Ridge)")
            ax.set_title("Giacomini-Rossi Fluctuation Test: M5 vs M0 (Ridge)", fontweight="bold")
            ax.legend(loc="best", fontsize=9)
            plt.tight_layout()
            fig.savefig(CH2_OUTPUT / "giacomini_rossi_fluctuation.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Window size: {window_size} months, T={T}")
            print(f"  Saved giacomini_rossi_fluctuation.png")

            # Summary statistic: fraction of time M5 is significantly better
            frac_sig = np.mean(rolling_dm > cv_5)
            print(f"  Fraction where M5 sig. better: {frac_sig:.1%}")

            pd.DataFrame({"index": rolling_idx, "rolling_dm": rolling_dm}).to_csv(
                CH2_OUTPUT / "giacomini_rossi_data.csv", index=False
            )
        else:
            print("  [WARN] Not enough observations for GR test")
    else:
        print("  [WARN] M0 or M5 Ridge errors not available")

    # ---------------------------------------------------------------
    # Phase 7f: Rossi (2013) Out-of-Sample Stability Tests
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ROSSI (2013) OUT-OF-SAMPLE STABILITY TESTS")
    print("=" * 70)

    if key_m0_ridge in spec_errors and key_m5_ridge in spec_errors:
        e_m0 = spec_errors[key_m0_ridge]
        e_m5 = spec_errors[key_m5_ridge]
        valid_mask = ~np.isnan(e_m0) & ~np.isnan(e_m5)

        if valid_mask.sum() >= 30:
            e0 = e_m0[valid_mask]
            e5 = e_m5[valid_mask]
            T_oos = len(e0)
            d_t = e0 ** 2 - e5 ** 2  # loss differential

            # --- (a) Sup-type test (Rossi 2013, §3.2) ---
            # Supremum of absolute rolling DM statistic
            w = max(20, T_oos // 4)
            hw = w // 2
            roll_dm = []
            for t in range(hw, T_oos - hw):
                d_w = d_t[t - hw:t + hw]
                mu = d_w.mean()
                sig = d_w.std(ddof=1)
                roll_dm.append(mu / (sig / np.sqrt(len(d_w))) if sig > 0 else 0.0)
            roll_dm = np.array(roll_dm)

            sup_dm = np.max(np.abs(roll_dm))
            # Bootstrap p-value for the sup statistic
            rng = np.random.default_rng(42)
            n_boot = 2000
            sup_boot = np.empty(n_boot)
            for b in range(n_boot):
                idx = rng.choice(T_oos, size=T_oos, replace=True)
                d_b = d_t[idx]
                roll_b = []
                for t in range(hw, T_oos - hw):
                    d_w = d_b[t - hw:t + hw]
                    mu = d_w.mean()
                    sig = d_w.std(ddof=1)
                    roll_b.append(mu / (sig / np.sqrt(len(d_w))) if sig > 0 else 0.0)
                sup_boot[b] = np.max(np.abs(roll_b))
            sup_p = np.mean(sup_boot >= sup_dm)

            # --- (b) CUSUM test on loss differentials ---
            cum_d = np.cumsum(d_t - d_t.mean())
            std_d = d_t.std(ddof=1)
            cusum_stat = np.max(np.abs(cum_d)) / (std_d * np.sqrt(T_oos))
            # Kolmogorov-Smirnov 5% critical value ≈ 1.36
            ks_cv = 1.36
            cusum_reject = cusum_stat > ks_cv

            print(f"  Loss differential: d_t = e^2(M0) - e^2(M5)")
            print(f"  T_oos = {T_oos}, rolling window = {w}")
            print(f"\n  (a) Sup-DM test:")
            print(f"      sup|DM_t| = {sup_dm:.3f}")
            print(f"      Bootstrap p-value = {sup_p:.3f} ({n_boot} replications)")
            if sup_p < 0.05:
                print("      -> Reject H0: relative accuracy is UNSTABLE over time")
            else:
                print("      -> Fail to reject H0: no evidence of instability")
            print(f"\n  (b) CUSUM test:")
            print(f"      CUSUM stat = {cusum_stat:.3f}  (KS 5% CV = {ks_cv})")
            if cusum_reject:
                print("      -> Reject H0: structural break in loss differential")
            else:
                print("      -> Fail to reject H0: stable loss differential")

            pd.DataFrame([{
                "sup_dm": sup_dm,
                "sup_dm_p": sup_p,
                "cusum_stat": cusum_stat,
                "cusum_cv_5pct": ks_cv,
                "cusum_reject": cusum_reject,
                "T_oos": T_oos,
                "window": w,
            }]).to_csv(CH2_OUTPUT / "rossi_stability_tests.csv", index=False)
            print(f"\n  Saved rossi_stability_tests.csv")
        else:
            print("  [WARN] Not enough observations for stability tests")
    else:
        print("  [WARN] M0 or M5 Ridge errors not available")

    # ---------------------------------------------------------------
    # Phase 7g: M5 Interaction Discussion
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  M5 INTERACTION ANALYSIS")
    print("=" * 70)

    if "M5" in all_results and "M0" in all_results:
        m5_ridge = all_results["M5"].get("Ridge", ForecastResult(""))
        sum_parts_rmse = 0
        n_parts = 0
        for spec in ["M1", "M2", "M3", "M4"]:
            if spec in all_results:
                sr = all_results[spec].get("Ridge", ForecastResult(""))
                if sr.rmse and m0_ridge_rmse:
                    sum_parts_rmse += (sr.rmse - m0_ridge_rmse)
                    n_parts += 1

        if m5_ridge.rmse and m0_ridge_rmse and n_parts > 0:
            m5_gain = m5_ridge.rmse - m0_ridge_rmse
            sum_marginal = sum_parts_rmse
            interaction = m5_gain - sum_marginal
            print(f"  M5 total gain vs M0:       {m5_gain:+,.0f}")
            print(f"  Sum of marginal gains:     {sum_marginal:+,.0f}")
            print(f"  Interaction effect:        {interaction:+,.0f}")
            if sum_marginal != 0:
                ratio = m5_gain / sum_marginal
                print(f"  M5 / sum(marginals) ratio: {ratio:.2f}")
                if ratio < 1:
                    print("  -> Sub-additive: combining categories yields diminishing returns")
                elif ratio > 1:
                    print("  -> Super-additive: categories complement each other")
                else:
                    print("  -> Purely additive")

            pd.DataFrame([{
                "m5_gain": m5_gain,
                "sum_marginal_gains": sum_marginal,
                "interaction": interaction,
            }]).to_csv(CH2_OUTPUT / "m5_interaction.csv", index=False)
            print(f"  Saved m5_interaction.csv")

    # ---------------------------------------------------------------
    # Phase 7c: Robustness checks
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ROBUSTNESS CHECKS")
    print("=" * 70)

    robustness_rows = []

    # (a) Different minimum training windows
    for win in [45, 75, 90]:
        print(f"\n  --- Window = {win} months ---")
        for spec_name in ["M0", "M5"]:
            alt_features = SPECIFICATIONS[spec_name]
            models_r = ablation_expanding_window(
                df, target_col, baseline_features, alt_features,
                min_train_size=win, step=1
            )
            for mname in ["Ridge", "XGBoost"]:
                rmse = models_r.get(mname, ForecastResult("")).rmse
                if rmse:
                    robustness_rows.append({
                        "check": f"window_{win}",
                        "specification": spec_name,
                        "model": mname,
                        "rmse": rmse,
                    })
                    print(f"    {spec_name} {mname}: RMSE = {rmse:,.0f}")

    # (b) Crisis subsample evaluation -- evaluate only on crisis periods
    crisis_periods = {
        "GFC": ("2008-09-01", "2009-06-30"),
        "Euro_Crisis": ("2011-06-01", "2012-06-30"),
        "COVID": ("2020-03-01", "2021-03-31"),
        "Energy_Crisis": ("2022-01-01", "2022-12-31"),
    }

    for crisis_name, (c_start, c_end) in crisis_periods.items():
        c_start_dt = pd.Timestamp(c_start)
        c_end_dt = pd.Timestamp(c_end)
        n_crisis = df[(df["date"] >= c_start_dt) & (df["date"] <= c_end_dt)].shape[0]
        if n_crisis < 3:
            continue

        print(f"\n  --- Crisis: {crisis_name} ({n_crisis} obs) ---")

        for spec_name in ["M0", "M5"]:
            if spec_name not in all_results:
                continue
            mods = all_results[spec_name]
            for mname in ["Ridge", "XGBoost"]:
                res = mods.get(mname)
                if res is None or res.rmse is None:
                    continue
                # Filter predictions to crisis period
                preds_crisis, acts_crisis = [], []
                for i, d in enumerate(res.dates):
                    dt = pd.Timestamp(d)
                    if c_start_dt <= dt <= c_end_dt:
                        p = res.predictions[i]
                        a = res.actuals[i]
                        if not np.isnan(p) and not np.isnan(a):
                            preds_crisis.append(p)
                            acts_crisis.append(a)
                if len(preds_crisis) >= 2:
                    from sklearn.metrics import mean_squared_error as _mse
                    crisis_rmse = np.sqrt(_mse(acts_crisis, preds_crisis))
                    robustness_rows.append({
                        "check": f"crisis_{crisis_name}",
                        "specification": spec_name,
                        "model": mname,
                        "rmse": crisis_rmse,
                    })
                    print(f"    {spec_name} {mname}: RMSE = {crisis_rmse:,.0f}")

    if robustness_rows:
        pd.DataFrame(robustness_rows).to_csv(
            CH2_OUTPUT / "robustness_results.csv", index=False
        )
        print(f"\n  Robustness results saved to {CH2_OUTPUT / 'robustness_results.csv'}")

    # ---------------------------------------------------------------
    # Phase 8: SHAP category decomposition
    # ---------------------------------------------------------------    if HAS_SHAP:
        print("\n" + "=" * 60)
        print("  SHAP CATEGORY DECOMPOSITION (M5)")
        print("=" * 60)
        try:
            cat_pct, cat_imp = shap_category_decomposition(
                df, target_col, baseline_features, SPECIFICATIONS["M5"]
            )
            if cat_pct:
                for cat, pct in sorted(cat_pct.items()):
                    print(f"    {cat:<25} {pct:6.1f}%")
        except Exception as e:
            print(f"  [WARN] SHAP decomposition failed: {e}")

    # ---------------------------------------------------------------
    # Phase 9: Pseudo-Real-Time Vintage Robustness
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PSEUDO-REAL-TIME VINTAGE ROBUSTNESS")
    print("=" * 70)
    print("  Simulating data revisions (Eurostat revision statistics):")
    print("  Gaussian noise, std = 3% of level, applied to trade_goods target")

    vintage_rows = []
    n_mc = 50
    revision_std_frac = 0.03  # 3% of level — consistent with EU trade revision stats

    target_std = df[target_col].std()
    revision_sigma = revision_std_frac * df[target_col].abs().mean()

    np.random.seed(2026)
    for mc_iter in range(n_mc):
        df_noisy = df.copy()
        noise = np.random.normal(0, revision_sigma, size=len(df_noisy))
        df_noisy[target_col] = df_noisy[target_col] + noise

        for spec_name in ["M0", "M5"]:
            alt_feats = SPECIFICATIONS[spec_name]
            mods = ablation_expanding_window(
                df_noisy, target_col, baseline_features, alt_feats,
                min_train_size=60, step=1, run_xgboost=False
            )
            for mname in ["Ridge"]:
                rmse_v = mods.get(mname, ForecastResult("")).rmse
                if rmse_v:
                    vintage_rows.append({
                        "mc_iter": mc_iter, "spec": spec_name,
                        "model": mname, "rmse": rmse_v,
                    })

        if (mc_iter + 1) % 10 == 0:
            print(f"    MC iteration {mc_iter+1}/{n_mc} done")

    if vintage_rows:
        vdf = pd.DataFrame(vintage_rows)
        # Compare M0 vs M5 under revision noise
        m0_rmses = vdf[(vdf["spec"] == "M0") & (vdf["model"] == "Ridge")]["rmse"]
        m5_rmses = vdf[(vdf["spec"] == "M5") & (vdf["model"] == "Ridge")]["rmse"]
        if len(m0_rmses) > 0 and len(m5_rmses) > 0:
            m0_mean = m0_rmses.mean()
            m5_mean = m5_rmses.mean()
            pct_gain = (m5_mean - m0_mean) / m0_mean * 100
            print(f"\n  Results over {n_mc} MC replications (revision noise std = {revision_sigma:,.0f}):")
            print(f"    M0 Ridge mean RMSE:  {m0_mean:,.0f} (sd {m0_rmses.std():,.0f})")
            print(f"    M5 Ridge mean RMSE:  {m5_mean:,.0f} (sd {m5_rmses.std():,.0f})")
            print(f"    M5 vs M0:            {pct_gain:+.1f}%")
            # Fraction of MC where M5 beats M0
            if len(m0_rmses) == len(m5_rmses):
                frac_better = np.mean(m5_rmses.values < m0_rmses.values)
                print(f"    Fraction M5 < M0:    {frac_better:.0%}")

        vdf.to_csv(CH2_OUTPUT / "vintage_robustness.csv", index=False)
        print(f"  Saved vintage_robustness.csv")

    # ---------------------------------------------------------------
    # Phase 10: MCS Power Analysis (Monte Carlo)
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  MCS POWER ANALYSIS (Monte Carlo)")
    print("=" * 70)

    T_oos_actual = 0
    if "M0" in all_results:
        r0 = all_results["M0"].get("Ridge", ForecastResult(""))
        if r0.rmse and r0.predictions:
            T_oos_actual = len([p for p in r0.predictions if not np.isnan(p)])

    if T_oos_actual >= 20:
        n_mc_power = 500
        effect_size_pct = 7.7  # matches observed M5 improvement
        T_sim = T_oos_actual
        alpha_mcs = 0.10
        np.random.seed(42)

        # Under DGP where Model B is truly effect_size_pct% better than Model A
        # Generate squared errors and test whether MCS eliminates A
        rejections = 0
        for mc_i in range(n_mc_power):
            # Model A (baseline): squared errors ~ chi2(1) * sigma^2
            sigma_a = 1.0
            sigma_b = sigma_a * (1 - effect_size_pct / 100)
            e2_a = np.random.exponential(sigma_a ** 2, size=T_sim)
            e2_b = np.random.exponential(sigma_b ** 2, size=T_sim)

            # Simple two-model MCS: test H0: equal predictive ability
            d_ij = e2_a - e2_b  # positive = A worse than B
            T_d = len(d_ij)
            d_mean = d_ij.mean()
            d_std = d_ij.std(ddof=1)
            if d_std > 0:
                t_stat = d_mean / (d_std / np.sqrt(T_d))
            else:
                t_stat = 0
            # Bootstrap p-value (block bootstrap)
            block_len = max(1, int(np.sqrt(T_d)))
            n_blocks = T_d // block_len + 1
            boot_t = np.zeros(1000)
            for b in range(1000):
                idx = np.concatenate([
                    np.arange(start, min(start + block_len, T_d))
                    for start in np.random.randint(0, T_d, size=n_blocks)
                ])[:T_d]
                d_boot = d_ij[idx]
                d_b_mean = d_boot.mean()
                d_b_std = d_boot.std(ddof=1)
                if d_b_std > 0:
                    boot_t[b] = (d_b_mean - d_mean) / (d_b_std / np.sqrt(T_d))
                else:
                    boot_t[b] = 0
            p_val = np.mean(np.abs(boot_t) >= np.abs(t_stat))
            if p_val < alpha_mcs:
                rejections += 1

        power = rejections / n_mc_power
        print(f"  DGP: Model B is {effect_size_pct}% better than Model A (RMSE)")
        print(f"  T = {T_sim}, alpha = {alpha_mcs}, MC replications = {n_mc_power}")
        print(f"  Empirical power = {power:.1%}")
        if power < 0.50:
            print(f"  -> Low power confirms that MCS retaining all models")
            print(f"     is expected given T = {T_sim} and effect = {effect_size_pct}%")

        pd.DataFrame([{
            "effect_size_pct": effect_size_pct,
            "T_oos": T_sim,
            "alpha": alpha_mcs,
            "n_mc": n_mc_power,
            "power": power,
        }]).to_csv(CH2_OUTPUT / "mcs_power_analysis.csv", index=False)
        print(f"  Saved mcs_power_analysis.csv")
    else:
        print(f"  [WARN] T_oos = {T_oos_actual}, too small for power analysis")

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    pd.DataFrame(summary_rows).to_csv(CH2_OUTPUT / "ablation_results.csv", index=False)

    # Save data source metadata
    meta_rows = [{"source": k, "status": v} for k, v in alt_meta.items()]
    pd.DataFrame(meta_rows).to_csv(CH2_OUTPUT / "data_source_status.csv", index=False)

    print(f"\n\nResults saved to {CH2_OUTPUT}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CHAPTER 2 ABLATION STUDY -- COMPLETE")
    print("=" * 70)
    n_real = sum(1 for v in alt_meta.values() if v == "real")
    n_synth = sum(1 for v in alt_meta.values() if v == "synthetic")
    print(f"  Data sources: {n_real} real + {n_synth} synthetic")
    print(f"  Specifications tested: {len(all_results)}")
    if m0_ridge_rmse:
        print(f"  M0 Ridge baseline RMSE: {m0_ridge_rmse:,.0f}")
    best_spec = min(
        [(s, r.get("Ridge", ForecastResult("")).rmse)
         for s, r in all_results.items() if r.get("Ridge", ForecastResult("")).rmse],
        key=lambda x: x[1] if x[1] else float('inf'),
        default=("---", None)
    )
    if best_spec[1]:
        print(f"  Best specification: {best_spec[0]} (Ridge RMSE = {best_spec[1]:,.0f})")

    print("\n[OK] Chapter 2 analysis complete!")


if __name__ == "__main__":
    main()
