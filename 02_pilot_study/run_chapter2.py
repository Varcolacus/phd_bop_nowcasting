"""
Chapter 2 — Ablation Study: Alternative Data for BoP Nowcasting
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

        # Variance estimation (Newey-West with 1 lag)
        S = (zd.T @ zd) / n_t
        if n_t > 1:
            zd_lag = zd[:-1]
            zd_lead = zd[1:]
            gamma1 = (zd_lead.T @ zd_lag) / n_t
            S = S + gamma1 + gamma1.T

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
                               min_train_size=60, step=1):
    """
    Run expanding-window evaluation for a given specification.
    Uses Ridge and XGBoost.

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
        "XGBoost": ForecastResult("XGBoost"),
    }

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
    # then fill remaining NaN with 0 (neutral) for sentiment and alt-data columns
    alt_cols = [c for c in df.columns if c not in
                ["date", target_col] + [f"{target_col}_lag1", f"{target_col}_lag3",
                 f"{target_col}_lag12", "eurusd", "hicp", "interest_rate",
                 "eurusd_lag1", "hicp_lag1", "interest_rate_lag1"]]
    for c in alt_cols:
        df[c] = df[c].ffill().fillna(0)
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
        ridge_vs_str = f"{ridge_vs:+.1f}%" if ridge_vs is not None else "—"
        xgb_vs_str = f"{xgb_vs:+.1f}%" if xgb_vs is not None else "—"

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

    # (b) Crisis subsample evaluation — evaluate only on crisis periods
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
    print("  CHAPTER 2 ABLATION STUDY — COMPLETE")
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
        default=("—", None)
    )
    if best_spec[1]:
        print(f"  Best specification: {best_spec[0]} (Ridge RMSE = {best_spec[1]:,.0f})")

    print("\n[OK] Chapter 2 analysis complete!")


if __name__ == "__main__":
    main()
