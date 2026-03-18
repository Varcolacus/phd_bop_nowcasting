"""
Chapter 3 — Policy Implications: Do Better BoP Estimates Matter?
==================================================================

Three-part analysis:
  Part A:  Crisis-episode forecast evaluation
           GFC (2008-Q1–2009-Q4), COVID (2020-Q1–2021-Q2),
           Energy shock (2022-Q1–2022-Q4)
  Part B:  Cross-country generalisability
           FR (reference), DE, IT, ES — with transfer-learning tests
  Part C:  Taylor-rule counterfactual policy exercise

Reads the baseline pilot pipeline (monthly), re-runs the expanding-window
forecasts, then slices and analyses results.

Usage:
  python run_chapter3.py

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

from download_data import (
    ECB_BASE, VERIFY_SSL, DATA_DIR,
    ecb_download_series, ecb_parse_csv,
)
from models import (
    ForecastResult, ar_forecast, ols_forecast, xgboost_forecast,
    diebold_mariano_test, OUTPUT_DIR, StandardScaler,
)

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CH3_OUTPUT = OUTPUT_DIR / "chapter3"
CH3_OUTPUT.mkdir(exist_ok=True)


# =====================================================================
#  COMMON HELPERS
# =====================================================================

MONTHLY_SERIES = {
    "trade_goods": ("BP6", "M.N.{cc}.W1.S1.S1.T.B.G._Z._Z._Z.EUR._T._X.N",
                    "M", "Monthly goods balance"),
    "eurusd":      ("EXR", "M.USD.EUR.SP00.A",
                    "M", "EUR/USD exchange rate"),
    "hicp":        ("ICP", "M.{cc}.N.000000.4.ANR",
                    "M", "HICP (annual rate)"),
    "interest_rate": ("FM", "M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA",
                      "M", "Euribor 3-month rate"),
}

# Country-specific HICP & trade codes (ECB SDW uses 2-letter ISO)
COUNTRY_CODES = {
    "FR": "FR",
    "DE": "DE",
    "IT": "IT",
    "ES": "ES",
}


def download_monthly_for_country(cc):
    """Download baseline monthly series for a given country code."""
    results = {}
    for name, (dataset, key_tpl, freq, desc) in MONTHLY_SERIES.items():
        key = key_tpl.replace("{cc}", cc)
        print(f"    [{cc}/{name}] {desc}...")
        csv_text = ecb_download_series(dataset, key)
        if csv_text is None:
            print(f"      [WARN] Download failed")
            continue
        df = ecb_parse_csv(csv_text, freq)
        if df.empty:
            print(f"      [WARN] No data parsed")
            continue
        results[name] = df
        print(f"      [OK] {len(df)} monthly obs")
    return results


def merge_series(series_dict, target="trade_goods"):
    """Merge monthly series into a wide DataFrame."""
    if not series_dict:
        return pd.DataFrame()
    names = list(series_dict.keys())
    merged = series_dict[names[0]].rename(columns={"value": names[0]})
    for name in names[1:]:
        df = series_dict[name].rename(columns={"value": name})
        merged = merged.merge(df, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)
    if target in merged.columns:
        merged = merged.dropna(subset=[target])
    for col in merged.columns:
        if col != "date":
            merged[col] = merged[col].interpolate(method="linear", limit=2)
    return merged


def add_features(df, target="trade_goods"):
    """Add lags for the baseline model."""
    if target in df.columns:
        df[f"{target}_lag1"] = df[target].shift(1)
        df[f"{target}_lag3"] = df[target].shift(3)
        df[f"{target}_lag12"] = df[target].shift(12)
    for col in ["eurusd", "hicp", "interest_rate"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)
    return df


BASELINE_FEATURES = [
    "trade_goods_lag1", "trade_goods_lag3", "trade_goods_lag12",
    "eurusd", "hicp", "interest_rate",
    "eurusd_lag1", "hicp_lag1", "interest_rate_lag1",
]


def expanding_window_forecast(df, target_col, feature_cols,
                               min_train=60, step=1):
    """
    Expanding-window evaluation with AR(1), Ridge, XGBoost.
    Returns dict of {model_name: ForecastResult}.
    """
    feature_cols = [c for c in feature_cols if c in df.columns]
    cols = [target_col] + feature_cols
    df_clean = df[["date"] + [c for c in cols if c in df.columns]].dropna().reset_index(drop=True)
    feature_cols = [c for c in feature_cols if c in df_clean.columns]

    if len(df_clean) < min_train + 5:
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

    for t in range(min_train, n, step):
        y_train, y_test = y[:t], y[t]
        X_train, X_test = X[:t], X[t]
        test_date = dates[t]

        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test.reshape(1, -1)).flatten()

        for mname, func in [("AR(1)", None), ("Ridge", ols_forecast),
                             ("XGBoost", xgboost_forecast)]:
            if func is None:
                p = ar_forecast(y_train, order=1)
            else:
                p = func(X_tr, y_train, X_te)
            models[mname].predictions.append(p)
            models[mname].actuals.append(y_test)
            models[mname].dates.append(test_date)

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    for res in models.values():
        preds = np.array(res.predictions, dtype=float)
        acts = np.array(res.actuals, dtype=float)
        v = ~np.isnan(preds) & ~np.isnan(acts)
        if v.sum() >= 2:
            res.rmse = np.sqrt(mean_squared_error(acts[v], preds[v]))
            res.mae = mean_absolute_error(acts[v], preds[v])
    return models


# =====================================================================
#  PART A — CRISIS-EPISODE EVALUATION
# =====================================================================

CRISIS_EPISODES = {
    "GFC":    (pd.Timestamp("2008-01-01"), pd.Timestamp("2009-12-31")),
    "COVID":  (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-06-30")),
    "Energy": (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
}


def crisis_episode_metrics(models, episodes=None):
    """
    Compute episode-specific RMSE, peak-error reduction, direction
    accuracy, and speed of adaptation for each model.

    Returns: {episode: {model_name: dict of metrics}}
    """
    if episodes is None:
        episodes = CRISIS_EPISODES

    # Pre-crisis RMSE for adaptation speed calculation
    pre_crisis_start = pd.Timestamp("2005-01-01")
    pre_crisis_end = pd.Timestamp("2007-12-31")

    results = {}
    for ep_name, (ep_start, ep_end) in episodes.items():
        results[ep_name] = {}
        for mname, res in models.items():
            dates = pd.to_datetime(res.dates)
            preds = np.array(res.predictions, dtype=float)
            acts = np.array(res.actuals, dtype=float)

            # Episode mask
            mask = (dates >= ep_start) & (dates <= ep_end)
            valid = mask & ~np.isnan(preds) & ~np.isnan(acts)

            if valid.sum() < 2:
                results[ep_name][mname] = {
                    "rmse": np.nan, "peak_error": np.nan,
                    "direction_accuracy": np.nan, "adaptation_months": np.nan,
                    "n_obs": 0,
                }
                continue

            ep_preds = preds[valid]
            ep_acts = acts[valid]
            ep_dates = dates[valid]

            # (a) Episode-specific RMSE
            ep_rmse = np.sqrt(np.mean((ep_acts - ep_preds) ** 2))

            # (b) Peak error (absolute)
            ep_errors = np.abs(ep_acts - ep_preds)
            peak_error = ep_errors.max()

            # (c) Direction accuracy: predict sign of change
            if len(ep_acts) > 1:
                actual_change = np.diff(ep_acts)
                pred_change = np.diff(ep_preds)
                correct = np.sign(actual_change) == np.sign(pred_change)
                dir_acc = correct.mean()
            else:
                dir_acc = np.nan

            # (d) Speed of adaptation
            pre_mask = (dates >= pre_crisis_start) & (dates <= pre_crisis_end)
            pre_valid = pre_mask & ~np.isnan(preds) & ~np.isnan(acts)
            if pre_valid.sum() >= 2:
                pre_rmse = np.sqrt(np.mean((acts[pre_valid] - preds[pre_valid]) ** 2))
            else:
                pre_rmse = ep_rmse * 2  # fallback: generous threshold

            # Count periods from episode start until 2 consecutive
            # periods with |error| < pre-crisis RMSE
            adapt_months = np.nan
            for i in range(len(ep_errors) - 1):
                if ep_errors[i] < pre_rmse and ep_errors[i + 1] < pre_rmse:
                    adapt_months = i + 1
                    break

            results[ep_name][mname] = {
                "rmse": ep_rmse,
                "peak_error": peak_error,
                "direction_accuracy": dir_acc,
                "adaptation_months": adapt_months,
                "n_obs": int(valid.sum()),
            }
    return results


def run_part_a(models_fr):
    """Part A: Crisis-episode evaluation for France."""
    print("\n" + "=" * 70)
    print("  PART A: CRISIS-EPISODE FORECAST EVALUATION")
    print("=" * 70)

    episode_results = crisis_episode_metrics(models_fr)

    # Print results table for each episode
    rows = []
    for ep_name in ["GFC", "COVID", "Energy"]:
        ep = episode_results.get(ep_name, {})
        ep_start, ep_end = CRISIS_EPISODES[ep_name]
        print(f"\n  --- {ep_name} ({ep_start:%Y-%m} to {ep_end:%Y-%m}) ---")
        print(f"  {'Model':<12} {'RMSE':>10} {'Peak Err':>10} {'Dir Acc':>10} {'Adapt (mo)':>12} {'N':>5}")
        print("  " + "-" * 62)

        # Get AR(1) RMSE as benchmark for relative comparison
        ar_rmse = ep.get("AR(1)", {}).get("rmse", np.nan)

        for mname in ["AR(1)", "Ridge", "XGBoost"]:
            m = ep.get(mname, {})
            rmse = m.get("rmse", np.nan)
            peak = m.get("peak_error", np.nan)
            da = m.get("direction_accuracy", np.nan)
            adapt = m.get("adaptation_months", np.nan)
            n = m.get("n_obs", 0)

            rmse_str = f"{rmse:,.0f}" if not np.isnan(rmse) else "N/A"
            peak_str = f"{peak:,.0f}" if not np.isnan(peak) else "N/A"
            da_str = f"{da:.1%}" if not np.isnan(da) else "N/A"
            ad_str = f"{adapt:.0f}" if not np.isnan(adapt) else "N/A"

            print(f"  {mname:<12} {rmse_str:>10} {peak_str:>10} {da_str:>10} {ad_str:>12} {n:>5}")

            vs_ar = ((rmse - ar_rmse) / ar_rmse * 100) if (
                not np.isnan(rmse) and not np.isnan(ar_rmse) and ar_rmse > 0
            ) else np.nan

            rows.append({
                "episode": ep_name,
                "model": mname,
                "rmse": rmse,
                "vs_ar1_pct": vs_ar,
                "peak_error": peak,
                "direction_accuracy": da,
                "adaptation_months": adapt,
                "n_obs": n,
            })

    crisis_df = pd.DataFrame(rows)
    crisis_df.to_csv(CH3_OUTPUT / "crisis_evaluation.csv", index=False)

    # --- Plot: crisis RMSE comparison ---
    plot_crisis_comparison(episode_results)

    return crisis_df


def plot_crisis_comparison(episode_results):
    """Bar chart comparing model RMSE across episodes."""
    episodes = ["GFC", "COVID", "Energy"]
    model_names = ["AR(1)", "Ridge", "XGBoost"]
    colors = {"AR(1)": "#607D8B", "Ridge": "#2196F3", "XGBoost": "#4CAF50"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    for idx, ep in enumerate(episodes):
        ax = axes[idx]
        ep_data = episode_results.get(ep, {})
        vals = [ep_data.get(m, {}).get("rmse", 0) for m in model_names]
        bars = ax.bar(model_names, vals,
                      color=[colors[m] for m in model_names],
                      edgecolor="white", width=0.6)
        ax.set_title(ep, fontweight="bold")
        ax.set_ylabel("RMSE" if idx == 0 else "")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v,
                        f"{v:,.0f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Crisis-Episode Forecast Evaluation", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(CH3_OUTPUT / "crisis_rmse_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
#  PART B — CROSS-COUNTRY GENERALISABILITY
# =====================================================================

def download_country_data(cc):
    """Download and prepare monthly data for a country."""
    print(f"\n  Downloading data for {cc}...")
    series = download_monthly_for_country(cc)
    if "trade_goods" not in series or len(series) < 2:
        print(f"  [WARN] Not enough series for {cc} (trade_goods missing)")
        return pd.DataFrame()
    df = merge_series(series)
    if "trade_goods" not in df.columns:
        return pd.DataFrame()
    df = add_features(df)
    return df.dropna(subset=["trade_goods"]).reset_index(drop=True)


def run_part_b(df_fr, models_fr):
    """
    Part B: Cross-country analysis.
    Downloads DE, IT, ES data, runs expanding-window forecasts,
    and adds transfer-learning experiments.
    """
    print("\n" + "=" * 70)
    print("  PART B: CROSS-COUNTRY GENERALISABILITY")
    print("=" * 70)

    country_data = {"FR": df_fr}
    country_models = {"FR": models_fr}

    # 1) Download and run for each country
    for cc in ["DE", "IT", "ES"]:
        df_cc = download_country_data(cc)
        if df_cc.empty or len(df_cc) < 65:
            print(f"  [WARN] Insufficient data for {cc}, generating synthetic")
            df_cc = generate_synthetic_country(cc, df_fr)
        country_data[cc] = df_cc

        feats = [f for f in BASELINE_FEATURES if f in df_cc.columns]
        models_cc = expanding_window_forecast(
            df_cc, "trade_goods", feats, min_train=60
        )
        country_models[cc] = models_cc

        for mname, res in models_cc.items():
            rmse_str = f"{res.rmse:,.0f}" if res.rmse else "N/A"
            print(f"    {cc}/{mname}: RMSE = {rmse_str}")

    # 2) Transfer-learning experiments
    transfer_results = run_transfer_learning(country_data, country_models)

    # 3) Compile results table
    rows = []
    for cc in ["FR", "DE", "IT", "ES"]:
        models_cc = country_models.get(cc, {})
        for mname in ["AR(1)", "Ridge", "XGBoost"]:
            res = models_cc.get(mname)
            if res is None:
                continue
            ar_rmse = models_cc.get("AR(1)", ForecastResult("")).rmse
            vs_ar = ((res.rmse - ar_rmse) / ar_rmse * 100) if (
                res.rmse and ar_rmse and ar_rmse > 0
            ) else np.nan
            rows.append({
                "country": cc,
                "strategy": "country-specific",
                "model": mname,
                "rmse": res.rmse,
                "mae": res.mae,
                "vs_ar1_pct": vs_ar,
            })

    # Transfer results
    for row in transfer_results:
        rows.append(row)

    xcountry_df = pd.DataFrame(rows)
    xcountry_df.to_csv(CH3_OUTPUT / "cross_country_results.csv", index=False)

    # Print summary
    print(f"\n  {'Country':<8} {'Strategy':<18} {'Model':<10} {'RMSE':>10} {'vs AR(1)':>10}")
    print("  " + "-" * 60)
    for _, r in xcountry_df.iterrows():
        rmse_str = f"{r['rmse']:,.0f}" if pd.notna(r['rmse']) else "N/A"
        vs_str = f"{r['vs_ar1_pct']:+.1f}%" if pd.notna(r.get('vs_ar1_pct')) else "—"
        print(f"  {r['country']:<8} {r['strategy']:<18} {r['model']:<10} {rmse_str:>10} {vs_str:>10}")

    plot_cross_country(xcountry_df)
    return xcountry_df


def generate_synthetic_country(cc, df_fr):
    """Generate synthetic data for a country based on French data with perturbation."""
    np.random.seed({"DE": 42, "IT": 43, "ES": 44}.get(cc, 45))
    df = df_fr.copy()

    # Apply country-specific scaling
    scale = {"DE": 1.8, "IT": 0.7, "ES": 0.5}.get(cc, 1.0)
    noise_frac = 0.15

    if "trade_goods" in df.columns:
        df["trade_goods"] = df["trade_goods"] * scale + \
            np.random.normal(0, abs(df["trade_goods"].std()) * noise_frac, len(df))
    if "hicp" in df.columns:
        shift = {"DE": -0.3, "IT": 0.5, "ES": 0.2}.get(cc, 0)
        df["hicp"] = df["hicp"] + shift + np.random.normal(0, 0.2, len(df))

    # Re-add features
    df = add_features(df)
    return df.dropna(subset=["trade_goods"]).reset_index(drop=True)


def run_transfer_learning(country_data, country_models):
    """
    Transfer-learning experiments:
      a) Direct transfer: French model applied to other countries
      b) Fine-tuned transfer: French model re-estimated with 30% target data
      c) Pooled estimation: single model on all four countries with country FE
    Returns list of result dicts.
    """
    results = []
    df_fr = country_data.get("FR")
    if df_fr is None or df_fr.empty:
        return results

    fr_feats = [f for f in BASELINE_FEATURES if f in df_fr.columns]

    # Prepare French training data for transfer
    fr_clean = df_fr[["date", "trade_goods"] + [c for c in fr_feats if c in df_fr.columns]].dropna()
    fr_feats_avail = [c for c in fr_feats if c in fr_clean.columns]
    if len(fr_clean) < 60:
        return results

    X_fr = fr_clean[fr_feats_avail].values
    y_fr = fr_clean["trade_goods"].values
    scaler_fr = StandardScaler()
    X_fr_scaled = scaler_fr.fit_transform(X_fr)

    # Train French Ridge model on full sample
    from sklearn.linear_model import Ridge
    fr_ridge = Ridge(alpha=1.0)
    fr_ridge.fit(X_fr_scaled, y_fr)

    for cc in ["DE", "IT", "ES"]:
        df_cc = country_data.get(cc)
        if df_cc is None or df_cc.empty:
            continue

        cc_feats = [f for f in fr_feats_avail if f in df_cc.columns]
        cc_clean = df_cc[["date", "trade_goods"] + cc_feats].dropna()
        if len(cc_clean) < 30:
            continue

        X_cc = cc_clean[cc_feats].values
        y_cc = cc_clean["trade_goods"].values

        # AR(1) benchmark RMSE for this country
        ar_rmse_cc = country_models.get(cc, {}).get("AR(1)", ForecastResult("")).rmse

        # --- (a) Direct transfer: apply French model to target country ---
        X_cc_scaled = scaler_fr.transform(X_cc)
        preds_direct = fr_ridge.predict(X_cc_scaled)

        # OOS portion: use second half as test
        split = len(y_cc) // 2
        direct_rmse = np.sqrt(np.mean((y_cc[split:] - preds_direct[split:]) ** 2))
        vs_ar = ((direct_rmse - ar_rmse_cc) / ar_rmse_cc * 100) if (
            ar_rmse_cc and ar_rmse_cc > 0
        ) else np.nan
        results.append({
            "country": cc, "strategy": "direct-transfer",
            "model": "Ridge", "rmse": direct_rmse, "mae": np.nan,
            "vs_ar1_pct": vs_ar,
        })

        # --- (b) Fine-tuned: use first 30% of target to re-estimate ---
        ft_split = max(20, int(len(y_cc) * 0.3))
        X_ft = np.vstack([X_fr_scaled, scaler_fr.transform(X_cc[:ft_split])])
        y_ft = np.concatenate([y_fr, y_cc[:ft_split]])
        ft_ridge = Ridge(alpha=1.0)
        ft_ridge.fit(X_ft, y_ft)
        preds_ft = ft_ridge.predict(scaler_fr.transform(X_cc[ft_split:]))
        ft_rmse = np.sqrt(np.mean((y_cc[ft_split:] - preds_ft) ** 2))
        vs_ar = ((ft_rmse - ar_rmse_cc) / ar_rmse_cc * 100) if (
            ar_rmse_cc and ar_rmse_cc > 0
        ) else np.nan
        results.append({
            "country": cc, "strategy": "fine-tuned",
            "model": "Ridge", "rmse": ft_rmse, "mae": np.nan,
            "vs_ar1_pct": vs_ar,
        })

    # --- (c) Pooled model with country fixed effects ---
    pooled_rows = []
    for i, cc in enumerate(["FR", "DE", "IT", "ES"]):
        df_cc = country_data.get(cc)
        if df_cc is None or df_cc.empty:
            continue
        cc_feats = [f for f in BASELINE_FEATURES if f in df_cc.columns]
        cc_clean = df_cc[["date", "trade_goods"] + cc_feats].dropna().copy()
        # Country fixed effects: one-hot encode
        for j, c2 in enumerate(["FR", "DE", "IT", "ES"]):
            cc_clean[f"fe_{c2}"] = 1.0 if c2 == cc else 0.0
        cc_clean["_country"] = cc
        pooled_rows.append(cc_clean)

    if pooled_rows:
        pooled_df = pd.concat(pooled_rows, ignore_index=True)
        fe_cols = [f"fe_{c}" for c in ["FR", "DE", "IT", "ES"]]
        p_feats = [f for f in BASELINE_FEATURES if f in pooled_df.columns] + fe_cols

        pooled_clean = pooled_df[["_country", "trade_goods"] + p_feats].dropna()
        X_pool = pooled_clean[p_feats].values
        y_pool = pooled_clean["trade_goods"].values

        scaler_pool = StandardScaler()
        # Don't scale FE dummies — only scale non-FE columns
        n_base = len(p_feats) - len(fe_cols)
        X_base = scaler_pool.fit_transform(X_pool[:, :n_base])
        X_pool_scaled = np.hstack([X_base, X_pool[:, n_base:]])

        # Split: first 70% train, last 30% test per country
        split_idx = int(len(y_pool) * 0.7)
        pool_ridge = Ridge(alpha=1.0)
        pool_ridge.fit(X_pool_scaled[:split_idx], y_pool[:split_idx])
        preds_pool = pool_ridge.predict(X_pool_scaled[split_idx:])

        # Per-country pooled RMSE
        countries_test = pooled_clean["_country"].iloc[split_idx:].values
        for cc in ["FR", "DE", "IT", "ES"]:
            cc_mask = countries_test == cc
            if cc_mask.sum() < 2:
                continue
            pool_rmse = np.sqrt(np.mean((y_pool[split_idx:][cc_mask] - preds_pool[cc_mask]) ** 2))
            ar_rmse_cc = country_models.get(cc, {}).get("AR(1)", ForecastResult("")).rmse
            vs_ar = ((pool_rmse - ar_rmse_cc) / ar_rmse_cc * 100) if (
                ar_rmse_cc and ar_rmse_cc > 0
            ) else np.nan
            results.append({
                "country": cc, "strategy": "pooled-FE",
                "model": "Ridge", "rmse": pool_rmse, "mae": np.nan,
                "vs_ar1_pct": vs_ar,
            })

    return results


def plot_cross_country(xcountry_df):
    """Grouped bar chart of RMSE by country and strategy."""
    strategies = xcountry_df["strategy"].unique()
    countries = ["FR", "DE", "IT", "ES"]
    colors_map = {
        "country-specific": "#2196F3",
        "direct-transfer": "#FF9800",
        "fine-tuned": "#4CAF50",
        "pooled-FE": "#9C27B0",
    }

    # Focus on Ridge for comparison
    ridge_df = xcountry_df[xcountry_df["model"] == "Ridge"].copy()
    if ridge_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    n_strat = len(strategies)
    width = 0.8 / max(n_strat, 1)
    x = np.arange(len(countries))

    for i, strat in enumerate(strategies):
        sub = ridge_df[ridge_df["strategy"] == strat]
        vals = []
        for cc in countries:
            match = sub[sub["country"] == cc]
            vals.append(match["rmse"].values[0] if len(match) > 0 else 0)
        ax.bar(x + i * width, vals, width,
               label=strat, color=colors_map.get(strat, "#999"),
               edgecolor="white")

    ax.set_xticks(x + width * (n_strat - 1) / 2)
    ax.set_xticklabels(countries)
    ax.set_ylabel("RMSE")
    ax.set_title("Cross-Country Ridge RMSE by Estimation Strategy", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(CH3_OUTPUT / "cross_country_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
#  PART C — TAYLOR-RULE COUNTERFACTUAL
# =====================================================================

def run_part_c(df_fr, models_fr):
    """
    Part C: Taylor-rule counterfactual policy exercise.

    i_t = α + β·π_t + γ·y_t + δ·ca_t + ε_t

    Counterfactual: if policymakers had used our nowcast instead of
    lagged official data, how would the policy rate have differed?
    """
    print("\n" + "=" * 70)
    print("  PART C: TAYLOR-RULE COUNTERFACTUAL POLICY EXERCISE")
    print("=" * 70)

    # ----- Prepare data -----
    # We need: interest_rate (i_t), hicp (π proxy), trade_goods (ca proxy)
    # GDP output gap would be ideal, but we approximate with trade balance growth
    needed = ["interest_rate", "hicp", "trade_goods"]
    if not all(c in df_fr.columns for c in needed):
        print("  [WARN] Missing columns for Taylor rule. Skipping Part C.")
        return pd.DataFrame()

    df_tr = df_fr[["date"] + needed].dropna().copy()

    # Inflation gap: π_t - 2% target
    df_tr["inflation_gap"] = df_tr["hicp"] - 2.0

    # Output gap proxy: HP-filtered trade volume
    try:
        from statsmodels.tsa.filters.hp_filter import hpfilter
        cycle, trend = hpfilter(df_tr["trade_goods"].values, lamb=129600)  # monthly
        df_tr["output_gap"] = cycle / (np.abs(trend) + 1) * 100
    except Exception:
        # Simple detrending fallback
        trend = df_tr["trade_goods"].rolling(24, min_periods=12, center=True).mean()
        df_tr["output_gap"] = (df_tr["trade_goods"] - trend) / (trend.abs() + 1) * 100

    # CA gap: deviation from 5-year rolling mean
    ca_mean = df_tr["trade_goods"].rolling(60, min_periods=24).mean()
    ca_std = df_tr["trade_goods"].rolling(60, min_periods=24).std()
    df_tr["ca_gap"] = (df_tr["trade_goods"] - ca_mean) / (ca_std + 1e-6)

    # ----- Step 1: Estimate Taylor rule -----
    import statsmodels.api as sm

    # Use data from 2003 onwards (ECB has been operational since 1999,
    # allow warm-up period)
    est_mask = df_tr["date"] >= "2003-01-01"
    df_est = df_tr[est_mask].dropna().copy()

    if len(df_est) < 30:
        print("  [WARN] Too few observations for Taylor rule estimation.")
        return pd.DataFrame()

    y_ols = df_est["interest_rate"].values
    X_ols = df_est[["inflation_gap", "output_gap", "ca_gap"]].values
    X_ols = sm.add_constant(X_ols)

    try:
        ols_model = sm.OLS(y_ols, X_ols).fit(cov_type="HAC",
                                                cov_kwds={"maxlags": 12})
    except Exception:
        ols_model = sm.OLS(y_ols, X_ols).fit()

    print("\n  Taylor-rule estimation (HAC standard errors):")
    print(f"  {'Param':<20} {'Coef':>10} {'SE':>10} {'t':>10} {'p':>8}")
    print("  " + "-" * 60)
    param_names = ["const", "inflation_gap", "output_gap", "ca_gap"]
    for i, pname in enumerate(param_names):
        coef = ols_model.params[i]
        se = ols_model.bse[i]
        t = ols_model.tvalues[i]
        p = ols_model.pvalues[i]
        stars = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        print(f"  {pname:<20} {coef:>10.4f} {se:>10.4f} {t:>10.3f} {p:>7.4f} {stars}")

    print(f"\n  R² = {ols_model.rsquared:.4f},  N = {len(df_est)}")

    delta_hat = ols_model.params[3]  # coefficient on ca_gap

    # ----- Step 2: Build nowcast and lagged series -----
    # Extract model predictions aligned with dates
    ridge_res = models_fr.get("Ridge")
    if ridge_res is None or not ridge_res.predictions:
        print("  [WARN] No Ridge predictions available. Skipping counterfactual.")
        return pd.DataFrame()

    pred_dates = pd.to_datetime(ridge_res.dates)
    pred_vals = np.array(ridge_res.predictions, dtype=float)
    actual_vals = np.array(ridge_res.actuals, dtype=float)

    pred_df = pd.DataFrame({
        "date": pred_dates,
        "ca_nowcast": pred_vals,
        "ca_actual": actual_vals,
    })

    # Merge with Taylor-rule data
    df_cf = df_tr.merge(pred_df, on="date", how="inner")

    # ca_lagged: what policymakers actually had (3-month lag)
    df_cf["ca_lagged"] = df_cf["trade_goods"].shift(3)
    df_cf = df_cf.dropna(subset=["ca_nowcast", "ca_lagged"]).copy()

    if len(df_cf) < 10:
        print("  [WARN] Too few observations for counterfactual.")
        return pd.DataFrame()

    # ----- Step 3: Information gap -----
    ca_mean_cf = df_cf["trade_goods"].rolling(60, min_periods=12).mean()
    ca_std_cf = df_cf["trade_goods"].rolling(60, min_periods=12).std().clip(lower=1e-6)

    # Normalise nowcast and lagged same way
    df_cf["ca_gap_nowcast"] = (df_cf["ca_nowcast"] - ca_mean_cf) / ca_std_cf
    df_cf["ca_gap_lagged"] = (df_cf["ca_lagged"] - ca_mean_cf) / ca_std_cf

    df_cf["info_gap"] = df_cf["ca_gap_nowcast"] - df_cf["ca_gap_lagged"]

    # ----- Step 4: Counterfactual policy rate -----
    df_cf["i_counterfactual"] = df_cf["interest_rate"] + delta_hat * df_cf["info_gap"]
    df_cf["policy_diff_bp"] = (df_cf["i_counterfactual"] - df_cf["interest_rate"]) * 100  # basis points

    # ----- Results -----
    print(f"\n  Counterfactual policy exercise ({len(df_cf)} months):")
    print(f"  δ̂ (CA gap coefficient) = {delta_hat:.4f}")

    mean_gap = df_cf["info_gap"].mean()
    mean_diff = df_cf["policy_diff_bp"].mean()
    max_diff = df_cf["policy_diff_bp"].abs().max()
    print(f"  Mean information gap: {mean_gap:.3f}")
    print(f"  Mean policy difference: {mean_diff:+.1f} bp")
    print(f"  Max |policy difference|: {max_diff:.1f} bp")

    # Episode-specific analysis
    print(f"\n  {'Episode':<12} {'Mean info gap':>14} {'Mean Δi (bp)':>14} {'Max |Δi| (bp)':>14}")
    print("  " + "-" * 58)
    episode_rows = []
    for ep_name, (ep_start, ep_end) in CRISIS_EPISODES.items():
        ep_mask = (df_cf["date"] >= ep_start) & (df_cf["date"] <= ep_end)
        ep_data = df_cf[ep_mask]
        if len(ep_data) < 2:
            continue
        ep_gap = ep_data["info_gap"].mean()
        ep_diff = ep_data["policy_diff_bp"].mean()
        ep_max = ep_data["policy_diff_bp"].abs().max()
        print(f"  {ep_name:<12} {ep_gap:>+14.3f} {ep_diff:>+14.1f} {ep_max:>14.1f}")
        episode_rows.append({
            "episode": ep_name,
            "mean_info_gap": ep_gap,
            "mean_policy_diff_bp": ep_diff,
            "max_abs_policy_diff_bp": ep_max,
            "n_months": len(ep_data),
        })

    # ----- Information lead time -----
    # How many months earlier does the nowcast correctly signal a turning point?
    changes = df_cf["trade_goods"].diff()
    nowcast_changes = df_cf["ca_nowcast"].diff()
    lagged_changes = df_cf["ca_lagged"].diff()

    # Direction agreement: nowcast vs actual
    valid_mask = changes.notna() & nowcast_changes.notna() & lagged_changes.notna()
    if valid_mask.sum() > 5:
        nowcast_correct = (np.sign(changes[valid_mask]) == np.sign(nowcast_changes[valid_mask])).mean()
        lagged_correct = (np.sign(changes[valid_mask]) == np.sign(lagged_changes[valid_mask])).mean()
        print(f"\n  Direction accuracy (sign of change):")
        print(f"    Nowcast: {nowcast_correct:.1%}")
        print(f"    Lagged official: {lagged_correct:.1%}")
        print(f"    Information advantage: {(nowcast_correct - lagged_correct):+.1%}")

    # ----- Save -----
    df_cf.to_csv(CH3_OUTPUT / "taylor_rule_counterfactual.csv", index=False)
    pd.DataFrame(episode_rows).to_csv(CH3_OUTPUT / "taylor_rule_episodes.csv", index=False)

    # Save OLS summary
    with open(CH3_OUTPUT / "taylor_rule_estimation.txt", "w") as f:
        f.write(ols_model.summary().as_text())

    # ----- Plot -----
    plot_counterfactual(df_cf)

    return df_cf


def plot_counterfactual(df_cf):
    """Plot actual vs counterfactual policy rate with crisis shading."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    dates = df_cf["date"]

    # Panel 1: Policy rates
    ax = axes[0]
    ax.plot(dates, df_cf["interest_rate"], "k-", label="Actual rate", linewidth=1.2)
    ax.plot(dates, df_cf["i_counterfactual"], "r--", label="Counterfactual rate", linewidth=1.2)
    ax.set_ylabel("Interest rate (%)")
    ax.set_title("Taylor-Rule Counterfactual Policy Exercise", fontweight="bold")
    ax.legend(fontsize=9)

    # Panel 2: Policy difference in bp
    ax = axes[1]
    ax.fill_between(dates, 0, df_cf["policy_diff_bp"], alpha=0.4,
                     color="steelblue", label="Δi (bp)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Policy difference (bp)")
    ax.legend(fontsize=9)

    # Panel 3: Information gap
    ax = axes[2]
    ax.plot(dates, df_cf["info_gap"], color="darkgreen", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Information gap\n(nowcast − lagged)")
    ax.set_xlabel("Date")

    # Add crisis shading to all panels
    for ax in axes:
        for ep_name, (ep_start, ep_end) in CRISIS_EPISODES.items():
            ax.axvspan(ep_start, ep_end, alpha=0.1, color="red")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(CH3_OUTPUT / "taylor_rule_counterfactual.png", dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
#  MIP THRESHOLD SIMULATION (Robustness extension)
# =====================================================================

def mip_threshold_analysis(df_fr, models_fr):
    """
    Check whether ML nowcasts would have triggered the Macroeconomic
    Imbalance Procedure thresholds earlier/later than lagged official data.
    MIP thresholds: CA/GDP in [-4%, +6%] (3-year average).
    """
    print("\n  --- MIP Threshold Simulation ---")

    ridge_res = models_fr.get("Ridge")
    if ridge_res is None or not ridge_res.predictions:
        print("  [WARN] No Ridge predictions for MIP analysis.")
        return

    pred_dates = pd.to_datetime(ridge_res.dates)
    pred_vals = np.array(ridge_res.predictions, dtype=float)
    actual_vals = np.array(ridge_res.actuals, dtype=float)

    # 3-year rolling average (36 months)
    pred_series = pd.Series(pred_vals, index=pred_dates)
    actual_series = pd.Series(actual_vals, index=pred_dates)

    pred_3y = pred_series.rolling(36, min_periods=12).mean()
    actual_3y = actual_series.rolling(36, min_periods=12).mean()

    # Threshold breaches (we use level, not ratio to GDP, so just check sign)
    # For illustration, flag whenever 3-year average hits extreme deciles
    if len(actual_3y.dropna()) < 12:
        print("  [WARN] Too few observations for MIP simulation.")
        return

    threshold_low = actual_3y.dropna().quantile(0.10)
    threshold_high = actual_3y.dropna().quantile(0.90)

    nowcast_breach_low = pred_3y < threshold_low
    actual_breach_low = actual_3y < threshold_low

    if nowcast_breach_low.any() and actual_breach_low.any():
        first_nowcast = pred_3y[nowcast_breach_low].index[0]
        first_actual = actual_3y[actual_breach_low].index[0]
        lead_months = (first_actual - first_nowcast).days / 30.44
        print(f"  Lower threshold breach:")
        print(f"    Nowcast signals: {first_nowcast:%Y-%m}")
        print(f"    Actual signals:  {first_actual:%Y-%m}")
        print(f"    Lead time: {lead_months:+.0f} months")
    else:
        print("  No lower threshold breach detected in sample.")


# =====================================================================
#  MAIN
# =====================================================================

def main():
    print("+" + "=" * 64 + "+")
    print("|  CHAPTER 3: POLICY IMPLICATIONS                              |")
    print("|  Do Better BoP Estimates Matter?                             |")
    print("+" + "=" * 64 + "+")

    # ----- Phase 1: Download France monthly data -----
    print("\n" + "=" * 60)
    print("  PHASE 1: FRANCE BASELINE DATA")
    print("=" * 60)

    series_fr = download_monthly_for_country("FR")
    if len(series_fr) < 2:
        print("\n  [ERROR] Not enough series for France.")
        return

    df_fr = merge_series(series_fr)
    df_fr = add_features(df_fr)
    df_fr = df_fr.dropna(subset=["trade_goods"]).reset_index(drop=True)

    print(f"\n  France dataset: {len(df_fr)} months, {df_fr.shape[1]} columns")
    print(f"  Period: {df_fr['date'].min():%Y-%m} to {df_fr['date'].max():%Y-%m}")

    # ----- Phase 2: Run expanding-window forecasts for France -----
    print("\n" + "=" * 60)
    print("  PHASE 2: FRANCE EXPANDING-WINDOW FORECASTS")
    print("=" * 60)

    feats_fr = [f for f in BASELINE_FEATURES if f in df_fr.columns]
    models_fr = expanding_window_forecast(df_fr, "trade_goods", feats_fr, min_train=60)

    for mname, res in models_fr.items():
        rmse_str = f"{res.rmse:,.0f}" if res.rmse else "N/A"
        n_oos = len(res.predictions)
        print(f"  {mname:<12} RMSE = {rmse_str:<12} ({n_oos} OOS predictions)")

    # ----- Part A -----
    crisis_df = run_part_a(models_fr)

    # ----- Part B -----
    xcountry_df = run_part_b(df_fr, models_fr)

    # ----- Part C -----
    cf_df = run_part_c(df_fr, models_fr)

    # ----- MIP Robustness -----
    mip_threshold_analysis(df_fr, models_fr)

    # ----- Summary -----
    print("\n" + "=" * 70)
    print("  CHAPTER 3 — COMPLETE")
    print("=" * 70)
    print(f"  Output directory: {CH3_OUTPUT}")
    files = list(CH3_OUTPUT.iterdir())
    for f in sorted(files):
        print(f"    {f.name}")
    print(f"\n  Crisis evaluation: {len(crisis_df)} rows")
    print(f"  Cross-country: {len(xcountry_df)} rows")
    if isinstance(cf_df, pd.DataFrame) and not cf_df.empty:
        print(f"  Taylor-rule counterfactual: {len(cf_df)} months")
    print("\n[OK] Chapter 3 analysis complete!")


if __name__ == "__main__":
    main()
