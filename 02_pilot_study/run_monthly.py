"""
Monthly Frequency Extension — BoP Nowcasting Pilot Study
==========================================================

Runs the same expanding-window model comparison from run_pilot.py
but at **monthly** frequency, using monthly BoP proxies and indicators
downloaded from the ECB Statistical Data Warehouse.

Because official BoP data are quarterly, this script uses monthly
*trade in goods* data (Eurostat/ECB) as the target, together with
monthly macro indicators (HICP, EUR/USD, industrial production proxy).

Usage:
  python run_monthly.py

Author: PhD Pilot Study
Date: March 2026
"""

import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

import pandas as pd
import numpy as np
import requests
import warnings
from io import StringIO

from download_data import (
    ECB_BASE, VERIFY_SSL, DATA_DIR,
    ecb_download_series, ecb_parse_csv,
)
from models import (
    ForecastResult, ar_forecast, ols_forecast, lasso_forecast,
    gradient_boosting_forecast, xgboost_forecast, lstm_forecast,
    diebold_mariano_test, print_results, save_results,
    shap_feature_importance, OUTPUT_DIR,
    HAS_TORCH, HAS_SHAP,
    StandardScaler,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*convergence.*")

MONTHLY_OUTPUT = OUTPUT_DIR / "monthly"
MONTHLY_OUTPUT.mkdir(exist_ok=True)


# -----------------------------------------------------------------------
# Monthly series definitions
# -----------------------------------------------------------------------
# Monthly trade in goods for France (BPM6 methodology)
# This is the closest monthly proxy available from the ECB SDW for a BoP
# component.  We also pull the same monthly indicators already used in
# the quarterly pipeline (EUR/USD, HICP, Euribor) but keep them monthly.

MONTHLY_SERIES = {
    # Monthly trade balance (goods) — BPM6
    "trade_goods": ("BP6", "M.N.FR.W1.S1.S1.T.B.G._Z._Z._Z.EUR._T._X.N",
                    "M", "Monthly goods balance"),
    # EUR/USD exchange rate
    "eurusd":      ("EXR", "M.USD.EUR.SP00.A",
                    "M", "EUR/USD exchange rate"),
    # France HICP — annual rate of change (monthly)
    "hicp":        ("ICP", "M.FR.N.000000.4.ANR",
                    "M", "France HICP (annual rate)"),
    # Euribor 3-month rate (monthly)
    "interest_rate": ("FM", "M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA",
                      "M", "Euribor 3-month rate"),
}


def download_monthly_data():
    """Download monthly series from the ECB SDW."""
    print("\n" + "=" * 60)
    print("  MONTHLY DATA PIPELINE")
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
        df.to_csv(DATA_DIR / f"monthly_{name}_raw.csv", index=False)
        results[name] = df
        print(f"    [OK] {len(df)} monthly obs "
              f"({df['date'].min():%Y-%m} to {df['date'].max():%Y-%m})")
    return results


def merge_monthly(series_dict):
    """Merge monthly series into a single wide DataFrame."""
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


def prepare_monthly_features(df, target="trade_goods"):
    """Add lags and derived features for the monthly dataset."""
    if target in df.columns:
        df[f"{target}_lag1"] = df[target].shift(1)
        df[f"{target}_lag3"] = df[target].shift(3)
        df[f"{target}_lag12"] = df[target].shift(12)
        df[f"{target}_mom"] = df[target].diff(1)   # month-on-month
        df[f"{target}_yoy"] = df[target].diff(12)   # year-on-year
    for col in ["eurusd", "hicp", "interest_rate"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)
    df = df.dropna().reset_index(drop=True)
    return df


def monthly_expanding_window(df, target_col, feature_cols,
                              min_train_size=60, step=1):
    """Expanding-window evaluation identical to quarterly version."""
    print(f"\nRunning monthly expanding window evaluation...")
    print(f"   Target: {target_col}")
    print(f"   Features: {len(feature_cols)} variables")
    print(f"   Min training size: {min_train_size} months")

    cols_needed = [target_col] + feature_cols
    df_clean = df[["date"] + cols_needed].dropna().reset_index(drop=True)

    if len(df_clean) < min_train_size + 5:
        print(f"   [WARN] Not enough data ({len(df_clean)} obs).")
        return {}

    y = df_clean[target_col].values
    X = df_clean[feature_cols].values
    dates = df_clean["date"].values
    n = len(y)

    print(f"   Total observations: {n}")
    print(f"   Out-of-sample periods: {n - min_train_size}")

    models = {
        "AR(1)":  ForecastResult("AR(1)"),
        "AR(12)": ForecastResult("AR(12)"),
        "Ridge":  ForecastResult("Ridge"),
        "LASSO":  ForecastResult("LASSO"),
        "GradientBoosting": ForecastResult("GradientBoosting"),
        "XGBoost": ForecastResult("XGBoost"),
    }
    if HAS_TORCH:
        models["LSTM"] = ForecastResult("LSTM")

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

        # AR(12)
        p = ar_forecast(y_train, order=12)
        models["AR(12)"].predictions.append(p)
        models["AR(12)"].actuals.append(y_test)
        models["AR(12)"].dates.append(test_date)

        # Ridge
        p = ols_forecast(X_train_scaled, y_train, X_test_scaled)
        models["Ridge"].predictions.append(p)
        models["Ridge"].actuals.append(y_test)
        models["Ridge"].dates.append(test_date)

        # LASSO
        p = lasso_forecast(X_train_scaled, y_train, X_test_scaled)
        models["LASSO"].predictions.append(p)
        models["LASSO"].actuals.append(y_test)
        models["LASSO"].dates.append(test_date)

        # GB
        p = gradient_boosting_forecast(X_train_scaled, y_train, X_test_scaled)
        models["GradientBoosting"].predictions.append(p)
        models["GradientBoosting"].actuals.append(y_test)
        models["GradientBoosting"].dates.append(test_date)

        # XGBoost
        p = xgboost_forecast(X_train_scaled, y_train, X_test_scaled)
        models["XGBoost"].predictions.append(p)
        models["XGBoost"].actuals.append(y_test)
        models["XGBoost"].dates.append(test_date)

        # LSTM
        if HAS_TORCH and "LSTM" in models:
            p = lstm_forecast(X_train_scaled, y_train, X_test_scaled,
                              lookback=6, epochs=80)
            models["LSTM"].predictions.append(p)
            models["LSTM"].actuals.append(y_test)
            models["LSTM"].dates.append(test_date)

    # Compute metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    for name, res in models.items():
        preds = np.array(res.predictions, dtype=float)
        acts = np.array(res.actuals, dtype=float)
        valid = ~np.isnan(preds) & ~np.isnan(acts)
        if valid.sum() < 2:
            continue
        preds_v = preds[valid]
        acts_v = acts[valid]
        res.rmse = np.sqrt(mean_squared_error(acts_v, preds_v))
        res.mae = mean_absolute_error(acts_v, preds_v)
        if len(acts_v) > 1:
            actual_dir = np.diff(acts_v) > 0
            pred_dir = np.diff(preds_v) > 0
            res.direction_accuracy = np.mean(actual_dir == pred_dir) * 100

    return models


def save_monthly_results(models):
    """Save monthly results to the monthly/ output subfolder."""
    rows = []
    for name, res in models.items():
        for i in range(len(res.predictions)):
            rows.append({
                "model": name,
                "date": res.dates[i],
                "actual": res.actuals[i],
                "predicted": res.predictions[i],
            })
    pd.DataFrame(rows).to_csv(MONTHLY_OUTPUT / "forecast_results.csv", index=False)

    summary = []
    for name, res in models.items():
        summary.append({
            "model": name, "rmse": res.rmse,
            "mae": res.mae, "direction_accuracy": res.direction_accuracy,
        })
    pd.DataFrame(summary).to_csv(MONTHLY_OUTPUT / "model_comparison.csv", index=False)
    print(f"\nMonthly results saved to {MONTHLY_OUTPUT}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    print("+" + "=" * 58 + "+")
    print("|  MONTHLY FREQUENCY EXTENSION — BoP Nowcasting Pilot     |")
    print("+" + "=" * 58 + "+")

    # Step 1: Download monthly data
    series = download_monthly_data()
    if len(series) < 2:
        print("\n[ERROR] Not enough monthly series downloaded.")
        return

    # Step 2: Merge & prepare
    df = merge_monthly(series)
    if df.empty or "trade_goods" not in df.columns:
        print("\n[ERROR] No trade_goods target in monthly data.")
        return

    target_col = "trade_goods"
    df = prepare_monthly_features(df, target=target_col)

    print(f"\nMonthly dataset: {df.shape[0]} months x {df.shape[1]} columns")
    print(f"   Period: {df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}")

    # Features: exclude target-contemporaneous transforms
    leakage = [f"{target_col}_mom", f"{target_col}_yoy"]
    feature_cols = [c for c in df.columns
                    if c != "date" and c != target_col
                    and c not in leakage
                    and df[c].dtype in [np.float64, np.int64, float]]

    print(f"   Target: {target_col}")
    print(f"   Features ({len(feature_cols)}):")
    for f in feature_cols:
        print(f"     - {f}")

    # Step 3: Expanding window
    models = monthly_expanding_window(df, target_col, feature_cols,
                                       min_train_size=60, step=1)
    if not models:
        print("[ERROR] Monthly evaluation failed.")
        return

    # Step 4: Print & save
    print_results(models, benchmark="AR(1)")
    save_monthly_results(models)

    # Step 5: SHAP
    if HAS_SHAP:
        print("\n" + "=" * 60)
        print("  SHAP FEATURE IMPORTANCE (Monthly XGBoost)")
        print("=" * 60)
        try:
            # re-use the quarterly SHAP function but direct output
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import xgboost as xgb
            import shap

            cols_needed = [target_col] + feature_cols
            df_c = df[["date"] + cols_needed].dropna().reset_index(drop=True)
            from sklearn.preprocessing import StandardScaler as SS
            sc = SS()
            X_s = sc.fit_transform(df_c[feature_cols].values)
            y_s = df_c[target_col].values
            m = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                                  min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
                                  random_state=42, verbosity=0)
            m.fit(X_s, y_s)
            sv = shap.TreeExplainer(m).shap_values(X_s)
            mean_abs = np.abs(sv).mean(axis=0)
            order = np.argsort(mean_abs)[::-1]
            top_n = min(15, len(feature_cols))
            top_idx = order[:top_n]
            fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
            ax.barh(range(top_n), mean_abs[top_idx][::-1], color="#9C27B0", edgecolor="white")
            ax.set_yticks(range(top_n))
            ax.set_yticklabels([feature_cols[i] for i in top_idx][::-1])
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title("Monthly XGBoost Feature Importance (SHAP)", fontweight="bold")
            plt.tight_layout()
            fig.savefig(MONTHLY_OUTPUT / "shap_importance_monthly.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("  Saved shap_importance_monthly.png")
            print(f"\n  Top {top_n} monthly features by mean |SHAP value|:")
            for rank, idx in enumerate(top_idx, 1):
                print(f"    {rank:>2}. {feature_cols[idx]:<25} {mean_abs[idx]:.2f}")
        except Exception as e:
            print(f"  [WARN] Monthly SHAP failed: {e}")

    print("\n" + "=" * 60)
    print("  [OK] MONTHLY EXTENSION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
