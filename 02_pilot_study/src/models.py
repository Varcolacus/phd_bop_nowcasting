"""
Baseline & ML Models for BoP Nowcasting Pilot Study
=====================================================

Implements:
  1. AR(1) baseline — simple autoregressive benchmark
  2. AR(4) baseline — richer autoregressive benchmark
  3. OLS with macro features — traditional econometric approach
  4. Gradient Boosting (scikit-learn) — ML benchmark
  5. XGBoost — primary ML model

All models are evaluated using expanding-window pseudo real-time
out-of-sample forecasting.

Author: PhD Pilot Study
Date: March 2026
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from dataclasses import dataclass, field

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import statsmodels.api as sm

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_modeling_data():
    """Load the prepared modeling dataset."""
    path = DATA_DIR / "modeling_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Modeling dataset not found at {path}. "
            "Run 01_download_data.py first."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    return df


# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    """Stores results from a single model's evaluation."""
    model_name: str
    predictions: list = field(default_factory=list)
    actuals: list = field(default_factory=list)
    dates: list = field(default_factory=list)
    rmse: float = None
    mae: float = None
    direction_accuracy: float = None


def ar_forecast(y_train, order=1):
    """
    Fit an AR(p) model and return 1-step-ahead forecast.
    Uses statsmodels OLS for simplicity and robustness.
    """
    if len(y_train) < order + 2:
        return np.nan

    # Build lagged features manually for robustness
    y = np.array(y_train)
    X = np.column_stack([y[order - i - 1: len(y) - i - 1] for i in range(order)])
    y_target = y[order:]

    if len(y_target) < 2:
        return np.nan

    X = sm.add_constant(X)
    try:
        model = sm.OLS(y_target, X).fit()
        # Forecast: use the last 'order' observations
        last_obs = np.array([y[-(i + 1)] for i in range(order)])
        x_new = np.concatenate([[1.0], last_obs])
        return model.predict(x_new)[0]
    except Exception:
        return np.nan


def ols_forecast(X_train, y_train, X_test_row):
    """
    Ridge regression with macro features. 1-step-ahead forecast.
    Uses Ridge (L2 penalty) to avoid overfitting when features > observations.
    """
    from sklearn.linear_model import RidgeCV

    if len(y_train) < 10:
        return np.nan

    try:
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        model.fit(X_train, y_train)
        return model.predict(X_test_row.reshape(1, -1))[0]
    except Exception:
        return np.nan


def gradient_boosting_forecast(X_train, y_train, X_test_row):
    """
    Scikit-learn Gradient Boosting. 1-step-ahead forecast.
    """
    if len(y_train) < 10:
        return np.nan

    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=5,
        random_state=42,
    )
    try:
        model.fit(X_train, y_train)
        return model.predict(X_test_row.reshape(1, -1))[0]
    except Exception:
        return np.nan


def xgboost_forecast(X_train, y_train, X_test_row):
    """
    XGBoost regression. 1-step-ahead forecast.
    """
    if len(y_train) < 10:
        return np.nan

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    try:
        model.fit(X_train, y_train)
        return model.predict(X_test_row.reshape(1, -1))[0]
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Expanding Window Evaluation
# ---------------------------------------------------------------------------

def expanding_window_evaluation(df, target_col, feature_cols,
                                 min_train_size=40, step=1):
    """
    Pseudo real-time expanding window out-of-sample evaluation.

    Parameters:
    -----------
    df : DataFrame with date, target, and feature columns
    target_col : str, name of the target variable (BoP component)
    feature_cols : list of str, names of macro feature columns
    min_train_size : int, minimum number of observations before
                     starting out-of-sample evaluation
    step : int, step size for expanding window

    Returns:
    --------
    dict of ForecastResult objects, one per model
    """
    print(f"\nRunning expanding window evaluation...")
    print(f"   Target: {target_col}")
    print(f"   Features: {len(feature_cols)} variables")
    print(f"   Min training size: {min_train_size} quarters")

    # Prepare data — drop rows where target or features are NaN
    cols_needed = [target_col] + feature_cols
    df_clean = df[["date"] + cols_needed].dropna().reset_index(drop=True)

    if len(df_clean) < min_train_size + 5:
        print(f"   [WARN] Not enough data ({len(df_clean)} obs). Need {min_train_size + 5}.")
        return {}

    y = df_clean[target_col].values
    X = df_clean[feature_cols].values
    dates = df_clean["date"].values

    n = len(y)
    n_test = n - min_train_size

    print(f"   Total observations: {n}")
    print(f"   Out-of-sample periods: {n_test}")

    # Initialize results for each model
    models = {
        "AR(1)": ForecastResult("AR(1)"),
        "AR(4)": ForecastResult("AR(4)"),
        "Ridge": ForecastResult("Ridge"),
        "GradientBoosting": ForecastResult("GradientBoosting"),
        "XGBoost": ForecastResult("XGBoost"),
    }

    # Scale features for ML models
    scaler = StandardScaler()

    for t in range(min_train_size, n, step):
        # Training data: everything up to time t
        y_train = y[:t]
        X_train = X[:t]

        # Test observation: time t
        y_test = y[t]
        X_test = X[t]
        test_date = dates[t]

        # Scale features (fit on training data only)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test.reshape(1, -1)).flatten()

        # --- AR(1) ---
        pred_ar1 = ar_forecast(y_train, order=1)
        models["AR(1)"].predictions.append(pred_ar1)
        models["AR(1)"].actuals.append(y_test)
        models["AR(1)"].dates.append(test_date)

        # --- AR(4) ---
        pred_ar4 = ar_forecast(y_train, order=4)
        models["AR(4)"].predictions.append(pred_ar4)
        models["AR(4)"].actuals.append(y_test)
        models["AR(4)"].dates.append(test_date)

        # --- Ridge ---
        pred_ols = ols_forecast(X_train_scaled, y_train, X_test_scaled)
        models["Ridge"].predictions.append(pred_ols)
        models["Ridge"].actuals.append(y_test)
        models["Ridge"].dates.append(test_date)

        # --- Gradient Boosting ---
        pred_gb = gradient_boosting_forecast(X_train_scaled, y_train, X_test_scaled)
        models["GradientBoosting"].predictions.append(pred_gb)
        models["GradientBoosting"].actuals.append(y_test)
        models["GradientBoosting"].dates.append(test_date)

        # --- XGBoost ---
        pred_xgb = xgboost_forecast(X_train_scaled, y_train, X_test_scaled)
        models["XGBoost"].predictions.append(pred_xgb)
        models["XGBoost"].actuals.append(y_test)
        models["XGBoost"].dates.append(test_date)

    # Compute metrics
    for name, res in models.items():
        preds = np.array(res.predictions, dtype=float)
        acts = np.array(res.actuals, dtype=float)

        # Remove NaN predictions
        valid = ~np.isnan(preds) & ~np.isnan(acts)
        if valid.sum() < 2:
            continue

        preds_v = preds[valid]
        acts_v = acts[valid]

        res.rmse = np.sqrt(mean_squared_error(acts_v, preds_v))
        res.mae = mean_absolute_error(acts_v, preds_v)

        # Direction accuracy: did the model predict the correct sign of change?
        if len(acts_v) > 1:
            actual_dir = np.diff(acts_v) > 0
            pred_dir = np.diff(preds_v) > 0
            res.direction_accuracy = np.mean(actual_dir == pred_dir) * 100

    return models


# ---------------------------------------------------------------------------
# Diebold-Mariano Test
# ---------------------------------------------------------------------------

def diebold_mariano_test(e1, e2, h=1):
    """
    Diebold-Mariano test for equal predictive accuracy.

    H0: Both forecasts have equal accuracy
    H1: Forecast 1 is more accurate than Forecast 2

    Parameters:
    -----------
    e1 : array, forecast errors from model 1
    e2 : array, forecast errors from model 2
    h : int, forecast horizon

    Returns:
    --------
    DM statistic and p-value
    """
    from scipy import stats

    d = e1 ** 2 - e2 ** 2  # Loss differential (squared errors)
    n = len(d)

    if n < 5:
        return np.nan, np.nan

    d_mean = np.mean(d)

    # Newey-West type variance estimator
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = gamma_0

    for k in range(1, h):
        if k < n:
            gamma_k = np.cov(d[k:], d[:-k])[0, 1]
            gamma_sum += 2 * gamma_k

    var_d = gamma_sum / n

    if var_d <= 0:
        return np.nan, np.nan

    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1))

    return dm_stat, p_value


# ---------------------------------------------------------------------------
# Results Summary
# ---------------------------------------------------------------------------

def print_results(models, benchmark="AR(1)"):
    """
    Print a summary table of model performance.
    """
    print("\n" + "=" * 70)
    print("  RESULTS: OUT-OF-SAMPLE FORECAST ACCURACY")
    print("=" * 70)

    print(f"\n  {'Model':<20} {'RMSE':>10} {'MAE':>10} {'Dir. Acc.':>10} {'vs AR(1)':>10}")
    print("  " + "-" * 60)

    benchmark_rmse = models.get(benchmark, ForecastResult(benchmark)).rmse

    for name, res in models.items():
        rmse_str = f"{res.rmse:.2f}" if res.rmse else "N/A"
        mae_str = f"{res.mae:.2f}" if res.mae else "N/A"
        dir_str = f"{res.direction_accuracy:.1f}%" if res.direction_accuracy else "N/A"

        # Relative RMSE vs benchmark
        if res.rmse and benchmark_rmse:
            rel = ((res.rmse - benchmark_rmse) / benchmark_rmse) * 100
            rel_str = f"{rel:+.1f}%"
        else:
            rel_str = "—"

        print(f"  {name:<20} {rmse_str:>10} {mae_str:>10} {dir_str:>10} {rel_str:>10}")

    # Diebold-Mariano tests vs AR(1)
    print(f"\n  Diebold-Mariano Test vs {benchmark}:")
    print(f"  {'Model':<20} {'DM Stat':>10} {'p-value':>10} {'Significant':>12}")
    print("  " + "-" * 52)

    bench_res = models.get(benchmark)
    if bench_res:
        bench_errors = np.array(bench_res.actuals) - np.array(bench_res.predictions)
        bench_valid = ~np.isnan(bench_errors)

        for name, res in models.items():
            if name == benchmark:
                continue

            model_errors = np.array(res.actuals) - np.array(res.predictions)
            model_valid = ~np.isnan(model_errors)

            # Align valid observations
            both_valid = bench_valid & model_valid
            if both_valid.sum() < 5:
                continue

            dm_stat, p_val = diebold_mariano_test(
                model_errors[both_valid],
                bench_errors[both_valid]
            )

            if not np.isnan(dm_stat):
                sig = "Yes *" if p_val < 0.10 else ("Yes **" if p_val < 0.05 else "No")
                print(f"  {name:<20} {dm_stat:>10.3f} {p_val:>10.4f} {sig:>12}")

    print("\n" + "=" * 70)


def save_results(models):
    """Save forecast results to CSV."""
    rows = []
    for name, res in models.items():
        for i in range(len(res.predictions)):
            rows.append({
                "model": name,
                "date": res.dates[i],
                "actual": res.actuals[i],
                "predicted": res.predictions[i],
            })

    df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "forecast_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nForecast results saved to {out_path}")

    # Summary metrics
    summary_rows = []
    for name, res in models.items():
        summary_rows.append({
            "model": name,
            "rmse": res.rmse,
            "mae": res.mae,
            "direction_accuracy": res.direction_accuracy,
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "model_comparison.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Model comparison saved to {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the full pilot study pipeline."""
    print("=" * 70)
    print("  BOP NOWCASTING PILOT STUDY")
    print("  AI vs. Econometric Baselines — Expanding Window Evaluation")
    print("=" * 70)

    # Load data
    df = load_modeling_data()
    print(f"\nLoaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"   Period: {df['date'].min()} to {df['date'].max()}")

    # Identify target and features
    # Target: BoP current account (or trade balance)
    # Strategy: use the first BoP-related column as target
    potential_targets = [c for c in df.columns
                        if "bop" in c.lower() and "yoy" not in c and "qoq" not in c]

    if not potential_targets:
        # Fallback: use the first numeric column that's not a date
        potential_targets = [c for c in df.columns
                            if c != "date" and df[c].dtype in [np.float64, np.int64]
                            and "yoy" not in c and "qoq" not in c]

    if not potential_targets:
        print("[ERROR] No suitable target variable found.")
        return

    target_col = potential_targets[0]

    # Features: all other numeric columns (levels + transformations)
    feature_cols = [c for c in df.columns
                    if c != "date" and c != target_col
                    and df[c].dtype in [np.float64, np.int64, float]]

    # Also add lagged target as a feature (if not already present)
    for lag_name, lag_val in [(f"{target_col}_lag1", 1), (f"{target_col}_lag4", 4)]:
        if lag_name not in df.columns:
            df[lag_name] = df[target_col].shift(lag_val)
        if lag_name not in feature_cols:
            feature_cols.append(lag_name)

    print(f"\n   Target variable: {target_col}")
    print(f"   Feature variables: {len(feature_cols)}")
    for f in feature_cols[:10]:
        print(f"     - {f}")
    if len(feature_cols) > 10:
        print(f"     ... and {len(feature_cols) - 10} more")

    # Run expanding window evaluation
    models = expanding_window_evaluation(
        df, target_col, feature_cols,
        min_train_size=40,
        step=1
    )

    if not models:
        print("[ERROR] Evaluation failed. Not enough data.")
        return

    # Print results
    print_results(models)

    # Save results
    save_results(models)

    print("\n[OK] Pilot study complete!")


if __name__ == "__main__":
    main()
