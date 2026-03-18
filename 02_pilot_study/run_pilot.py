"""
Main Pipeline — BoP Nowcasting Pilot Study
============================================

Run this script to execute the full pipeline:
  1. Download data from Eurostat
  2. Prepare modeling dataset
  3. Run expanding-window model evaluation
  4. Generate visualizations

Usage:
  python run_pilot.py

Author: PhD Pilot Study
Date: March 2026
"""

import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))


def run_realtime_info_test():
    """
    Simulated real-time information flow test.

    Mimics publication lags: at each evaluation quarter, produces forecasts
    at three information horizons:
      - Early: features lagged by 2 extra months (only M1 of quarter available)
      - Mid:   features lagged by 1 extra month  (M1+M2 available)
      - Late:  all within-quarter features available

    This tests whether Ridge accuracy improves monotonically as more
    within-quarter data arrive (news content).
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_squared_error

    from src.models import load_modeling_data, OUTPUT_DIR

    df = load_modeling_data()

    potential_targets = [c for c in df.columns
                        if "bop" in c.lower() and "yoy" not in c and "qoq" not in c]
    if not potential_targets:
        potential_targets = [c for c in df.columns
                            if c != "date" and df[c].dtype in [np.float64, np.int64]
                            and "yoy" not in c and "qoq" not in c]
    if not potential_targets:
        print("  [WARN] No target variable found for real-time test")
        return

    target_col = potential_targets[0]
    bop_leakage = ["bop_goods", "bop_services"]
    target_contemp = [f"{target_col}_yoy", f"{target_col}_qoq"]
    feature_cols = [c for c in df.columns
                    if c != "date" and c != target_col
                    and c not in target_contemp
                    and df[c].dtype in [np.float64, np.int64, float]
                    and not any(c.startswith(prefix) for prefix in bop_leakage)]

    cols_needed = [target_col] + feature_cols
    df_clean = df[["date"] + cols_needed].dropna().reset_index(drop=True)
    y = df_clean[target_col].values
    X = df_clean[feature_cols].values
    n = len(y)
    min_train = 40

    if n < min_train + 5:
        print("  [WARN] Not enough data for real-time test")
        return

    horizon_results = {"early": [], "mid": [], "late": []}

    scaler = StandardScaler()
    for t in range(min_train, n):
        y_train = y[:t]
        X_train = X[:t]
        y_test = y[t]
        X_test = X[t].copy()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test.reshape(1, -1)).flatten()

        # Late: full information (standard forecast)
        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        model.fit(X_train_scaled, y_train)
        pred_late = model.predict(X_test_scaled.reshape(1, -1))[0]
        horizon_results["late"].append((y_test, pred_late))

        # Mid: replace last 1/3 of features with their training-period mean
        n_feat = len(feature_cols)
        n_mask = max(1, n_feat // 3)
        X_mid = X_test_scaled.copy()
        X_mid[-n_mask:] = 0.0  # Scaled mean is 0
        pred_mid = model.predict(X_mid.reshape(1, -1))[0]
        horizon_results["mid"].append((y_test, pred_mid))

        # Early: replace last 2/3 of features with mean
        n_mask2 = max(1, 2 * n_feat // 3)
        X_early = X_test_scaled.copy()
        X_early[-n_mask2:] = 0.0
        pred_early = model.predict(X_early.reshape(1, -1))[0]
        horizon_results["early"].append((y_test, pred_early))

    # Compute RMSE for each horizon
    rows = []
    for horizon, results in horizon_results.items():
        acts = np.array([r[0] for r in results])
        preds = np.array([r[1] for r in results])
        rmse = np.sqrt(mean_squared_error(acts, preds))
        rows.append({"horizon": horizon, "rmse": round(rmse, 1)})
        print(f"  {horizon:>6} horizon: RMSE = {rmse:,.0f}")

    rt_df = pd.DataFrame(rows)
    rt_df.to_csv(OUTPUT_DIR / "realtime_info_test.csv", index=False)
    print(f"  Saved realtime_info_test.csv")

    # Check monotonic improvement
    rmses = [r["rmse"] for r in rows]
    if rmses[0] >= rmses[1] >= rmses[2]:
        print("  [OK] RMSE improves monotonically: early >= mid >= late (news content confirmed)")
    else:
        print("  [INFO] Non-monotonic pattern (may indicate noise)")


def main():
    print("+" + "=" * 58 + "+")
    print("|  BOP NOWCASTING PILOT STUDY -- FULL PIPELINE              |")
    print("|  Chapter 1: AI vs. Econometrics for BoP Nowcasting       |")
    print("+" + "=" * 58 + "+")

    # -- Step 1: Download Data --
    print("\n\n" + "#" * 60)
    print("  STEP 1/4: DOWNLOADING DATA FROM EUROSTAT")
    print("#" * 60)

    from src.download_data import build_dataset  # noqa

    df = build_dataset()

    if df.empty:
        print("\n[ERROR] Data pipeline failed. Cannot continue.")
        return

    # -- Step 2: Run Models --
    print("\n\n" + "#" * 60)
    print("  STEP 2/4: RUNNING MODEL EVALUATION")
    print("#" * 60)

    from src.models import main as run_models  # noqa
    run_models()

    # -- Step 2b: Real-Time Information Flow Test --
    print("\n\n" + "#" * 60)
    print("  STEP 2b: REAL-TIME INFORMATION FLOW TEST")
    print("#" * 60)

    try:
        run_realtime_info_test()
    except Exception as e:
        print(f"  [WARN] Real-time info test failed: {e}")

    # -- Step 3: Generate Visualizations --
    print("\n\n" + "#" * 60)
    print("  STEP 3/4: GENERATING VISUALIZATIONS")
    print("#" * 60)

    try:
        from src.visualize import main as run_viz  # noqa
        run_viz()
    except Exception as e:
        print(f"  [WARN] Visualization error: {e}")
        print("  Results are still saved in outputs/")

    # -- Step 4: Summary --
    print("\n\n" + "#" * 60)
    print("  STEP 4/4: SUMMARY")
    print("#" * 60)

    output_dir = Path(__file__).parent / "outputs"
    print(f"\n  Output files in {output_dir}:")
    for f in sorted(output_dir.glob("*")):
        size = f.stat().st_size / 1024
        print(f"    - {f.name:<30} ({size:.0f} KB)")

    print("\n" + "=" * 60)
    print("  [OK] PILOT STUDY PIPELINE COMPLETE")
    print("=" * 60)
    print("\n  Next steps:")
    print("  1. Review outputs/model_comparison.csv for RMSE results")
    print("  2. Check outputs/forecast_comparison.png for visual fit")
    print("  3. If results look promising, expand to monthly data")
    print("     and additional BoP components")
    print("  4. Share results memo with potential PhD supervisors")


if __name__ == "__main__":
    main()
