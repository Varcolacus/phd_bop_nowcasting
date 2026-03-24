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
    Simulated real-time information flow test with a stylised
    release calendar.

    At each evaluation quarter, produces forecasts at three horizons
    reflecting when different data categories typically become available:

      - Early (T+30 d):  only financial/survey variables available
        (released daily/monthly with no lag).
      - Mid   (T+60 d):  + hard activity data (IP, retail, trade with
        ~45-day publication lag).
      - Late  (T+85 d):  full information set (all features available).

    Features are grouped by typical publication lag rather than by
    arbitrary column position, giving a more realistic ragged-edge test.
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

    # ---- Stylised release-calendar grouping ----
    # Early: financial & survey (available <30 days after quarter-end)
    early_kw = ["eurusd", "stoxx", "euribor", "interest", "pmi",
                "confidence", "bci", "google", "bdi", "sentiment"]
    # Mid: hard activity (available ~45-60 days)
    # Late: everything else (trade, GDP, satellite, lagged BoP)
    early_idx = [i for i, c in enumerate(feature_cols)
                 if any(k in c.lower() for k in early_kw)]
    mid_idx = list(range(len(feature_cols)))   # all features
    late_idx = list(range(len(feature_cols)))   # identical to mid (full info)

    # Mask indices: Early = only early_idx available, rest → 0
    mask_early = set(range(len(feature_cols))) - set(early_idx)
    # Mid: only late-arriving features masked (trade values, GDP, satellite)
    late_kw = ["gdp", "satellite", "nightlight", "lag"]
    mask_mid = {i for i, c in enumerate(feature_cols)
                if any(k in c.lower() for k in late_kw)}

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

        model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        model.fit(X_train_scaled, y_train)

        # Late: full information
        pred_late = model.predict(X_test_scaled.reshape(1, -1))[0]
        horizon_results["late"].append((y_test, pred_late))

        # Mid: mask late-arriving features
        X_mid = X_test_scaled.copy()
        for idx in mask_mid:
            X_mid[idx] = 0.0
        pred_mid = model.predict(X_mid.reshape(1, -1))[0]
        horizon_results["mid"].append((y_test, pred_mid))

        # Early: only financial/survey features
        X_early = X_test_scaled.copy()
        for idx in mask_early:
            X_early[idx] = 0.0
        pred_early = model.predict(X_early.reshape(1, -1))[0]
        horizon_results["early"].append((y_test, pred_early))

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

    rmses = [r["rmse"] for r in rows]
    if rmses[0] >= rmses[1] >= rmses[2]:
        print("  [OK] RMSE improves monotonically: early >= mid >= late")
    else:
        print("  [INFO] Non-monotonic pattern (may indicate noise)")


def _run_enhanced_tests(df_raw, OUTPUT_DIR,
                        clark_west_test, r2_oos, forecast_combination,
                        diebold_mariano_test, model_confidence_set,
                        load_modeling_data, expanding_window_evaluation,
                        ForecastResult):
    """Run Clark-West, R²_OOS, DM p-values, forecast combination, and MCS."""
    import numpy as np
    import pandas as pd

    # Re-load model results from saved CSVs to reconstruct per-period errors
    forecast_df = pd.read_csv(OUTPUT_DIR / "forecast_results.csv", parse_dates=["date"])
    comparison_df = pd.read_csv(OUTPUT_DIR / "model_comparison.csv")

    model_names = forecast_df["model"].unique()
    benchmark_name = "AR(1)"

    # Build per-model prediction/actual arrays
    models_data = {}
    for mname in model_names:
        sub = forecast_df[forecast_df["model"] == mname].sort_values("date")
        models_data[mname] = {
            "actual": sub["actual"].values.astype(float),
            "predicted": sub["predicted"].values.astype(float),
        }

    bench = models_data.get(benchmark_name)
    if bench is None:
        print("  [WARN] No AR(1) benchmark found.")
        return

    bench_errors = bench["actual"] - bench["predicted"]

    # ----- DM p-values + Clark-West test -----
    print("\n  Clark-West (2007) & DM tests vs AR(1):")
    print(f"  {'Model':<20} {'DM stat':>10} {'DM p':>8} {'CW stat':>10} {'CW p':>8} {'R²_OOS':>10}")
    print("  " + "-" * 70)

    enhanced_rows = []
    for mname in model_names:
        if mname == benchmark_name:
            continue
        m = models_data[mname]
        n = min(len(m["actual"]), len(bench["actual"]))
        acts = m["actual"][:n]
        preds = m["predicted"][:n]
        bench_preds = bench["predicted"][:n]
        bench_acts = bench["actual"][:n]

        e_model = acts - preds
        e_bench = bench_acts - bench_preds

        valid = ~np.isnan(e_model) & ~np.isnan(e_bench)
        if valid.sum() < 10:
            continue

        dm_stat, dm_p = diebold_mariano_test(e_model[valid], e_bench[valid])
        cw_stat, cw_p = clark_west_test(e_model[valid], e_bench[valid],
                                         acts[valid], bench_preds[valid])
        r2 = r2_oos(acts[valid], preds[valid], bench_preds[valid])

        dm_s = f"{dm_stat:.3f}" if not np.isnan(dm_stat) else "N/A"
        dm_ps = f"{dm_p:.4f}" if not np.isnan(dm_p) else "N/A"
        cw_s = f"{cw_stat:.3f}" if not np.isnan(cw_stat) else "N/A"
        cw_ps = f"{cw_p:.4f}" if not np.isnan(cw_p) else "N/A"
        r2_s = f"{r2:.4f}" if not np.isnan(r2) else "N/A"

        print(f"  {mname:<20} {dm_s:>10} {dm_ps:>8} {cw_s:>10} {cw_ps:>8} {r2_s:>10}")

        enhanced_rows.append({
            "model": mname, "dm_stat": dm_stat, "dm_p": dm_p,
            "cw_stat": cw_stat, "cw_p": cw_p, "r2_oos": r2,
        })

    pd.DataFrame(enhanced_rows).to_csv(OUTPUT_DIR / "enhanced_tests.csv", index=False)
    print(f"  Saved enhanced_tests.csv")

    # ----- Holm-Bonferroni Multiple Testing Correction -----
    from src.models import holm_bonferroni as _hb
    if enhanced_rows:
        dm_ps_raw = [r["dm_p"] for r in enhanced_rows if not np.isnan(r["dm_p"])]
        cw_ps_raw = [r["cw_p"] for r in enhanced_rows if not np.isnan(r["cw_p"])]
        if dm_ps_raw:
            dm_adj = _hb(dm_ps_raw)
            cw_adj = _hb(cw_ps_raw) if cw_ps_raw else []
            print(f"\n  Holm-Bonferroni adjusted p-values ({len(dm_ps_raw)} comparisons):")
            print(f"  {'Model':<20} {'DM raw':>8} {'DM adj':>8} {'CW raw':>8} {'CW adj':>8}")
            print("  " + "-" * 50)
            j_dm, j_cw = 0, 0
            for r in enhanced_rows:
                dm_r = f"{r['dm_p']:.4f}" if not np.isnan(r["dm_p"]) else "N/A"
                cw_r = f"{r['cw_p']:.4f}" if not np.isnan(r["cw_p"]) else "N/A"
                dm_a = f"{dm_adj[j_dm]:.4f}" if not np.isnan(r["dm_p"]) else "N/A"
                cw_a = f"{cw_adj[j_cw]:.4f}" if cw_ps_raw and not np.isnan(r["cw_p"]) else "N/A"
                print(f"  {r['model']:<20} {dm_r:>8} {dm_a:>8} {cw_r:>8} {cw_a:>8}")
                if not np.isnan(r["dm_p"]):
                    r["dm_p_adj"] = dm_adj[j_dm]
                    j_dm += 1
                if not np.isnan(r["cw_p"]):
                    r["cw_p_adj"] = cw_adj[j_cw] if cw_ps_raw else np.nan
                    j_cw += 1
            # Re-save with adjusted p-values
            pd.DataFrame(enhanced_rows).to_csv(
                OUTPUT_DIR / "enhanced_tests.csv", index=False)
            print(f"  Updated enhanced_tests.csv with adjusted p-values")

    # ----- Forecast Combination -----
    print("\n  Forecast Combination (inverse-RMSE weights):")

    # Rebuild ForecastResult objects for combination
    fr_models = {}
    for mname in model_names:
        sub = forecast_df[forecast_df["model"] == mname].sort_values("date")
        fr = ForecastResult(mname)
        fr.predictions = sub["predicted"].tolist()
        fr.actuals = sub["actual"].tolist()
        fr.dates = sub["date"].tolist()
        rmse_row = comparison_df[comparison_df["model"] == mname]
        fr.rmse = rmse_row["rmse"].values[0] if len(rmse_row) > 0 else None
        fr_models[mname] = fr

    combo = forecast_combination(fr_models, method="inverse_rmse",
                                  exclude=["AR(1)", "LSTM", "GRU"])
    if combo.rmse:
        bench_rmse = fr_models[benchmark_name].rmse
        vs = (combo.rmse - bench_rmse) / bench_rmse * 100 if bench_rmse else 0
        print(f"  Combination RMSE: {combo.rmse:,.0f} ({vs:+.1f}% vs AR(1))")

        combo_row = pd.DataFrame([{
            "model": "Combination",
            "rmse": combo.rmse,
            "mae": combo.mae,
            "direction_accuracy": combo.direction_accuracy,
        }])
        # Append to model_comparison.csv
        comp_df = pd.read_csv(OUTPUT_DIR / "model_comparison.csv")
        comp_df = comp_df[comp_df["model"] != "Combination"]
        comp_df = pd.concat([comp_df, combo_row], ignore_index=True)
        comp_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
        print(f"  Updated model_comparison.csv with Combination")

    # ----- Model Confidence Set -----
    print("\n  Model Confidence Set (Hansen, Lunde & Nason 2011):")
    try:
        surviving, elim = model_confidence_set(fr_models, alpha=0.10,
                                                n_boot=1000, block_length=4)
        print(f"  Superior set (α=0.10): {surviving}")
        if elim:
            print(f"  Eliminated: {elim}")
        mcs_df = pd.DataFrame([{"model": m, "in_mcs": m in surviving}
                                for m in fr_models.keys()])
        mcs_df.to_csv(OUTPUT_DIR / "model_confidence_set.csv", index=False)
        print(f"  Saved model_confidence_set.csv")
    except Exception as e:
        print(f"  [WARN] MCS failed: {e}")


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

    # -- Step 1b: Unit-Root Diagnostics --
    print("\n\n" + "#" * 60)
    print("  STEP 1b: UNIT-ROOT DIAGNOSTICS (ADF + KPSS)")
    print("#" * 60)

    try:
        from src.models import unit_root_tests, OUTPUT_DIR as _OD
        import pandas as _pd_ur
        _urt_cols = [c for c in df.columns
                     if c != "date" and df[c].dtype in [float, 'float64']]
        urt_df = unit_root_tests(df, _urt_cols)
        if not urt_df.empty:
            urt_df.to_csv(_OD / "unit_root_tests.csv", index=False)
            print(f"  Saved unit_root_tests.csv ({len(urt_df)} series)")
            n_stat = (urt_df["conclusion"] == "Stationary").sum()
            n_ur = (urt_df["conclusion"] == "Unit root").sum()
            n_inc = (urt_df["conclusion"] == "Inconclusive").sum()
            print(f"  Stationary: {n_stat}, Unit root: {n_ur}, "
                  f"Inconclusive: {n_inc}")
            for _, r in urt_df.iterrows():
                tag = "✓" if r["conclusion"] == "Stationary" else (
                    "✗" if r["conclusion"] == "Unit root" else "?")
                print(f"    {tag} {r['variable']:<25} ADF p={r['adf_pvalue']:.3f}  "
                      f"KPSS p={r['kpss_pvalue']:.3f}  → {r['conclusion']}")
    except Exception as e:
        print(f"  [WARN] Unit-root tests failed: {e}")

    # -- Step 2: Run Models --
    print("\n\n" + "#" * 60)
    print("  STEP 2/4: RUNNING MODEL EVALUATION")
    print("#" * 60)

    from src.models import (  # noqa
        main as run_models,
        clark_west_test, r2_oos, forecast_combination,
        diebold_mariano_test, model_confidence_set,
        load_modeling_data, expanding_window_evaluation,
        conformal_prediction_intervals,
        ForecastResult, OUTPUT_DIR,
    )
    run_models()

    # -- Step 2a: Enhanced statistical tests (Clark-West, R²_OOS, combinations) --
    print("\n\n" + "#" * 60)
    print("  STEP 2a: ENHANCED STATISTICAL TESTS")
    print("#" * 60)

    try:
        import pandas as _pd
        _run_enhanced_tests(df, OUTPUT_DIR,
                            clark_west_test, r2_oos, forecast_combination,
                            diebold_mariano_test, model_confidence_set,
                            load_modeling_data, expanding_window_evaluation,
                            ForecastResult)
    except Exception as e:
        print(f"  [WARN] Enhanced tests failed: {e}")

    # -- Step 2a-iv: Conformal Prediction Intervals --
    print("\n\n" + "#" * 60)
    print("  STEP 2a-iv: CONFORMAL PREDICTION INTERVALS")
    print("#" * 60)

    try:
        import pandas as _cpd
        forecast_df_cp = _cpd.read_csv(OUTPUT_DIR / "forecast_results.csv",
                                       parse_dates=["date"])
        fr_cp = {}
        for mname in forecast_df_cp["model"].unique():
            sub = forecast_df_cp[forecast_df_cp["model"] == mname].sort_values("date")
            fr = ForecastResult(mname)
            fr.predictions = sub["predicted"].tolist()
            fr.actuals = sub["actual"].tolist()
            fr.dates = sub["date"].tolist()
            fr_cp[mname] = fr
        conf_df = conformal_prediction_intervals(fr_cp, alpha=0.10)
        if not conf_df.empty:
            conf_df.to_csv(OUTPUT_DIR / "conformal_intervals.csv", index=False)
            print(f"  Saved conformal_intervals.csv")
            print(f"  {'Model':<20} {'Nominal':>8} {'Empirical':>10} {'Width':>8}")
            print("  " + "-" * 50)
            for _, r in conf_df.iterrows():
                print(f"  {r['model']:<20} {r['nominal_coverage']:>8.0%}"
                      f" {r['empirical_coverage']:>10.1%} {r['mean_width']:>8,.0f}")
    except Exception as e:
        print(f"  [WARN] Conformal intervals failed: {e}")

    # -- Step 2a-ii: Sub-Period RMSE Breakdown --
    print("\n\n" + "#" * 60)
    print("  STEP 2a-ii: SUB-PERIOD RMSE BREAKDOWN")
    print("#" * 60)

    try:
        import pandas as _spd
        import numpy as _np
        forecast_df = _spd.read_csv(OUTPUT_DIR / "forecast_results.csv",
                                    parse_dates=["date"])
        periods = [
            ("2010-2014", "2010-01-01", "2014-12-31"),
            ("2015-2019", "2015-01-01", "2019-12-31"),
            ("2020-2022", "2020-01-01", "2022-12-31"),
        ]
        sp_rows = []
        for pname, pstart, pend in periods:
            mask = (forecast_df["date"] >= pstart) & (forecast_df["date"] <= pend)
            sub = forecast_df[mask]
            ar1_rmse = None
            for mname in sub["model"].unique():
                ms = sub[sub["model"] == mname]
                errs = ms["actual"].values - ms["predicted"].values
                rmse = _np.sqrt(_np.mean(errs ** 2))
                if mname == "AR(1)":
                    ar1_rmse = rmse
                sp_rows.append({"period": pname, "model": mname,
                                "rmse": round(rmse), "n_obs": len(ms)})
            # Add pct improvement vs AR(1)
            if ar1_rmse and ar1_rmse > 0:
                for row in sp_rows:
                    if row["period"] == pname:
                        row["pct_vs_ar1"] = round(
                            (row["rmse"] - ar1_rmse) / ar1_rmse * 100, 1)

        sp_df = _spd.DataFrame(sp_rows)
        sp_df.to_csv(OUTPUT_DIR / "subperiod_rmse.csv", index=False)
        print(f"  Saved subperiod_rmse.csv ({len(sp_rows)} rows)")
        # Print summary table
        for pname, _, _ in periods:
            psub = sp_df[sp_df["period"] == pname]
            ar1_r = psub[psub["model"] == "AR(1)"]["rmse"].values
            ridge_r = psub[psub["model"] == "Ridge"]["rmse"].values
            n_o = psub["n_obs"].iloc[0] if len(psub) > 0 else 0
            if len(ar1_r) > 0 and len(ridge_r) > 0:
                pct = (ridge_r[0] - ar1_r[0]) / ar1_r[0] * 100
                print(f"  {pname}: AR(1)={ar1_r[0]:.0f}, Ridge={ridge_r[0]:.0f} "
                      f"({pct:+.1f}%), N={n_o}")
    except Exception as e:
        print(f"  [WARN] Sub-period analysis failed: {e}")

    # -- Step 2a-iii: Hyperparameter Sensitivity --
    print("\n\n" + "#" * 60)
    print("  STEP 2a-iii: HYPERPARAMETER SENSITIVITY")
    print("#" * 60)

    try:
        from sklearn.ensemble import GradientBoostingRegressor
        import xgboost as xgb_mod
        import numpy as _npx

        hp_df_data = load_modeling_data()
        hp_target = "bop_ca"
        hp_features = [c for c in hp_df_data.columns
                       if c not in ["date", hp_target]]
        hp_y = hp_df_data[hp_target].values
        hp_X = hp_df_data[hp_features].values

        min_train = 40
        sens_rows = []

        # Configurations to test
        configs = [
            ("GB_default", "GB", {"n_estimators": 100, "max_depth": 3,
             "learning_rate": 0.05, "min_samples_leaf": 5}),
            ("GB_lr01", "GB", {"n_estimators": 100, "max_depth": 3,
             "learning_rate": 0.1, "min_samples_leaf": 5}),
            ("GB_depth5", "GB", {"n_estimators": 100, "max_depth": 5,
             "learning_rate": 0.05, "min_samples_leaf": 5}),
            ("XGB_default", "XGB", {"n_estimators": 100, "max_depth": 3,
             "learning_rate": 0.1, "min_child_weight": 5,
             "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0}),
            ("XGB_lr005", "XGB", {"n_estimators": 100, "max_depth": 3,
             "learning_rate": 0.05, "min_child_weight": 5,
             "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0}),
            ("XGB_noreg", "XGB", {"n_estimators": 100, "max_depth": 3,
             "learning_rate": 0.1, "min_child_weight": 5,
             "colsample_bytree": 1.0, "reg_alpha": 0.0, "reg_lambda": 0.0}),
        ]

        for cfg_name, model_type, params in configs:
            preds, actuals = [], []
            for t in range(min_train, len(hp_y)):
                X_tr, y_tr = hp_X[:t], hp_y[:t]
                X_te = hp_X[t:t+1]
                if model_type == "GB":
                    m = GradientBoostingRegressor(random_state=42, **params)
                else:
                    m = xgb_mod.XGBRegressor(random_state=42, verbosity=0,
                                             **params)
                m.fit(X_tr, y_tr)
                preds.append(m.predict(X_te)[0])
                actuals.append(hp_y[t])
            errs = _npx.array(actuals) - _npx.array(preds)
            rmse = _npx.sqrt(_npx.mean(errs ** 2))
            sens_rows.append({"config": cfg_name, "model_type": model_type,
                              "rmse": round(rmse), "params": str(params)})
            print(f"  {cfg_name}: RMSE = {rmse:,.0f}")

        sens_df = _spd.DataFrame(sens_rows)
        sens_df.to_csv(OUTPUT_DIR / "hyperparameter_sensitivity.csv", index=False)
        print(f"  Saved hyperparameter_sensitivity.csv")
    except Exception as e:
        print(f"  [WARN] Hyperparameter sensitivity failed: {e}")

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
