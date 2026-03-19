"""Quick test for ablation bootstrap CI."""
import sys, os
sys.path.insert(0, 'src')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import pandas as pd
import numpy as np
from pathlib import Path
from models import (ForecastResult, ar_forecast, ols_forecast, xgboost_forecast,
                    block_bootstrap_rmse_ci)

CH2_OUTPUT = Path('outputs/chapter2')

df = pd.read_csv(CH2_OUTPUT / 'combined_dataset.csv')
df['date'] = pd.to_datetime(df['date'])
print(f'Dataset: {len(df)} rows, {len(df.columns)} cols')

target_col = 'trade_goods'
baseline_features = [c for c in [
    f"{target_col}_lag1", f"{target_col}_lag3", f"{target_col}_lag12",
    "eurusd", "hicp", "interest_rate",
    "eurusd_lag1", "hicp_lag1", "interest_rate_lag1",
] if c in df.columns]
print(f'Baseline features ({len(baseline_features)}): {baseline_features}')

from run_chapter2 import ablation_expanding_window, SPECIFICATIONS

all_results = {}
for spec_name in ['M0', 'M1', 'M2', 'M3', 'M4', 'M5']:
    alt_features = SPECIFICATIONS[spec_name]
    models = ablation_expanding_window(df, target_col, baseline_features, alt_features,
                                       min_train_size=60, step=1)
    if models:
        all_results[spec_name] = models
        ridge = models.get('Ridge')
        if ridge and ridge.rmse:
            print(f'  {spec_name} Ridge: RMSE={ridge.rmse:.1f}, n={len(ridge.predictions)}')

print()
print('=== BOOTSTRAP CI ===')
m0_ridge = all_results['M0']['Ridge']
m0_date_idx = {str(d): i for i, d in enumerate(m0_ridge.dates)}
print(f'M0: {len(m0_ridge.dates)} dates')

for spec_name in ['M1', 'M2', 'M3', 'M4', 'M5']:
    mk_ridge = all_results[spec_name]['Ridge']
    common_m0, common_mk = [], []
    for j, d in enumerate(mk_ridge.dates):
        if str(d) in m0_date_idx:
            common_m0.append(m0_date_idx[str(d)])
            common_mk.append(j)
    print(f'{spec_name}: {len(mk_ridge.dates)} dates, {len(common_m0)} common')

    if len(common_m0) >= 12:
        ab = ForecastResult('M0_Ridge')
        ab.actuals = [m0_ridge.actuals[i] for i in common_m0]
        ab.predictions = [m0_ridge.predictions[i] for i in common_m0]
        ab.rmse = m0_ridge.rmse

        am = ForecastResult(f'{spec_name}_Ridge')
        am.actuals = [mk_ridge.actuals[i] for i in common_mk]
        am.predictions = [mk_ridge.predictions[i] for i in common_mk]
        am.rmse = mk_ridge.rmse

        virtual = {'M0_Ridge': ab, f'{spec_name}_Ridge': am}
        try:
            boot_df = block_bootstrap_rmse_ci(virtual, benchmark='M0_Ridge',
                                               n_boot=1000, block_length=6, seed=42)
            if not boot_df.empty:
                r = boot_df.iloc[0]
                print(f'  -> {r["pct_improvement"]:+.1f}% [{r["ci_lower"]:+.1f}, {r["ci_upper"]:+.1f}]')
            else:
                print(f'  -> empty result')
        except Exception as e:
            print(f'  -> ERROR: {e}')
