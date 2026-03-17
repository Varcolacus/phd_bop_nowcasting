# PhD Research: AI/ML for Balance of Payments Nowcasting

## Overview

This repository contains the planning documents and pilot study code for a PhD thesis on using machine learning to nowcast France's Balance of Payments (BoP).

**Research question:** Can AI/ML models outperform traditional econometric methods for nowcasting BoP components?

## Structure

```
phd/
  00_overview/              # Topic exploration & comparison
  01_selected_topic/        # Thesis outline, publication strategy, action plan
  02_pilot_study/           # Working pilot study code
    run_pilot.py            # Full pipeline orchestrator
    src/
      download_data.py      # Downloads real data from ECB SDW (7 series)
      models.py             # AR(1), AR(4), Ridge, GradientBoosting, XGBoost
      visualize.py          # 4 diagnostic charts
    data/                   # Downloaded datasets (auto-generated, git-ignored)
    outputs/                # Results: CSVs + charts (charts git-ignored)
```

## Pilot Study Results (Real ECB Data)

91 quarterly observations (2000-Q1 to 2022-Q3), expanding window out-of-sample evaluation:

| Model | RMSE | vs AR(1) | Statistically Significant |
|---|---|---|---|
| AR(1) | 8,913 | baseline | -- |
| AR(4) | 7,512 | -15.7% | No |
| GradientBoosting | **4,433** | **-50.3%** | **Yes** |
| XGBoost | **4,542** | **-49.0%** | **Yes** |

ML models beat AR baselines by ~50% RMSE, statistically significant via Diebold-Mariano test.

## Data Sources

- **ECB Statistical Data Warehouse** (public, no auth required):
  - BP6: France Balance of Payments (BPM6)
  - EXR: EUR/USD exchange rate
  - MNA: France GDP
  - ICP: France HICP
  - FM: Euribor 3-month rate

## How to Run

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib requests xgboost
cd 02_pilot_study
python run_pilot.py
```

The pipeline will automatically download data from the ECB and run all models.

## Next Steps

- [ ] Fix Ridge feature leakage (remove BoP components from features)
- [ ] Add LSTM/neural network model
- [ ] Add SHAP feature importance analysis
- [ ] Extend to monthly frequency
- [ ] Write pilot study memo for PhD supervisor search
- [ ] Chapters 2 & 3 (NLP for BoP, real-time data revisions)

## Author

Banque de France, Balance of Payments department
