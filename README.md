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
    run_pilot.py            # Full pipeline orchestrator (quarterly)
    run_monthly.py          # Monthly frequency extension pipeline
    run_chapter2.py         # Chapter 2: ablation study with alternative data
    run_chapter3.py         # Chapter 3: policy implications analysis
    src/
      download_data.py      # Downloads real data from ECB SDW (7 series)
      models.py             # AR(1), AR(4/12), Ridge, GB, XGBoost, LSTM
      alternative_data.py   # Alt data: BDI, FX vol, CDS, TPU, Google Trends, NTL
      nlp_sentiment.py      # FinBERT NLP pipeline for ECB press conferences
      visualize.py          # 4 diagnostic charts
    data/                   # Downloaded datasets (auto-generated, git-ignored)
    outputs/                # Results: CSVs, charts, pilot study memo
      pilot_study_memo.txt  # Comprehensive 5-page pilot study memo
      monthly/              # Monthly frequency results
      chapter2/             # Ablation study results & SHAP decomposition
      chapter3/             # Crisis eval, cross-country, Taylor-rule results
  03_chapters/              # Historical working-paper drafts (superseded by 04_latex/chapters/)
    chapter2_alternative_data.txt      # Ch.2: Alternative data for BoP
    chapter3_policy_implications.txt   # Ch.3: Policy implications
```

## Pilot Study Results (Real ECB Data)

91 quarterly observations (2000-Q1 to 2022-Q3), expanding window out-of-sample evaluation:

| Model | RMSE | vs AR(1) | Statistically Significant |
|---|---|---|---|
| AR(1) | 8,913 | baseline | -- |
| AR(4) | 7,512 | -15.7% | No |
| Ridge | **6,538** | **-26.6%** | **Yes** |
| GradientBoosting | 7,562 | -15.1% | No |
| XGBoost | 7,671 | -13.9% | No |
| LSTM | 8,999 | +1.0% | No |

After fixing feature leakage (removed BoP sub-components and contemporaneous target transforms from features), Ridge is the best-performing model, the only one statistically significant vs AR(1) via Diebold-Mariano test.

**SHAP top features:** `bop_ca_lag4`, `bop_ca_lag1`, `hicp`, `gdp_yoy`, `gdp`

## Data Sources

- **ECB Statistical Data Warehouse** (public, no auth required):
  - BP6: France Balance of Payments (BPM6)
  - EXR: EUR/USD exchange rate
  - MNA: France GDP
  - ICP: France HICP
  - FM: Euribor 3-month rate

## How to Run

```bash
pip install -r requirements.txt
cd 02_pilot_study
python run_pilot.py         # Quarterly pilot study (Chapter 1)
python run_monthly.py       # Monthly frequency extension
python run_chapter2.py      # Alternative data ablation study (Chapter 2)
python run_chapter3.py      # Policy implications analysis (Chapter 3)
```

The pipelines automatically download data from the ECB and run all models.

## Chapter 2 Results — Alternative Data Ablation Study

180 monthly observations, 108 OOS. Ablation over 6 specifications (M0–M5):

| Spec | Description | Ridge RMSE | vs M0 | GW p-value |
|---|---|---|---|---|
| M0 | Baseline | 1,998 | — | — |
| M1 | +Trade-activity (BDI, CTI) | 2,061 | +3.1% | 0.39 |
| M2 | +Financial-flow (FX vol, CDS) | 2,007 | +0.5% | 0.57 |
| **M3** | **+Text/Sentiment (TPU, Google Trends, NLP)** | **1,894** | **-5.2%** | **0.0006** |
| M4 | +Satellite (NTL) | 2,000 | +0.1% | 0.31 |
| M5 | All combined | 1,981 | -0.8% | 0.52 |

Text/sentiment data (Category C) provides the largest and statistically significant improvement (Giacomini-White test).

## Chapter 3 Results — Policy Implications

**Crisis evaluation:** XGBoost shows 72.7% direction accuracy during Energy shock vs 36.4% for AR(1). Ridge cuts RMSE by 45% vs AR(1) during the 2022 energy crisis.

**Cross-country:** Methodology transfers to DE, IT (real ECB data), ES (synthetic). Fine-tuned transfer learning outperforms direct transfer by 20-50%.

**Taylor-rule counterfactual:** δ̂ = 0.494 (p=0.001), max policy difference of 164bp. Nowcast direction accuracy 57.7% vs 52.9% for lagged official data.

## Next Steps

- [x] Fix Ridge feature leakage (remove BoP components from features)
- [x] Add LSTM/neural network model
- [x] Add SHAP feature importance analysis
- [x] Extend to monthly frequency
- [x] Write pilot study memo for PhD supervisor search
- [x] Chapters 2 & 3 working paper drafts (skeleton + methodology)
- [x] Chapter 2 executable code: alternative data pipeline, NLP sentiment, ablation study
- [x] Chapter 3 executable code: crisis evaluation, cross-country, Taylor-rule counterfactual

## Author

Banque de France, Balance of Payments department
