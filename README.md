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
    run_all.py              # End-to-end orchestrator (all chapters + LaTeX)
    validate_tables.py      # Cross-validate LaTeX tables against CSV outputs
    src/
      download_data.py      # Downloads real data from ECB SDW (7 series)
      models.py             # AR(1), AR(4/12), Ridge, LASSO, GB, XGBoost, LSTM, GRU
      alternative_data.py   # Alt data: BDI, FX vol, CDS, TPU, Google Trends, NTL
      nlp_sentiment.py      # FinBERT NLP pipeline for ECB press conferences
      visualize.py          # 4 diagnostic charts
    tests/
      test_statistical.py   # Pytest tests for DM, CW, Holm-Bonferroni, conformal, R²_OOS
    data/                   # Downloaded datasets (auto-generated, git-ignored)
    outputs/                # Results: CSVs, charts, pilot study memo
      pilot_study_memo.txt  # Comprehensive 5-page pilot study memo
      monthly/              # Monthly frequency results
      chapter2/             # Ablation study results & SHAP decomposition
      chapter3/             # Crisis eval, cross-country, Taylor-rule results
  03_chapters/              # Historical working-paper drafts (superseded by 04_latex/chapters/)
    chapter2_alternative_data.txt      # Ch.2: Alternative data for BoP
    chapter3_policy_implications.txt   # Ch.3: Policy implications
  04_latex/                 # LaTeX thesis (pdflatex + biber)
    thesis.tex              # Main file
    chapters/               # Per-chapter .tex files
    figures/                # 12 PNG figures
    references.bib          # BibLaTeX bibliography
  requirements.txt          # Dependency ranges
  requirements-lock.txt     # Pinned exact versions for reproducibility
```

## Pilot Study Results (Real ECB Data)

92 quarterly observations (2000-Q1 to 2022-Q4), expanding window out-of-sample evaluation:

| Model | RMSE | vs AR(1) | Statistically Significant |
|---|---|---|---|
| AR(1) | 8,913 | baseline | -- |
| AR(4) | 7,512 | -15.7% | No |
| DFM | 7,229 | -18.9% | Yes (p=0.049) |
| Bridge | 7,685 | -13.8% | No |
| Ridge | **6,567** | **-26.3%** | **Yes (p=0.013)** |
| LASSO | 6,720 | -24.6% | Yes (p=0.021) |
| GradientBoosting | 7,344 | -17.6% | Marginal (p=0.062) |
| XGBoost | 7,476 | -16.1% | Marginal (p=0.082) |
| LSTM | 9,004 | +1.0% | No |
| GRU | 9,000 | +1.0% | No |
| Combination | 7,050 | -20.9% | Yes (p=0.035) |

Ridge regression is the best-performing individual model. The inverse-RMSE forecast combination provides robust ensemble performance. Conformal prediction intervals (90% nominal) achieve near-nominal empirical coverage.

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

# Individual chapters
python run_pilot.py         # Quarterly pilot study (Chapter 1)
python run_monthly.py       # Monthly frequency extension
python run_chapter2.py      # Alternative data ablation study (Chapter 2)
python run_chapter3.py      # Policy implications analysis (Chapter 3)

# Or run everything + compile LaTeX thesis
python run_all.py           # All chapters + pdflatex/biber
python run_all.py --skip-latex  # All chapters, skip LaTeX compilation

# Validate LaTeX tables match CSV outputs
python validate_tables.py

# Run tests
python -m pytest tests/ -v
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
