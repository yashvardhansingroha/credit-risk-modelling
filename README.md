# Lending Club Credit Risk Pipeline

End-to-end retail credit risk stack: PostgreSQL ETL on ~2.2M Lending Club loans (1,344,976 closed loans with a known outcome are modelled),
stratified logistic PD model, WOE/IV application scorecard, PSI monitoring, risk-adjusted
pricing, IFRS 9 ECL with stress scenarios, Power BI DAX measures, and a **Streamlit
dashboard** that replicates the DAX logic on a light stratified sample.

## Headline results

Full book of **1,344,976 closed loans** (941,483 train / 403,493 test), latest run, from `output/model_metrics.json`.

| | Result |
|---|---|
| PD model | Logistic regression, **ROC-AUC 0.705**, Gini 0.41, KS 0.30 |
| Scorecard | WOE/IV application scorecard; strongest feature `term_months` (IV 0.174) |
| Stability | Train vs test PSI ≈ 0.00 for score, DTI and grade (stable) |
| IFRS 9-style ECL | $5.62B on $19.39B EAD (28.96% coverage, LGD fixed at 60%) |
| Stress test | ECL +44% at 1.5× PD (adverse), +71% at 2× PD (severe), +10% with +5pp PD (recession) |
| Pricing | 54.5% of exposure ($10.57B) priced below its risk-adjusted rate |

Why 2.2M vs 1.34M: the raw file has ~2.2M loans; only loans with a final outcome (Fully Paid, Charged Off or Default) are kept, so the model is trained and scored on those 1,344,976.

## Prerequisites

1. **PostgreSQL** on `localhost:5432`: `CREATE DATABASE mn_analytics_db;`
2. Lending Club `accepted_2007_to_2018Q4.csv.gz` (default path in `config.py`)
3. `.env` in repo root:
   `CREDIT_RISK_DATABASE_URL=postgresql://postgres:PW@localhost:5432/mn_analytics_db`
   `CREDIT_RISK_CSV_PATH=C:\Users\ysing\Downloads\accepted_2007_to_2018Q4.csv.gz`
4. `pip install -r requirements.txt`

## Run order

```bash
python 1_data_ingestion.py      # bulk load ~2.2M rows (minutes)
psql -U postgres -d mn_analytics_db -f 2_etl_pipeline.sql  # analytical views
python 3_ml_risk_model.py       # PD model + WOE scorecard + scored export
python 5_portfolio_analytics.py # PSI, pricing adequacy, stress testing
python create_light_sample.py   # 25k stratified sample for the dashboard
streamlit run app.py            # dashboard (self-bootstraps sample if missing)
pytest tests/ -q                # WOE/IV + ECL math + sample fidelity
```

## Model & portfolio metrics (detail)

| Metric | Value |
|---|---|
| ROC-AUC | **0.7054** (Gini 0.4109 = 2×AUC−1) |
| F1 / threshold | 0.425 @ 0.51 (F1-optimised, `class_weight='balanced'`) |
| KS statistic | 0.2987 |
| Portfolio EAD / ECL | **$19.39B / $5.62B** (coverage 28.96%, LGD fixed 60%) |
| Underpriced exposure | 54.5% of EAD ($10.57B) |
| Strongest IV feature | `term_months` (IV 0.174) |

## Dashboard (`app.py`) — 4 tabs

1. **IFRS 9 ECL Portfolio** — EAD, ECL, coverage ratio, weighted PD, observed default
   rate, AUC/Gini, underpriced share; ECL by grade + vintage default-rate curve.
2. **Grade Migration** — grade × vintage matrix (default rate / exposure share / avg PD).
3. **Macro Stress Simulator** — PD-multiplier and LGD sliders recompute stressed ECL
   (PD capped at 0.99), Δ vs base, per-grade base-vs-stressed bars, scenario table.
4. **Scorecard & PSI** — score-decile calibration, train-vs-test PSI (0.10/0.25
   thresholds), IV by feature, KS + full-book model metrics.

The app never loads the 378 MB scored book. It runs on `output/sample_scored_loans.csv`
(25,000 rows, 2.8 MB) — a grade × default × PD-decile stratified sample whose ROC-AUC
(0.7054), Gini (0.41) and ECL/exposure ratio (28.96%) match the full book; dollar KPIs
are grossed back up via the generator's scale factor (`output/sample_meta.json`).
Generation is deterministic (fixed-seed grid search) and the app self-bootstraps it.

## Artifacts (`output/`)

| File | Description |
|------|-------------|
| `powerbi_scored_loan_data.csv` | Scored book for Power BI (PD, ECL, score, pricing) |
| `sample_scored_loans.csv` / `sample_meta.json` | Light dashboard sample + provenance |
| `model_metrics.json` | AUC, F1, Gini, KS, ECL, PSI, stress summary |
| `woe_iv_summary.csv` / `scorecard_points.csv` | WOE bins, IV contributions, points |
| `stress_test_results.csv` | Base / adverse / severe / recession ECL |
| `pricing_by_grade.csv` / `ecl_by_grade.csv` | Pricing adequacy + ECL concentration |

## Power BI & Analytics

Import output/powerbi_scored_loan_data.csv into Power BI, then copy DAX measures from
[4_powerbi_dax.txt](4_powerbi_dax.txt). For executive risk methodology and portfolio governance details,
see [reports/RISK_MEMO.md](reports/RISK_MEMO.md).


## Project layout

```
├── 1_data_ingestion.py / 2_etl_pipeline.sql / 3_ml_risk_model.py
├── 4_powerbi_dax.txt / 5_portfolio_analytics.py / scorecard.py / config.py
├── create_light_sample.py / app.py / tests/test_credit_risk.py
└── output/ / reports/ / requirements.txt / README.md
```

## Notes

- LGD fixed at **60%** (conservative unsecured consumer, `config.py`); EAD = `loan_amnt`
  (no CCF). Stress scenarios are illustrative, not regulatory ICAAP/CCAR packs.
- AUC ~0.70–0.71 is the expected logistic baseline for these features; further gains
  usually need underwriting fields (e.g. FICO) or a non-linear challenger model.
