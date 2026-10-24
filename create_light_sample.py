"""Deterministic stratified 25,000-row sample of the scored Lending Club book.

Reads output/powerbi_scored_loan_data.csv (1.34M rows), draws a grade x default_flag x
PD-decile stratified sample, and writes output/sample_scored_loans.csv (<5 MB) plus
output/sample_meta.json. A fixed-seed grid search (seeds 0-499) keeps the sample that
best reproduces the full-book ROC-AUC (tol 0.0010) and ECL/exposure ratio (tol 0.05%).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "output" / "powerbi_scored_loan_data.csv"
OUT = ROOT / "output" / "sample_scored_loans.csv"
META = ROOT / "output" / "sample_meta.json"

N_SAMPLE = 25_000
LGD = 0.60
KEEP = [
    "id", "loan_amnt", "int_rate", "grade", "home_ownership", "purpose",
    "annual_inc", "dti", "revol_util", "term_months", "vintage_year",
    "default_flag", "probability_of_default", "expected_loss",
    "application_score", "split", "underpriced_flag", "pricing_adequacy",
]
ROUND = {"annual_inc": 0, "dti": 2, "revol_util": 1, "int_rate": 2,
         "probability_of_default": 6, "expected_loss": 2, "pricing_adequacy": 4}


def allocate(counts: pd.Series, n: int) -> pd.Series:
    """Exact proportional allocation (largest-remainder) summing to n."""
    quota = counts / counts.sum() * n
    base = quota.astype(int)
    for k in quota.sub(base).sort_values(ascending=False).index[: n - base.sum()]:
        base[k] += 1
    return base


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"{SRC} not found - run 3_ml_risk_model.py first.")
    df = pd.read_csv(SRC, usecols=KEEP)
    full_auc = roc_auc_score(df["default_flag"], df["probability_of_default"])
    full_exp, full_ecl = float(df["loan_amnt"].sum()), float(df["expected_loss"].sum())
    strata = (df["grade"].astype(str) + df["default_flag"].astype(str)
              + pd.qcut(df["probability_of_default"], 10, labels=False).astype(str))
    pos = dict(df.groupby(strata).indices)
    counts = allocate(strata.value_counts(), N_SAMPLE)
    best = (np.inf, None, 0.0, 0.0, -1)
    for seed in range(500):
        rng = np.random.default_rng(seed)
        take = np.sort(np.concatenate([rng.choice(pos[k], c, replace=False) for k, c in counts.items()]))
        d = df.iloc[take].round(ROUND)
        auc = roc_auc_score(d["default_flag"], d["probability_of_default"])
        err = abs(d["expected_loss"].sum() / d["loan_amnt"].sum() - full_ecl / full_exp) / (full_ecl / full_exp)
        fit = abs(auc - full_auc) / 0.0010 + err / 0.0005
        if fit < best[0]:
            best = (fit, take, auc, err, seed)
    _, take, auc, ecl_err, seed = best
    assert abs(auc - full_auc) <= 0.0010 and ecl_err <= 0.0005, (auc, ecl_err)
    s = df.iloc[take].round(ROUND).reset_index(drop=True)
    scale = full_exp / float(s["loan_amnt"].sum())
    s.to_csv(OUT, index=False)
    meta = {
        "seed": seed, "n_sample": len(s), "n_full": len(df), "lgd": LGD,
        "stratification": "grade x default_flag x PD-decile, largest-remainder allocation",
        "sample_auc": round(float(auc), 4), "sample_gini": round(2 * float(auc) - 1, 4),
        "full_auc": round(float(full_auc), 4), "full_gini": round(2 * float(full_auc) - 1, 4),
        "sample_exposure": round(float(s["loan_amnt"].sum()), 2),
        "sample_ecl": round(float(s["expected_loss"].sum()), 2),
        "full_exposure": round(full_exp, 2), "full_ecl": round(full_ecl, 2),
        "scale_factor": round(scale, 6),
        "scaled_ecl": round(float(s["expected_loss"].sum()) * scale, 2),
    }
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {OUT} ({len(s):,} rows, seed={seed}, {OUT.stat().st_size / 1e6:.2f} MB)")
    print(f"AUC {auc:.4f} (full {full_auc:.4f}) | Gini {2 * auc - 1:.4f} | "
          f"scaled ECL ${meta['scaled_ecl'] / 1e9:.4f}B vs full ${full_ecl / 1e9:.4f}B")


if __name__ == "__main__":
    main()
