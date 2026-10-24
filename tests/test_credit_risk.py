"""Unit tests: WOE/IV math, IFRS 9 ECL math, and sample fidelity to the full book."""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "output" / "sample_scored_loans.csv"

@pytest.fixture(scope="module")
def loans():
    if not SAMPLE.exists():  # self-bootstrap: regenerate the light sample
        subprocess.run([sys.executable, str(ROOT / "create_light_sample.py")], check=True, cwd=ROOT)
    return pd.read_csv(SAMPLE)

def woe_iv(bins: pd.Series, y: pd.Series) -> float:
    """Canonical IV = sum over bins of (pct_good - pct_bad) * ln(pct_good / pct_bad)."""
    frame, iv = pd.DataFrame({"b": bins.to_numpy(), "y": y.to_numpy()}), 0.0
    for _, g in frame.groupby("b"):
        pg = (g["y"] == 0).sum() / (frame["y"] == 0).sum()
        pb = (g["y"] == 1).sum() / (frame["y"] == 1).sum()
        iv += (pg - pb) * math.log(pg / pb)
    return iv

def meta_doc():
    return json.loads((ROOT / "output" / "sample_meta.json").read_text())

def test_woe_iv_hand_computed():
    """Bin 0: 3 good/1 bad, bin 1: 2 good/2 bad -> IV = 0.2929633 closed form."""
    bins = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    y = pd.Series([0, 0, 0, 1, 0, 0, 1, 1])
    assert woe_iv(bins, y) == pytest.approx(0.2929633, abs=1e-4)

def test_iv_symmetric_under_label_flip():
    """Swapping good/bad labels leaves IV unchanged (WOE flips sign, IV does not)."""
    bins = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    y = pd.Series([0, 0, 0, 1, 0, 0, 1, 1])
    assert woe_iv(bins, y) == pytest.approx(woe_iv(bins, 1 - y))

def test_iv_dti_matches_full_book(loans):
    """Sample dti IV (5 quantile bins) within 15% of the fitted scorecard artifact."""
    full = json.loads((ROOT / "output" / "model_metrics.json").read_text())
    dti_iv = next(f["iv"] for f in full["feature_iv"] if f["feature"] == "dti")
    iv = woe_iv(pd.qcut(loans["dti"], 5, labels=False, duplicates="drop"), loans["default_flag"])
    assert abs(iv - dti_iv) / dti_iv < 0.15

def test_ecl_identity_per_loan(loans):
    """IFRS 9 ECL per loan = EAD x PD x LGD(0.60), matching the exported column."""
    ecl = loans["loan_amnt"] * loans["probability_of_default"] * 0.60
    assert np.allclose(ecl, loans["expected_loss"], rtol=1e-3, atol=0.05)

def test_portfolio_ecl_scaled_to_full_book(loans):
    """25k-row sample grossed up by its scale factor reproduces the $5.62B book ECL."""
    meta = meta_doc()
    scaled = loans["expected_loss"].sum() * meta["scale_factor"]
    assert scaled == pytest.approx(meta["full_ecl"], rel=0.001)
    assert scaled / meta["full_exposure"] == pytest.approx(
        meta["full_ecl"] / meta["full_exposure"], abs=1e-4)

def test_discrimination_preserved(loans):
    """Sample ROC-AUC within 0.0010 of full book; Gini = 2*AUC-1 stays 0.41."""
    meta = meta_doc()
    auc = roc_auc_score(loans["default_flag"], loans["probability_of_default"])
    assert auc == pytest.approx(meta["sample_auc"], abs=5e-4)
    assert abs(auc - meta["full_auc"]) <= 0.0010
    assert abs(2 * auc - 1 - 0.41) <= 0.005

def test_sample_shape_and_determinism(loans):
    """Exactly 25,000 rows, one row per loan id, grades A-G all present."""
    assert len(loans) == 25_000
    assert loans["id"].is_unique
    assert set(loans["grade"]) == set("ABCDEFG")
