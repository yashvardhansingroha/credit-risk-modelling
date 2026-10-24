"""Streamlit dashboard replicating the Power BI DAX measures (4_powerbi_dax.txt).

Runs on output/sample_scored_loans.csv, a 25,000-row stratified sample of the 1.34M-loan
scored book; dollar KPIs are grossed up to the full book via the sample's scale factor.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent; OUT = ROOT / "output"

@st.cache_data(show_spinner="Loading scored portfolio...")
def load_data():
    if not all((OUT / f).exists() for f in ("sample_scored_loans.csv", "sample_meta.json")):
        subprocess.run([sys.executable, str(ROOT / "create_light_sample.py")], check=True, cwd=ROOT)
    df = pd.read_csv(OUT / "sample_scored_loans.csv")
    meta = json.loads((OUT / "sample_meta.json").read_text(encoding="utf-8"))
    full = json.loads((OUT / "model_metrics.json").read_text(encoding="utf-8"))
    return df, meta, full

def auc_ks(y, pds):
    """Mann-Whitney AUC and two-sample KS from the PD ranking (vectorised, no sklearn)."""
    y, r = y.to_numpy(), pds.rank(method="average").to_numpy()
    n1, n0 = float(y.sum()), float(len(y) - y.sum())
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)
    o = np.argsort(pds.to_numpy()); cum = np.cumsum(y[o])
    ks = float(np.max(np.abs(cum / n1 - (np.arange(1, len(y) + 1) - cum) / n0)))
    return float(auc), ks

def psi(exp, act, bins=10):
    """Population Stability Index on train-quantile bins (mirrors 5_portfolio_analytics.py)."""
    exp, act = exp[~np.isnan(exp)], act[~np.isnan(act)]
    edges = np.unique(np.quantile(exp, np.linspace(0, 1, bins + 1)))
    e = np.clip(np.histogram(exp, bins=edges)[0] / len(exp), 1e-6, None)
    a = np.clip(np.histogram(act, bins=edges)[0] / len(act), 1e-6, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))

st.set_page_config(page_title="Lending Club Credit Risk", page_icon="🏦", layout="wide")
st.title("🏦 Lending Club Credit Risk — IFRS 9 ECL & Scorecard Monitoring")
df, meta, full = load_data()
SCALE, LGD = meta["scale_factor"], meta["lgd"]
EXPOSURE = float(df["loan_amnt"].sum() * SCALE)
AUC, KS = auc_ks(df["default_flag"], df["probability_of_default"])
BASE_ECL = float((df["loan_amnt"] * df["probability_of_default"] * LGD).sum() * SCALE)

with st.sidebar:
    st.markdown(f"**🏦 Credit Risk** — Lending Club 2007–2018\n\n"
                f"**Sample** {meta['n_sample']:,} of {meta['n_full']:,} loans — grade × default × "
                f"PD-decile stratified (seed {meta['seed']})\n\n**Scale factor** ×{SCALE:,.0f} to "
                f"full book\n\n**LGD** {LGD:.0%} — conservative unsecured consumer assumption")

t1, t2, t3, t4 = st.tabs(["📈 IFRS 9 ECL Portfolio", "🔀 Grade Migration", "🌡️ Macro Stress Simulator", "🎯 Scorecard & PSI"])

with t1:
    c = st.columns(5)
    c[0].metric("Total Exposure (EAD)", f"${EXPOSURE/1e9:.2f}B")
    c[1].metric("Portfolio ECL (PD×LGD×EAD)", f"${BASE_ECL/1e9:.2f}B")
    c[2].metric("ECL Coverage Ratio", f"{BASE_ECL/EXPOSURE:.2%}", f"full book {full['ecl_coverage_ratio']:.2%}")
    c[3].metric("Exposure-weighted PD", f"{(df['loan_amnt']*df['probability_of_default']).sum()/df['loan_amnt'].sum():.2%}")
    c[4].metric("Observed Default Rate", f"{df['default_flag'].mean():.2%}")
    c = st.columns(4)
    c[0].metric("ROC-AUC (sample)", f"{AUC:.4f}", f"full book {full['roc_auc']:.4f}")
    c[1].metric("Gini (2×AUC−1)", f"{2*AUC-1:.4f}")
    c[2].metric("Underpriced Exposure", f"{df.loc[df['underpriced_flag']==1,'loan_amnt'].sum()*SCALE/EXPOSURE:.1%}",
                f"full book {full['underpriced_exposure_pct']:.1%}")
    c[3].metric("Avg Pricing Adequacy", f"{df['pricing_adequacy'].mean():+.2%}")
    left, right = st.columns(2)
    left.subheader("ECL by grade (scaled $B)")
    left.bar_chart((df.groupby("grade")["expected_loss"].sum() * SCALE / 1e9).round(3))
    right.subheader("Default rate by origination vintage"); right.line_chart(df.groupby("vintage_year")["default_flag"].mean().round(4))

with t2:
    st.subheader("Grade × vintage migration matrix")
    metric = st.selectbox("Metric", ["Observed default rate", "Exposure share", "Average PD"])
    val, agg = {"Observed default rate": ("default_flag", "mean"), "Exposure share": ("loan_amnt", "sum"),
                "Average PD": ("probability_of_default", "mean")}[metric]
    piv = df.pivot_table(index="grade", columns="vintage_year", values=val, aggfunc=agg)
    if metric == "Exposure share":
        piv = piv / df["loan_amnt"].sum()
    st.dataframe(piv.map(lambda v: "—" if pd.isna(v) else f"{v:.1%}"), use_container_width=True)
    st.line_chart(df.pivot_table(index="vintage_year", columns="grade", values="default_flag", aggfunc="mean").round(4))

with t3:
    st.subheader("IFRS 9 stress — PD multiplier and LGD sliders (stressed PD capped at 0.99)")
    col1, col2 = st.columns(2)
    mult = col1.slider("PD multiplier (severe scenario = 2.0)", 0.5, 3.0, 1.0, 0.05)
    lgd = col2.slider("LGD (downturn severity)", 0.10, 1.00, 0.60, 0.05)
    st_pd = np.minimum(df["probability_of_default"] * mult, 0.99)
    stressed = float((df["loan_amnt"] * st_pd * lgd).sum() * SCALE)
    c = st.columns(4)
    c[0].metric("Stressed ECL", f"${stressed/1e9:.2f}B")
    c[1].metric("Δ vs base", f"{(stressed-BASE_ECL)/1e9:+.2f}B", f"{stressed/BASE_ECL-1:+.1%}")
    c[2].metric("ECL coverage", f"{stressed/EXPOSURE:.2%}")
    c[3].metric("Stressed EAD-weighted PD", f"{(df['loan_amnt']*st_pd).sum()/df['loan_amnt'].sum():.2%}")
    by_grade = pd.DataFrame({"Base": df.groupby("grade")["expected_loss"].sum() * SCALE / 1e9,
                             "Stressed": (df["loan_amnt"]*st_pd*lgd).groupby(df["grade"]).sum() * SCALE / 1e9}).round(3)
    st.bar_chart(by_grade)
    sd = pd.read_csv(OUT / "stress_test_results.csv")
    sd["total_ecl"], sd["delta_vs_base"] = (sd["total_ecl"]/1e9).round(3), (sd["delta_vs_base"]/1e9).round(3)
    st.dataframe(sd.set_index("scenario"))

with t4:
    st.subheader("Application scorecard — score-decile calibration (300–850 scale)")
    cal = (df.assign(decile=pd.qcut(df["application_score"], 10, labels=False, duplicates="drop"))
           .groupby("decile").agg(n_loans=("id", "size"), avg_score=("application_score", "mean"),
                                  obs_default_rate=("default_flag", "mean"), model_avg_pd=("probability_of_default", "mean")))
    rho = cal["obs_default_rate"].corr(pd.Series(cal.index), method="spearman")
    st.caption(f"Score-decile Spearman rank correlation vs observed default rate: {rho:.2f} (−1 = perfectly rank-ordering)")
    disp = cal.assign(avg_score=cal["avg_score"].map("{:.0f}".format),
                      obs_default_rate=cal["obs_default_rate"].map("{:.2%}".format),
                      model_avg_pd=cal["model_avg_pd"].map("{:.2%}".format))
    st.dataframe(disp, use_container_width=True)
    st.subheader("Population Stability Index — train vs test (thresholds 0.10 / 0.25)")
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    rows = [{"Feature": f, "PSI": round(psi(tr[c].to_numpy(float), te[c].to_numpy(float)), 4)} for f, c in
            [("Application score", "application_score"), ("Model PD", "probability_of_default"), ("DTI", "dti")]]
    cats = sorted(df["grade"].unique())
    eg = tr["grade"].value_counts(normalize=True).reindex(cats, fill_value=1e-6).to_numpy()
    ag = te["grade"].value_counts(normalize=True).reindex(cats, fill_value=1e-6).to_numpy()
    rows.append({"Feature": "Grade mix", "PSI": round(float(np.sum((ag-eg)*np.log(ag/eg))), 4)})
    psidf = pd.DataFrame(rows)
    psidf["Status"] = np.where(psidf["PSI"] < 0.10, "🟢 stable", np.where(psidf["PSI"] < 0.25, "🟠 moderate", "🔴 significant"))
    st.dataframe(psidf, hide_index=True, use_container_width=True)
    st.subheader("Information Value by feature (fitted on train — output/woe_iv_summary.csv)")
    st.bar_chart(pd.read_csv(OUT / "woe_iv_summary.csv").groupby("feature")["feature_iv"].first().sort_values().round(4))
    st.caption(f"Sample KS {KS:.3f} · full-book model: AUC {full['roc_auc']:.4f}, Gini {full['gini']:.4f}, "
               f"F1 {full['f1']:.3f} @ threshold {full['decision_threshold']}, train/test {full['train_size']:,}/{full['test_size']:,}")
