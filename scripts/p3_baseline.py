"""Peds track, Lesson P3 - Baseline logistic regression vs a simple bedside score.
Reads p2_features.csv (from p2_features.py)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X = pd.read_csv("p2_features.csv")
y = X.pop("inhospital_mortality").to_numpy()

# --- 1. Predictors chosen clinically (EPV: 119 deaths / 11 = ~11) ---------------------
X["caprefill"] = (X["caprefill_adm"] == "Yes").astype(float).where(X["caprefill_adm"].notna())
X["respdistress"] = (X["respdistress_adm"] == "Yes").astype(float).where(X["respdistress_adm"].notna())
X["hiv_pos"] = (X["hivstatus_adm"] == "HIV positive").astype(float).where(X["hivstatus_adm"].notna())
X["malaria_pos"] = (X["malariastatuspos_adm"] == "Yes").astype(float).where(X["malariastatuspos_adm"].notna())
X["not_feeding"] = (X["feedingstatus_adm"] == "Not feeding at all").astype(float)
NUM = ["bcs_total", "waz", "lactate_mmolpl_adm", "spo2_min", "rr_ratio"]
BIN = ["lactate_measured", "caprefill", "respdistress", "hiv_pos", "malaria_pos", "not_feeding"]
PRED = NUM + BIN
print(f"{len(PRED)} predictors, EPV = {y.sum() / len(PRED):.1f}")

# --- 2. Pipeline: impute -> scale -> logistic regression ------------------------------
prep = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM),
    ("bin", SimpleImputer(strategy="most_frequent"), BIN)])
model = Pipeline([("prep", prep), ("lr", LogisticRegression(C=1e6, max_iter=5000))])  # ~unpenalised


def tpr_at_fpr(y_true, score, max_fpr=0.20):
    fpr, tpr, _ = roc_curve(y_true, score)
    return tpr[fpr <= max_fpr].max()


# --- 3. Bedside score: 1 point per danger sign (no fitting) ---------------------------
bedside = (X["bcs_coma"] + X["sam"] + X["hypoxaemia"] + X["caprefill"].fillna(0)
           + X["respdistress"].fillna(0) + X["not_feeding"])
print("\nBedside score distribution & mortality:")
print(pd.DataFrame({"n": bedside.value_counts().sort_index(),
                    "mortality_%": (pd.Series(y).groupby(bedside).mean() * 100).round(1)}))

# --- 4. Repeated stratified cross-validation ------------------------------------------
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
rows = []
for tr, te in cv.split(X, y):
    p = model.fit(X.iloc[tr][PRED], y[tr]).predict_proba(X.iloc[te][PRED])[:, 1]
    b = bedside.iloc[te].to_numpy()
    for name, s in [("logistic (11)", p), ("bedside score", b)]:
        rows.append([name, roc_auc_score(y[te], s), average_precision_score(y[te], s),
                     tpr_at_fpr(y[te], s)])
res = pd.DataFrame(rows, columns=["model", "AUROC", "AUPRC", "TPR@FPR<=20%"])
print("\n5-fold x 10 CV (mean, SD):")
print(res.groupby("model").agg(["mean", "std"]).round(3).to_string())
print(f"AUPRC of a random guess = prevalence = {y.mean():.3f}")

# --- 5. Odds ratios on the full data --------------------------------------------------
model.fit(X[PRED], y)
coef = pd.Series(model["lr"].coef_[0], index=PRED)
print("\nOdds ratios (numeric: per 1 SD; binary: yes vs no):")
print(np.exp(coef).round(2).sort_values().to_string())
print("Apparent (training) AUROC:", round(roc_auc_score(y, model.predict_proba(X[PRED])[:, 1]), 3))

# --- 6. Kitchen sink: every column, no penalty vs L2 penalty --------------------------
allnum = X.select_dtypes("number").columns.tolist()
allcat = X.select_dtypes(exclude="number").columns.tolist()
sink_prep = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), allnum),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=10))]), allcat)])
n_param = sink_prep.fit_transform(X).shape[1]
print(f"\nKitchen sink: {n_param} parameters, EPV = {y.sum() / n_param:.2f}")
for name, C in [("no penalty", 1e6), ("L2, C=0.01", 0.01)]:
    sink = Pipeline([("prep", sink_prep), ("lr", LogisticRegression(C=C, max_iter=5000))])
    tr_auc, cv_auc = [], []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0).split(X, y):
        sink.fit(X.iloc[tr], y[tr])
        tr_auc.append(roc_auc_score(y[tr], sink.predict_proba(X.iloc[tr])[:, 1]))
        cv_auc.append(roc_auc_score(y[te], sink.predict_proba(X.iloc[te])[:, 1]))
    print(f"  {name:12s} train AUROC {np.mean(tr_auc):.3f} | CV AUROC {np.mean(cv_auc):.3f}")

# What does the penalised kitchen sink lean on? (watch for treatment / data-quality proxies)
sink = Pipeline([("prep", sink_prep), ("lr", LogisticRegression(C=0.01, max_iter=5000))]).fit(X, y)
co = pd.Series(sink["lr"].coef_[0], index=sink["prep"].get_feature_names_out())
print("\nTop 15 |coefficients| (L2 kitchen sink):")
print(co[co.abs().sort_values(ascending=False).index[:15]].round(2).to_string())
