"""Peds track, Lesson P4 - Calibration, class imbalance, choosing a threshold, decision curves.
Reads p2_features.csv; reuses the 11-predictor logistic model from P3."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X = pd.read_csv("p2_features.csv")
y = X.pop("inhospital_mortality").to_numpy()
yes = lambda col, val: (X[col] == val).astype(float).where(X[col].notna())
X["caprefill"] = yes("caprefill_adm", "Yes")
X["respdistress"] = yes("respdistress_adm", "Yes")
X["hiv_pos"] = yes("hivstatus_adm", "HIV positive")
X["malaria_pos"] = yes("malariastatuspos_adm", "Yes")
X["not_feeding"] = (X["feedingstatus_adm"] == "Not feeding at all").astype(float)
NUM = ["bcs_total", "waz", "lactate_mmolpl_adm", "spo2_min", "rr_ratio"]
BIN = ["lactate_measured", "caprefill", "respdistress", "hiv_pos", "malaria_pos", "not_feeding"]
PRED = NUM + BIN
bedside = (X["bcs_coma"] + X["sam"] + X["hypoxaemia"] + X["caprefill"].fillna(0)
           + X["respdistress"].fillna(0) + X["not_feeding"]).to_numpy()


def make_model(class_weight=None):
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM),
        ("bin", SimpleImputer(strategy="most_frequent"), BIN)])
    return Pipeline([("prep", prep), ("lr", LogisticRegression(C=1e6, max_iter=5000, class_weight=class_weight))])


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calib_stats(y, p):
    """Calibration-in-the-large (O/E), calibration slope & intercept, Brier and scaled Brier."""
    lr = LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), y)
    brier = brier_score_loss(y, p)
    return {"mean predicted": p.mean(), "observed": y.mean(), "O/E": y.mean() / p.mean(),
            "slope": lr.coef_[0, 0], "intercept": lr.intercept_[0],
            "Brier": brier, "scaled Brier": 1 - brier / (y.mean() * (1 - y.mean())),
            "AUROC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p)}


# --- Out-of-fold predictions, with a threshold chosen on each TRAINING fold -------------
skf = StratifiedKFold(5, shuffle=True, random_state=1)
oof = {"plain": np.zeros(len(y)), "balanced": np.zeros(len(y)), "oversampled": np.zeros(len(y))}
flag = np.zeros(len(y), bool)
rng = np.random.default_rng(0)
for tr, te in skf.split(X, y):
    m = make_model().fit(X.iloc[tr][PRED], y[tr])
    oof["plain"][te] = m.predict_proba(X.iloc[te][PRED])[:, 1]
    # threshold: 80th centile of predicted risk among TRAINING survivors -> FPR <= 20% (on training)
    p_tr = m.predict_proba(X.iloc[tr][PRED])[:, 1]
    thr = np.quantile(p_tr[y[tr] == 0], 0.80)
    flag[te] = oof["plain"][te] >= thr
    print(f"fold threshold = {thr:.3f}")
    oof["balanced"][te] = make_model("balanced").fit(X.iloc[tr][PRED], y[tr]).predict_proba(X.iloc[te][PRED])[:, 1]
    # random oversampling of deaths INSIDE the training fold only
    pos = tr[y[tr] == 1]
    tr_os = np.concatenate([tr, rng.choice(pos, size=(y[tr] == 0).sum() - len(pos), replace=True)])
    oof["oversampled"][te] = make_model().fit(X.iloc[tr_os][PRED], y[tr_os]).predict_proba(X.iloc[te][PRED])[:, 1]

print("\nCalibration & discrimination (out-of-fold):")
print(pd.DataFrame({k: calib_stats(y, p) for k, p in oof.items()}).round(3).to_string())

# --- Reliability table (deciles of predicted risk) --------------------------------------
dec = pd.qcut(oof["plain"], 10, labels=False)
print("\nDeciles:\n", pd.DataFrame({"mean predicted %": pd.Series(oof["plain"]).groupby(dec).mean() * 100,
                                    "observed %": pd.Series(y).groupby(dec).mean() * 100}).round(1).to_string())

# --- Operating point ---------------------------------------------------------------------
tp, fp = (flag & (y == 1)).sum(), (flag & (y == 0)).sum()
fn, tn = (~flag & (y == 1)).sum(), (~flag & (y == 0)).sum()
print(f"\nThreshold from training folds, applied to test folds:")
print(f"  TP {tp}  FP {fp}  FN {fn}  TN {tn}")
print(f"  sensitivity {tp/(tp+fn):.2f} | specificity {tn/(tn+fp):.2f} | FPR {fp/(fp+tn):.2f}")
print(f"  PPV {tp/(tp+fp):.3f} (NNE {(tp+fp)/tp:.1f}) | NPV {tn/(tn+fn):.3f} | flagged {flag.mean():.1%}")
b = bedside >= 2
print(f"  bedside>=2: sens {(b & (y==1)).sum()/y.sum():.2f} | FPR {(b & (y==0)).sum()/(y==0).sum():.2f} "
      f"| PPV {(b & (y==1)).sum()/b.sum():.3f}")


# --- Decision curve analysis --------------------------------------------------------------
def net_benefit(y, flagged, pt):
    n = len(y)
    return (flagged & (y == 1)).sum() / n - (flagged & (y == 0)).sum() / n * pt / (1 - pt)


print("\nNet benefit (per 100 children):")
for pt in [0.02, 0.05, 0.10, 0.15, 0.20]:
    print(f"  pt={pt:.2f}  model {100*net_benefit(y, oof['plain'] >= pt, pt):5.2f} | "
          f"bedside>=2 {100*net_benefit(y, b, pt):5.2f} | treat all {100*net_benefit(y, np.ones(len(y), bool), pt):5.2f}")
