"""Peds track, Lesson P5 - Gradient boosting vs logistic regression, bootstrap CIs, SHAP.
Reads p2_features.csv. SHAP values are computed exactly (interventional Shapley over all
2^11 feature coalitions) for teaching; shap.TreeExplainer gives the same values, faster."""
import itertools
import math
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
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


def logreg():
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM),
        ("bin", SimpleImputer(strategy="most_frequent"), BIN)])
    return Pipeline([("prep", prep), ("lr", LogisticRegression(C=1e6, max_iter=5000))])


def gbm():
    # handles NaN natively; shallow trees + strong regularisation for 119 events
    return HistGradientBoostingClassifier(learning_rate=0.03, max_iter=200, max_depth=3,
                                          min_samples_leaf=40, l2_regularization=1.0, random_state=0)


MODELS = {"logistic": logreg, "boosting": gbm}

# --- 1. Same CV splits for both models -> paired comparison ------------------------------
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
rows, oof = [], {k: np.zeros(len(y)) for k in MODELS}
t = time.time()
for i, (tr, te) in enumerate(cv.split(X, y)):
    for name, make in MODELS.items():
        p = make().fit(X.iloc[tr][PRED], y[tr]).predict_proba(X.iloc[te][PRED])[:, 1]
        rows.append([i, name, roc_auc_score(y[te], p), average_precision_score(y[te], p), brier_score_loss(y[te], p)])
        if i < 5:
            oof[name][te] = p          # first repeat = one out-of-fold prediction per child
res = pd.DataFrame(rows, columns=["fold", "model", "AUROC", "AUPRC", "Brier"])
print(f"CV done in {time.time() - t:.0f}s")
print(res.groupby("model")[["AUROC", "AUPRC", "Brier"]].agg(["mean", "std"]).round(3).to_string())
diff = res.pivot(index="fold", columns="model", values="AUROC").eval("boosting - logistic")
print(f"Paired AUROC difference (boosting - logistic): {diff.mean():+.3f}, boosting better in {(diff > 0).mean():.0%} of folds")

# --- 2. Bootstrap 95% CI on out-of-fold predictions --------------------------------------
rng = np.random.default_rng(0)
boot = []
for _ in range(1000):
    b = rng.integers(0, len(y), len(y))
    if y[b].sum() == 0:
        continue
    boot.append([roc_auc_score(y[b], oof["logistic"][b]), roc_auc_score(y[b], oof["boosting"][b])])
boot = np.array(boot)
for j, name in enumerate(MODELS):
    print(f"{name:9s} AUROC {roc_auc_score(y, oof[name]):.3f} (95% CI {np.percentile(boot[:, j], 2.5):.3f}-{np.percentile(boot[:, j], 97.5):.3f})")
d = boot[:, 1] - boot[:, 0]
print(f"difference {d.mean():+.3f} (95% CI {np.percentile(d, 2.5):+.3f} to {np.percentile(d, 97.5):+.3f})")
for name in MODELS:
    print(f"{name:9s} calibration: mean predicted {oof[name].mean():.3f} vs observed {y.mean():.3f}")


# --- 3. Exact SHAP (interventional Shapley values) ---------------------------------------
def exact_shap(predict, X_explain, X_background):
    """phi_j = sum over coalitions S not containing j of |S|!(p-|S|-1)!/p! * [v(S+j) - v(S)],
    v(S) = mean prediction when features in S come from the child and the rest from background."""
    p = X_explain.shape[1]
    Xe, Xb = X_explain.to_numpy(float), X_background.to_numpy(float)
    nb = len(Xb)
    masks = np.array(list(itertools.product([0, 1], repeat=p)), bool)       # 2^p coalitions
    v = np.zeros((len(Xe), len(masks)))
    for k, m in enumerate(masks):
        Z = np.repeat(Xb[None], len(Xe), 0)                                    # (n, nb, p)
        Z[:, :, m] = Xe[:, None, m]
        v[:, k] = predict(pd.DataFrame(Z.reshape(-1, p), columns=X_explain.columns)).reshape(len(Xe), nb).mean(1)
    index = {tuple(m): k for k, m in enumerate(masks)}
    phi = np.zeros((len(Xe), p))
    for k, m in enumerate(masks):
        s = m.sum()
        for j in np.where(~m)[0]:
            m2 = m.copy(); m2[j] = True
            w = math.factorial(s) * math.factorial(p - s - 1) / math.factorial(p)
            phi[:, j] += w * (v[:, index[tuple(m2)]] - v[:, k])
    return phi, v[:, 0]                                                        # v(empty) = base value


def log_odds(model):
    return lambda Z: np.log(np.clip(pr := model.predict_proba(Z)[:, 1], 1e-9, 1) / np.clip(1 - pr, 1e-9, 1))


fitted = {name: make().fit(X[PRED], y) for name, make in MODELS.items()}
bg = X[PRED].sample(50, random_state=0)
ex = X[PRED].sample(300, random_state=1)
t = time.time()
shap = {name: exact_shap(log_odds(m), ex, bg)[0] for name, m in fitted.items()}
print(f"\nExact SHAP for 300 children x 2 models in {time.time() - t:.0f}s")
imp = pd.DataFrame({name: np.abs(s).mean(0) for name, s in shap.items()}, index=PRED)
print("Mean |SHAP| (log-odds):\n", imp.sort_values("boosting", ascending=False).round(3).to_string())
