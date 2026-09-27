"""Peds track, Lesson P6 - Synthetic data & privacy, subgroup fairness, transportability.
Reads p2_features.csv; reuses the 11-predictor logistic model from P3-P5."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import brentq
from sklearn.tree import DecisionTreeRegressor

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


rng = np.random.default_rng(0)

# === 1. Synthetic data: make our own CART synthesis and test utility + privacy =============
D = X[PRED].assign(died=y)
real_tr, real_te = train_test_split(D, test_size=0.3, stratify=D["died"], random_state=0)


def cart_synthesize(real, n, min_leaf=10, seed=0):
    """Sequential CART (the synthpop method): column 1 from its marginal; each later column
    drawn from real donors in the same tree leaf, given the synthetic columns so far."""
    r = np.random.default_rng(seed)
    cols = list(real.columns)
    syn = pd.DataFrame(index=range(n), columns=cols, dtype=float)
    syn[cols[0]] = r.choice(real[cols[0]].to_numpy(), n)
    for j, col in enumerate(cols[1:], 1):
        prev = cols[:j]
        tree = DecisionTreeRegressor(min_samples_leaf=min_leaf, random_state=seed)
        target = real[col].fillna(-999)
        tree.fit(real[prev].fillna(-999), target)
        leaf_real, leaf_syn = tree.apply(real[prev].fillna(-999)), tree.apply(syn[prev].fillna(-999))
        donors = pd.Series(real[col].to_numpy()).groupby(leaf_real).apply(lambda s: s.to_numpy())
        syn[col] = [r.choice(donors[l]) for l in leaf_syn]
    return syn


syn = cart_synthesize(real_tr, len(real_tr))
print("Mortality real vs synthetic:", round(real_tr["died"].mean(), 3), round(syn["died"].mean(), 3))
print(pd.DataFrame({"real mean": real_tr.mean(), "synthetic mean": syn.mean()}).round(2).to_string())


def auroc_on_real_test(train):
    m = logreg().fit(train[PRED], train["died"].astype(int))
    return roc_auc_score(real_te["died"], m.predict_proba(real_te[PRED])[:, 1])


print(f"\nTRTR (train real, test real)      AUROC {auroc_on_real_test(real_tr):.3f}")
print(f"TSTR (train synthetic, test real) AUROC {auroc_on_real_test(syn):.3f}")


def dcr(a, b, scale):
    """Distance from each row of a to its closest row in b (standardised, NaN -> -1 flag)."""
    A = ((a - scale.mean()) / scale.std()).fillna(-9).to_numpy()
    B = ((b - scale.mean()) / scale.std()).fillna(-9).to_numpy()
    return np.array([np.sqrt(((B - r) ** 2).sum(1)).min() for r in A])


d_syn = dcr(syn[PRED], real_tr[PRED], real_tr[PRED])
d_hold = dcr(real_te[PRED], real_tr[PRED], real_tr[PRED])
print(f"\nDistance to closest real training record: synthetic median {np.median(d_syn):.2f}, "
      f"real hold-out median {np.median(d_hold):.2f}")
print(f"Exact copies of a training child: synthetic {np.mean(d_syn == 0):.1%}, hold-out {np.mean(d_hold == 0):.1%}")
syn_leaky = cart_synthesize(real_tr, len(real_tr), min_leaf=1)
d_leaky = dcr(syn_leaky[PRED], real_tr[PRED], real_tr[PRED])
print(f"min_leaf=1 synthesis: exact copies {np.mean(d_leaky == 0):.1%}, TSTR AUROC {auroc_on_real_test(syn_leaky):.3f}")

# === 2. Subgroup performance ================================================================
oof = np.zeros(len(y))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=1).split(X, y):
    oof[te] = logreg().fit(X.iloc[tr][PRED], y[tr]).predict_proba(X.iloc[te][PRED])[:, 1]
fpr, tpr, thr = roc_curve(y, oof)
t20 = thr[np.where(fpr <= 0.20)[0][-1]]
flag = oof >= t20

groups = {
    "sex": X["sex_adm"],
    "age": pd.cut(X["agecalc_adm"], [0, 12, 24, 61], right=False, labels=["6-11 m", "12-23 m", "24-60 m"]),
    "HIV": X["hivstatus_adm"],
    "SAM": X["sam"].map({0: "no SAM", 1: "SAM"}),
    "malaria": X["malariastatuspos_adm"].map({"Yes": "malaria +", "No": "malaria -"}),
    "mother's education": X["momedu_adm"],
}
rows = []
for g, s in groups.items():
    for lvl in s.dropna().unique():
        m = (s == lvl).to_numpy()
        yy, pp, ff = y[m], oof[m], flag[m]
        boots = []
        for _ in range(300):
            b = rng.integers(0, m.sum(), m.sum())
            if 0 < yy[b].sum() < len(b):
                boots.append(roc_auc_score(yy[b], pp[b]))
        rows.append([g, lvl, m.sum(), yy.sum(), roc_auc_score(yy, pp) if 0 < yy.sum() < m.sum() else np.nan,
                     np.percentile(boots, 2.5), np.percentile(boots, 97.5), yy.mean() / pp.mean(),
                     ff[yy == 1].mean(), ff[yy == 0].mean()])
sub = pd.DataFrame(rows, columns=["group", "level", "n", "deaths", "AUROC", "lo", "hi", "O/E", "sens", "FPR"])
print("\n", sub.round(2).to_string())

# === 3. Transportability ====================================================================
model = logreg().fit(X[PRED], y)
p_all = model.predict_proba(X[PRED])[:, 1]
logit = lambda p: np.log(p / (1 - p))

# 3a. prevalence shift: a hospital with 1.5% mortality (resample survivors/deaths)
for target in [0.015, 0.10]:
    n_d = int(round(target * 2000)); idx = np.concatenate([rng.choice(np.where(y == 1)[0], n_d),
                                                           rng.choice(np.where(y == 0)[0], 2000 - n_d)])
    yy, pp = y[idx], oof[idx]              # out-of-fold risks, so AUROC is not in-sample
    # intercept-only recalibration ("update the baseline risk") on half, evaluate on the other half
    half = rng.permutation(len(idx)); a, b = half[:1000], half[1000:]
    c = brentq(lambda c: (yy[a] - 1 / (1 + np.exp(-(logit(pp[a]) + c)))).sum(), -5, 5)
    p_new = 1 / (1 + np.exp(-(logit(pp[b]) + c)))
    print(f"\nSite with {target:.1%} mortality: mean predicted {pp[b].mean():.3f} vs observed {yy[b].mean():.3f}; "
          f"AUROC {roc_auc_score(yy[b], pp[b]):.3f}; after intercept update ({c:+.2f}) mean predicted {p_new.mean():.3f}")

# 3b. practice shift: hospital that measures lactate in every child
Xs = X[PRED].copy(); Xs["lactate_measured"] = 1.0
unm = X["lactate_measured"].to_numpy() == 0
print(f"\nLactate measured in everyone -> previously unmeasured children: mean predicted "
      f"{p_all[unm].mean():.3f} -> {model.predict_proba(Xs)[:, 1][unm].mean():.3f} (observed {y[unm].mean():.3f})")

# 3c. case-mix shift: train on infants/toddlers (<24 m), test on older children (>=24 m)
young = (X["agecalc_adm"] < 24).to_numpy()
m_y = logreg().fit(X[young][PRED], y[young])
p_old = m_y.predict_proba(X[~young][PRED])[:, 1]
print(f"Train <24 m (n={young.sum()}, mort {y[young].mean():.3f}) -> test >=24 m (n={(~young).sum()}, "
      f"mort {y[~young].mean():.3f}): AUROC {roc_auc_score(y[~young], p_old):.3f}, mean predicted {p_old.mean():.3f}")
