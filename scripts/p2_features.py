"""Peds track, Lesson P2 - Feature engineering: leakage removal, cleaning, WHO z-scores,
Blantyre Coma Score, age-adjusted vitals. Writes p2_features.csv for P3.

WHO growth-standard LMS tables (who_growth/*.txt) come from the WHO `anthro` R package
(github.com/WorldHealthOrganization/anthro, data-raw/growthstandards, GPL-3).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

df = pd.read_csv("PSDC_SyntheticTrainingData_Dataset_ODR.tab", sep="\t")
y = (df["inhospital_mortality"] == 1).astype(int)

# --- 1. Leakage & structural columns -------------------------------------------------
LEAKAGE = ["lengthadm", "lactate2_mmolpl_adm"]            # known only after admission
IDS = ["studyid_adm"]
SKIP_LOGIC = ["nonexclbreastfed_adm", "spo2other_adm", "vaccmeaslessource_adm",
              "vaccpneumocsource_adm", "vaccdptsource_adm", "momagefirstpreg_adm"]
X = df.drop(columns=LEAKAGE + IDS + SKIP_LOGIC + ["inhospital_mortality"])
print(f"Kept {X.shape[1]} of {df.shape[1]} columns")

# --- 2. Clean physiologically impossible values -> missing ---------------------------
LIMITS = {"glucose_mmolpl_adm": (0.5, 40), "hematocrit_gpdl_adm": (5, 65),
          "hr_bpm_adm": (40, 250), "rr_brpm_app_adm": (8, 120), "sysbp_mmhg_adm": (40, 180),
          "temp_c_adm": (32, 43), "spo2site1_pc_oxi_adm": (40, 100), "spo2site2_pc_oxi_adm": (40, 100)}
for col, (lo, hi) in LIMITS.items():
    bad = X[col].notna() & ~X[col].between(lo, hi)
    print(f"  {col}: {bad.sum()} implausible -> NaN")
    X.loc[bad, col] = np.nan

# --- 3. WHO z-scores (LMS method) ----------------------------------------------------
def lms_z(value, L, M, S, restricted=True):
    """WHO LMS z-score. restricted=True applies WHO's tail adjustment beyond +/-3 SD
    (used for weight-based indicators and MUAC, not for length/height)."""
    z = np.where(L == 0, np.log(value / M) / S, ((value / M) ** L - 1) / (L * S))
    if restricted:
        sd = lambda k: M * (1 + L * S * k) ** (1 / L)
        z = np.where(z > 3, 3 + (value - sd(3)) / (sd(3) - sd(2)), z)
        z = np.where(z < -3, -3 + (value - sd(-3)) / (sd(-2) - sd(-3)), z)
    return z


def ref(name):
    return pd.read_csv(f"who_growth/{name}.txt", sep="\t")


sex = X["sex_adm"].map({"Male": 1, "Female": 2})
age_days = (X["agecalc_adm"] * 30.4375).round().astype(int)
under2 = X["agecalc_adm"] < 24          # WHO: recumbent length < 24 m, standing height >= 24 m


def by_age(table, value):
    t = ref(table).set_index(["sex", "age"])
    lms = t.reindex(pd.MultiIndex.from_arrays([sex, age_days])).to_numpy()[:, :3].astype(float)
    return lms_z(value.to_numpy(float), *lms.T, restricted=(table != "lenanthro"))


X["waz"] = by_age("weianthro", X["weight_kg_adm"])            # weight-for-age
X["haz"] = by_age("lenanthro", X["height_cm_adm"])            # length/height-for-age
X["muacz"] = by_age("acanthro", X["muac_mm_adm"] / 10)       # MUAC-for-age (table in cm)

# weight-for-length (<2 y) or weight-for-height (>=2 y), looked up by body size to 0.1 cm
wfl, wfh = ref("wflanthro").set_index(["sex", "length"]), ref("wfhanthro").set_index(["sex", "height"])
size = X["height_cm_adm"].round(1)
lms = np.full((len(X), 3), np.nan)
for tab, mask in [(wfl, under2), (wfh, ~under2)]:
    idx = pd.MultiIndex.from_arrays([sex[mask], size[mask]])
    lms[mask.to_numpy()] = tab.reindex(idx).to_numpy()[:, :3].astype(float)
X["whz"] = lms_z(X["weight_kg_adm"].to_numpy(float), *lms.T)

# WHO flags for biologically implausible z-scores -> missing
for col, (lo, hi) in {"waz": (-6, 5), "haz": (-6, 6), "whz": (-5, 5), "muacz": (-5, 5)}.items():
    bad = X[col].notna() & ~X[col].between(lo, hi)
    print(f"  {col}: {bad.sum()} outside WHO plausible range -> NaN")
    X.loc[bad, col] = np.nan

# Clinical malnutrition categories (WHO)
X["underweight"] = (X["waz"] < -2).astype(int)                    # moderate or severe
X["wasting"] = ((X["whz"] < -2) | (X["muac_mm_adm"] < 125)).astype(int)
X["sam"] = ((X["whz"] < -3) | (X["muac_mm_adm"] < 115)).astype(int)  # severe acute malnutrition

# --- 4. Blantyre Coma Score (0-5) ----------------------------------------------------
eye = X["bcseye_adm"].map({"Watches or follows": 1, "Fails to watch or follow": 0})
verbal = X["bcsverbal_adm"].map({"Cries appropriately with pain, or, if verbal, speaks": 2,
                                 "Moan or abnormal cry with pain": 1, "No vocal response to pain": 0})
motor = X["bcsmotor_adm"].map({"Localizes painful stimulus": 2,
                               "Withdraws limb from painful stimulus": 1,
                               "No response or inappropriate response": 0})
X["bcs_total"] = eye + verbal + motor
X["bcs_coma"] = (X["bcs_total"] <= 2).astype(int)      # BCS <= 2 = unrousable coma

# --- 5. Age-adjusted vitals ----------------------------------------------------------
# WHO IMCI fast breathing: >= 50/min if 2-11 months, >= 40/min if 12-59 months
rr_cut = np.where(X["agecalc_adm"] < 12, 50, 40)
X["rr_ratio"] = X["rr_brpm_app_adm"] / rr_cut          # >1 = fast breathing for age
X["fast_breathing"] = (X["rr_ratio"] >= 1).astype(int)
X["spo2_min"] = X[["spo2site1_pc_oxi_adm", "spo2site2_pc_oxi_adm"]].min(axis=1)
X["hypoxaemia"] = (X["spo2_min"] < 90).astype(int)
X["shock_index"] = X["hr_bpm_adm"] / X["sysbp_mmhg_adm"]

# --- 6. Missingness indicators for informative labs ----------------------------------
X["lactate_measured"] = X["lactate_mmolpl_adm"].notna().astype(int)
X["hct_measured"] = X["hematocrit_gpdl_adm"].notna().astype(int)

# --- 7. Leakage screen: single-feature AUROC ------------------------------------------
num = X.select_dtypes("number")
auc = {c: roc_auc_score(y[num[c].notna()], num[c].dropna()) for c in num if num[c].nunique() > 1}
auc = pd.Series(auc).map(lambda a: max(a, 1 - a)).sort_values(ascending=False)
print("\nTop single-feature AUROC (direction-free):\n", auc.head(12).round(3).to_string())
print("lengthadm AUROC (excluded, for comparison):",
      round(max(a := roc_auc_score(y, df["lengthadm"]), 1 - a), 3))

X["inhospital_mortality"] = y
X.to_csv("p2_features.csv", index=False)
print(f"\nSaved p2_features.csv: {X.shape}")
