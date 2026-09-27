"""Peds track, Lesson P1b - Deeper data overview: types, coding, ranges, plausibility, charts."""
import matplotlib
matplotlib.use("Agg")                      # draw charts to files (no pop-up window)
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------- 1. Load
df = pd.read_csv("PSDC_SyntheticTrainingData_Dataset_ODR.tab", sep="\t")
dd = pd.read_csv("data_dictionary.txt", sep="|", skipinitialspace=True)
dd.columns = [c.strip() for c in dd.columns]
dd["group"] = dd["Variable Type"].str.extract(r"\((.*)\)")[0].str.strip()
died = df["inhospital_mortality"] == 1

# ---------------------------------------------------------------- 2. Structure
print("=" * 70, "\n1. STRUCTURE")
print(f"{df.shape[0]} rows (one per child) x {df.shape[1]} columns")
print("Duplicate study IDs:", df["studyid_adm"].duplicated().sum())
print("\nVariables per group (from data dictionary):")
print(dd["group"].value_counts().to_string())
print("\nPandas data types:", df.dtypes.astype(str).value_counts().to_dict())

# ---------------------------------------------------------------- 3. One child, as the model sees it
print("=" * 70, "\n2. ONE CHILD (first 25 columns)")
print(df.iloc[0, :25].to_string())

# ---------------------------------------------------------------- 4. How categories are coded
print("=" * 70, "\n3. CATEGORY CODING (examples)")
for col in ["bcseye_adm", "bcsverbal_adm", "feedingstatus_adm", "momedu_adm", "admitabx_adm___8"]:
    print(f"\n{col}: {df[col].value_counts(dropna=False).to_dict()}")
checkbox = [c for c in df.columns if "___" in c]
print(f"\n{len(checkbox)} 'check all that apply' columns (name___N = one tick box each)")

# ---------------------------------------------------------------- 5. Continuous: ranges & plausibility
print("=" * 70, "\n4. CONTINUOUS VARIABLES - range check")
limits = {  # plausible limits for children 6-60 months
    "agecalc_adm": (6, 60), "weight_kg_adm": (3, 30), "height_cm_adm": (50, 125),
    "muac_mm_adm": (70, 220), "hr_bpm_adm": (40, 250), "rr_brpm_app_adm": (8, 120),
    "sysbp_mmhg_adm": (40, 180), "temp_c_adm": (32, 43), "spo2site1_pc_oxi_adm": (40, 100),
    "hematocrit_gpdl_adm": (5, 65), "lactate_mmolpl_adm": (0.3, 25), "glucose_mmolpl_adm": (0.5, 40)}
rows = []
for col, (lo, hi) in limits.items():
    s = df[col]
    rows.append([col, s.min(), s.quantile(.05), s.median(), s.quantile(.95), s.max(),
                 ((s < lo) | (s > hi)).sum(), f"{s.isna().mean():.0%}"])
print(pd.DataFrame(rows, columns=["variable", "min", "p5", "median", "p95", "max",
                                  "implausible", "missing"]).round(1).to_string(index=False))

# ---------------------------------------------------------------- 6. Outcome by age band
print("=" * 70, "\n5. MORTALITY BY AGE BAND")
band = pd.cut(df["agecalc_adm"], [6, 12, 24, 36, 60], include_lowest=True,
              labels=["6-12 m", "12-24 m", "24-36 m", "36-60 m"])
print(pd.DataFrame({"children": band.value_counts(sort=False),
                    "deaths": died.groupby(band, observed=True).sum(),
                    "mortality_%": (died.groupby(band, observed=True).mean() * 100).round(1)}).to_string())

# ---------------------------------------------------------------- 7. Charts
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for a, (col, title) in zip(ax.flat[:4], [("agecalc_adm", "Age (months)"),
                                         ("weight_kg_adm", "Weight (kg)"),
                                         ("rr_brpm_app_adm", "Respiratory rate"),
                                         ("lactate_mmolpl_adm", "Lactate (mmol/L)")]):
    a.hist(df.loc[~died, col].dropna(), bins=40, density=True, alpha=.55, label="survived")
    a.hist(df.loc[died, col].dropna(), bins=40, density=True, alpha=.55, label="died")
    a.set_title(title); a.legend()
# weight vs age: why raw weight is misleading
a = ax.flat[4]
a.scatter(df.loc[~died, "agecalc_adm"], df.loc[~died, "weight_kg_adm"], s=4, alpha=.3, label="survived")
a.scatter(df.loc[died, "agecalc_adm"], df.loc[died, "weight_kg_adm"], s=14, c="red", label="died")
a.set_xlabel("Age (months)"); a.set_ylabel("Weight (kg)"); a.set_title("Weight depends on age"); a.legend()
# mortality by Blantyre motor response
a = ax.flat[5]
m = (died.groupby(df["bcsmotor_adm"]).mean() * 100).sort_values()
a.barh(m.index.str[:22], m.values)
a.set_xlabel("Mortality %"); a.set_title("Mortality by Blantyre motor response")
fig.suptitle("Pediatric Sepsis Challenge (synthetic) - overview, n=2,686, deaths=119")
fig.tight_layout(); fig.savefig("p1b_overview.png", dpi=110)
print("\nSaved chart -> p1b_overview.png")
