"""Peds track, Lesson P1 - First look at the 2024 Pediatric Sepsis Challenge (synthetic) dataset."""
import pandas as pd

df = pd.read_csv("PSDC_SyntheticTrainingData_Dataset_ODR.tab", sep="\t")
print(f"Shape: {df.shape[0]} children x {df.shape[1]} columns")

# --- Outcome ---
y = df["inhospital_mortality"]
print("\nOutcome values:", y.value_counts(dropna=False).to_dict())
dead = (y.astype(str).str.lower().isin(["1", "yes", "true"]))
print(f"In-hospital deaths: {dead.sum()} ({dead.mean():.1%})")
print("Length of admission (days): median", df["lengthadm"].median(),
      "| IQR", df["lengthadm"].quantile(.25), "-", df["lengthadm"].quantile(.75))

# --- Missingness ---
miss = df.isna().mean().sort_values(ascending=False)
print("\nMost-missing columns:\n", (miss.head(8) * 100).round(1).to_string())
print(f"Overall missing cells: {df.isna().mean().mean():.1%}")

# --- Continuous vitals & labs: survivors vs deaths ---
cont = ["agecalc_adm", "weight_kg_adm", "muac_mm_adm", "hr_bpm_adm", "rr_brpm_app_adm",
        "sysbp_mmhg_adm", "temp_c_adm", "spo2site1_pc_oxi_adm", "hematocrit_gpdl_adm",
        "lactate_mmolpl_adm", "glucose_mmolpl_adm"]
tab = df[cont].groupby(dead.map({False: "survived", True: "died"})).median().T.round(1)
print("\nMedian values by outcome:\n", tab.to_string())

# --- Categorical signs: mortality rate per category ---
for col in ["sex_adm", "respdistress_adm", "caprefill_adm", "bcsmotor_adm",
            "hivstatus_adm", "malariastatuspos_adm", "oxygenavail_adm"]:
    c = df[col].fillna("missing")
    g = pd.DataFrame({"n": c.value_counts(),
                      "mortality_%": (dead.groupby(c).mean() * 100).round(1)})
    print(f"\n{col}:\n{g.to_string()}")
