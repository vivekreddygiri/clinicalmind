import pandas as pd
import numpy as np
import pickle
import os
from modules.preprocessor      import (load_data, build_patient_base,
                                        get_patient_hadm_map)
from modules.label_engine       import (generate_labels,
                                        get_icd_history_per_patient)
from modules.feature_engineer   import (extract_lab_features,
                                        extract_prescription_features,
                                        extract_diagnosis_flags,
                                        build_feature_matrix)
from modules.risk_model         import train_all_models

os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

print("=" * 60)
print("   ClinicalMind — Training Pipeline")
print("=" * 60)

# ── Step 1: Load all CSVs ─────────────────────────────────────
print("\n[1/7] Loading data...")
(patients, admissions, diagnoses, d_icd,
 labevents, d_labitems, noteevents,
 prescriptions, procedures) = load_data()

# ── Step 2: Build patient base ────────────────────────────────
print("\n[2/7] Building patient base...")
base_df, adm_clean = build_patient_base(patients, admissions)

# ── Step 3: Generate risk labels ──────────────────────────────
print("\n[3/7] Generating risk labels from ICD-9 codes...")
labels_df    = generate_labels(diagnoses)
icd_hist_df  = get_icd_history_per_patient(diagnoses, d_icd)

# ── Step 4: Feature engineering ───────────────────────────────
print("\n[4/7] Extracting lab features...")
lab_df       = extract_lab_features(labevents)

print("\n[5/7] Extracting prescription features...")
rx_df        = extract_prescription_features(prescriptions)

print("\n[6/7] Extracting diagnosis flags...")
diag_flag_df = extract_diagnosis_flags(diagnoses)

# ── Step 5: Build full feature matrix ─────────────────────────
print("\n[7/7] Building feature matrix...")
feature_matrix = build_feature_matrix(base_df, lab_df, rx_df, diag_flag_df)

# Merge labels into feature matrix
full_df = feature_matrix.merge(labels_df, on="SUBJECT_ID", how="inner")
print(f"\nFinal dataset: {len(full_df)} patients | {full_df.shape[1]} columns")

# ── Step 6: Train models ───────────────────────────────────────
print("\n── Training XGBoost Models ──")
models, results = train_all_models(full_df)

# ── Step 7: Save supporting data for dashboard ────────────────
print("\nSaving supporting data...")

base_df.to_pickle("outputs/base_df.pkl")
lab_df.to_pickle("outputs/lab_df.pkl")
rx_df.to_pickle("outputs/rx_df.pkl")
icd_hist_df.to_pickle("outputs/icd_hist_df.pkl")
feature_matrix.to_pickle("outputs/feature_matrix.pkl")
full_df.to_pickle("outputs/full_df.pkl")

# Save noteevents lean version (only needed columns to save RAM)
noteevents_lean = noteevents[
    ["SUBJECT_ID", "CATEGORY", "TEXT"]
].copy()
noteevents_lean.columns = noteevents_lean.columns.str.upper()
noteevents_lean.to_pickle("outputs/noteevents_lean.pkl")

prescriptions_lean = prescriptions[
    ["SUBJECT_ID", "DRUG"]
].copy()
prescriptions_lean.columns = prescriptions_lean.columns.str.upper()
prescriptions_lean.to_pickle("outputs/prescriptions_lean.pkl")

# ── Save additional data for longitudinal analysis ────────────
print("Saving longitudinal data...")

# Admissions with dates — needed for timeline
admissions.columns = admissions.columns.str.upper()
admissions_lean = admissions[
    ["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "ADMISSION_TYPE"]
].copy()
admissions_lean["ADMITTIME"] = pd.to_datetime(
    admissions_lean["ADMITTIME"], errors="coerce"
)
admissions_lean.to_pickle("outputs/admissions_lean.pkl")

# Lab events per admission — filter to only key labs to save space
labevents.columns = labevents.columns.str.upper()
KEY_LAB_IDS = [
    50809, 50931,   # Glucose
    50912,          # Creatinine
    51006,          # BUN
    50852,          # HbA1c
    50963,          # BNP
    51222,          # Hemoglobin
    50971,          # Potassium
    50983,          # Sodium
    51265,          # Platelets
    51301,          # WBC
]
labevents_hadm = labevents[
    labevents["ITEMID"].isin(KEY_LAB_IDS)
][["SUBJECT_ID", "HADM_ID", "ITEMID", "VALUENUM"]].copy()
labevents_hadm["VALUENUM"] = pd.to_numeric(
    labevents_hadm["VALUENUM"], errors="coerce"
)
labevents_hadm = labevents_hadm.dropna(subset=["VALUENUM", "HADM_ID"])
labevents_hadm.to_pickle("outputs/labevents_hadm.pkl")

# Prescriptions per admission
prescriptions.columns = prescriptions.columns.str.upper()
prescriptions_hadm = prescriptions[
    ["SUBJECT_ID", "HADM_ID", "DRUG"]
].copy()
prescriptions_hadm["DRUG"] = prescriptions_hadm["DRUG"].astype(str).str.lower()
prescriptions_hadm.to_pickle("outputs/prescriptions_hadm.pkl")

# Diagnoses per admission
diagnoses.columns = diagnoses.columns.str.upper()
diagnoses_hadm = diagnoses[
    ["SUBJECT_ID", "HADM_ID", "ICD9_CODE"]
].copy()
diagnoses_hadm["ICD9_CODE"] = diagnoses_hadm["ICD9_CODE"].astype(str).str.strip()
diagnoses_hadm.to_pickle("outputs/diagnoses_hadm.pkl")

print("Longitudinal data saved.")

print("\n" + "=" * 60)
print("   Training Complete!")
print("=" * 60)
print("\nModel Performance Summary:")
for target, res in results.items():
    print(f"  {target:<25} AUC={res['auc']:.4f}  F1={res['f1']:.4f}")

print("\nAll outputs saved to /outputs/")
print("You can now run:  streamlit run app.py")