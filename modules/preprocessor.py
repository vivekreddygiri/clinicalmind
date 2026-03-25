import pandas as pd
import numpy as np
import os

DATA_PATH = "data/"

def load_data():
    print("Loading MIMIC-III tables...")

    patients     = pd.read_csv(os.path.join(DATA_PATH, "PATIENTS.csv"),       low_memory=False)
    admissions   = pd.read_csv(os.path.join(DATA_PATH, "ADMISSIONS.csv"),     low_memory=False)
    diagnoses    = pd.read_csv(os.path.join(DATA_PATH, "DIAGNOSES_ICD.csv"),  low_memory=False)
    d_icd        = pd.read_csv(os.path.join(DATA_PATH, "D_ICD_DIAGNOSES.csv"),low_memory=False)
    labevents    = pd.read_csv(os.path.join(DATA_PATH, "LABEVENTS.csv"),       low_memory=False)
    d_labitems   = pd.read_csv(os.path.join(DATA_PATH, "D_LABITEMS.csv"),     low_memory=False)
    noteevents   = pd.read_csv(os.path.join(DATA_PATH, "NOTEEVENTS.csv"),     low_memory=False)
    prescriptions= pd.read_csv(os.path.join(DATA_PATH, "PRESCRIPTIONS.csv"), low_memory=False)
    procedures   = pd.read_csv(os.path.join(DATA_PATH, "PROCEDURES_ICD.csv"),low_memory=False)

    print("All tables loaded.")
    return patients, admissions, diagnoses, d_icd, labevents, d_labitems, noteevents, prescriptions, procedures


def clean_patients(patients):
    patients.columns = patients.columns.str.upper()
    patients = patients[["SUBJECT_ID", "GENDER", "DOB"]].drop_duplicates()
    patients["DOB"] = pd.to_datetime(patients["DOB"], errors="coerce")
    return patients


def clean_admissions(admissions):
    admissions.columns = admissions.columns.str.upper()
    admissions = admissions[[
        "SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME",
        "ADMISSION_TYPE", "DIAGNOSIS", "HOSPITAL_EXPIRE_FLAG"
    ]].drop_duplicates()
    admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"], errors="coerce")
    admissions["DISCHTIME"] = pd.to_datetime(admissions["DISCHTIME"], errors="coerce")
    admissions["LOS_DAYS"] = (admissions["DISCHTIME"] - admissions["ADMITTIME"]).dt.days
    return admissions


def compute_age(patients, admissions):
    # Merge to get first admission date per patient
    first_adm = admissions.groupby("SUBJECT_ID")["ADMITTIME"].min().reset_index()
    patients = patients.merge(first_adm, on="SUBJECT_ID", how="left")

    # MIMIC-III shifts DOB far into the past for patients >89 (privacy protection)
    # This causes int64 overflow when subtracting timestamps directly
    # Safe fix: compute age using .year only
    def safe_age(row):
        try:
            if pd.isnull(row["DOB"]) or pd.isnull(row["ADMITTIME"]):
                return np.nan
            years = row["ADMITTIME"].year - row["DOB"].year
            return min(round(float(years), 1), 89)
        except Exception:
            return np.nan

    patients["AGE"] = patients.apply(safe_age, axis=1)
    patients["AGE"] = patients["AGE"].clip(lower=0, upper=89)
    patients = patients[["SUBJECT_ID", "GENDER", "AGE"]]
    return patients


def build_patient_base(patients, admissions):
    pat = clean_patients(patients)
    adm = clean_admissions(admissions)
    pat = compute_age(pat, adm)

    # Aggregate admission-level info per patient
    adm_agg = adm.groupby("SUBJECT_ID").agg(
        NUM_ADMISSIONS   = ("HADM_ID",              "nunique"),
        AVG_LOS          = ("LOS_DAYS",             "mean"),
        MAX_LOS          = ("LOS_DAYS",             "max"),
        EMERGENCY_COUNT  = ("ADMISSION_TYPE",        lambda x: (x == "EMERGENCY").sum()),
    ).reset_index()

    base = pat.merge(adm_agg, on="SUBJECT_ID", how="left")
    print(f"Patient base built: {len(base)} patients")
    return base, adm


def get_patient_hadm_map(admissions):
    """Returns mapping of SUBJECT_ID -> list of HADM_IDs"""
    adm = admissions.copy()
    adm.columns = adm.columns.str.upper()
    return adm.groupby("SUBJECT_ID")["HADM_ID"].apply(list).to_dict()