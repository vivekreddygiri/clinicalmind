import pandas as pd

# ICD-9 code prefixes for each disease
DIABETES_CODES   = ["250"]                         # 250.xx = Diabetes Mellitus
CKD_CODES        = ["585", "403", "404"]           # 585 = CKD, 403/404 = Hypertensive CKD
HEART_FAIL_CODES = ["428", "402", "I50"]           # 428.xx = Heart Failure

def starts_with_any(code, prefixes):
    code = str(code).strip()
    return any(code.startswith(p) for p in prefixes)

def generate_labels(diagnoses_df):
    df = diagnoses_df.copy()
    df.columns = df.columns.str.upper()
    df["ICD9_CODE"] = df["ICD9_CODE"].astype(str).str.strip()

    # Per patient — did they EVER have this diagnosis?
    grouped = df.groupby("SUBJECT_ID")["ICD9_CODE"].apply(list).reset_index()

    def label_patient(codes):
        diabetes   = int(any(starts_with_any(c, DIABETES_CODES)   for c in codes))
        ckd        = int(any(starts_with_any(c, CKD_CODES)        for c in codes))
        heart_fail = int(any(starts_with_any(c, HEART_FAIL_CODES) for c in codes))
        return pd.Series({
            "LABEL_DIABETES":    diabetes,
            "LABEL_CKD":         ckd,
            "LABEL_HEARTFAIL":   heart_fail
        })

    labels = grouped["ICD9_CODE"].apply(label_patient)
    labels["SUBJECT_ID"] = grouped["SUBJECT_ID"].values

    print("Label distribution:")
    print(f"  Diabetes:     {labels['LABEL_DIABETES'].sum()} patients")
    print(f"  CKD:          {labels['LABEL_CKD'].sum()} patients")
    print(f"  Heart Failure:{labels['LABEL_HEARTFAIL'].sum()} patients")

    return labels[["SUBJECT_ID", "LABEL_DIABETES", "LABEL_CKD", "LABEL_HEARTFAIL"]]


def get_icd_history_per_patient(diagnoses_df, d_icd_df):
    """Returns human-readable diagnosis history per patient"""
    diag = diagnoses_df.copy()
    d_icd = d_icd_df.copy()
    diag.columns = diag.columns.str.upper()
    d_icd.columns = d_icd.columns.str.upper()

    merged = diag.merge(
        d_icd[["ICD9_CODE", "SHORT_TITLE"]],
        on="ICD9_CODE", how="left"
    )
    history = merged.groupby("SUBJECT_ID")["SHORT_TITLE"].apply(
        lambda x: list(x.dropna().unique())
    ).reset_index()
    history.columns = ["SUBJECT_ID", "DIAGNOSIS_HISTORY"]
    return history