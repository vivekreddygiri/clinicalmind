import pandas as pd
import numpy as np
from modules.risk_model import TARGET_NAMES

# ── Lab item ID to feature name mapping ───────────────────────
LAB_ITEMS = {
    "GLUCOSE":    [50809, 50931],
    "CREATININE": [50912],
    "BUN":        [51006],
    "HBA1C":      [50852],
    "BNP":        [50963],
    "HEMOGLOBIN": [51222],
    "POTASSIUM":  [50971],
    "SODIUM":     [50983],
    "PLATELETS":  [51265],
    "WBC":        [51301],
}

PRESCRIPTION_FLAGS = {
    "MED_INSULIN":     ["insulin"],
    "MED_METFORMIN":   ["metformin"],
    "MED_DIURETIC":    ["furosemide", "lasix", "torsemide"],
    "MED_ACE":         ["lisinopril", "enalapril", "ramipril"],
    "MED_STATIN":      ["atorvastatin", "simvastatin", "rosuvastatin"],
    "MED_BETABLOCKER": ["metoprolol", "carvedilol", "bisoprolol"],
    "MED_DIALYSIS":    ["dialysis"],
}

DIAGNOSIS_FLAGS = {
    "HAS_HYPERTENSION":  ["401", "402"],
    "HAS_OBESITY":       ["278"],
    "HAS_ANEMIA":        ["280", "281", "282"],
    "HAS_COPD":          ["496", "491"],
    "HAS_LIVER_DISEASE": ["571", "572"],
}


def build_admission_features(
    hadm_id, subject_id,
    labevents_hadm, prescriptions_hadm,
    diagnoses_hadm, base_row
):
    """
    Build a feature vector for ONE admission.
    Uses only data from that specific admission.
    """
    features = {}

    # Demographics (same across all admissions)
    features["AGE"] = float(base_row["AGE"]) if "AGE" in base_row else 0.0
    # Encode gender — stored as "M"/"F" string, convert to 1/0
    gender_raw = base_row["GENDER"] if "GENDER" in base_row else "F"
    features["GENDER"] = 1.0 if str(gender_raw).strip().upper() == "M" else 0.0

    # ── Lab features for this admission ───────────────────────
    # Cast both sides to int to handle float64 vs int64 mismatch
    adm_labs = labevents_hadm[
        labevents_hadm["HADM_ID"].astype(int) == int(hadm_id)
    ]

    for feat_name, item_ids in LAB_ITEMS.items():
        subset = adm_labs[adm_labs["ITEMID"].isin(item_ids)]["VALUENUM"]
        if len(subset) > 0:
            features[f"{feat_name}_MEAN"]  = subset.mean()
            features[f"{feat_name}_MAX"]   = subset.max()
            features[f"{feat_name}_MIN"]   = subset.min()
            features[f"{feat_name}_STD"]   = subset.std() if len(subset) > 1 else 0.0
            features[f"{feat_name}_COUNT"] = len(subset)
        else:
            features[f"{feat_name}_MEAN"]  = np.nan
            features[f"{feat_name}_MAX"]   = np.nan
            features[f"{feat_name}_MIN"]   = np.nan
            features[f"{feat_name}_STD"]   = np.nan
            features[f"{feat_name}_COUNT"] = 0.0

    # ── Prescription features for this admission ───────────────
    adm_rx   = prescriptions_hadm[
        prescriptions_hadm["HADM_ID"] == hadm_id
    ]["DRUG"]
    all_drugs = " ".join(adm_rx.dropna().tolist())

    for flag_name, keywords in PRESCRIPTION_FLAGS.items():
        features[flag_name] = float(
            any(kw in all_drugs for kw in keywords)
        )

    # ── Diagnosis flags for this admission ─────────────────────
    adm_diag = diagnoses_hadm[
        diagnoses_hadm["HADM_ID"] == hadm_id
    ]["ICD9_CODE"]
    all_codes = " ".join(adm_diag.astype(str).tolist())

    for flag_name, codes in DIAGNOSIS_FLAGS.items():
        features[flag_name] = float(any(c in all_codes for c in codes))

    features["NUM_UNIQUE_DIAGNOSES"] = float(adm_diag.nunique())

    # Admission-level stats (single admission context)
    features["NUM_ADMISSIONS"]  = 1.0
    features["AVG_LOS"]         = 0.0
    features["MAX_LOS"]         = 0.0
    features["EMERGENCY_COUNT"] = 0.0

    return features


def compute_longitudinal_risk(
    subject_id,
    models,
    feature_cols,
    admissions_lean,
    labevents_hadm,
    prescriptions_hadm,
    diagnoses_hadm,
    base_df
):
    """
    Compute risk score per admission for one patient.
    Returns a DataFrame with columns:
    ADMITTIME, HADM_ID, risk per disease
    """
    # Get this patient's admissions sorted by date
    pat_adm = admissions_lean[
        admissions_lean["SUBJECT_ID"] == subject_id
    ].sort_values("ADMITTIME").copy()

    if len(pat_adm) == 0:
        return pd.DataFrame()

    # Get base demographics
    base_row = base_df[base_df["SUBJECT_ID"] == subject_id]
    if len(base_row) == 0:
        return pd.DataFrame()
    base_row = base_row.iloc[0]

    # Get this patient's per-admission data
    pat_labs = labevents_hadm[labevents_hadm["SUBJECT_ID"] == subject_id]
    pat_rx   = prescriptions_hadm[prescriptions_hadm["SUBJECT_ID"] == subject_id]
    pat_diag = diagnoses_hadm[diagnoses_hadm["SUBJECT_ID"] == subject_id]

    records = []

    for _, adm_row in pat_adm.iterrows():
        hadm_id    = adm_row["HADM_ID"]
        admit_time = adm_row["ADMITTIME"]

        try:
            # Build feature vector for this admission
            feat_dict = build_admission_features(
                hadm_id      = hadm_id,
                subject_id   = subject_id,
                labevents_hadm    = pat_labs,
                prescriptions_hadm= pat_rx,
                diagnoses_hadm    = pat_diag,
                base_row          = base_row,
            )

            # Align to model feature columns
            row_df = pd.DataFrame([feat_dict])
            for col in feature_cols:
                if col not in row_df.columns:
                    row_df[col] = 0.0
            row_df = row_df[feature_cols].astype(float)

            # Fill NaN with 0
            row_df = row_df.fillna(0.0)

            # Predict risk for each disease
            record = {
                "HADM_ID":   hadm_id,
                "ADMITTIME": admit_time,
            }
            for target, model in models.items():
                prob = model.predict_proba(row_df)[0][1]
                record[target] = round(float(prob) * 100, 2)

            records.append(record)

        except Exception as e:
            print(f"Skipping admission {hadm_id}: {e}")
            continue

    if not records:
        return pd.DataFrame()

    result_df = pd.DataFrame(records).sort_values("ADMITTIME")
    result_df["ADMISSION_NUM"] = range(1, len(result_df) + 1)
    return result_df