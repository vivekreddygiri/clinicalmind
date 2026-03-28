import pandas as pd
import numpy as np

# Key lab item IDs from D_LABITEMS (standard MIMIC-III IDs)
LAB_ITEMS = {
    "GLUCOSE":      [50809, 50931],   # Glucose (blood & urine)
    "CREATININE":   [50912],          # Creatinine
    "BUN":          [51006],          # Blood Urea Nitrogen
    "HBA1C":        [50852],          # HbA1c
    "POTASSIUM":    [50971],          # Potassium
    "SODIUM":       [50983],          # Sodium
    "HEMOGLOBIN":   [51222],          # Hemoglobin
    "PLATELETS":    [51265],          # Platelets
    "WBC":          [51301],          # White Blood Cells
    "BNP":          [50963],          # BNP (Heart Failure marker)
}

# Prescription keyword flags
PRESCRIPTION_FLAGS = {
    "MED_INSULIN":      ["insulin"],
    "MED_METFORMIN":    ["metformin"],
    "MED_DIURETIC":     ["furosemide", "lasix", "torsemide"],
    "MED_ACE":          ["lisinopril", "enalapril", "ramipril"],
    "MED_STATIN":       ["atorvastatin", "simvastatin", "rosuvastatin"],
    "MED_BETABLOCKER":  ["metoprolol", "carvedilol", "bisoprolol"],
    "MED_DIALYSIS":     ["dialysis"],
}


def extract_lab_features(labevents_df):
    lab = labevents_df.copy()
    lab.columns = lab.columns.str.upper()
    lab["VALUENUM"] = pd.to_numeric(lab["VALUENUM"], errors="coerce")
    lab = lab.dropna(subset=["VALUENUM", "SUBJECT_ID", "ITEMID"])

    result = pd.DataFrame()

    for feat_name, item_ids in LAB_ITEMS.items():
        subset = lab[lab["ITEMID"].isin(item_ids)]
        agg = subset.groupby("SUBJECT_ID")["VALUENUM"].agg(
            **{
                f"{feat_name}_MEAN": "mean",
                f"{feat_name}_MAX":  "max",
                f"{feat_name}_MIN":  "min",
                f"{feat_name}_STD":  "std",
                f"{feat_name}_COUNT":"count",
            }
        ).reset_index()

        if result.empty:
            result = agg
        else:
            result = result.merge(agg, on="SUBJECT_ID", how="outer")

    print(f"Lab features extracted for {len(result)} patients")
    return result


def extract_prescription_features(prescriptions_df):
    rx = prescriptions_df.copy()
    rx.columns = rx.columns.str.upper()
    rx["DRUG"] = rx["DRUG"].astype(str).str.lower()

    grouped = rx.groupby("SUBJECT_ID")["DRUG"].apply(
        lambda drugs: " ".join(drugs)
    ).reset_index()
    grouped.columns = ["SUBJECT_ID", "ALL_DRUGS"]

    for flag_name, keywords in PRESCRIPTION_FLAGS.items():
        grouped[flag_name] = grouped["ALL_DRUGS"].apply(
            lambda text: int(any(kw in text for kw in keywords))
        )

    grouped.drop(columns=["ALL_DRUGS"], inplace=True)
    print(f"Prescription features extracted for {len(grouped)} patients")
    return grouped


def extract_diagnosis_flags(diagnoses_df):
    diag = diagnoses_df.copy()
    diag.columns = diag.columns.str.upper()

    # Count unique diagnoses per patient
    diag_count = diag.groupby("SUBJECT_ID")["ICD9_CODE"].nunique().reset_index()
    diag_count.columns = ["SUBJECT_ID", "NUM_UNIQUE_DIAGNOSES"]

    # Flag common comorbidities
    all_codes = diag.groupby("SUBJECT_ID")["ICD9_CODE"].apply(
        lambda x: " ".join(x.astype(str))
    ).reset_index()
    all_codes.columns = ["SUBJECT_ID", "ALL_CODES"]

    flag_map = {
        "HAS_HYPERTENSION":  ["401", "402"],
        "HAS_OBESITY":       ["278"],
        "HAS_ANEMIA":        ["280", "281", "282"],
        "HAS_COPD":          ["496", "491"],
        "HAS_LIVER_DISEASE": ["571", "572"],
    }

    for flag, codes in flag_map.items():
        all_codes[flag] = all_codes["ALL_CODES"].apply(
            lambda x: int(any(c in x for c in codes))
        )

    all_codes = all_codes.merge(diag_count, on="SUBJECT_ID")
    all_codes.drop(columns=["ALL_CODES"], inplace=True)
    return all_codes


def build_feature_matrix(base_df, lab_df, rx_df, diag_flag_df,
                         symptom_df=None):
    matrix = base_df.copy()
    matrix = matrix.merge(lab_df,       on="SUBJECT_ID", how="left")
    matrix = matrix.merge(rx_df,        on="SUBJECT_ID", how="left")
    matrix = matrix.merge(diag_flag_df, on="SUBJECT_ID", how="left")

    # Merge symptom features if provided
    if symptom_df is not None:
        matrix = matrix.merge(symptom_df, on="SUBJECT_ID", how="left")
        # Fill missing symptom flags with 0
        symptom_cols = [c for c in symptom_df.columns if c != "SUBJECT_ID"]
        matrix[symptom_cols] = matrix[symptom_cols].fillna(0)

    # Encode gender
    matrix["GENDER"] = (matrix["GENDER"] == "M").astype(int)

    # Fill missing numeric values with median
    num_cols = matrix.select_dtypes(include=[np.number]).columns
    matrix[num_cols] = matrix[num_cols].fillna(matrix[num_cols].median())

    print(f"Feature matrix shape: {matrix.shape}")
    return matrix

# ── Symptom ICD-9 Code Mapping ────────────────────────────────
SYMPTOM_ICD9_MAP = {
    "SYMPTOM_FATIGUE":            ["78079", "7807", "7800"],
    "SYMPTOM_FREQUENT_URINATION": ["78841", "78842", "78843"],
    "SYMPTOM_EXCESSIVE_THIRST":   ["7835"],
    "SYMPTOM_CHEST_PAIN":         ["78650", "78651", "78652", "78659", "7865"],
    "SYMPTOM_SHORTNESS_BREATH":   ["78605", "78609", "7860"],
    "SYMPTOM_SWOLLEN_LEGS":       ["7823"],
    "SYMPTOM_BLURRED_VISION":     ["3688", "3689", "3680"],
    "SYMPTOM_NAUSEA":             ["78702", "78701", "7870"],
    "SYMPTOM_DECREASED_URINE":    ["7885"],
    "SYMPTOM_WEIGHT_GAIN":        ["7831"],
    "SYMPTOM_DIZZINESS":          ["7804"],
    "SYMPTOM_PALPITATIONS":       ["7851"],
}


def extract_symptom_features(diagnoses_df):
    """
    Extract symptom presence per patient using ICD-9 symptom codes (780-799).
    Returns binary flags — 1 if patient ever had this symptom coded, 0 otherwise.
    This replaces manual symptom weights with data-driven learned weights.
    """
    diag = diagnoses_df.copy()
    diag.columns = diag.columns.str.upper()

    # Clean ICD-9 codes — remove dots for matching
    diag["ICD9_CLEAN"] = (
        diag["ICD9_CODE"]
        .astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
        .str.upper()
    )

    # Aggregate all codes per patient
    all_codes = diag.groupby("SUBJECT_ID")["ICD9_CLEAN"].apply(
        lambda x: " ".join(x.tolist())
    ).reset_index()
    all_codes.columns = ["SUBJECT_ID", "ALL_CODES"]

    # Binary flag per symptom
    for symptom_flag, icd_codes in SYMPTOM_ICD9_MAP.items():
        all_codes[symptom_flag] = all_codes["ALL_CODES"].apply(
            lambda text: int(any(code in text for code in icd_codes))
        )

    result = all_codes.drop(columns=["ALL_CODES"])

    # Print symptom prevalence
    print("Symptom feature prevalence:")
    for col in result.columns:
        if col.startswith("SYMPTOM_"):
            count = result[col].sum()
            pct   = count / len(result) * 100
            print(f"  {col:<35}: {count:>6} patients ({pct:.1f}%)")

    return result