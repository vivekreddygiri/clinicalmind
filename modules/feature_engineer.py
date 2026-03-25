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


def build_feature_matrix(base_df, lab_df, rx_df, diag_flag_df):
    matrix = base_df.copy()
    matrix = matrix.merge(lab_df,      on="SUBJECT_ID", how="left")
    matrix = matrix.merge(rx_df,       on="SUBJECT_ID", how="left")
    matrix = matrix.merge(diag_flag_df,on="SUBJECT_ID", how="left")

    # Encode gender
    matrix["GENDER"] = (matrix["GENDER"] == "M").astype(int)

    # Fill missing lab values with median
    num_cols = matrix.select_dtypes(include=[np.number]).columns
    matrix[num_cols] = matrix[num_cols].fillna(matrix[num_cols].median())

    print(f"Feature matrix shape: {matrix.shape}")
    return matrix