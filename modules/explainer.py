import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

OUTPUT_PATH = "outputs/shap_plots/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

FEATURE_LABELS = {
    "GLUCOSE_MEAN":         "Average Blood Glucose",
    "GLUCOSE_MAX":          "Peak Blood Glucose",
    "CREATININE_MEAN":      "Average Creatinine",
    "CREATININE_MAX":       "Peak Creatinine",
    "BUN_MEAN":             "Average Blood Urea Nitrogen",
    "HBA1C_MEAN":           "Average HbA1c",
    "BNP_MEAN":             "Average BNP (Heart Marker)",
    "HEMOGLOBIN_MEAN":      "Average Hemoglobin",
    "MED_INSULIN":          "Insulin Prescribed",
    "MED_METFORMIN":        "Metformin Prescribed",
    "MED_DIURETIC":         "Diuretic Prescribed",
    "MED_ACE":              "ACE Inhibitor Prescribed",
    "MED_BETABLOCKER":      "Beta Blocker Prescribed",
    "HAS_HYPERTENSION":     "History of Hypertension",
    "HAS_OBESITY":          "History of Obesity",
    "AGE":                  "Patient Age",
    "NUM_ADMISSIONS":       "Number of Hospitalizations",
    "AVG_LOS":              "Average Length of Stay",
    "EMERGENCY_COUNT":      "Emergency Admissions",
    "NUM_UNIQUE_DIAGNOSES": "Number of Unique Diagnoses",
}


def get_shap_values(model, X_df):
    """Compute SHAP values for a dataframe of patients."""
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)
    return explainer, shap_values


def get_top_shap_features(model, patient_row_df, feature_cols, top_n=5):
    """
    Get top N SHAP features for a single patient prediction.
    Returns list of (feature_name, shap_value, feature_value).
    """
    explainer   = shap.TreeExplainer(model)
    shap_vals   = explainer.shap_values(patient_row_df[feature_cols])

    shap_series = pd.Series(shap_vals[0], index=feature_cols)
    top_features = shap_series.abs().nlargest(top_n).index.tolist()

    results = []
    for feat in top_features:
        results.append({
            "feature":       feat,
            "label":         FEATURE_LABELS.get(feat, feat),
            "shap_value":    round(float(shap_series[feat]), 4),
            "feature_value": round(float(patient_row_df[feat].values[0]), 3),
        })
    return results


def shap_to_natural_language(shap_features, disease_name):
    """
    Convert top SHAP features into a plain English explanation.
    E.g. 'Risk is elevated primarily due to high average glucose
    and insulin prescription history.'
    """
    positive_drivers = [
        f["label"] for f in shap_features if f["shap_value"] > 0
    ]
    negative_drivers = [
        f["label"] for f in shap_features if f["shap_value"] < 0
    ]

    explanation = f"For {disease_name}, the prediction is driven by:\n"

    if positive_drivers:
        explanation += (
            f"  • Risk-increasing factors: "
            f"{', '.join(positive_drivers[:3])}.\n"
        )
    if negative_drivers:
        explanation += (
            f"  • Risk-reducing factors: "
            f"{', '.join(negative_drivers[:2])}.\n"
        )

    if not positive_drivers and not negative_drivers:
        explanation += "  • Insufficient feature data for detailed explanation.\n"

    return explanation


def generate_shap_plot(model, patient_row_df, feature_cols, target, subject_id):
    """Generate and save a SHAP waterfall plot for one patient."""
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(patient_row_df[feature_cols])

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_vals[0], max_display=10, show=False)
    plt.title(f"SHAP Explanation — {target} — Patient {subject_id}")
    plt.tight_layout()

    path = os.path.join(OUTPUT_PATH, f"{subject_id}_{target}_shap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def explain_patient(subject_id, patient_row_df, models, feature_cols):
    """
    Full explanation for one patient across all 3 diseases.
    Returns dict of target -> {shap_features, nl_explanation, plot_path}
    """
    from modules.risk_model import TARGET_NAMES

    explanations = {}
    for target, model in models.items():
        shap_feats = get_top_shap_features(model, patient_row_df, feature_cols)
        nl_exp     = shap_to_natural_language(shap_feats, TARGET_NAMES[target])
        plot_path  = generate_shap_plot(
            model, patient_row_df, feature_cols, target, subject_id
        )
        explanations[target] = {
            "shap_features":    shap_feats,
            "nl_explanation":   nl_exp,
            "shap_plot_path":   plot_path,
        }
    return explanations