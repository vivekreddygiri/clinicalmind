import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import pickle
import os

MODELS_PATH = "outputs/models/"
os.makedirs(MODELS_PATH, exist_ok=True)

TARGETS = ["LABEL_DIABETES", "LABEL_CKD", "LABEL_HEARTFAIL"]
TARGET_NAMES = {
    "LABEL_DIABETES":  "Type 2 Diabetes",
    "LABEL_CKD":       "Chronic Kidney Disease",
    "LABEL_HEARTFAIL": "Heart Failure"
}


def get_feature_columns(df):
    """Return all feature columns (exclude ID and label columns)."""
    exclude = ["SUBJECT_ID"] + TARGETS
    return [c for c in df.columns if c not in exclude]


def train_all_models(feature_matrix_with_labels):
    """
    Train one XGBoost model per disease.
    Returns dict of target -> trained model, and evaluation results.
    """
    df = feature_matrix_with_labels.copy()
    feature_cols = get_feature_columns(df)

    X = df[feature_cols].astype(float)
    results = {}
    models  = {}

    for target in TARGETS:
        if target not in df.columns:
            print(f"Skipping {target} — label not found.")
            continue

        y = df[target].astype(int)

        print(f"\n── Training: {TARGET_NAMES[target]} ──")
        print(f"   Positive cases: {y.sum()} / {len(y)}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Handle class imbalance
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        model = xgb.XGBClassifier(
            n_estimators      = 300,
            max_depth         = 6,
            learning_rate     = 0.05,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            scale_pos_weight  = scale_pos_weight,
            eval_metric       = "auc",
            random_state      = 42,
            n_jobs            = -1,
        )

        model.fit(
            X_train, y_train,
            eval_set              = [(X_test, y_test)],
            verbose               = False,
        )

        # Evaluate
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auc = roc_auc_score(y_test, y_prob)
        f1  = f1_score(y_test, y_pred, zero_division=0)

        print(f"   AUC-ROC : {auc:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))

        results[target] = {
            "auc":    auc,
            "f1":     f1,
            "report": classification_report(y_test, y_pred, zero_division=0)
        }
        models[target] = model

        # Save model
        model_path = os.path.join(MODELS_PATH, f"{target}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"   Saved → {model_path}")

    # Save feature column order (needed for inference)
    with open(os.path.join(MODELS_PATH, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)

    return models, results


def load_all_models():
    """Load saved models from disk."""
    models = {}
    for target in TARGETS:
        path = os.path.join(MODELS_PATH, f"{target}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[target] = pickle.load(f)
    with open(os.path.join(MODELS_PATH, "feature_cols.pkl"), "rb") as f:
        feature_cols = pickle.load(f)
    return models, feature_cols


def predict_patient_risk(patient_features_dict, models, feature_cols):
    """
    Given a dict of feature values for ONE patient,
    return risk probabilities for all 3 diseases.
    """
    row = pd.DataFrame([patient_features_dict])

    # Align columns — fill missing with 0
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_cols].astype(float)

    risk_scores = {}
    for target, model in models.items():
        prob = model.predict_proba(row)[0][1]
        risk_scores[target] = round(float(prob), 4)

    return risk_scores


def risk_level(prob):
    """Convert probability to risk label."""
    if prob >= 0.70:
        return "HIGH", "🔴"
    elif prob >= 0.45:
        return "MODERATE", "🟠"
    elif prob >= 0.25:
        return "LOW-MODERATE", "🟡"
    else:
        return "LOW", "🟢"

# Symptom name to feature column mapping
# Used in app.py to set symptom flags before prediction
SYMPTOM_FEATURE_MAP = {
    "Fatigue":               "SYMPTOM_FATIGUE",
    "Frequent Urination":    "SYMPTOM_FREQUENT_URINATION",
    "Excessive Thirst":      "SYMPTOM_EXCESSIVE_THIRST",
    "Chest Pain":            "SYMPTOM_CHEST_PAIN",
    "Shortness of Breath":   "SYMPTOM_SHORTNESS_BREATH",
    "Swollen Legs":          "SYMPTOM_SWOLLEN_LEGS",
    "Blurred Vision":        "SYMPTOM_BLURRED_VISION",
    "Nausea":                "SYMPTOM_NAUSEA",
    "Decreased Urine Output":"SYMPTOM_DECREASED_URINE",
    "Rapid Weight Gain":     "SYMPTOM_WEIGHT_GAIN",
    "Dizziness":             "SYMPTOM_DIZZINESS",
    "Palpitations":          "SYMPTOM_PALPITATIONS",
}