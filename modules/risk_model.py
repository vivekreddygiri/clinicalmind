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
            use_label_encoder = False,
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

# Symptom to disease relevance mapping
# Values represent how much each symptom nudges each disease risk
SYMPTOM_WEIGHTS = {
    "Fatigue": {
        "LABEL_DIABETES":  0.08,
        "LABEL_CKD":       0.06,
        "LABEL_HEARTFAIL": 0.07
    },
    "Frequent Urination": {
        "LABEL_DIABETES":  0.15,
        "LABEL_CKD":       0.10,
        "LABEL_HEARTFAIL": 0.02
    },
    "Excessive Thirst": {
        "LABEL_DIABETES":  0.15,
        "LABEL_CKD":       0.05,
        "LABEL_HEARTFAIL": 0.01
    },
    "Chest Pain": {
        "LABEL_DIABETES":  0.03,
        "LABEL_CKD":       0.02,
        "LABEL_HEARTFAIL": 0.15
    },
    "Shortness of Breath": {
        "LABEL_DIABETES":  0.02,
        "LABEL_CKD":       0.05,
        "LABEL_HEARTFAIL": 0.18
    },
    "Swollen Legs": {
        "LABEL_DIABETES":  0.01,
        "LABEL_CKD":       0.08,
        "LABEL_HEARTFAIL": 0.15
    },
    "Blurred Vision": {
        "LABEL_DIABETES":  0.12,
        "LABEL_CKD":       0.04,
        "LABEL_HEARTFAIL": 0.01
    },
    "Nausea": {
        "LABEL_DIABETES":  0.04,
        "LABEL_CKD":       0.08,
        "LABEL_HEARTFAIL": 0.04
    },
    "Decreased Urine Output": {
        "LABEL_DIABETES":  0.03,
        "LABEL_CKD":       0.18,
        "LABEL_HEARTFAIL": 0.08
    },
    "Rapid Weight Gain": {
        "LABEL_DIABETES":  0.02,
        "LABEL_CKD":       0.06,
        "LABEL_HEARTFAIL": 0.14
    },
    "Dizziness": {
        "LABEL_DIABETES":  0.05,
        "LABEL_CKD":       0.04,
        "LABEL_HEARTFAIL": 0.07
    },
    "Palpitations": {
        "LABEL_DIABETES":  0.02,
        "LABEL_CKD":       0.02,
        "LABEL_HEARTFAIL": 0.12
    },
}


def apply_symptom_adjustment(risk_scores, selected_symptoms):
    """
    Adjust raw model risk scores based on current symptoms.
    Uses a capped additive approach so scores stay within [0, 1].
    """
    if not selected_symptoms:
        return risk_scores  # No symptoms = no change

    adjusted = {}
    for target, base_prob in risk_scores.items():
        total_boost = 0.0
        for symptom in selected_symptoms:
            if symptom in SYMPTOM_WEIGHTS:
                total_boost += SYMPTOM_WEIGHTS[symptom].get(target, 0.0)

        # Cap total boost at 0.35 so symptoms alone can't push to 100%
        total_boost = min(total_boost, 0.35)

        # Weighted blend: 70% model history + 30% symptom signal
        new_prob = base_prob + (1 - base_prob) * total_boost
        adjusted[target] = round(min(float(new_prob), 0.99), 4)

    return adjusted