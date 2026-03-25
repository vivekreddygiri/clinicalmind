import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    confusion_matrix, roc_curve, classification_report
)
from modules.risk_model import TARGETS, TARGET_NAMES, load_all_models

print("=" * 65)
print("   ClinicalMind — Model Evaluation Report")
print("=" * 65)

# ── Load data and models ──────────────────────────────────────
df           = pd.read_pickle("outputs/full_df.pkl")
models, feature_cols = load_all_models()

exclude = ["SUBJECT_ID"] + TARGETS
X = df[[c for c in df.columns if c not in exclude]].astype(float)

# Align columns
for col in feature_cols:
    if col not in X.columns:
        X[col] = 0
X = X[feature_cols]

# ── Per-model evaluation ──────────────────────────────────────
results = {}
for target in TARGETS:
    if target not in df.columns:
        continue

    y        = df[target].astype(int)
    model    = models[target]
    y_prob   = model.predict_proba(X)[:, 1]
    y_pred   = (y_prob >= 0.5).astype(int)

    auc      = roc_auc_score(y, y_prob)
    f1       = f1_score(y, y_pred, zero_division=0)
    acc      = accuracy_score(y, y_pred)
    cm       = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0

    results[target] = {
        "auc":         auc,
        "f1":          f1,
        "accuracy":    acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv":         ppv,
        "cm":          cm,
        "y":           y,
        "y_prob":      y_prob,
    }

    name = TARGET_NAMES[target]
    print(f"\n{'─'*65}")
    print(f"  {name}")
    print(f"{'─'*65}")
    print(f"  AUC-ROC      : {auc:.4f}   (>0.90 = Excellent)")
    print(f"  F1 Score     : {f1:.4f}   (>0.70 = Good)")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  Sensitivity  : {sensitivity:.4f}  (True Positive Rate)")
    print(f"  Specificity  : {specificity:.4f}  (True Negative Rate)")
    print(f"  Precision    : {ppv:.4f}  (Positive Predictive Value)")
    print(f"\n  Confusion Matrix:")
    print(f"              Predicted NO   Predicted YES")
    print(f"  Actual NO  :   {tn:<10}     {fp}")
    print(f"  Actual YES :   {fn:<10}     {tp}")
    print(f"\n{classification_report(y, y_pred, zero_division=0)}")

# ── Plot: ROC Curves + Confusion Matrices ────────────────────
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor("#0f1117")
gs  = gridspec.GridSpec(2, 3, figure=fig)

colors = {
    "LABEL_DIABETES":  "#4f8ef7",
    "LABEL_CKD":       "#f7a14f",
    "LABEL_HEARTFAIL": "#f74f4f",
}

# Row 1 — ROC Curves
ax_roc = fig.add_subplot(gs[0, :])
ax_roc.set_facecolor("#1a1f2e")
ax_roc.plot([0, 1], [0, 1], "w--", alpha=0.3, label="Random (AUC=0.50)")

for target, res in results.items():
    fpr, tpr, _ = roc_curve(res["y"], res["y_prob"])
    name        = TARGET_NAMES[target]
    color       = colors[target]
    ax_roc.plot(
        fpr, tpr,
        color     = color,
        linewidth = 2.5,
        label     = f"{name}  (AUC = {res['auc']:.4f})"
    )

ax_roc.set_xlabel("False Positive Rate", color="white")
ax_roc.set_ylabel("True Positive Rate", color="white")
ax_roc.set_title("ROC Curves — All Three Models", color="white", fontsize=14)
ax_roc.tick_params(colors="white")
ax_roc.legend(facecolor="#252b3b", labelcolor="white", fontsize=11)
ax_roc.spines[:].set_color("#2d3448")

# Row 2 — Confusion Matrices
for i, (target, res) in enumerate(results.items()):
    ax = fig.add_subplot(gs[1, i])
    ax.set_facecolor("#1a1f2e")
    cm   = res["cm"]
    name = TARGET_NAMES[target]
    col  = colors[target]

    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted NO", "Predicted YES"], color="white")
    ax.set_yticklabels(["Actual NO", "Actual YES"],       color="white")
    ax.set_title(f"{name}\nAUC={res['auc']:.3f}  F1={res['f1']:.3f}",
                 color=col, fontsize=10)
    ax.tick_params(colors="white")

    for r in range(2):
        for c in range(2):
            bg_val = cm[r, c] / cm.max()
            txt_color = "white" if bg_val > 0.4 else "#0f1117"
            ax.text(c, r, f"{cm[r, c]:,}",
                    ha="center", va="center",
                    color=txt_color, fontsize=13, fontweight="bold")

plt.suptitle("ClinicalMind — Full Model Evaluation", 
             color="white", fontsize=16, y=1.01)
plt.tight_layout()

save_path = "outputs/model_evaluation.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight",
            facecolor="#0f1117")
plt.show()
print(f"\nEvaluation plot saved → {save_path}")
print("=" * 65)