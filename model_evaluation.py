import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    confusion_matrix, roc_curve, classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
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

# ── Cross validation setup ────────────────────────────────────
CV_FOLDS = 5
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

print(f"\nRunning {CV_FOLDS}-Fold Stratified Cross Validation...")
print("(This will take a few minutes)\n")

# ── Per-model evaluation ──────────────────────────────────────
results    = {}
cv_results = {}

for target in TARGETS:
    if target not in df.columns:
        continue

    y     = df[target].astype(int)
    model = models[target]

    # ── Single split metrics ──────────────────────────────────
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc  = roc_auc_score(y, y_prob)
    f1   = f1_score(y, y_pred, zero_division=0)
    acc  = accuracy_score(y, y_pred)
    cm   = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0

    # ── Cross validation ──────────────────────────────────────
    print(f"  Running CV for {TARGET_NAMES[target]}...")
    cv_auc = cross_val_score(
        model, X, y,
        cv      = skf,
        scoring = "roc_auc",
        n_jobs  = -1
    )
    cv_f1 = cross_val_score(
        model, X, y,
        cv      = skf,
        scoring = "f1",
        n_jobs  = -1
    )
    cv_acc = cross_val_score(
        model, X, y,
        cv      = skf,
        scoring = "accuracy",
        n_jobs  = -1
    )

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
        "cv_auc":      cv_auc,
        "cv_f1":       cv_f1,
        "cv_acc":      cv_acc,
    }

    name = TARGET_NAMES[target]
    print(f"\n{'─'*65}")
    print(f"  {name}")
    print(f"{'─'*65}")
    print(f"  ── Single Split Metrics ──────────────────────────────")
    print(f"  AUC-ROC      : {auc:.4f}")
    print(f"  F1 Score     : {f1:.4f}")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  Sensitivity  : {sensitivity:.4f}  (True Positive Rate)")
    print(f"  Specificity  : {specificity:.4f}  (True Negative Rate)")
    print(f"  Precision    : {ppv:.4f}  (Positive Predictive Value)")
    print(f"\n  ── {CV_FOLDS}-Fold Cross Validation ──────────────────────────")
    print(f"  CV AUC       : {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}"
          f"  (min={cv_auc.min():.4f}  max={cv_auc.max():.4f})")
    print(f"  CV F1        : {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")
    print(f"  CV Accuracy  : {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"              Predicted NO   Predicted YES")
    print(f"  Actual NO  :   {tn:<10}     {fp}")
    print(f"  Actual YES :   {fn:<10}     {tp}")
    print(f"\n{classification_report(y, y_pred, zero_division=0)}")

# ── Summary Table ─────────────────────────────────────────────
print(f"\n{'═'*65}")
print("   FINAL SUMMARY — Single Split + Cross Validation")
print(f"{'═'*65}")
print(f"  {'Disease':<22} {'AUC':>7} {'CV AUC':>14} {'F1':>7} "
      f"{'CV F1':>13} {'Sensitivity':>12}")
print(f"  {'─'*65}")
for target in TARGETS:
    r    = results[target]
    name = TARGET_NAMES[target]
    print(
        f"  {name:<22} "
        f"{r['auc']:>7.4f} "
        f"{r['cv_auc'].mean():>7.4f}"
        f"+/-{r['cv_auc'].std():.4f}  "
        f"{r['f1']:>7.4f} "
        f"{r['cv_f1'].mean():>6.4f}"
        f"+/-{r['cv_f1'].std():.4f}  "
        f"{r['sensitivity']:>12.4f}"
    )

# ══════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════
print("\nGenerating evaluation charts...")

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor("#0f1117")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)

colors = {
    "LABEL_DIABETES":  "#4f8ef7",
    "LABEL_CKD":       "#f7a14f",
    "LABEL_HEARTFAIL": "#ff4b4b",
}

# ── Row 1: ROC Curves ─────────────────────────────────────────
ax_roc = fig.add_subplot(gs[0, :])
ax_roc.set_facecolor("#1a1f2e")
ax_roc.plot([0,1],[0,1], "w--", alpha=0.3, label="Random (AUC=0.50)")

for target, res in results.items():
    fpr, tpr, _ = roc_curve(res["y"], res["y_prob"])
    name        = TARGET_NAMES[target]
    color       = colors[target]
    cv          = res["cv_auc"]
    ax_roc.plot(
        fpr, tpr,
        color     = color,
        linewidth = 2.5,
        label     = (
            f"{name}  "
            f"AUC={res['auc']:.4f} | "
            f"CV={cv.mean():.4f}±{cv.std():.4f}"
        )
    )

ax_roc.set_xlabel("False Positive Rate", color="white")
ax_roc.set_ylabel("True Positive Rate",  color="white")
ax_roc.set_title(
    "ROC Curves — All Three Models with 5-Fold CV Scores",
    color="white", fontsize=13
)
ax_roc.tick_params(colors="white")
ax_roc.legend(facecolor="#252b3b", labelcolor="white", fontsize=10)
ax_roc.spines[:].set_color("#2d3448")

# ── Row 2: Confusion Matrices ─────────────────────────────────
for i, (target, res) in enumerate(results.items()):
    ax = fig.add_subplot(gs[1, i])
    ax.set_facecolor("#1a1f2e")
    cm   = res["cm"]
    name = TARGET_NAMES[target]
    col  = colors[target]

    ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted NO", "Predicted YES"], color="white")
    ax.set_yticklabels(["Actual NO",    "Actual YES"],    color="white")
    ax.set_title(
        f"{name}\nAUC={res['auc']:.3f}  F1={res['f1']:.3f}",
        color=col, fontsize=10
    )
    ax.tick_params(colors="white")

    for r in range(2):
        for c in range(2):
            bg_val    = cm[r, c] / cm.max()
            txt_color = "white" if bg_val > 0.4 else "#0f1117"
            ax.text(c, r, f"{cm[r, c]:,}",
                    ha="center", va="center",
                    color=txt_color, fontsize=13, fontweight="bold")

# ── Row 3: CV AUC per fold + CV Metrics bar chart ─────────────

# Chart A — CV AUC per fold (line per disease)
ax_fold = fig.add_subplot(gs[2, 0:2])
ax_fold.set_facecolor("#1a1f2e")

for target, res in results.items():
    name = TARGET_NAMES[target]
    col  = colors[target]
    fold_scores = res["cv_auc"]
    ax_fold.plot(
        range(1, CV_FOLDS + 1),
        fold_scores,
        marker    = "o",
        linewidth = 2,
        color     = col,
        label     = f"{name}  (mean={fold_scores.mean():.4f})"
    )
    # Mean line
    ax_fold.axhline(
        fold_scores.mean(),
        color     = col,
        linestyle = "--",
        alpha     = 0.4,
        linewidth = 1
    )

ax_fold.set_xlabel("Fold Number",  color="#9ca3af")
ax_fold.set_ylabel("AUC-ROC",      color="#9ca3af")
ax_fold.set_title(
    f"{CV_FOLDS}-Fold CV — AUC Per Fold",
    color="white", fontsize=11, fontweight="bold"
)
ax_fold.set_xticks(range(1, CV_FOLDS + 1))
ax_fold.tick_params(colors="#9ca3af")
ax_fold.spines[:].set_color("#2d3448")
ax_fold.legend(facecolor="#252b3b", labelcolor="white", fontsize=8)
ax_fold.set_ylim(0.88, 1.01)
ax_fold.grid(color="#2d3448", alpha=0.5)

# Chart B — CV Mean ± Std bar chart
ax_cv = fig.add_subplot(gs[2, 2])
ax_cv.set_facecolor("#1a1f2e")

disease_short = ["Diabetes", "CKD", "Heart\nFailure"]
x     = np.arange(len(TARGETS))
width = 0.35

cv_means = [results[t]["cv_auc"].mean() for t in TARGETS]
cv_stds  = [results[t]["cv_auc"].std()  for t in TARGETS]
bar_cols = [colors[t] for t in TARGETS]

bars = ax_cv.bar(
    x, cv_means,
    width      = width,
    color      = bar_cols,
    alpha      = 0.9,
    yerr       = cv_stds,
    error_kw   = {
        "ecolor":    "white",
        "capsize":   5,
        "linewidth": 1.5
    }
)

for bar, mean, std in zip(bars, cv_means, cv_stds):
    ax_cv.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + std + 0.003,
        f"{mean:.4f}\n±{std:.4f}",
        ha="center", va="bottom",
        color="white", fontsize=8
    )

ax_cv.set_xticks(x)
ax_cv.set_xticklabels(disease_short, color="white", fontsize=9)
ax_cv.set_ylabel("CV AUC (Mean ± Std)", color="#9ca3af")
ax_cv.set_title(
    "CV Stability Summary",
    color="white", fontsize=11, fontweight="bold"
)
ax_cv.set_ylim(0.88, 1.04)
ax_cv.tick_params(colors="#9ca3af")
ax_cv.spines[:].set_color("#2d3448")

plt.suptitle(
    "ClinicalMind — Complete Model Evaluation\n"
    f"Single Split Metrics + {CV_FOLDS}-Fold Stratified Cross Validation",
    color="white", fontsize=14, fontweight="bold", y=1.01
)

save_path = "outputs/model_evaluation.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.show()
print(f"\nEvaluation chart saved → {save_path}")
print("=" * 65)
print("   Evaluation Complete")
print("=" * 65)