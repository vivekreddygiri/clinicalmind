import os
from datetime import datetime
from modules.risk_model import TARGET_NAMES, risk_level

OUTPUT_PATH = "outputs/reports/"
os.makedirs(OUTPUT_PATH, exist_ok=True)


def build_full_report(
    summary_dict,
    risk_scores,
    explanations,
    current_symptoms=None
):
    """
    Build the complete ClinicalMind report for one patient.
    Returns the report as a formatted string.
    """
    s           = summary_dict
    now         = datetime.now().strftime("%Y-%m-%d %H:%M")
    symptoms    = current_symptoms or ["Not provided"]
    symptoms_str = ", ".join(symptoms)

    # ── Section 1: Header ──────────────────────────────────────────────────
    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║           ClinicalMind — Patient Intelligence Report             ║
║           Generated : {now}                          ║
╚══════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [1]  PATIENT DEMOGRAPHICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Patient ID       : {s['subject_id']}
  Age              : {s['age']} years
  Gender           : {s['gender']}
  Total Admissions : {s['num_admissions']}
  Avg Stay         : {s['avg_los_days']} days

"""

    # ── Section 2: Clinical Journey Summary ───────────────────────────────
    diag_str  = "\n      • ".join(s["diagnosis_history"][:15]) if s["diagnosis_history"] else "None recorded"
    meds_str  = "\n      • ".join(s["medications"][:15])        if s["medications"]       else "None recorded"
    procs_str = "\n      • ".join(s["procedures"][:10])         if s["procedures"]        else "None documented"

    report += f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [2]  CLINICAL JOURNEY SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  DIAGNOSIS HISTORY:
      • {diag_str}

  MEDICATIONS:
      • {meds_str}

  PROCEDURES DOCUMENTED:
      • {procs_str}

  CLINICAL NARRATIVE:
      {s['clinical_narrative']}

"""

    # ── Section 3: Lab Highlights ─────────────────────────────────────────
    report += """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [3]  KEY LAB VALUES (Averages Across All Admissions)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"""

    for lab, val in list(s["lab_highlights"].items())[:12]:
        report += f"      {lab:<25}: {val}\n"

    # ── Section 4: Current Symptoms ───────────────────────────────────────
    report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [4]  CURRENT SYMPTOMS (Physician Input)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      {symptoms_str}

"""

    # ── Section 5: Risk Profile ───────────────────────────────────────────
    report += """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [5]  CHRONIC DISEASE RISK PROFILE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"""

    for target, prob in risk_scores.items():
        level, icon = risk_level(prob)
        name = TARGET_NAMES.get(target, target)
        bar  = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        report += (
            f"  {icon}  {name:<30}"
            f"  {level:<15}"
            f"  {prob*100:.1f}%  [{bar}]\n"
        )

    # ── Section 6: Explainability ─────────────────────────────────────────
    report += """\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [6]  RISK DRIVERS — Why This Prediction?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"""

    for target, exp in explanations.items():
        name = TARGET_NAMES.get(target, target)
        report += f"\n  {name}:\n"
        report += f"  {exp['nl_explanation']}\n"
        report += f"  Top contributing factors:\n"
        for feat in exp["shap_features"][:5]:
            direction = "↑ increases" if feat["shap_value"] > 0 else "↓ reduces"
            report += (
                f"    • {feat['label']:<35}"
                f" = {feat['feature_value']}"
                f"  ({direction} risk)\n"
            )

    # ── Section 7: Recommendation Flag ───────────────────────────────────
    report += """\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [7]  CLINICAL RECOMMENDATION FLAGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"""

    recs = generate_recommendations(risk_scores)
    for rec in recs:
        report += f"  ⚠  {rec}\n"

    report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⚕  DISCLAIMER: This report is generated by an AI system trained
     on retrospective ICU data (MIMIC-III). It is intended to
     assist clinical decision-making, not replace physician judgment.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return report


def generate_recommendations(risk_scores):
    """Rule-based clinical recommendation flags based on risk scores."""
    recs = []
    if risk_scores.get("LABEL_DIABETES", 0) >= 0.70:
        recs.append("Order HbA1c and fasting glucose. Consider endocrinology referral.")
    elif risk_scores.get("LABEL_DIABETES", 0) >= 0.45:
        recs.append("Monitor glucose levels. Review diet and lifestyle factors.")

    if risk_scores.get("LABEL_CKD", 0) >= 0.70:
        recs.append("Order creatinine, eGFR, urine albumin. Nephrology referral advised.")
    elif risk_scores.get("LABEL_CKD", 0) >= 0.45:
        recs.append("Track creatinine trend. Avoid nephrotoxic medications.")

    if risk_scores.get("LABEL_HEARTFAIL", 0) >= 0.70:
        recs.append("Order BNP, echocardiogram. Cardiology referral strongly advised.")
    elif risk_scores.get("LABEL_HEARTFAIL", 0) >= 0.45:
        recs.append("Monitor fluid status and BNP. Review cardiac medications.")

    if not recs:
        recs.append("No high-risk flags at this time. Routine follow-up recommended.")

    return recs


def save_report(report_text, subject_id):
    """Save the report as a .txt file."""
    filename = f"patient_{subject_id}_report.txt"
    path     = os.path.join(OUTPUT_PATH, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Report saved → {path}")
    return path

