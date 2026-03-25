import pandas as pd
import random
import os
import shutil

# ── Config ────────────────────────────────────────────────────
NOTES_OUTPUT_DIR = "sample_notes"

# ── Load data ─────────────────────────────────────────────────
df         = pd.read_pickle("outputs/full_df.pkl")
noteevents = pd.read_pickle("outputs/noteevents_lean.pkl")

# ── Clear old notes folder and recreate ───────────────────────
if os.path.exists(NOTES_OUTPUT_DIR):
    shutil.rmtree(NOTES_OUTPUT_DIR)
os.makedirs(NOTES_OUTPUT_DIR)

# ── Sample 10 random patients ─────────────────────────────────
sample_ids = random.sample(list(df["SUBJECT_ID"].values), 10)

print("\n" + "="*40)
print("   ClinicalMind — Sample Patient IDs")
print("="*40)

for i, pid in enumerate(sample_ids, 1):
    row       = df[df["SUBJECT_ID"] == pid].iloc[0]
    diabetes  = "✓" if row["LABEL_DIABETES"]  == 1 else "✗"
    ckd       = "✓" if row["LABEL_CKD"]        == 1 else "✗"
    heartfail = "✓" if row["LABEL_HEARTFAIL"]  == 1 else "✗"

    print(f"  {i:>2}. Patient {pid:<8}  "
          f"Diabetes:{diabetes}  CKD:{ckd}  HeartFail:{heartfail}")

    # ── Get this patient's notes ──────────────────────────────
    patient_notes = noteevents[noteevents["SUBJECT_ID"] == pid].copy()

    # ── Build note text ───────────────────────────────────────
    lines = []
    lines.append("=" * 60)
    lines.append(f"  NOTEEVENTS — Patient {pid}")
    lines.append(f"  Diabetes:{diabetes}  CKD:{ckd}  HeartFail:{heartfail}")
    lines.append("=" * 60)

    if len(patient_notes) == 0:
        lines.append("\n  No clinical notes found for this patient.")
    else:
        lines.append(f"\n  Total notes: {len(patient_notes)}\n")

        # Group by category
        categories = patient_notes["CATEGORY"].unique()
        for cat in categories:
            cat_notes = patient_notes[patient_notes["CATEGORY"] == cat]
            lines.append(f"\n{'─'*60}")
            lines.append(f"  CATEGORY: {cat}  ({len(cat_notes)} notes)")
            lines.append(f"{'─'*60}")

            for j, (_, note_row) in enumerate(cat_notes.iterrows(), 1):
                lines.append(f"\n  --- Note {j} ---")
                text = str(note_row["TEXT"]).strip()
                # Clean anonymization tags
                import re
                text = re.sub(r"\[\*\*.*?\*\*\]", "[REDACTED]", text)
                text = re.sub(r"\n+", "\n", text)
                lines.append(text)

    lines.append("\n" + "=" * 60)
    lines.append("  END OF NOTES")
    lines.append("=" * 60)

    # ── Save to file ──────────────────────────────────────────
    filename  = f"patient_{pid}_notes.txt"
    filepath  = os.path.join(NOTES_OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

print("=" * 40)
print("  ✓ = has that condition in record")
print("  ✗ = does not have that condition")
print("=" * 40)
print(f"\n  Notes saved to: {NOTES_OUTPUT_DIR}/")
print(f"  Files: patient_XXXXXX_notes.txt x10")
print("=" * 40 + "\n")