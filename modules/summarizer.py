import pandas as pd
import re
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import os

# Load BART directly (bypasses pipeline task name issues across versions)
print("Loading summarization model...")
LOCAL_MODEL_PATH  = "models/bart-large-cnn"
_MODEL_NAME       = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) \
                    else "facebook/bart-large-cnn"

print(f"  Source: {'Local' if os.path.exists(LOCAL_MODEL_PATH) else 'Internet'}")

_TOKENIZER = BartTokenizer.from_pretrained(_MODEL_NAME, local_files_only=os.path.exists(LOCAL_MODEL_PATH))
_MODEL     = BartForConditionalGeneration.from_pretrained(_MODEL_NAME, local_files_only=os.path.exists(LOCAL_MODEL_PATH))
_MODEL.eval()
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL.to(_DEVICE)
print(f"Summarization model loaded on {_DEVICE}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_note(text):
    """Remove boilerplate, anonymization tags, excessive whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\[\*\*.*?\*\*\]", "", text)   # Remove [** anonymized **] tags
    text = re.sub(r"\n+", " ", text)               # Collapse newlines
    text = re.sub(r"\s+", " ", text)               # Collapse spaces
    text = text.strip()
    return text


def chunk_text(text, max_tokens=900):
    """
    BART has a 1024 token limit.
    We chunk the text into safe-sized pieces and summarize each,
    then combine for a final summary.
    """
    words = text.split()
    chunks = []
    current = []
    count = 0

    for word in words:
        current.append(word)
        count += 1
        if count >= max_tokens:
            chunks.append(" ".join(current))
            current = []
            count = 0

    if current:
        chunks.append(" ".join(current))

    return chunks


def summarize_chunks(chunks):
    """Summarize each chunk using BART directly."""
    summaries = []
    for chunk in chunks:
        if len(chunk.split()) < 30:
            summaries.append(chunk)
            continue
        try:
            inputs = _TOKENIZER(
                chunk,
                return_tensors  = "pt",
                max_length      = 1024,
                truncation      = True
            ).to(_DEVICE)

            with torch.no_grad():
                summary_ids = _MODEL.generate(
                    inputs["input_ids"],
                    num_beams  = 4,
                    max_length = 200,
                    min_length = 60,
                    early_stopping = True
                )

            text = _TOKENIZER.decode(
                summary_ids[0],
                skip_special_tokens = True
            )
            summaries.append(text)
        except Exception as e:
            print(f"Chunk summarization error: {e}")
            summaries.append(chunk[:300])
    return " ".join(summaries)


# ── Note Aggregation ──────────────────────────────────────────────────────────

NOTE_PRIORITY = [
    "discharge summary",
    "physician",
    "nursing",
    "radiology",
    "ecg",
    "echo",
]

def get_patient_notes(noteevents_df, subject_id, max_notes=10):
    """
    Get the most clinically relevant notes for a patient.
    Prioritizes discharge summaries, then physician notes.
    """
    notes = noteevents_df[noteevents_df["SUBJECT_ID"] == subject_id].copy()
    notes["CATEGORY"] = notes["CATEGORY"].astype(str).str.lower()

    # Sort by priority category
    priority_order = {cat: i for i, cat in enumerate(NOTE_PRIORITY)}
    notes["PRIORITY"] = notes["CATEGORY"].map(priority_order).fillna(99)
    notes = notes.sort_values("PRIORITY").head(max_notes)

    combined = " ".join(notes["TEXT"].apply(clean_note).tolist())
    return combined


# ── Structured Extraction ─────────────────────────────────────────────────────

def extract_medications_from_notes(text):
    """Simple regex-based medication extraction from note text."""
    med_patterns = [
        r'\b(metformin|insulin|lisinopril|atorvastatin|furosemide|'
        r'aspirin|warfarin|heparin|metoprolol|carvedilol|'
        r'amlodipine|losartan|omeprazole|pantoprazole|'
        r'amiodarone|digoxin|spironolactone|hydralazine)\b'
    ]
    found = set()
    for pattern in med_patterns:
        matches = re.findall(pattern, text.lower())
        found.update(matches)
    return list(found) if found else ["Not extracted"]


def extract_procedures_from_notes(text):
    """Simple keyword-based procedure extraction."""
    procedure_keywords = [
        "dialysis", "intubation", "ventilation", "catheterization",
        "biopsy", "echocardiogram", "colonoscopy", "endoscopy",
        "surgery", "transfusion", "angioplasty", "bypass",
        "pacemaker", "defibrillator", "bronchoscopy"
    ]
    found = [p for p in procedure_keywords if p in text.lower()]
    return found if found else ["None documented"]


# ── Master Summary Builder ────────────────────────────────────────────────────

def build_clinical_summary(
    subject_id,
    noteevents_df,
    diagnosis_history,       # list of diagnosis strings
    prescription_list,       # list of drug names
    lab_summary,             # dict of lab name -> mean value
    num_admissions,
    avg_los,
    age,
    gender
):
    """
    Build a full structured clinical summary for one patient.
    Returns a dict with all summary sections.
    """

    # 1. Get and summarize notes
    raw_notes = get_patient_notes(noteevents_df, subject_id)

    if len(raw_notes.split()) > 50:
        chunks = chunk_text(raw_notes, max_tokens=900)
        note_summary = summarize_chunks(chunks)
    else:
        note_summary = raw_notes if raw_notes else "No clinical notes available."

    # 2. Extract from notes
    meds_from_notes  = extract_medications_from_notes(raw_notes)
    procs_from_notes = extract_procedures_from_notes(raw_notes)

    # 3. Combine prescription sources
    all_medications = list(set(
        [m.title() for m in prescription_list] +
        [m.title() for m in meds_from_notes]
    ))

    # 4. Format lab summary
    formatted_labs = {
        k: f"{v:.2f}" for k, v in lab_summary.items()
        if v is not None and str(v) != "nan"
    }

    # 5. Assemble structured summary dict
    summary = {
        "subject_id":        subject_id,
        "age":               age,
        "gender":            "Male" if gender == 1 else "Female",
        "num_admissions":    int(num_admissions),
        "avg_los_days":      round(float(avg_los), 1),
        "diagnosis_history": diagnosis_history,
        "medications":       all_medications,
        "procedures":        procs_from_notes,
        "lab_highlights":    formatted_labs,
        "clinical_narrative": note_summary,
    }

    return summary


def format_summary_text(summary_dict):
    """
    Convert the summary dict into a clean readable text block
    for display in the report and dashboard.
    """
    s = summary_dict
    diag_str  = ", ".join(s["diagnosis_history"][:15]) if s["diagnosis_history"] else "None recorded"
    meds_str  = ", ".join(s["medications"][:15])        if s["medications"]       else "None recorded"
    procs_str = ", ".join(s["procedures"][:10])         if s["procedures"]        else "None documented"

    lab_str = ""
    for lab, val in list(s["lab_highlights"].items())[:10]:
        lab_str += f"\n      {lab}: {val}"

    text = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CLINICAL JOURNEY SUMMARY — Patient {s['subject_id']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  DEMOGRAPHICS
  ├─ Age    : {s['age']} years
  ├─ Gender : {s['gender']}
  ├─ Total Admissions : {s['num_admissions']}
  └─ Avg Length of Stay : {s['avg_los_days']} days

  DIAGNOSIS HISTORY
  └─ {diag_str}

  MEDICATIONS
  └─ {meds_str}

  PROCEDURES
  └─ {procs_str}

  KEY LAB VALUES (Averages across admissions)
  {lab_str}

  CLINICAL NARRATIVE
  └─ {s['clinical_narrative']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return text


def batch_summarize(
    subject_ids,
    noteevents_df,
    icd_history_df,
    prescriptions_df,
    lab_features_df,
    base_df
):
    """
    Run summarization for a list of patients.
    Returns dict of subject_id -> summary_dict
    """
    noteevents_df.columns    = noteevents_df.columns.str.upper()
    icd_history_df.columns   = icd_history_df.columns.str.upper()
    prescriptions_df.columns = prescriptions_df.columns.str.upper()
    lab_features_df.columns  = lab_features_df.columns.str.upper()
    base_df.columns          = base_df.columns.str.upper()

    summaries = {}

    for sid in tqdm(subject_ids, desc="Summarizing patients"):
        try:
            # Diagnosis history
            icd_row = icd_history_df[icd_history_df["SUBJECT_ID"] == sid]
            diag_hist = icd_row["DIAGNOSIS_HISTORY"].values[0] if len(icd_row) else []

            # Prescriptions
            rx = prescriptions_df[prescriptions_df["SUBJECT_ID"] == sid]["DRUG"]
            rx_list = rx.dropna().unique().tolist()

            # Lab summary (mean values)
            lab_row = lab_features_df[lab_features_df["SUBJECT_ID"] == sid]
            lab_means = {}
            if len(lab_row):
                mean_cols = [c for c in lab_row.columns if "_MEAN" in c]
                for col in mean_cols:
                    lab_means[col.replace("_MEAN", "")] = lab_row[col].values[0]

            # Demographics
            base_row = base_df[base_df["SUBJECT_ID"] == sid]
            age    = base_row["AGE"].values[0]    if len(base_row) else 0
            gender = base_row["GENDER"].values[0] if len(base_row) else 0
            n_adm  = base_row["NUM_ADMISSIONS"].values[0] if len(base_row) else 0
            los    = base_row["AVG_LOS"].values[0]        if len(base_row) else 0

            summary = build_clinical_summary(
                subject_id        = sid,
                noteevents_df     = noteevents_df,
                diagnosis_history = diag_hist,
                prescription_list = rx_list,
                lab_summary       = lab_means,
                num_admissions    = n_adm,
                avg_los           = los,
                age               = age,
                gender            = gender
            )

            summaries[sid] = summary

        except Exception as e:
            print(f"Error summarizing patient {sid}: {e}")
            continue

    return summaries