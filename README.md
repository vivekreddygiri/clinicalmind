# ClinicalMind 🏥

**Intelligent Patient Journey Summarizer & Chronic Disease Risk Profiler**

An end-to-end clinical AI system built on MIMIC-III that:

- Generates detailed structured clinical summaries from EHR notes
- Predicts risk of Type 2 Diabetes, CKD, and Heart Failure using XGBoost
- Explains predictions using SHAP (Explainable AI)
- Adjusts risk based on current patient symptoms
- Delivers results through an interactive Streamlit dashboard

## Tech Stack

- **NLP**: facebook/bart-large-cnn (HuggingFace Transformers)
- **ML**: XGBoost with SHAP explainability
- **Dashboard**: Streamlit + Plotly
- **Dataset**: MIMIC-III Clinical Database (46,520 patients)

## Model Performance

| Disease                | AUC-ROC | F1 Score | Sensitivity |
| ---------------------- | ------- | -------- | ----------- |
| Type 2 Diabetes        | 0.9758  | 0.8317   | 91.89%      |
| Chronic Kidney Disease | 0.9893  | 0.8128   | 96.22%      |
| Heart Failure          | 0.9966  | 0.9426   | 99.14%      |

## Project Structure

```
clinicalmind/
├── modules/
│   ├── preprocessor.py       # Data loading and cleaning
│   ├── label_engine.py       # ICD-9 risk label generation
│   ├── feature_engineer.py   # Feature extraction
│   ├── summarizer.py         # BART clinical summarization
│   ├── risk_model.py         # XGBoost training and inference
│   ├── explainer.py          # SHAP explainability
│   └── report_generator.py   # Clinical report builder
├── app.py                    # Streamlit dashboard
├── train.py                  # Training pipeline
├── evaluate_model.py         # Model evaluation + plots
├── get_sample_patients.py    # Sample patient IDs utility
├── save_model_locally.py     # Save BART model offline
└── requirements.txt
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/clinicalmind.git
cd clinicalmind
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add MIMIC-III data

Place your MIMIC-III CSV files in the `data/` folder.
Access to MIMIC-III requires credentialing at https://physionet.org

### 4. Save BART model locally

```bash
python save_model_locally.py
```

### 5. Run training pipeline

```bash
python train.py
```

### 6. Launch dashboard

```bash
streamlit run app.py
```

## Dataset

This project uses [MIMIC-III](https://physionet.org/content/mimiciii/1.4/),
a freely available critical care database. Access requires credentialing
through PhysioNet. Data files are not included in this repository.

## Disclaimer

ClinicalMind is a research prototype built for academic purposes.
It is not intended for real clinical use. All predictions must be
verified by qualified medical professionals.

## Author

B.Tech CSE Capstone Project
