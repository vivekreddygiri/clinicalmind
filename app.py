import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
from modules.risk_model       import load_all_models, predict_patient_risk, risk_level, TARGET_NAMES
from modules.summarizer       import build_clinical_summary, format_summary_text
from modules.explainer        import explain_patient, get_top_shap_features, shap_to_natural_language
from modules.report_generator import build_full_report, save_report

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "ClinicalMind",
    page_icon  = "🏥",
    layout     = "wide"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1rem; }
    .risk-high     { color: #ff4b4b; font-weight: bold; font-size: 1.1rem; }
    .risk-moderate { color: #ffa500; font-weight: bold; font-size: 1.1rem; }
    .risk-low-mod  { color: #ffd700; font-weight: bold; font-size: 1.1rem; }
    .risk-low      { color: #00cc66; font-weight: bold; font-size: 1.1rem; }
    .section-header {
        background: linear-gradient(90deg, #1a1f2e, #252b3b);
        padding: 0.5rem 1rem;
        border-left: 4px solid #4f8ef7;
        border-radius: 4px;
        margin: 1rem 0 0.5rem 0;
    }
    .metric-card {
        background: #1a1f2e;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #2d3448;
        text-align: center;
    }
    .stTextArea textarea { font-family: monospace; }
</style>
""", unsafe_allow_html=True)


# ── Load Resources (cached) ───────────────────────────────────
@st.cache_resource
def load_models():
    return load_all_models()

@st.cache_data
def load_outputs():
    base_df        = pd.read_pickle("outputs/base_df.pkl")
    lab_df         = pd.read_pickle("outputs/lab_df.pkl")
    rx_df          = pd.read_pickle("outputs/rx_df.pkl")
    icd_hist_df    = pd.read_pickle("outputs/icd_hist_df.pkl")
    feature_matrix = pd.read_pickle("outputs/feature_matrix.pkl")
    noteevents     = pd.read_pickle("outputs/noteevents_lean.pkl")
    prescriptions  = pd.read_pickle("outputs/prescriptions_lean.pkl")
    return base_df, lab_df, rx_df, icd_hist_df, feature_matrix, noteevents, prescriptions

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/caduceus.png", width=60)
    st.title("ClinicalMind")
    st.caption("AI-Powered Clinical Decision Support")
    st.markdown("---")

    st.subheader("🔍 Patient Lookup")
    subject_id_input = st.number_input(
        "Enter Patient ID (SUBJECT_ID)",
        min_value = 1,
        step      = 1,
        value     = 10006
    )

    st.markdown("---")
    st.subheader("🩺 Current Symptoms")
    st.caption("Enter symptoms reported at current visit")

    symptom_options = [
        "Fatigue", "Frequent Urination", "Excessive Thirst",
        "Chest Pain", "Shortness of Breath", "Swollen Legs",
        "Blurred Vision", "Nausea", "Decreased Urine Output",
        "Rapid Weight Gain", "Dizziness", "Palpitations"
    ]

    # Store symptom selections in session state
    # so they don't trigger re-analysis on change
    selected_symptoms = st.multiselect(
        "Select symptoms",
        symptom_options,
        default = st.session_state.get("symptom_input", []),
        key     = "symptom_input"
    )
    custom_symptom = st.text_input(
        "Add custom symptom",
        key = "custom_symptom_input"
    )

    st.markdown("---")
    analyze_btn = st.button(
        "🔬 Analyze Patient",
        use_container_width = True,
        type                = "primary"
    )

    # ONLY update analysis when button is explicitly clicked
    if analyze_btn:
        all_symptoms = list(selected_symptoms)
        if custom_symptom.strip():
            all_symptoms.append(custom_symptom.strip())

        st.session_state["analyzed"]   = True
        st.session_state["patient_id"] = int(subject_id_input)
        st.session_state["symptoms"]   = all_symptoms

    # Read from session state — never directly from widgets
    show_results    = st.session_state.get("analyzed", False)
    active_patient  = st.session_state.get("patient_id", int(subject_id_input))
    active_symptoms = st.session_state.get("symptoms", [])

    # Show what was used in the last analysis
    if show_results and active_symptoms:
        st.markdown("---")
        st.caption("Symptoms used in current analysis:")
        for sym in active_symptoms:
            st.markdown(f"  - {sym}")

# ── Main Panel ────────────────────────────────────────────────
st.title("🏥 ClinicalMind")
st.caption("Intelligent Patient Journey Summarizer & Chronic Disease Risk Profiler")
st.markdown("---")

if not show_results:
    st.info("👈 Enter a Patient ID and click **Analyze Patient** to begin.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📋 Clinical Summarization**
        Full structured summary of patient's
        clinical history from EHR notes.
        """)
    with col2:
        st.markdown("""
        **🔬 Risk Prediction**
        XGBoost models predict risk of
        Diabetes, CKD, and Heart Failure.
        """)
    with col3:
        st.markdown("""
        **💡 Explainable AI**
        SHAP values reveal exactly why
        each prediction was made.
        """)

else:
    subject_id = active_patient
    selected_symptoms = active_symptoms

    # Load everything
    with st.spinner("Loading patient data..."):
        try:
            models, feature_cols = load_models()
            (base_df, lab_df, rx_df,
             icd_hist_df, feature_matrix,
             noteevents, prescriptions) = load_outputs()
        except Exception as e:
            st.error(f"Failed to load outputs: {e}")
            st.stop()

    # Check patient exists
    patient_row = feature_matrix[feature_matrix["SUBJECT_ID"] == subject_id]
    if len(patient_row) == 0:
        st.error(f"Patient ID {subject_id} not found in the dataset.")
        st.stop()

    # ── Build Summary ─────────────────────────────────────────
    with st.spinner("Generating clinical summary..."):
        base_row  = base_df[base_df["SUBJECT_ID"] == subject_id]
        lab_row   = lab_df[lab_df["SUBJECT_ID"] == subject_id]
        icd_row   = icd_hist_df[icd_hist_df["SUBJECT_ID"] == subject_id]
        rx_row    = prescriptions[prescriptions["SUBJECT_ID"] == subject_id]

        diag_hist = icd_row["DIAGNOSIS_HISTORY"].values[0] if len(icd_row) else []
        rx_list   = rx_row["DRUG"].dropna().unique().tolist() if len(rx_row) else []

        lab_means = {}
        if len(lab_row):
            mean_cols = [c for c in lab_row.columns if "_MEAN" in c]
            for col in mean_cols:
                lab_means[col.replace("_MEAN", "")] = lab_row[col].values[0]

        age    = base_row["AGE"].values[0]    if len(base_row) else 0
        gender = base_row["GENDER"].values[0] if len(base_row) else 0
        n_adm  = base_row["NUM_ADMISSIONS"].values[0] if len(base_row) else 0
        los    = base_row["AVG_LOS"].values[0]        if len(base_row) else 0

        summary = build_clinical_summary(
            subject_id        = subject_id,
            noteevents_df     = noteevents,
            diagnosis_history = diag_hist,
            prescription_list = rx_list,
            lab_summary       = lab_means,
            num_admissions    = n_adm,
            avg_los           = los,
            age               = age,
            gender            = gender
        )

    # ── Predict Risk ──────────────────────────────────────────
    with st.spinner("Running risk prediction models..."):
        patient_features = patient_row.iloc[0].to_dict()
        patient_features.pop("SUBJECT_ID", None)
        raw_risk_scores = predict_patient_risk(patient_features, models, feature_cols)

        # Apply symptom adjustment on top of model scores
        from modules.risk_model import apply_symptom_adjustment
        risk_scores = apply_symptom_adjustment(raw_risk_scores, selected_symptoms)

    # ── Explain ───────────────────────────────────────────────
    with st.spinner("Computing SHAP explanations..."):
        aligned_row = patient_row.copy()
        for col in feature_cols:
            if col not in aligned_row.columns:
                aligned_row[col] = 0
        aligned_row = aligned_row[feature_cols]

        explanations = {}
        for target, model in models.items():
            shap_feats = get_top_shap_features(
                model, aligned_row, feature_cols
            )
            nl_exp = shap_to_natural_language(
                shap_feats, TARGET_NAMES[target]
            )
            explanations[target] = {
                "shap_features":  shap_feats,
                "nl_explanation": nl_exp
            }

    # ════════════════════════════════════════════════════════
    # TAB LAYOUT
    # ════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Clinical Summary",
        "🔬 Risk Profile",
        "💡 Explainability",
        "📄 Full Report"
    ])

    # ── Tab 1: Clinical Summary ───────────────────────────────
    with tab1:
        st.markdown(
            f"<div class='section-header'>Patient {subject_id} — Clinical Overview</div>",
            unsafe_allow_html=True
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Age",            f"{summary['age']} yrs")
        m2.metric("Gender",         summary["gender"])
        m3.metric("Admissions",     summary["num_admissions"])
        m4.metric("Avg Stay",       f"{summary['avg_los_days']} days")

        st.markdown("---")

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("🔖 Diagnosis History")
            if summary["diagnosis_history"]:
                for d in summary["diagnosis_history"][:15]:
                    st.markdown(f"- {d}")
            else:
                st.write("No diagnosis history available.")

            st.subheader("💊 Medications")
            if summary["medications"]:
                for m in summary["medications"][:15]:
                    st.markdown(f"- {m}")
            else:
                st.write("No medication records found.")

        with col_right:
            st.subheader("🔬 Key Lab Values")
            if summary["lab_highlights"]:
                lab_df_display = pd.DataFrame(
                    list(summary["lab_highlights"].items()),
                    columns=["Lab Test", "Avg Value"]
                )
                st.dataframe(lab_df_display, use_container_width=True)
            else:
                st.write("No lab data available.")

            st.subheader("🏥 Procedures")
            if summary["procedures"]:
                for p in summary["procedures"]:
                    st.markdown(f"- {p.title()}")
            else:
                st.write("None documented.")

        st.markdown("---")
        st.subheader("📝 Clinical Narrative")
        st.info(summary["clinical_narrative"])

        if selected_symptoms:
            st.subheader("🩺 Reported Current Symptoms")
            cols = st.columns(4)
            for i, sym in enumerate(selected_symptoms):
                cols[i % 4].warning(sym)

    # ── Tab 2: Risk Profile ───────────────────────────────────
    with tab2:
        st.markdown(
            "<div class='section-header'>Chronic Disease Risk Profile</div>",
            unsafe_allow_html=True
        )

        disease_labels = {
            "LABEL_DIABETES":  "Type 2 Diabetes",
            "LABEL_CKD":       "Chronic Kidney Disease",
            "LABEL_HEARTFAIL": "Heart Failure"
        }
        colors = {
            "HIGH":         "#ff4b4b",
            "MODERATE":     "#ffa500",
            "LOW-MODERATE": "#ffd700",
            "LOW":          "#00cc66"
        }

        for target, prob in risk_scores.items():
            level, icon = risk_level(prob)
            name        = disease_labels.get(target, target)
            color       = colors.get(level, "#888888")

            with st.container():
                c1, c2, c3 = st.columns([3, 1, 1])
                with c1:
                    st.markdown(f"**{icon} {name}**")
                    st.progress(float(prob))
                with c2:
                    st.markdown(
                        f"<span style='color:{color};font-size:1.2rem;font-weight:bold'>"
                        f"{level}</span>",
                        unsafe_allow_html=True
                    )
                with c3:
                    st.markdown(
                        f"<span style='font-size:1.4rem;font-weight:bold'>"
                        f"{prob*100:.1f}%</span>",
                        unsafe_allow_html=True
                    )
                st.markdown("---")

        # Radar chart
        st.subheader("📊 Risk Radar")
        categories = [disease_labels[t] for t in risk_scores.keys()]
        values     = [risk_scores[t] * 100 for t in risk_scores.keys()]

        fig = go.Figure(go.Scatterpolar(
            r      = values + [values[0]],
            theta  = categories + [categories[0]],
            fill   = "toself",
            line_color = "#4f8ef7",
            fillcolor  = "rgba(79,142,247,0.2)"
        ))
        fig.update_layout(
            polar = dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            font_color    = "#ffffff",
            height        = 400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Recommendations
        st.subheader("⚠️ Clinical Recommendation Flags")
        from modules.report_generator import generate_recommendations
        recs = generate_recommendations(risk_scores)
        for rec in recs:
            st.warning(rec)

    # ── Tab 3: Explainability ─────────────────────────────────
    with tab3:
        st.markdown(
            "<div class='section-header'>SHAP Explainability — Why This Prediction?</div>",
            unsafe_allow_html=True
        )

        for target, exp in explanations.items():
            name = TARGET_NAMES[target]
            prob = risk_scores[target]
            level, icon = risk_level(prob)

            with st.expander(
                f"{icon} {name} — {prob*100:.1f}% Risk",
                expanded=True
            ):
                st.markdown(f"**Natural Language Explanation:**")
                st.info(exp["nl_explanation"])

                st.markdown("**Top Risk Drivers:**")

                feat_names = [f["label"]         for f in exp["shap_features"]]
                shap_vals  = [f["shap_value"]     for f in exp["shap_features"]]
                feat_vals  = [f["feature_value"]  for f in exp["shap_features"]]
                bar_colors = [
                    "#ff4b4b" if v > 0 else "#00cc66"
                    for v in shap_vals
                ]

                fig = go.Figure(go.Bar(
                    x           = shap_vals,
                    y           = feat_names,
                    orientation = "h",
                    marker_color= bar_colors,
                    text        = [f"val={v}" for v in feat_vals],
                    textposition= "outside"
                ))
                fig.update_layout(
                    xaxis_title   = "SHAP Value (impact on prediction)",
                    paper_bgcolor = "rgba(0,0,0,0)",
                    plot_bgcolor  = "rgba(15,17,23,0.5)",
                    font_color    = "#ffffff",
                    height        = 300,
                    margin        = dict(l=10, r=10, t=10, b=30)
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Full Report ────────────────────────────────────
    with tab4:
        st.markdown(
            "<div class='section-header'>Full Clinical Intelligence Report</div>",
            unsafe_allow_html=True
        )

        full_report = build_full_report(
            summary_dict     = summary,
            risk_scores      = risk_scores,
            explanations     = explanations,
            current_symptoms = selected_symptoms
        )

        st.text_area(
            "Report Preview",
            value  = full_report,
            height = 700
        )

        report_path = save_report(full_report, subject_id)

        with open(report_path, "r", encoding="utf-8") as f:
            report_bytes = f.read()

        st.download_button(
            label    = "⬇️ Download Full Report (.txt)",
            data     = report_bytes,
            file_name= f"clinicalmind_patient_{subject_id}.txt",
            mime     = "text/plain"
        )