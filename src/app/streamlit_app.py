"""
FDA Regulatory Pathway Predictor — Streamlit App

Three pages:
  1. Pathway Predictor  — enter device details, get pathway + confidence
  2. EDA Dashboard      — explore training data and insights
  3. Model Performance  — review evaluation metrics and model card

Run with:
    uv run streamlit run src/app/streamlit_app.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parents[2] / "artifacts"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FDA Pathway Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    """Load model.pkl; return payload dict or None."""
    model_path = ARTIFACTS_DIR / "model.pkl"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def artifact_exists(name: str) -> bool:
    return (ARTIFACTS_DIR / name).exists()


def read_text(name: str) -> str:
    path = ARTIFACTS_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return f"*{name} not found. Run the pipeline first.*"


def read_json(name: str) -> dict | None:
    path = ARTIFACTS_DIR / name
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


MEDICAL_SPECIALTIES = [
    "Cardiovascular",
    "Orthopedic",
    "Neurology",
    "General Hospital",
    "Radiology",
    "Oncology",
    "Ophthalmology",
    "General Plastic Surgery",
    "Other",
]

DEVICE_CLASSES = ["Class I", "Class II", "Class III"]

PATHWAY_INFO = {
    "510k": {
        "color": "#2196F3",
        "icon": "",
        "description": (
            "510(k) Clearance — demonstrate substantial equivalence to a predicate device. "
            "Typical for Class II devices. Usually faster and less costly than PMA."
        ),
    },
    "PMA": {
        "color": "#F44336",
        "icon": "",
        "description": (
            "Premarket Approval (PMA) — required for Class III devices that support or "
            "sustain human life or present an unreasonable risk. Requires clinical evidence."
        ),
    },
    "De Novo": {
        "color": "#4CAF50",
        "icon": "",
        "description": (
            "De Novo — for novel, low-to-moderate risk devices with no predicate. "
            "Creates a new device type and can itself serve as a future predicate."
        ),
    },
}


# ---------------------------------------------------------------------------
# Page 1 — Pathway Predictor
# ---------------------------------------------------------------------------

def page_predictor():
    st.title(" FDA Regulatory Pathway Predictor")
    st.markdown(
        "Enter your device's characteristics below to receive an estimated "
        "regulatory pathway recommendation."
    )

    model_payload = load_model()
    if model_payload is None:
        st.warning(
            "No trained model found. Run the pipeline first:  \n"
            "`uv run python -m src.flow.main_flow`"
        )
        return

    st.markdown("---")
    st.subheader("Device Characteristics")

    col1, col2 = st.columns(2)

    with col1:
        device_name = st.text_input(
            "Device Name",
            placeholder="e.g. Cardiac Monitor, Hip Implant…",
        )
        device_class = st.selectbox("Device Class", DEVICE_CLASSES, index=1)
        medical_specialty = st.selectbox("Medical Specialty", MEDICAL_SPECIALTIES)

    with col2:
        implant_flag = st.checkbox("Implantable Device")
        life_sustain_flag = st.checkbox("Life-Sustaining / Life-Supporting Device")
        st.markdown("")
        st.markdown("")
        predict_btn = st.button("Predict Pathway", type="primary", use_container_width=True)

    st.markdown("---")

    if predict_btn:
        if not device_name.strip():
            st.error("Please enter a device name.")
            return

        # Build feature row
        import numpy as np
        from sklearn.preprocessing import LabelEncoder

        model = model_payload["model"]
        feature_cols = model_payload["feature_cols"]
        le_target: LabelEncoder = model_payload["label_encoder"]

        # Encode inputs the same way as training
        # We need to encode device_class and medical_specialty
        # Use the same LabelEncoder classes from training if available,
        # or map manually via a fallback
        try:
            contract = read_json("dataset_contract.json")
            # Reconstruct encoders from clean_data if available
            clean_path = ARTIFACTS_DIR / "clean_data.csv"
            if clean_path.exists():
                df_train = pd.read_csv(clean_path)
                le_class = LabelEncoder().fit(df_train["device_class"].astype(str))
                le_spec = LabelEncoder().fit(
                    df_train["medical_specialty_description"].astype(str)
                )
            else:
                # Fallback: fit on a known set
                le_class = LabelEncoder().fit(DEVICE_CLASSES + ["Unknown"])
                le_spec = LabelEncoder().fit(MEDICAL_SPECIALTIES)

            def safe_transform(le, value):
                if value in le.classes_:
                    return int(le.transform([value])[0])
                # Use the most common class index as fallback
                return 0

            features = []
            if "device_class_enc" in feature_cols:
                features.append(safe_transform(le_class, device_class))
            if "medical_specialty_description_enc" in feature_cols:
                features.append(safe_transform(le_spec, medical_specialty))
            if "implant_flag_bin" in feature_cols:
                features.append(1 if implant_flag else 0)
            if "life_sustain_support_flag_bin" in feature_cols:
                features.append(1 if life_sustain_flag else 0)

            X_pred = np.array(features).reshape(1, -1)
            y_pred_enc = model.predict(X_pred)[0]
            predicted_pathway = le_target.inverse_transform([y_pred_enc])[0]

            # Confidence (probability)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_pred)[0]
                class_probs = {
                    le_target.inverse_transform([i])[0]: float(p)
                    for i, p in enumerate(proba)
                }
            else:
                class_probs = {predicted_pathway: 1.0}

            # Display result
            info = PATHWAY_INFO.get(predicted_pathway, {})
            st.success(f"**Predicted Pathway: {info.get('icon', '')} {predicted_pathway}**")
            st.markdown(f"> {info.get('description', '')}")

            # Confidence chart
            st.subheader("Confidence by Pathway")
            prob_df = pd.DataFrame(
                {"Pathway": list(class_probs.keys()), "Confidence": list(class_probs.values())}
            ).sort_values("Confidence", ascending=False)

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 3))
            colors = [PATHWAY_INFO.get(p, {}).get("color", "#888") for p in prob_df["Pathway"]]
            bars = ax.barh(prob_df["Pathway"], prob_df["Confidence"], color=colors, edgecolor="white")
            ax.set_xlim(0, 1.0)
            ax.set_xlabel("Probability")
            ax.set_title(f"Pathway Confidence — {device_name}")
            for bar, val in zip(bars, prob_df["Confidence"]):
                ax.text(
                    min(val + 0.02, 0.95),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1%}",
                    va="center",
                    fontweight="bold",
                )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Summary table
            with st.expander("Full probability breakdown"):
                st.dataframe(
                    prob_df.style.format({"Confidence": "{:.2%}"}).bar(
                        subset=["Confidence"], color="#2196F3"
                    ),
                    use_container_width=True,
                )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            logger.exception("Prediction error")

    # Disclaimer
    st.markdown("---")
    st.info(
        "**Disclaimer:** This tool is for planning purposes only. "
        "Consult qualified regulatory counsel before making regulatory decisions."
    )


# ---------------------------------------------------------------------------
# Page 2 — EDA Dashboard
# ---------------------------------------------------------------------------

def page_eda():
    st.title(" EDA Dashboard")
    st.markdown(
        "Exploratory data analysis of the training dataset used to build the model."
    )

    tab1, tab2, tab3 = st.tabs(["EDA Report", "Insights", "Dataset Contract"])

    with tab1:
        st.subheader("EDA Report")
        if artifact_exists("eda_report.html"):
            html_content = (ARTIFACTS_DIR / "eda_report.html").read_text(encoding="utf-8")
            components.html(html_content, height=900, scrolling=True)
        else:
            st.warning("eda_report.html not found. Run the pipeline first.")

    with tab2:
        st.subheader("Key Insights")
        st.markdown(read_text("insights.md"))

    with tab3:
        st.subheader("Dataset Contract")
        contract = read_json("dataset_contract.json")
        if contract:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Records", f"{contract.get('row_count', 'N/A'):,}")
            col2.metric("Feature Columns", len(contract.get("feature_columns", [])))
            col3.metric("Target Classes", len(contract.get("target_classes", [])))

            st.markdown("#### Pathway Distribution")
            dist = contract.get("pathway_distribution", {})
            if dist:
                total = sum(dist.values())
                dist_df = pd.DataFrame(
                    [{"Pathway": k, "Count": v, "Pct": f"{100*v/total:.1f}%"}
                     for k, v in dist.items()]
                )
                st.dataframe(dist_df, use_container_width=True)

            st.markdown("#### Column Details")
            dtypes = contract.get("dtypes", {})
            nulls = contract.get("null_counts", {})
            if dtypes:
                detail_df = pd.DataFrame([
                    {"Column": col, "Type": dtypes.get(col, ""), "Nulls": nulls.get(col, 0)}
                    for col in contract.get("columns", [])
                ])
                st.dataframe(detail_df, use_container_width=True)
        else:
            st.warning("dataset_contract.json not found. Run the pipeline first.")


# ---------------------------------------------------------------------------
# Page 3 — Model Performance
# ---------------------------------------------------------------------------

def page_model_performance():
    st.title(" Model Performance")
    st.markdown("Full evaluation metrics, confusion matrix, and model card.")

    tab1, tab2, tab3 = st.tabs(["Evaluation Report", "Model Card", "Confusion Matrix"])

    with tab1:
        st.subheader("Evaluation Report")
        st.markdown(read_text("evaluation_report.md"))

    with tab2:
        st.subheader("Model Card")
        st.markdown(read_text("model_card.md"))

    with tab3:
        st.subheader("Confusion Matrix")
        cm_path = ARTIFACTS_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix — Best Model", use_container_width=True)
        else:
            st.warning("confusion_matrix.png not found. Run the pipeline first.")

        # Show label mapping if available
        lm = read_json("label_mapping.json")
        if lm:
            with st.expander("Label encoding mapping"):
                st.json(lm)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

PAGES = {
    " Pathway Predictor": page_predictor,
    " EDA Dashboard": page_eda,
    " Model Performance": page_model_performance,
}


def main():
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/"
            "FDA_logo.svg/200px-FDA_logo.svg.png",
            width=120,
        )
        st.markdown("## FDA Pathway Predictor")
        st.markdown("*ML-powered regulatory pathway estimation*")
        st.markdown("---")
        selected = st.radio("Navigate to", list(PAGES.keys()), label_visibility="collapsed")
        st.markdown("---")
        st.markdown(
            "<small>Built with CrewAI · scikit-learn · Streamlit</small>",
            unsafe_allow_html=True,
        )

    PAGES[selected]()


if __name__ == "__main__":
    main()
