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
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parents[2] / "artifacts"

# Ensure src/ is importable (needed on Streamlit Cloud)
_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FDA Pathway Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* ── Global ─────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] { background: #f4f6fb; }
[data-testid="stMain"] { padding-top: 0 !important; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"] { display: none; }

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #1a1f36 !important;
    border-right: 1px solid #2a3060;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: #c5cae9 !important; }
[data-testid="stSidebar"] .stMarkdown small { color: #5c6bc0 !important; }

/* Sidebar nav buttons */
[data-testid="stSidebar"] .stButton > button {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 10px;
    color: #9fa8da !important;
    font-size: 14px;
    font-weight: 500;
    text-align: left;
    padding: 10px 14px;
    transition: all 0.18s ease;
    width: 100%;
    margin: 2px 0;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(63,81,181,0.25) !important;
    border-color: rgba(63,81,181,0.4);
    color: #fff !important;
}
[data-testid="stSidebar"] .stButton > button:focus:not(:active) {
    background: rgba(63,81,181,0.35) !important;
    border-color: rgba(63,81,181,0.6);
    color: #fff !important;
    box-shadow: none;
}

/* ── Cards ───────────────────────────────────────────────────────── */
.card {
    background: #fff;
    border-radius: 14px;
    padding: 24px 28px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    margin-bottom: 20px;
    border: 1px solid #eef0f7;
}

/* ── Page hero banner ────────────────────────────────────────────── */
.page-hero {
    background: linear-gradient(135deg, #3f51b5 0%, #5c6bc0 100%);
    color: #fff;
    border-radius: 16px;
    padding: 30px 36px;
    margin-bottom: 28px;
}
.page-hero h1 { font-size: 26px; font-weight: 800; margin: 0 0 8px; letter-spacing: -0.3px; }
.page-hero p  { margin: 0; opacity: .88; font-size: 15px; line-height: 1.6; }

/* ── Section divider ─────────────────────────────────────────────── */
.section-header {
    font-size: 15px;
    font-weight: 700;
    color: #37474f;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin: 28px 0 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #e8eaf6;
    margin-left: 6px;
}

/* ── Pathway reference tiles ─────────────────────────────────────── */
.pathway-tile {
    background: #fff;
    border-radius: 12px;
    padding: 18px 20px;
    border-left: 5px solid;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    height: 100%;
    min-height: 140px;
}
.pathway-tile h4 { margin: 0 0 7px; font-size: 15px; font-weight: 700; }
.pathway-tile p  { margin: 0; font-size: 12.5px; color: #546e7a; line-height: 1.55; }
.pathway-tile .meta {
    margin-top: 10px;
    font-size: 11.5px;
    font-weight: 600;
    color: #78909c;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ── Result card ─────────────────────────────────────────────────── */
.result-card {
    border-radius: 14px;
    padding: 28px 30px;
    border-left: 7px solid;
    box-shadow: 0 6px 24px rgba(0,0,0,0.10);
}
.result-card .rc-eyebrow {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    opacity: .6;
    margin-bottom: 4px;
}
.result-card .rc-name {
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -0.5px;
    line-height: 1.1;
    margin: 0;
}
.result-card .rc-meta {
    font-size: 12.5px;
    font-weight: 600;
    opacity: .55;
    margin-top: 8px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.result-card .rc-desc {
    font-size: 14.5px;
    margin-top: 14px;
    line-height: 1.65;
    opacity: .85;
}

/* ── Metric pill ─────────────────────────────────────────────────── */
.metric-pill {
    background: #fff;
    border-radius: 12px;
    padding: 20px 22px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    border: 1px solid #eef0f7;
}
.metric-pill .mp-value { font-size: 30px; font-weight: 700; }
.metric-pill .mp-label {
    font-size: 11px;
    color: #78909c;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 4px;
}

/* ── Confidence bars ─────────────────────────────────────────────── */
.conf-wrap { margin: 8px 0; }
.conf-row {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 5px;
}
.conf-bg {
    background: #eef0f7;
    border-radius: 6px;
    height: 11px;
    overflow: hidden;
}
.conf-fill { height: 100%; border-radius: 6px; }

/* ── Disclaimer banner ───────────────────────────────────────────── */
.disclaimer {
    background: #fff8e1;
    border: 1px solid #ffe082;
    border-left: 5px solid #ffc107;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 13px;
    color: #5d4037;
    margin-top: 28px;
    display: flex;
    gap: 10px;
    align-items: flex-start;
    line-height: 1.55;
}

/* ── Input tweaks ────────────────────────────────────────────────── */
.stTextInput > div > div > input { border-radius: 8px; }
.stSelectbox > div > div > div  { border-radius: 8px; }

/* ── Primary button ──────────────────────────────────────────────── */
div:not([data-testid="stSidebar"]) .stButton > button[kind="primary"] {
    border-radius: 10px;
    font-weight: 700;
    letter-spacing: 0.02em;
    padding: 12px 20px;
    font-size: 15px;
}
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Auto-run pipeline if artifacts are missing (e.g. first deploy on Streamlit Cloud)
# ---------------------------------------------------------------------------

def _pipeline_ready() -> bool:
    return (ARTIFACTS_DIR / "model.pkl").exists() and (ARTIFACTS_DIR / "clean_data.csv").exists()


def _run_pipeline() -> None:
    """Run full pipeline inline — called once on first deploy."""
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    from src.flow.main_flow import run_pipeline
    run_pipeline()


if not _pipeline_ready():
    with st.spinner("First-time setup: fetching FDA data and training model… (this takes ~1 min)"):
        try:
            _run_pipeline()
            st.success("Pipeline complete! Reloading…")
            st.rerun()
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            st.stop()


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
    "Cardiovascular", "Orthopedic", "Neurology", "General Hospital",
    "Radiology", "Oncology", "Ophthalmology", "General Plastic Surgery", "Other",
]

DEVICE_CLASSES = ["Class I", "Class II", "Class III"]

CLASS_DESCRIPTIONS = {
    "Class I":   "Low risk — general controls only (e.g. bandages, tongue depressors).",
    "Class II":  "Moderate risk — special controls + 510(k) (e.g. infusion pumps, X-ray systems).",
    "Class III": "High risk — PMA required (e.g. pacemakers, implantable defibrillators).",
}

PATHWAY_INFO = {
    "510k": {
        "color": "#1976D2",
        "bg":    "#e3f2fd",
        "icon":  "🔵",
        "label": "510(k) Clearance",
        "description": (
            "Demonstrate substantial equivalence to a predicate device. "
            "Typical for Class II devices. Usually faster and less costly than PMA."
        ),
        "time": "~3–12 months",
        "risk": "Moderate",
    },
    "PMA": {
        "color": "#C62828",
        "bg":    "#ffebee",
        "icon":  "🔴",
        "label": "Premarket Approval",
        "description": (
            "Required for Class III devices that support or sustain human life or "
            "present an unreasonable risk of illness or injury. Requires clinical evidence."
        ),
        "time": "~2–3 years",
        "risk": "High",
    },
    "De Novo": {
        "color": "#2E7D32",
        "bg":    "#e8f5e9",
        "icon":  "🟢",
        "label": "De Novo Classification",
        "description": (
            "For novel, low-to-moderate risk devices with no predicate. "
            "Creates a new device type and can itself serve as a future predicate."
        ),
        "time": "~12–24 months",
        "risk": "Low–Moderate",
    },
}


def _render_confidence_bars(class_probs: dict) -> None:
    sorted_probs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
    html = ""
    for pathway, prob in sorted_probs:
        color = PATHWAY_INFO.get(pathway, {}).get("color", "#888")
        icon  = PATHWAY_INFO.get(pathway, {}).get("icon", "")
        html += f"""
        <div class="conf-wrap">
            <div class="conf-row">
                <span>{icon} {pathway}</span>
                <span style="color:{color}">{prob:.1%}</span>
            </div>
            <div class="conf-bg">
                <div class="conf-fill" style="width:{prob*100:.1f}%;background:{color};"></div>
            </div>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page 1 — Pathway Predictor
# ---------------------------------------------------------------------------

def page_predictor() -> None:
    st.markdown("""
    <div class="page-hero">
        <h1>🏥 FDA Regulatory Pathway Predictor</h1>
        <p>Enter your medical device characteristics to receive an ML-powered estimate
        of the most appropriate FDA regulatory submission pathway.</p>
    </div>
    """, unsafe_allow_html=True)

    model_payload = load_model()
    if model_payload is None:
        st.warning(
            "No trained model found. Run the pipeline first:  \n"
            "`uv run python -m src.flow.main_flow`"
        )
        return

    # ── Pathway overview tiles ──
    st.markdown('<div class="section-header">📋 Pathway Overview</div>', unsafe_allow_html=True)
    tile_cols = st.columns(3)
    for col, (key, info) in zip(tile_cols, PATHWAY_INFO.items()):
        with col:
            st.markdown(f"""
            <div class="pathway-tile" style="border-color:{info['color']};">
                <h4 style="color:{info['color']}">{info['icon']} {key}</h4>
                <p>{info['description']}</p>
                <div class="meta">⏱ {info['time']} &nbsp;|&nbsp; ⚠️ Risk: {info['risk']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input form ──
    st.markdown('<div class="section-header">🔬 Device Characteristics</div>', unsafe_allow_html=True)

    col_form, col_flags = st.columns([3, 2], gap="large")

    with col_form:
        device_name = st.text_input(
            "Device Name",
            placeholder="e.g. Cardiac Monitor, Hip Implant, Glucose Meter…",
            help="Enter the commercial or generic name of your medical device.",
        )
        c1, c2 = st.columns(2)
        with c1:
            device_class = st.selectbox("Device Class", DEVICE_CLASSES, index=1)
        with c2:
            medical_specialty = st.selectbox("Medical Specialty", MEDICAL_SPECIALTIES)
        st.caption(f"ℹ️ {CLASS_DESCRIPTIONS.get(device_class, '')}")

    with col_flags:
        st.markdown("**Device Properties**")
        implant_flag      = st.toggle("Implantable Device",
                                      help="Device is intended to be surgically implanted in the body.")
        life_sustain_flag = st.toggle("Life-Sustaining / Life-Supporting",
                                      help="Device sustains or supports human life.")
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡ Predict Pathway", type="primary", use_container_width=True)
        st.caption("Estimates are based on historical FDA data patterns.")

    # ── Result ──
    if predict_btn:
        if not device_name.strip():
            st.error("⚠️ Please enter a device name before predicting.")
            return

        import numpy as np
        from sklearn.preprocessing import LabelEncoder

        model       = model_payload["model"]
        feature_cols = model_payload["feature_cols"]
        le_target: LabelEncoder = model_payload["label_encoder"]

        try:
            clean_path = ARTIFACTS_DIR / "clean_data.csv"
            if clean_path.exists():
                df_train = pd.read_csv(clean_path)
                le_class = LabelEncoder().fit(df_train["device_class"].astype(str))
                le_spec  = LabelEncoder().fit(df_train["medical_specialty_description"].astype(str))
            else:
                le_class = LabelEncoder().fit(DEVICE_CLASSES + ["Unknown"])
                le_spec  = LabelEncoder().fit(MEDICAL_SPECIALTIES)

            def safe_transform(le, value):
                return int(le.transform([value])[0]) if value in le.classes_ else 0

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

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_pred)[0]
                class_probs = {
                    le_target.inverse_transform([i])[0]: float(p)
                    for i, p in enumerate(proba)
                }
            else:
                class_probs = {predicted_pathway: 1.0}

            info           = PATHWAY_INFO.get(predicted_pathway, {})
            top_confidence = class_probs.get(predicted_pathway, 1.0)

            st.markdown('<div class="section-header">🎯 Prediction Result</div>', unsafe_allow_html=True)

            res_left, res_right = st.columns([3, 2], gap="large")

            with res_left:
                st.markdown(f"""
                <div class="result-card"
                     style="background:{info.get('bg','#f5f5f5')};
                            border-color:{info.get('color','#888')};
                            color:#1a1f36;">
                    <div class="rc-eyebrow">Recommended Pathway</div>
                    <div class="rc-name">{info.get('icon','')} {predicted_pathway}</div>
                    <div class="rc-meta">
                        {info.get('label','')} &nbsp;·&nbsp;
                        {info.get('time','')} &nbsp;·&nbsp;
                        Risk: {info.get('risk','')}
                    </div>
                    <div class="rc-desc">{info.get('description','')}</div>
                </div>
                """, unsafe_allow_html=True)

            with res_right:
                st.markdown(f"""
                <div class="metric-pill" style="margin-bottom:16px;">
                    <div class="mp-value" style="color:{info.get('color','#3f51b5')}">
                        {top_confidence:.0%}
                    </div>
                    <div class="mp-label">Model Confidence</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("**Confidence by Pathway**")
                _render_confidence_bars(class_probs)

            with st.expander("📊 Full probability breakdown"):
                prob_df = pd.DataFrame(
                    {"Pathway": list(class_probs.keys()), "Confidence": list(class_probs.values())}
                ).sort_values("Confidence", ascending=False)
                st.dataframe(
                    prob_df.style.format({"Confidence": "{:.2%}"}).bar(
                        subset=["Confidence"], color="#3f51b5"
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            logger.exception("Prediction error")

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <span style="font-size:20px;flex-shrink:0;">⚠️</span>
        <span><b>Disclaimer:</b> This tool is for planning and educational purposes only.
        Predictions are based on historical openFDA data patterns and do not constitute
        regulatory advice. Consult qualified regulatory counsel before making any
        regulatory submission decisions.</span>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page 2 — EDA Dashboard
# ---------------------------------------------------------------------------

def page_eda() -> None:
    st.markdown("""
    <div class="page-hero">
        <h1>📊 EDA Dashboard</h1>
        <p>Exploratory data analysis of the openFDA training dataset
        used to build the classification model.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 EDA Report", "💡 Key Insights", "📋 Dataset Contract"])

    with tab1:
        if artifact_exists("eda_report.html"):
            html_content = (ARTIFACTS_DIR / "eda_report.html").read_text(encoding="utf-8")
            components.html(html_content, height=900, scrolling=True)
        else:
            st.warning("eda_report.html not found. Run the pipeline first.")

    with tab2:
        st.markdown(read_text("insights.md"))

    with tab3:
        contract = read_json("dataset_contract.json")
        if contract:
            m1, m2, m3 = st.columns(3)
            with m1:
                row_count = contract.get("row_count", "N/A")
                formatted = f"{row_count:,}" if isinstance(row_count, int) else row_count
                st.markdown(f"""
                <div class="metric-pill">
                    <div class="mp-value" style="color:#3f51b5">{formatted}</div>
                    <div class="mp-label">Total Records</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-pill">
                    <div class="mp-value" style="color:#3f51b5">{len(contract.get('feature_columns', []))}</div>
                    <div class="mp-label">Feature Columns</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-pill">
                    <div class="mp-value" style="color:#3f51b5">{len(contract.get('target_classes', []))}</div>
                    <div class="mp-label">Target Classes</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Pathway Distribution")
            dist = contract.get("pathway_distribution", {})
            if dist:
                total = sum(dist.values())
                dist_df = pd.DataFrame([
                    {"Pathway": k, "Count": v, "Share": f"{100*v/total:.1f}%"}
                    for k, v in dist.items()
                ])
                st.dataframe(dist_df, use_container_width=True, hide_index=True)

            st.markdown("#### Column Details")
            dtypes = contract.get("dtypes", {})
            nulls  = contract.get("null_counts", {})
            if dtypes:
                detail_df = pd.DataFrame([
                    {"Column": col, "Type": dtypes.get(col, ""), "Nulls": nulls.get(col, 0)}
                    for col in contract.get("columns", [])
                ])
                st.dataframe(detail_df, use_container_width=True, hide_index=True)
        else:
            st.warning("dataset_contract.json not found. Run the pipeline first.")


# ---------------------------------------------------------------------------
# Page 3 — Model Performance
# ---------------------------------------------------------------------------

def page_model_performance() -> None:
    st.markdown("""
    <div class="page-hero">
        <h1>📉 Model Performance</h1>
        <p>Full evaluation metrics, confusion matrix, and model documentation
        for the trained FDA pathway classifier.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Evaluation Report", "🃏 Model Card", "🔲 Confusion Matrix"])

    with tab1:
        st.markdown(read_text("evaluation_report.md"))

    with tab2:
        st.markdown(read_text("model_card.md"))

    with tab3:
        cm_path = ARTIFACTS_DIR / "confusion_matrix.png"
        if cm_path.exists():
            _, center_col, _ = st.columns([1, 2, 1])
            with center_col:
                st.image(str(cm_path), caption="Confusion Matrix — Best Model", use_container_width=True)
        else:
            st.warning("confusion_matrix.png not found. Run the pipeline first.")

        lm = read_json("label_mapping.json")
        if lm:
            with st.expander("Label encoding mapping"):
                st.json(lm)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Page 4 — API Explorer
# ---------------------------------------------------------------------------

# All searchable fields per openFDA device endpoint.
# Format: (field_name, type, description, example_value)
_FDA_FIELDS: dict[str, list[tuple[str, str, str, str]]] = {
    "device/510k": [
        ("device_name",                   "string",  "Commercial/trade name of the device",                      "Cardiac Monitor"),
        ("k_number",                      "string",  "510(k) number (prefix DEN = De Novo)",                     "K203487"),
        ("product_code",                  "string",  "Three-letter FDA product code",                             "DZE"),
        ("applicant",                     "string",  "Company submitting the 510(k)",                             "Medtronic"),
        ("device_class",                  "string",  "Device risk class (1, 2, or 3)",                            "2"),
        ("medical_specialty_description", "string",  "Medical specialty area",                                    "Cardiovascular"),
        ("decision_code",                 "string",  "Decision outcome (SESE=cleared, DENO=De Novo, etc.)",       "SESE"),
        ("decision_date",                 "date",    "Date of FDA decision (YYYY-MM-DD or range with [TO])",      "2023-01-01"),
        ("date_received",                 "date",    "Date application was received",                             "2022-06-15"),
        ("implant_flag",                  "string",  "Y if device is implantable, N otherwise",                   "Y"),
        ("life_sustain_support_flag",     "string",  "Y if device is life-sustaining/supporting",                 "N"),
        ("clearance_type",                "string",  "Type of clearance (Traditional, Abbreviated, etc.)",        "Traditional"),
        ("expedited_review_flag",         "string",  "Y if expedited review was granted",                         "N"),
        ("third_party_flag",              "string",  "Y if reviewed by accredited third party",                   "N"),
        ("statement_or_summary",          "string",  "Whether a statement or summary was submitted",               "Summary"),
        ("city",                          "string",  "City of the applicant",                                     "Minneapolis"),
        ("state",                         "string",  "State of the applicant (2-letter code)",                    "MN"),
        ("country_code",                  "string",  "Country code of the applicant",                             "US"),
        ("openfda.regulation_number",     "string",  "CFR regulation number (e.g. 870.3710)",                     "870.3710"),
        ("openfda.device_name",           "string",  "Device name from openFDA harmonized data",                  "Electrocardiograph"),
        ("openfda.medical_specialty_description", "string", "Specialty from openFDA namespace",                   "Cardiovascular"),
        ("openfda.fei_number",            "string",  "FDA Establishment Identifier",                              "1234567"),
    ],
    "device/pma": [
        ("trade_name",                    "string",  "Trade name of the device",                                  "HeartMate 3"),
        ("generic_name",                  "string",  "Generic/common name of the device",                         "Ventricular Assist Device"),
        ("pma_number",                    "string",  "PMA application number",                                    "P180030"),
        ("product_code",                  "string",  "Three-letter FDA product code",                             "DQN"),
        ("applicant",                     "string",  "Company submitting the PMA",                                "Abbott"),
        ("device_class",                  "string",  "Device risk class (almost always 3 for PMA)",               "3"),
        ("advisory_committee",            "string",  "Advisory committee short code",                             "CV"),
        ("advisory_committee_description","string",  "Full name of advisory committee",                           "Cardiovascular"),
        ("decision_code",                 "string",  "Decision code (APPR=approved, DENY=denied, etc.)",          "APPR"),
        ("decision_date",                 "date",    "Date of FDA decision",                                      "2022-11-10"),
        ("date_received",                 "date",    "Date application was received",                             "2021-03-01"),
        ("expedited_review_flag",         "string",  "Y if expedited review was granted",                         "Y"),
        ("supplement_type",               "string",  "Type of PMA supplement (if applicable)",                    "PanelTrack"),
        ("supplement_reason",             "string",  "Reason for the supplement",                                  "New Indication"),
        ("city",                          "string",  "City of the applicant",                                     "Sylmar"),
        ("state",                         "string",  "State of the applicant",                                    "CA"),
        ("country_code",                  "string",  "Country code of the applicant",                             "US"),
        ("openfda.regulation_number",     "string",  "CFR regulation number",                                     "870.3545"),
        ("openfda.fei_number",            "string",  "FDA Establishment Identifier",                              "3005847770"),
    ],
    "device/classification": [
        ("device_name",                   "string",  "Device type name",                                          "Pacemaker"),
        ("product_code",                  "string",  "Three-letter FDA product code",                             "DTB"),
        ("device_class",                  "string",  "Device risk class (1, 2, or 3)",                            "3"),
        ("medical_specialty_description", "string",  "Medical specialty area",                                    "Cardiovascular"),
        ("regulation_number",             "string",  "CFR regulation (Title 21)",                                 "870.3610"),
        ("submission_type_id",            "string",  "Submission type required (2=510k, 3=PMA, 6=De Novo, etc.)", "2"),
        ("definition",                    "string",  "Official FDA definition of the device type",                 "electrode"),
        ("implant_flag",                  "string",  "Y if device type is implantable",                           "Y"),
        ("life_sustain_support_flag",     "string",  "Y if life-sustaining",                                      "N"),
        ("gmp_exempt_flag",               "string",  "Y if device is exempt from GMP requirements",               "N"),
        ("openfda.fei_number",            "string",  "FDA Establishment Identifier",                              "1234567"),
    ],
}

_DECISION_CODES = {
    "510k": {
        "SESE": "Substantially Equivalent — Cleared",
        "DENO": "De Novo Classification Granted",
        "NSUB": "Not Substantially Equivalent",
        "WTDR": "Withdrawn",
        "HOLD": "On Hold",
    },
    "pma": {
        "APPR": "Approved",
        "DENY": "Denied",
        "WTDR": "Withdrawn",
        "APRL": "Approved with conditions",
    },
}

_SEARCH_SYNTAX = """
### openFDA Search Syntax

| Syntax | Meaning | Example |
|--------|---------|---------|
| `field:value` | Exact match | `device_class:2` |
| `field:"multi word"` | Phrase match | `device_name:"cardiac monitor"` |
| `field:[A+TO+Z]` | Range | `decision_date:[2022-01-01+TO+2023-12-31]` |
| `field:value1+AND+field2:value2` | Both conditions | `device_class:3+AND+implant_flag:Y` |
| `field:value1+field:value2` | Either condition (OR) | `medical_specialty_description:Cardiovascular+medical_specialty_description:Orthopedic` |
| `_exists_:field` | Field exists | `_exists_:k_number` |

**Base URL:** `https://api.fda.gov/{endpoint}.json?search={query}&limit={n}`

**Example:** `https://api.fda.gov/device/510k.json?search=device_class:2+AND+implant_flag:Y&limit=10`
"""


def page_api_explorer() -> None:
    st.markdown("""
    <div class="page-hero">
        <h1>🔍 API Search Reference</h1>
        <p>All searchable openFDA fields for each device endpoint — with a live query builder
        to construct and preview API calls directly against the openFDA API.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_fields, tab_builder, tab_decisions = st.tabs([
        "📋 Field Reference", "⚙️ Query Builder", "🏷️ Decision Codes"
    ])

    # ── Tab 1 — Field Reference ──
    with tab_fields:
        st.markdown(_SEARCH_SYNTAX)
        st.markdown("---")

        endpoint_label = st.selectbox(
            "Endpoint",
            list(_FDA_FIELDS.keys()),
            format_func=lambda x: f"/{x}",
            key="ref_endpoint",
        )
        fields = _FDA_FIELDS[endpoint_label]

        search_filter = st.text_input(
            "Filter fields", placeholder="e.g. date, implant, device…", key="field_filter"
        )

        rows = fields
        if search_filter.strip():
            q = search_filter.strip().lower()
            rows = [r for r in fields if q in r[0].lower() or q in r[2].lower()]

        df_fields = pd.DataFrame(rows, columns=["Field Name", "Type", "Description", "Example Value"])

        def make_query(row):
            val = row["Example Value"]
            field = row["Field Name"]
            if " " in val:
                return f'{field}:"{val}"'
            return f"{field}:{val}"

        df_fields["Example Search Query"] = df_fields.apply(make_query, axis=1)

        st.markdown(f"**{len(rows)} fields** for `/{endpoint_label}`")
        st.dataframe(
            df_fields,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Field Name":          st.column_config.TextColumn(width="medium"),
                "Type":                st.column_config.TextColumn(width="small"),
                "Description":         st.column_config.TextColumn(width="large"),
                "Example Value":       st.column_config.TextColumn(width="small"),
                "Example Search Query":st.column_config.TextColumn(width="medium"),
            },
        )

    # ── Tab 2 — Query Builder ──
    with tab_builder:
        st.markdown("Build a query visually, then preview the live API response.")

        b_col1, b_col2 = st.columns([2, 3])

        with b_col1:
            endpoint = st.selectbox(
                "Endpoint",
                list(_FDA_FIELDS.keys()),
                format_func=lambda x: f"/{x}",
                key="builder_endpoint",
            )
            fields_for_ep = _FDA_FIELDS[endpoint]
            field_names = [f[0] for f in fields_for_ep]
            field_descs = {f[0]: f[2] for f in fields_for_ep}
            field_examples = {f[0]: f[3] for f in fields_for_ep}

            st.markdown("**Conditions** (AND logic)")

            conditions: list[str] = []
            for i in range(3):
                c1, c2 = st.columns([2, 2])
                with c1:
                    field = st.selectbox(
                        f"Field {i+1}", ["(none)"] + field_names,
                        key=f"qb_field_{i}",
                    )
                with c2:
                    placeholder = field_examples.get(field, "value") if field != "(none)" else ""
                    value = st.text_input(
                        f"Value {i+1}", placeholder=placeholder,
                        key=f"qb_value_{i}",
                        help=field_descs.get(field, "") if field != "(none)" else "",
                    )
                if field != "(none)" and value.strip():
                    v = f'"{value.strip()}"' if " " in value.strip() else value.strip()
                    conditions.append(f"{field}:{v}")

            limit = st.slider("Result limit", 1, 100, 10, key="qb_limit")

        with b_col2:
            if conditions:
                search_str = "+AND+".join(conditions)
                base = f"https://api.fda.gov/{endpoint}.json"
                full_url = f"{base}?search={search_str}&limit={limit}"
            else:
                full_url = f"https://api.fda.gov/{endpoint}.json?limit={limit}"

            st.markdown("**Generated URL**")
            st.code(full_url, language="text")

            run_btn = st.button("▶ Run Query", type="primary", key="qb_run")

            if run_btn:
                import requests as _req
                with st.spinner("Fetching from openFDA…"):
                    try:
                        resp = _req.get(full_url, timeout=15)
                        if resp.status_code == 200:
                            data = resp.json()
                            results = data.get("results", [])
                            meta = data.get("meta", {})
                            total = meta.get("results", {}).get("total", "?")
                            st.success(f"✅ {len(results)} records returned (total matching: {total:,})" if isinstance(total, int) else f"✅ {len(results)} records returned")

                            if results:
                                # Flatten top-level keys only (skip nested dicts)
                                flat_rows = []
                                for rec in results:
                                    flat_rows.append({
                                        k: v for k, v in rec.items()
                                        if not isinstance(v, (dict, list))
                                    })
                                st.dataframe(
                                    pd.DataFrame(flat_rows),
                                    use_container_width=True,
                                    hide_index=True,
                                )
                                with st.expander("Raw JSON (first result)"):
                                    st.json(results[0])
                        elif resp.status_code == 404:
                            st.warning("No records matched this query.")
                        else:
                            st.error(f"API error {resp.status_code}: {resp.text[:300]}")
                    except Exception as exc:
                        st.error(f"Request failed: {exc}")
            else:
                st.info("Fill in at least one field + value above, then click **Run Query**.")

    # ── Tab 3 — Decision Codes ──
    with tab_decisions:
        st.markdown("Reference table of `decision_code` values returned by each endpoint.")

        for ep, codes in _DECISION_CODES.items():
            st.markdown(f"#### `device/{ep}` — `decision_code` values")
            code_df = pd.DataFrame(
                [{"Code": k, "Meaning": v} for k, v in codes.items()]
            )
            st.dataframe(code_df, use_container_width=True, hide_index=True)
            st.markdown("")

        st.markdown("""
        #### `submission_type_id` — `device/classification`
        | Value | Meaning |
        |-------|---------|
        | `1` | PMA |
        | `2` | 510(k) |
        | `3` | PDP |
        | `4` | PMA Supplement |
        | `5` | Pre-submission |
        | `6` | De Novo |
        | `7` | Transitional |
        | `8` | STED |
        """)


NAV_ITEMS = [
    ("🏥", "Pathway Predictor",  page_predictor),
    ("📊", "EDA Dashboard",      page_eda),
    ("📉", "Model Performance",  page_model_performance),
    ("🔍", "API Explorer",       page_api_explorer),
]


def main() -> None:
    inject_css()

    if "page" not in st.session_state:
        st.session_state.page = 0

    with st.sidebar:
        st.markdown("""
        <div style="padding:4px 0 22px;">
            <div style="font-size:36px;line-height:1;">🏥</div>
            <div style="font-size:16px;font-weight:700;color:#fff;margin:8px 0 2px;">
                FDA Pathway Predictor
            </div>
            <div style="font-size:12px;color:#7986cb;">
                ML-powered regulatory estimation
            </div>
        </div>
        <div style="height:1px;background:rgba(255,255,255,0.08);margin-bottom:16px;"></div>
        """, unsafe_allow_html=True)

        st.markdown(
            "<div style='font-size:11px;font-weight:700;text-transform:uppercase;"
            "letter-spacing:0.08em;color:#5c6bc0;margin-bottom:6px;'>Navigation</div>",
            unsafe_allow_html=True,
        )
        for i, (icon, label, _) in enumerate(NAV_ITEMS):
            if st.button(f"{icon}  {label}", key=f"nav_{i}", use_container_width=True):
                st.session_state.page = i
                st.rerun()

        st.markdown(
            "<div style='height:1px;background:rgba(255,255,255,0.08);margin:20px 0 16px;'></div>",
            unsafe_allow_html=True,
        )

        model_status = "🟢 Model ready" if load_model() else "🔴 Model not found"
        st.markdown(
            f"<div style='font-size:13px;'>{model_status}</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='position:absolute;bottom:24px;font-size:11px;color:#3d4f8c;'>"
            "Built with CrewAI · scikit-learn · Streamlit</div>",
            unsafe_allow_html=True,
        )

    NAV_ITEMS[st.session_state.page][2]()


if __name__ == "__main__":
    main()
