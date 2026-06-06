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

/* ── Pathway tile "Learn more" buttons ───────────────────────────── */
[data-testid="stHorizontalBlock"] .stButton > button[kind="secondary"] {
    background: transparent;
    border: 1px solid #e0e3f0;
    border-radius: 0 0 10px 10px;
    color: #5c6bc0;
    font-size: 12.5px;
    font-weight: 600;
    padding: 6px 12px;
    margin-top: -2px;
    transition: all 0.15s ease;
}
[data-testid="stHorizontalBlock"] .stButton > button[kind="secondary"]:hover {
    background: #eef0fb;
    border-color: #5c6bc0;
    color: #3f51b5;
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


# ---------------------------------------------------------------------------
# FDA Product Code Classification Database (foiclass.zip)
# Updated every Sunday by FDA: https://www.fda.gov/medical-devices/classify-your-medical-device/download-product-code-classification-files
# ---------------------------------------------------------------------------

_FOICLASS_URL = "https://www.accessdata.fda.gov/premarket/ftparea/foiclass.zip"

# Pipe-delimited columns in known order (no header in file)
_FOICLASS_COLS = [
    "REVIEW_PANEL", "MEDICALSPECIALTY", "PRODUCTCODE", "DEVICENAME",
    "DEVICECLASS", "UNCLASSIFIED_REASON", "GMP_EXEMPT_FLAG",
    "THIRD_PARTY_FLAG", "REVIEW_CODE", "REGULATION_NUMBER",
    "SUBMISSION_TYPE_ID", "DEFINITION", "PHYSICALSTATE",
    "TECHNICALMETHOD", "TARGET_AREA",
]

_SPECIALTY_CODE_MAP = {
    "AN": "Anesthesiology",       "CV": "Cardiovascular",
    "CH": "Clinical Chemistry",   "DE": "Dental",
    "EN": "Ear, Nose, Throat",    "GU": "Gastroenterology, Urology",
    "HO": "General Hospital",     "HE": "Hematology",
    "IM": "Immunology",           "MG": "Medical Genetics",
    "MI": "Microbiology",         "NE": "Neurology",
    "OB": "Obstetrics/Gynecology","OP": "Ophthalmic",
    "OR": "Orthopedic",           "PA": "Pathology",
    "PM": "Physical Medicine",    "RA": "Radiology",
    "SU": "General, Plastic Surgery", "TX": "Clinical Toxicology",
}

_CLASS_CODE_MAP = {
    "1": "Class I", "2": "Class II", "3": "Class III",
    "f": "Class I", "F": "Class I", "U": "Not Sure",
}

_SUBMISSION_CODE_MAP = {
    "1": "PMA", "2": "510(k)", "3": "PDP",
    "4": "PMA Supplement", "5": "Pre-submission", "6": "De Novo",
}


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def _load_fda_classification_db() -> pd.DataFrame | None:
    """Download and parse foiclass.zip — cached 7 days (FDA updates weekly)."""
    import io
    import zipfile
    import requests as _req
    try:
        resp = _req.get(_FOICLASS_URL, timeout=30)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            raw = z.open(z.namelist()[0]).read().decode("latin-1")
        lines = [l for l in raw.splitlines() if l.strip()]
        rows  = [l.split("|") for l in lines]
        n     = max(len(r) for r in rows)
        cols  = _FOICLASS_COLS[:n] + [f"extra_{i}" for i in range(n - len(_FOICLASS_COLS))]
        rows  = [r + [""] * (n - len(r)) for r in rows]   # pad short rows
        df = pd.DataFrame(rows, columns=cols)
        df = df.apply(lambda s: s.str.strip() if s.dtype == object else s)
        df["Class"]      = df["DEVICECLASS"].map(_CLASS_CODE_MAP).fillna("Not Sure")
        df["Specialty"]  = df["MEDICALSPECIALTY"].map(_SPECIALTY_CODE_MAP).fillna("Other")
        df["Submission"] = df["SUBMISSION_TYPE_ID"].map(_SUBMISSION_CODE_MAP).fillna("Unknown")
        return df
    except Exception:
        return None


MEDICAL_SPECIALTIES = [
    "Anesthesiology",
    "Cardiovascular",
    "Clinical Chemistry",
    "Clinical Toxicology",
    "Dental",
    "Ear, Nose, Throat",
    "Gastroenterology, Urology",
    "General Hospital",
    "General, Plastic Surgery",
    "Hematology",
    "Immunology",
    "Medical Genetics",
    "Microbiology",
    "Neurology",
    "Obstetrics/Gynecology",
    "Ophthalmic",
    "Orthopedic",
    "Pathology",
    "Physical Medicine",
    "Radiology",
    "Other",
]

DEVICE_CLASSES = ["Not Sure", "Class I", "Class II", "Class III"]

CLASS_DESCRIPTIONS = {
    "Not Sure":  "Select if you don't know your device's class yet. We'll show predictions for all three classes.",
    "Class I":   "Low risk — general controls only (e.g. bandages, tongue depressors).",
    "Class II":  "Moderate risk — special controls + 510(k) (e.g. infusion pumps, X-ray systems).",
    "Class III": "High risk — PMA required (e.g. pacemakers, implantable defibrillators).",
}

CLASS_LOOKUP_HINT = """
**How to find your device class:**
1. Search the [FDA Product Classification Database](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfPCD/classification.cfm) by device name or product code
2. Search cleared 510(k)s for similar devices at [510(k) Database](https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm)
3. Look up your CFR regulation number (21 CFR Part 862–892)
4. Request a **Pre-Submission (Q-Sub)** meeting with FDA CDRH for an official determination
"""

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


# ---------------------------------------------------------------------------
# Downstream submission checklists — shown after a prediction
# ---------------------------------------------------------------------------

PATHWAY_CHECKLISTS: dict[str, list[dict]] = {
    "510k": [
        {
            "phase": "1 · Pre-Submission",
            "icon": "📋",
            "items": [
                "Identify and document a legally marketed predicate device (same intended use)",
                "Confirm device is Class II (or reclassified Class III with 510k history)",
                "Request an FDA Pre-Sub meeting (Q-Sub) to align on testing strategy",
                "Draft the Substantial Equivalence (SE) comparison table",
                "Identify applicable FDA guidance documents and recognized standards",
                "Review CDRH's iRASC database for recent 510(k) decisions in your product code",
            ],
        },
        {
            "phase": "2 · Technical Documentation",
            "icon": "🔬",
            "items": [
                "Device description — intended use, indications for use, design specs",
                "Predicate comparison — side-by-side table of technological characteristics",
                "Substantial equivalence argument — why differences don't raise new Q&A issues",
                "Performance testing — bench, biocompatibility (ISO 10993), EMC, software (if applicable)",
                "Sterilization validation (if device is sterile)",
                "Shelf-life / packaging testing",
                "Software documentation per FDA SW guidance (if SiMD or SaMD)",
                "Cybersecurity documentation (if network-connected)",
            ],
        },
        {
            "phase": "3 · Labeling",
            "icon": "🏷️",
            "items": [
                "Draft labeling in FDA final format (21 CFR Part 801)",
                "Include device name, intended use, indications, contraindications, warnings",
                "Directions for use / Instructions for Use (IFU)",
                "Ensure labeling matches the predicate's intended use scope",
                "Prepare 510(k) Summary OR 510(k) Statement (one required)",
            ],
        },
        {
            "phase": "4 · Submission Package",
            "icon": "📦",
            "items": [
                "Cover letter with CDRH contact info and submission type",
                "Table of contents (eCopy format per FDA guidance)",
                "Device description section",
                "Predicate comparison & SE argument section",
                "Performance testing reports",
                "Labeling (draft)",
                "510(k) Summary or Statement",
                "Pay user fee (check current MDUFA fee schedule)",
                "Submit via FDA eSubmitter or CDRH Customer Collaboration Portal",
            ],
        },
        {
            "phase": "5 · FDA Review (90–180 days)",
            "icon": "⏳",
            "items": [
                "Respond promptly to Additional Information (AI) requests — typically 90-day clock",
                "Track submission status on FDA's 510(k) database",
                "Prepare for Interactive Review requests (voluntary pilot)",
                "Engage FDA reviewer via teleconference if questions arise",
            ],
        },
        {
            "phase": "6 · Post-Clearance",
            "icon": "✅",
            "items": [
                "Register establishment and list device in FDA FURLS (within 30 days)",
                "Implement Quality System Regulation (QSR / 21 CFR Part 820) — MDR reporting",
                "File Medical Device Reports (MDRs) for any reportable adverse events",
                "Submit a new 510(k) before making significant changes to design, labeling, or manufacturing",
                "Maintain complaint files and CAPA records",
            ],
        },
    ],
    "PMA": [
        {
            "phase": "1 · Early Planning",
            "icon": "🗺️",
            "items": [
                "Confirm device requires PMA (Class III, life-sustaining, or no predicate)",
                "Request Breakthrough Device Designation if eligible (for faster interaction)",
                "Schedule Pre-Submission (Q-Sub) meeting to discuss clinical study design",
                "Review existing Advisory Committee meeting transcripts for your device type",
                "Engage a Regulatory Affairs consultant with PMA experience",
                "Develop an IDE (Investigational Device Exemption) application if clinical trials needed",
            ],
        },
        {
            "phase": "2 · Investigational Device Exemption (IDE)",
            "icon": "🧪",
            "items": [
                "Prepare IDE application — device description, investigational plan, risk analysis",
                "Obtain IRB approval at each clinical site",
                "Submit IDE to FDA and await approval before initiating pivotal studies",
                "Conduct Feasibility study (if needed) under IDE",
                "Design pivotal clinical trial with primary and secondary endpoints",
                "Power calculations and statistical analysis plan (SAP)",
            ],
        },
        {
            "phase": "3 · Clinical & Non-Clinical Studies",
            "icon": "🏥",
            "items": [
                "Biocompatibility testing per ISO 10993 series",
                "Sterilization validation (ISO 11135 / 11137 as applicable)",
                "Electrical safety and EMC testing (IEC 60601 series)",
                "Software verification & validation per IEC 62304 / FDA SW guidance",
                "Animal studies (if required) under IACUC approval",
                "Pivotal clinical investigation — GCP-compliant, FDA-registered sites",
                "Compile Clinical Study Report (CSR)",
            ],
        },
        {
            "phase": "4 · PMA Module Assembly",
            "icon": "📦",
            "items": [
                "Administrative section — applicant info, device description, table of contents",
                "Summary of Safety and Effectiveness Data (SSED) — public document",
                "Non-clinical studies section",
                "Clinical investigations section with CSR",
                "Manufacturing information — facility info, QS documentation, process validation",
                "Proposed labeling in final format",
                "References and bibliography",
                "Pay PMA user fee (significantly higher than 510(k))",
            ],
        },
        {
            "phase": "5 · FDA Review (180-day statutory goal)",
            "icon": "⏳",
            "items": [
                "Prepare for Major Deficiency Letter — respond within FDA-specified timeframe",
                "Prepare for Advisory Panel meeting (public — prepare slides and Q&A)",
                "Respond to all panel questions and FDA information requests",
                "Negotiate any approval conditions (labeling changes, post-approval studies)",
                "Track status on FDA's PMA database",
            ],
        },
        {
            "phase": "6 · Post-Approval",
            "icon": "✅",
            "items": [
                "Complete any Post-Approval Studies (PAS) committed to FDA",
                "Register establishment and list device within 30 days of approval",
                "File Annual PMA Reports (30-day window each year)",
                "Submit PMA Supplements for any changes to design, manufacturing, or labeling",
                "Implement MDR (Medical Device Reporting) program — 5-day, 15-day, 30-day reports",
                "Maintain full QSR / 21 CFR 820 compliance — FDA inspections possible",
            ],
        },
    ],
    "De Novo": [
        {
            "phase": "1 · Eligibility Determination",
            "icon": "🔍",
            "items": [
                "Confirm device is novel — no legally marketed predicate device exists",
                "Confirm device is low-to-moderate risk (Class I or II level)",
                "Determine entry path: Direct De Novo OR Post-NSE De Novo (after 510(k) refusal)",
                "Request a Pre-Submission (Q-Sub) meeting with CDRH to confirm De Novo eligibility",
                "Identify or propose a new device type name and three-letter product code",
                "Review existing De Novo orders in CDRH database for similar device types",
            ],
        },
        {
            "phase": "2 · Risk Analysis & Special Controls Proposal",
            "icon": "⚖️",
            "items": [
                "Conduct comprehensive risk analysis (ISO 14971) — identify all hazards",
                "Propose Special Controls that mitigate each identified risk",
                "Draft General Controls and Special Controls framework for the new device type",
                "Prepare risk/benefit analysis demonstrating low-to-moderate residual risk",
                "Propose classification regulation language (similar to 21 CFR Part 870–892 format)",
            ],
        },
        {
            "phase": "3 · Technical Documentation",
            "icon": "🔬",
            "items": [
                "Device description — intended use, indications, design specifications, principles of operation",
                "Performance testing supporting the special controls proposal",
                "Biocompatibility (ISO 10993) if in contact with patient",
                "Software documentation if SiMD/SaMD (FDA SW guidance + IEC 62304)",
                "Cybersecurity documentation if network-connected",
                "Sterilization validation (if sterile device)",
                "Labeling compliant with proposed special controls",
            ],
        },
        {
            "phase": "4 · De Novo Request Package",
            "icon": "📦",
            "items": [
                "Cover letter with device description and proposed classification",
                "Table of contents",
                "De Novo summary — risk/benefit, special controls, proposed classification",
                "Performance testing reports",
                "Full technical documentation",
                "Draft proposed labeling",
                "Draft classification regulation language",
                "Pay De Novo user fee (check current MDUFA schedule)",
                "Submit via FDA eSubmitter or Customer Collaboration Portal",
            ],
        },
        {
            "phase": "5 · FDA Review (150-day statutory goal)",
            "icon": "⏳",
            "items": [
                "Respond to FDA's Additional Information (AI) requests promptly",
                "Negotiate Special Controls language with FDA reviewer",
                "Negotiate proposed classification regulation text",
                "Track status via CDRH Customer Collaboration Portal",
            ],
        },
        {
            "phase": "6 · Post-Grant",
            "icon": "✅",
            "items": [
                "De Novo order is published — your device type becomes a predicate for future 510(k)s",
                "Register establishment and list device in FDA FURLS within 30 days",
                "Implement QSR / 21 CFR Part 820 quality system",
                "File MDRs for any reportable adverse events",
                "Submit a new 510(k) for future device modifications (device now has a predicate)",
                "Monitor any mandatory Post-Market Surveillance studies committed to FDA",
            ],
        },
    ],
}


def _render_checklist(pathway: str) -> None:
    """Render the submission checklist for the predicted pathway."""
    checklist = PATHWAY_CHECKLISTS.get(pathway)
    if not checklist:
        return

    info = PATHWAY_INFO.get(pathway, {})
    color = info.get("color", "#3f51b5")

    st.markdown(
        f'<div class="section-header">📝 Submission Checklist — {info.get("icon","")} {pathway}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Use this as a high-level planning guide. Each phase should be confirmed "
        "against the latest FDA guidance documents and your specific device's requirements."
    )

    for phase_data in checklist:
        phase_icon = phase_data["icon"]
        phase_name = phase_data["phase"]
        items = phase_data["items"]

        with st.expander(f"{phase_icon} {phase_name}", expanded=False):
            for item in items:
                st.markdown(
                    f'<div style="display:flex;gap:10px;align-items:flex-start;'
                    f'padding:7px 0;border-bottom:1px solid #f0f2f9;">'
                    f'<span style="color:{color};font-size:16px;flex-shrink:0;margin-top:1px;">◻</span>'
                    f'<span style="font-size:14px;line-height:1.5;color:#37474f;">{item}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown(
        f'<div style="background:#f8f9ff;border:1px solid #e8eaf6;border-left:4px solid {color};'
        f'border-radius:8px;padding:12px 16px;margin-top:12px;font-size:13px;color:#546e7a;">'
        f'<b>Next step:</b> Review the official FDA guidance for {pathway} submissions and '
        f'request a Pre-Submission (Q-Sub) meeting with CDRH to align your strategy before investing in testing.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Pathway detail dialogs
# ---------------------------------------------------------------------------

@st.dialog("🔵 510(k) Clearance — Full Details", width="large")
def _dialog_510k() -> None:
    st.markdown("""
    **510(k) Clearance** is the most common FDA premarket submission pathway.
    You must demonstrate that your device is **substantially equivalent** to a legally
    marketed predicate device — same intended use and same/different technology
    that does not raise new safety or effectiveness questions.

    ---
    #### When to use 510(k)
    - Device is Class II (moderate risk)
    - A legally marketed predicate device exists
    - Device does not have a new intended use
    - Any new technology does not raise different safety/effectiveness questions

    ---
    #### Required elements
    | Element | Details |
    |---------|---------|
    | Device description | Intended use, indications, technical specs |
    | Predicate comparison | Side-by-side comparison table |
    | Performance testing | Bench, animal, or clinical data as appropriate |
    | Labeling | Draft labeling in final format |
    | 510(k) summary or statement | Public-facing summary of SE decision |

    ---
    #### Typical timeline
    | Phase | Duration |
    |-------|----------|
    | Preparation | 3–12 months |
    | FDA review (standard) | 90 days |
    | FDA review (complex) | Up to 180 days |
    | **Total** | **3–18 months** |

    ---
    #### Key decision codes
    | Code | Meaning |
    |------|---------|
    | `SESE` | Substantially Equivalent — Cleared |
    | `NSUB` | Not Substantially Equivalent — Not Cleared |
    | `DENG` | De Novo Granted |
    | `WTDR` | Withdrawn |
    """)
    st.link_button(
        "Open Official FDA 510(k) Guidance",
        "https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-correct-submission/premarket-notification-510k",
        use_container_width=True,
    )


@st.dialog("🔴 Premarket Approval (PMA) — Full Details", width="large")
def _dialog_pma() -> None:
    st.markdown("""
    **Premarket Approval (PMA)** is the most rigorous FDA device review pathway.
    Required for Class III devices — those that **support or sustain human life**,
    are implanted, or present a potential unreasonable risk of illness or injury.

    ---
    #### When to use PMA
    - Device is Class III (high risk)
    - Device is life-sustaining or life-supporting
    - Device is implanted and presents unreasonable risk
    - No 510(k) predicate exists and De Novo is not appropriate

    ---
    #### Required elements
    | Element | Details |
    |---------|---------|
    | Clinical data | Valid scientific evidence from clinical investigations |
    | Non-clinical studies | Bench, animal, biocompatibility testing |
    | Manufacturing information | Full quality system documentation |
    | Device description | Complete technical specifications |
    | Proposed labeling | Final labeling in FDA format |
    | Summary of safety & effectiveness | Public-facing SSED document |

    ---
    #### Typical timeline
    | Phase | Duration |
    |-------|----------|
    | Pre-submission meetings | 3–6 months |
    | Clinical trials | 1–4 years |
    | FDA review | 180 days (statutory goal) |
    | **Total** | **2–7 years** |

    ---
    #### Key decision codes
    | Code | Meaning |
    |------|---------|
    | `APPR` | Approved |
    | `APRL` | Approved with conditions |
    | `DENY` | Denied |
    | `WTDR` | Withdrawn |
    """)
    st.link_button(
        "Open Official FDA PMA Guidance",
        "https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-correct-submission/premarket-approval-pma",
        use_container_width=True,
    )


@st.dialog("🟢 De Novo Classification — Full Details", width="large")
def _dialog_de_novo() -> None:
    st.markdown("""
    **De Novo** is a risk-based classification pathway for **novel, low-to-moderate risk devices**
    that have no legally marketed predicate. It establishes a new device type and
    can itself become a predicate for future 510(k) submissions.

    ---
    #### When to use De Novo
    - Device is novel — no legally marketed predicate exists
    - Device is Class I or Class II risk level (not high risk)
    - An NSE determination was received after a 510(k) submission
    - Device uses new technology with a new intended use

    ---
    #### Required elements
    | Element | Details |
    |---------|---------|
    | Device description | Full technical description and intended use |
    | Classification proposal | Proposed device type name and product code |
    | Special controls proposal | Proposed mitigations for identified risks |
    | Performance testing | Evidence supporting low/moderate risk |
    | Risk analysis | Complete risk/benefit analysis |
    | Draft labeling | Final labeling reflecting special controls |

    ---
    #### Typical timeline
    | Phase | Duration |
    |-------|----------|
    | Preparation | 6–18 months |
    | FDA review | 150 days (statutory goal) |
    | Negotiation of special controls | Variable |
    | **Total** | **12–24 months** |

    ---
    #### Two entry paths
    | Path | Description |
    |------|-------------|
    | Direct De Novo | Submit De Novo request without a prior 510(k) |
    | Post-NSE De Novo | Submit after receiving a Not Substantially Equivalent 510(k) decision |
    """)
    st.link_button(
        "Open Official FDA De Novo Guidance",
        "https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-correct-submission/de-novo-classification-request",
        use_container_width=True,
    )


_PATHWAY_DIALOGS = {
    "510k":    _dialog_510k,
    "PMA":     _dialog_pma,
    "De Novo": _dialog_de_novo,
}


def _device_lookup_section() -> None:
    """Search FDA classification DB and pre-fill form fields via session state."""
    with st.expander("🔍 Look Up Your Device in the FDA Classification Database", expanded=True):
        st.caption(
            "Search the official FDA product code database (foiclass.zip, updated weekly) "
            "to automatically fill in device class, specialty, and properties."
        )

        load_col, _ = st.columns([3, 2])
        with load_col:
            query = st.text_input(
                "Search by device name, product code, or keyword",
                placeholder="e.g. IV labeling, catheter, glucose monitor, K203487…",
                key="fda_class_search",
            )

        if not query.strip():
            st.info("Type a device name or product code above to search ~7,000 FDA device types.", icon="💡")
            return

        with st.spinner("Searching FDA classification database…"):
            df_cls = _load_fda_classification_db()

        if df_cls is None:
            st.error(
                "Could not load FDA classification database. "
                "Check your internet connection and try again."
            )
            return

        q = query.strip().lower()
        mask = (
            df_cls["DEVICENAME"].str.lower().str.contains(q, na=False)
            | df_cls["PRODUCTCODE"].str.lower().str.contains(q, na=False)
            | df_cls["DEFINITION"].str.lower().str.contains(q, na=False)
        )
        hits = df_cls[mask].copy()

        if hits.empty:
            st.warning(
                f"No FDA device types matched **'{query}'**. "
                "Try a broader term (e.g. 'label' instead of 'IV labeling system')."
            )
            return

        hits = hits.head(50)
        st.markdown(
            f"**{len(hits)} result{'s' if len(hits) != 1 else ''}** — "
            "click a row to auto-fill the form below.",
        )

        display = hits[[
            "PRODUCTCODE", "DEVICENAME", "Class", "Specialty",
            "Submission", "REGULATION_NUMBER", "DEFINITION",
        ]].rename(columns={
            "PRODUCTCODE":        "Code",
            "DEVICENAME":         "Device Type",
            "REGULATION_NUMBER":  "Regulation",
            "DEFINITION":         "Definition",
        })

        event = st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            height=213,          # ~5 rows visible; scroll for more
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "Code":       st.column_config.TextColumn(width="small"),
                "Device Type":st.column_config.TextColumn(width="large"),
                "Class":      st.column_config.TextColumn(width="small"),
                "Specialty":  st.column_config.TextColumn(width="medium"),
                "Submission": st.column_config.TextColumn(width="medium"),
                "Regulation": st.column_config.TextColumn(width="medium"),
                "Definition": st.column_config.TextColumn(width="large"),
            },
        )

        sel_rows = event.selection.rows if hasattr(event, "selection") else []
        if sel_rows:
            row = hits.iloc[sel_rows[0]]

            # Determine implant / life-sustain from extra columns if present
            implant_val    = False
            life_sust_val  = False
            for col in df_cls.columns:
                val = str(row.get(col, "")).upper()
                if "IMPLANT" in col.upper() and val == "Y":
                    implant_val = True
                if "LIFE" in col.upper() and val == "Y":
                    life_sust_val = True

            # Map specialty — ensure it's in our dropdown list
            specialty_val = row["Specialty"] if row["Specialty"] in MEDICAL_SPECIALTIES else "Other"

            # Write prefill values into session state
            st.session_state["_prefill_class"]    = row["Class"]
            st.session_state["_prefill_specialty"] = specialty_val
            st.session_state["_prefill_implant"]   = implant_val
            st.session_state["_prefill_life"]      = life_sust_val
            st.session_state["_prefill_name"]      = row["DEVICENAME"].title()

            info_color = PATHWAY_INFO.get(
                {"510(k)": "510k", "PMA": "PMA", "De Novo": "De Novo"}.get(row["Submission"], ""), {}
            ).get("color", "#3f51b5")

            st.markdown(
                f'<div style="background:#f0f4ff;border-left:5px solid {info_color};'
                f'border-radius:8px;padding:14px 18px;margin-top:8px;font-size:14px;">'
                f'<b>✅ Selected:</b> {row["DEVICENAME"]} '
                f'(<code>{row["PRODUCTCODE"]}</code> · {row["REGULATION_NUMBER"]})<br>'
                f'<span style="color:#546e7a;">Class: <b>{row["Class"]}</b> &nbsp;|&nbsp; '
                f'Specialty: <b>{row["Specialty"]}</b> &nbsp;|&nbsp; '
                f'Typical Pathway: <b>{row["Submission"]}</b></span><br>'
                f'<span style="font-size:12px;color:#78909c;">{row["DEFINITION"][:250]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.success("Form fields auto-filled below — review and click **Predict Pathway**.", icon="⬇️")


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
            if st.button(
                "Learn more",
                key=f"tile_more_{key}",
                use_container_width=True,
            ):
                _PATHWAY_DIALOGS[key]()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── FDA Classification Lookup ──
    st.markdown('<div class="section-header">🔍 Step 1 — Find Your Device Type</div>', unsafe_allow_html=True)
    _device_lookup_section()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input form (reads prefill values from session state) ──
    st.markdown('<div class="section-header">🔬 Step 2 — Confirm Details & Predict</div>', unsafe_allow_html=True)

    _pf_class     = st.session_state.get("_prefill_class", "Not Sure")
    _pf_specialty = st.session_state.get("_prefill_specialty", MEDICAL_SPECIALTIES[0])
    _pf_implant   = st.session_state.get("_prefill_implant", False)
    _pf_life      = st.session_state.get("_prefill_life", False)
    _pf_name      = st.session_state.get("_prefill_name", "")

    col_form, col_flags = st.columns([3, 2], gap="large")

    with col_form:
        device_name = st.text_input(
            "Device Name",
            value=_pf_name,
            placeholder="e.g. Cardiac Monitor, Hip Implant, IV Labeling System…",
            help="Enter the commercial or generic name of your medical device.",
        )
        c1, c2 = st.columns(2)
        with c1:
            cls_idx = DEVICE_CLASSES.index(_pf_class) if _pf_class in DEVICE_CLASSES else 0
            device_class = st.selectbox("Device Class", DEVICE_CLASSES, index=cls_idx)
        with c2:
            spec_idx = MEDICAL_SPECIALTIES.index(_pf_specialty) if _pf_specialty in MEDICAL_SPECIALTIES else 0
            medical_specialty = st.selectbox("Medical Specialty", MEDICAL_SPECIALTIES, index=spec_idx)
        if device_class == "Not Sure":
            st.info(CLASS_DESCRIPTIONS["Not Sure"], icon="❓")
        else:
            st.caption(f"ℹ️ {CLASS_DESCRIPTIONS.get(device_class, '')}")

    with col_flags:
        st.markdown("**Device Properties**")
        implant_flag      = st.toggle("Implantable Device", value=_pf_implant,
                                      help="Device is intended to be surgically implanted in the body.")
        life_sustain_flag = st.toggle("Life-Sustaining / Life-Supporting", value=_pf_life,
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

        model        = model_payload["model"]
        feature_cols = model_payload["feature_cols"]
        le_target: LabelEncoder = model_payload["label_encoder"]

        try:
            clean_path = ARTIFACTS_DIR / "clean_data.csv"
            if clean_path.exists():
                df_train = pd.read_csv(clean_path)
                le_class = LabelEncoder().fit(df_train["device_class"].astype(str))
                le_spec  = LabelEncoder().fit(df_train["medical_specialty_description"].astype(str))
            else:
                le_class = LabelEncoder().fit(["Class I", "Class II", "Class III", "Unknown"])
                le_spec  = LabelEncoder().fit(MEDICAL_SPECIALTIES)

            def safe_transform(le, value):
                return int(le.transform([value])[0]) if value in le.classes_ else 0

            def _predict_for_class(cls: str) -> tuple[str, dict]:
                feats = []
                if "device_class_enc" in feature_cols:
                    feats.append(safe_transform(le_class, cls))
                if "medical_specialty_description_enc" in feature_cols:
                    feats.append(safe_transform(le_spec, medical_specialty))
                if "implant_flag_bin" in feature_cols:
                    feats.append(1 if implant_flag else 0)
                if "life_sustain_support_flag_bin" in feature_cols:
                    feats.append(1 if life_sustain_flag else 0)
                X = np.array(feats).reshape(1, -1)
                y_enc = model.predict(X)[0]
                pathway = le_target.inverse_transform([y_enc])[0]
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)[0]
                    probs = {le_target.inverse_transform([i])[0]: float(p) for i, p in enumerate(proba)}
                else:
                    probs = {pathway: 1.0}
                return pathway, probs

            st.markdown('<div class="section-header">🎯 Prediction Result</div>', unsafe_allow_html=True)

            # ── "Not Sure" mode: show all 3 classes side by side ──
            if device_class == "Not Sure":
                st.info(
                    "You selected **Not Sure** for device class. "
                    "Here's the predicted pathway for each class — "
                    "once you determine your class, focus on that column.",
                    icon="ℹ️",
                )

                known_classes = ["Class I", "Class II", "Class III"]
                cols = st.columns(3)
                predicted_pathways = {}

                for col, cls in zip(cols, known_classes):
                    pathway, probs = _predict_for_class(cls)
                    predicted_pathways[cls] = pathway
                    info = PATHWAY_INFO.get(pathway, {})
                    top_conf = probs.get(pathway, 1.0)
                    with col:
                        st.markdown(f"""
                        <div class="result-card"
                             style="background:{info.get('bg','#f5f5f5')};
                                    border-color:{info.get('color','#888')};
                                    color:#1a1f36;padding:20px 22px;">
                            <div class="rc-eyebrow">{cls}</div>
                            <div class="rc-name" style="font-size:24px;">{info.get('icon','')} {pathway}</div>
                            <div class="rc-meta">{info.get('time','')} · Risk: {info.get('risk','')}</div>
                            <div class="rc-desc" style="font-size:13px;">{info.get('description','')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="metric-pill" style="margin-top:10px;">
                            <div class="mp-value" style="color:{info.get('color','#3f51b5')};font-size:22px;">{top_conf:.0%}</div>
                            <div class="mp-label">Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("❓ How to find your device class"):
                    st.markdown(CLASS_LOOKUP_HINT)

                # Show checklist for the most likely pathway (Class II result, most common)
                dominant = predicted_pathways.get("Class II", list(predicted_pathways.values())[0])
                st.markdown(
                    f"*Showing checklist for the **Class II** prediction ({dominant}) "
                    f"as a starting point — update once you confirm your class.*"
                )
                st.markdown("<br>", unsafe_allow_html=True)
                _render_checklist(dominant)

            # ── Normal single-class prediction ──
            else:
                predicted_pathway, class_probs = _predict_for_class(device_class)
                info           = PATHWAY_INFO.get(predicted_pathway, {})
                top_confidence = class_probs.get(predicted_pathway, 1.0)

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

                st.markdown("<br>", unsafe_allow_html=True)
                _render_checklist(predicted_pathway)

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

@st.cache_data(show_spinner=False)
def _load_clean_df() -> pd.DataFrame | None:
    path = ARTIFACTS_DIR / "clean_data.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _eda_charts(df: pd.DataFrame, year_label: str) -> None:
    """Render inline EDA charts for a (possibly filtered) dataframe."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    PATHWAY_COLORS = {
        "510k":     "#1976D2",
        "PMA":      "#C62828",
        "De Novo":  "#2E7D32",
    }

    def pathway_color_list(index):
        return [PATHWAY_COLORS.get(p, "#888") for p in index]

    c1, c2 = st.columns(2)

    # ── Chart 1: Pathway distribution ──
    with c1:
        counts = df["pathway"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(counts.index, counts.values,
                      color=pathway_color_list(counts.index), edgecolor="white", linewidth=1.2)
        ax.set_title(f"Pathway Distribution — {year_label}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Count")
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts.values) * 0.01,
                    str(v), ha="center", fontweight="bold", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Chart 2: Device class vs pathway ──
    with c2:
        if "device_class" in df.columns:
            ct = pd.crosstab(df["device_class"], df["pathway"])
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ct.plot(kind="bar", ax=ax,
                    color=[PATHWAY_COLORS.get(c, "#888") for c in ct.columns],
                    edgecolor="white", linewidth=1.2)
            ax.set_title(f"Device Class vs Pathway — {year_label}", fontsize=12, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Count")
            ax.legend(title="Pathway", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    c3, c4 = st.columns(2)

    # ── Chart 3: Top medical specialties ──
    with c3:
        if "medical_specialty_description" in df.columns:
            top = df["medical_specialty_description"].value_counts().head(8)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            palette = sns.color_palette("husl", len(top))
            ax.barh(top.index[::-1], top.values[::-1], color=palette[::-1])
            ax.set_title(f"Top Specialties — {year_label}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Count")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ── Chart 4: Implant flag vs pathway ──
    with c4:
        if "implant_flag" in df.columns:
            ct2 = pd.crosstab(df["implant_flag"], df["pathway"])
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ct2.plot(kind="bar", ax=ax,
                     color=[PATHWAY_COLORS.get(c, "#888") for c in ct2.columns],
                     edgecolor="white", linewidth=1.2)
            ax.set_title(f"Implant Flag vs Pathway — {year_label}", fontsize=12, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Count")
            ax.legend(title="Pathway", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


def page_eda() -> None:
    st.markdown("""
    <div class="page-hero">
        <h1>📊 EDA Dashboard</h1>
        <p>Exploratory data analysis of the openFDA training dataset.
        Use the year filter to drill into a specific decision year.</p>
    </div>
    """, unsafe_allow_html=True)

    df_full = _load_clean_df()
    if df_full is None:
        st.warning("clean_data.csv not found. Run the pipeline first.")
        return

    # ── Year filter ──
    has_year = "decision_year" in df_full.columns and df_full["decision_year"].notna().any()

    if has_year:
        years_available = sorted(df_full["decision_year"].dropna().astype(int).unique(), reverse=True)
        filter_col, info_col = st.columns([2, 5])
        with filter_col:
            selected_year = st.selectbox(
                "Filter by Decision Year",
                ["All years"] + [str(y) for y in years_available],
                key="eda_year",
            )
        df = df_full if selected_year == "All years" else df_full[
            df_full["decision_year"] == int(selected_year)
        ]
        year_label = selected_year
        with info_col:
            st.markdown(
                f"<div style='padding-top:28px;color:#546e7a;font-size:13px;'>"
                f"Showing <b>{len(df):,}</b> of <b>{len(df_full):,}</b> records</div>",
                unsafe_allow_html=True,
            )
        if len(df) == 0:
            st.warning(f"No records found for {selected_year}.")
            return
    else:
        df = df_full
        year_label = "All years"
        st.info(
            "No decision year data found in the current dataset. "
            "Re-run the pipeline to fetch real openFDA records with date fields.",
            icon="ℹ️",
        )

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Charts", "📋 Distribution", "💡 Insights", "🗂 Dataset Contract"
    ])

    # ── Tab 1 — Dynamic charts ──
    with tab1:
        _eda_charts(df, year_label)

    # ── Tab 2 — Distribution table ──
    with tab2:
        counts = df["pathway"].value_counts()
        total = len(df)

        m1, m2, m3, m4 = st.columns(4)
        pills = [
            (f"{total:,}", "Total Records", "#3f51b5"),
            (str(counts.get("510k", 0)), "510(k)", "#1976D2"),
            (str(counts.get("PMA", 0)), "PMA", "#C62828"),
            (str(counts.get("De Novo", 0)), "De Novo", "#2E7D32"),
        ]
        for col, (val, label, color) in zip([m1, m2, m3, m4], pills):
            with col:
                st.markdown(f"""
                <div class="metric-pill">
                    <div class="mp-value" style="color:{color}">{val}</div>
                    <div class="mp-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        dist_col, class_col = st.columns(2)
        with dist_col:
            st.markdown("**Pathway Distribution**")
            dist_df = pd.DataFrame([
                {"Pathway": k, "Count": v, "Share": f"{100*v/total:.1f}%"}
                for k, v in counts.items()
            ])
            st.dataframe(dist_df, use_container_width=True, hide_index=True)

        with class_col:
            st.markdown("**Device Class Breakdown**")
            if "device_class" in df.columns:
                class_counts = df["device_class"].value_counts()
                class_df = pd.DataFrame([
                    {"Class": k, "Count": v, "Share": f"{100*v/total:.1f}%"}
                    for k, v in class_counts.items()
                ])
                st.dataframe(class_df, use_container_width=True, hide_index=True)

        if "medical_specialty_description" in df.columns:
            st.markdown("**Top Medical Specialties**")
            spec_counts = df["medical_specialty_description"].value_counts().head(10)
            spec_df = pd.DataFrame([
                {"Specialty": k, "Count": v, "Share": f"{100*v/total:.1f}%"}
                for k, v in spec_counts.items()
            ])
            st.dataframe(spec_df, use_container_width=True, hide_index=True)

    # ── Tab 3 — Insights (static) ──
    with tab3:
        st.markdown(read_text("insights.md"))

    # ── Tab 4 — Dataset Contract ──
    with tab4:
        contract = read_json("dataset_contract.json")
        if contract:
            st.markdown("*The dataset contract reflects the full dataset from the last pipeline run.*")
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
        "DENG": "De Novo Granted",
        "SESK": "Substantially Equivalent with conditions",
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
