from __future__ import annotations

import logging
from pathlib import Path

from crewai import Agent
from crewai.tools import tool

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parents[3] / "artifacts"


# ---------------------------------------------------------------------------
# CrewAI tools — thin wrappers around the actual pipeline functions
# ---------------------------------------------------------------------------

@tool("Fetch FDA Data")
def fetch_fda_data_tool(dummy: str = "") -> str:
    """
    Fetch medical device records from the openFDA 510(k) and PMA endpoints.
    Saves raw records to artifacts/raw_data.csv.
    Returns a summary of records fetched.
    """
    from src.tools.fda_api_tool import build_raw_dataframe
    df = build_raw_dataframe()
    counts = df["pathway"].value_counts().to_dict()
    return (
        f"Fetched {len(df)} records from openFDA. "
        f"Pathway distribution: {counts}. "
        f"Saved to artifacts/raw_data.csv."
    )


@tool("Clean FDA Data")
def clean_fda_data_tool(dummy: str = "") -> str:
    """
    Load artifacts/raw_data.csv, deduplicate and normalize the records,
    save artifacts/clean_data.csv and artifacts/dataset_contract.json.
    Returns a cleaning summary.
    """
    import pandas as pd
    from src.tools.data_processing import clean_data

    raw_path = ARTIFACTS_DIR / "raw_data.csv"
    df = pd.read_csv(raw_path)
    clean_df = clean_data(df)
    return (
        f"Cleaned {len(df)} → {len(clean_df)} rows. "
        f"Pathway distribution: {clean_df['pathway'].value_counts().to_dict()}. "
        f"Saved clean_data.csv and dataset_contract.json."
    )


@tool("Generate EDA Report")
def generate_eda_tool(dummy: str = "") -> str:
    """
    Load artifacts/clean_data.csv and produce exploratory data analysis:
    artifacts/eda_report.html (embedded charts) and artifacts/insights.md.
    Returns a summary of charts generated.
    """
    import pandas as pd
    from src.tools.data_processing import generate_eda

    clean_path = ARTIFACTS_DIR / "clean_data.csv"
    df = pd.read_csv(clean_path)
    generate_eda(df)
    return (
        f"EDA complete on {len(df)} records. "
        f"Saved eda_report.html and insights.md."
    )


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

def ingestion_agent() -> Agent:
    return Agent(
        role="Data Ingestion Specialist",
        goal=(
            "Fetch medical device regulatory records from the openFDA API "
            "and save them as a raw CSV for downstream processing."
        ),
        backstory=(
            "You are an expert in consuming FDA open data APIs. You handle "
            "pagination, rate limits, and exponential backoff to reliably "
            "retrieve large batches of device records."
        ),
        tools=[fetch_fda_data_tool],
        allow_delegation=False,
        verbose=True,
    )


def cleaning_agent() -> Agent:
    return Agent(
        role="Data Cleaning Specialist",
        goal=(
            "Deduplicate and normalize raw device records, handle missing values, "
            "and produce a clean dataset with a strict schema contract."
        ),
        backstory=(
            "You are a meticulous data engineer who transforms messy API responses "
            "into well-structured, analysis-ready datasets. You ensure every "
            "downstream step can trust the data it receives."
        ),
        tools=[clean_fda_data_tool],
        allow_delegation=False,
        verbose=True,
    )


def eda_agent() -> Agent:
    return Agent(
        role="Exploratory Data Analyst",
        goal=(
            "Produce insightful visualizations and written observations about "
            "the cleaned device dataset, highlighting class distributions, "
            "feature correlations, and modeling recommendations."
        ),
        backstory=(
            "You are a data storyteller who translates numbers into clear, "
            "actionable findings. You generate publication-quality charts and "
            "concise markdown summaries for non-technical stakeholders."
        ),
        tools=[generate_eda_tool],
        allow_delegation=False,
        verbose=True,
    )
