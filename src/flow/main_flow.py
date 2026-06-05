"""
FDA Regulatory Pathway Predictor — CrewAI Flow entrypoint.

Run with:
    uv run python -m src.flow.main_flow

Two execution modes:
  - LLM mode  : set OPENAI_API_KEY in .env → full CrewAI agent reasoning
  - Direct mode: no key needed → calls pipeline functions directly (same artifacts)
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parents[2] / "artifacts"

# ---------------------------------------------------------------------------
# Attempt to import CrewAI Flow
# ---------------------------------------------------------------------------
try:
    from crewai.flow.flow import Flow, listen, start
    FLOW_AVAILABLE = True
except ImportError:
    FLOW_AVAILABLE = False
    logger.warning("crewai.flow not available — using sequential direct mode.")

HAS_LLM = bool(
    os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
)


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------

def _validate_crew1_outputs() -> None:
    """Assert all Crew 1 artifacts exist and meet minimum quality standards."""
    clean_path = ARTIFACTS_DIR / "clean_data.csv"
    contract_path = ARTIFACTS_DIR / "dataset_contract.json"

    assert clean_path.exists(), f"Missing artifact: {clean_path}"
    assert contract_path.exists(), f"Missing artifact: {contract_path}"

    df = pd.read_csv(clean_path)
    assert "pathway" in df.columns, "Target column 'pathway' missing from clean_data.csv"
    assert df["pathway"].nunique() >= 2, "Need at least 2 pathway classes"

    if len(df) < 100:
        logger.warning(
            "Only %d records found — falling back to synthetic sample data.", len(df)
        )
        raise AssertionError(f"Too few records ({len(df)}) — fallback to sample data")

    logger.info(
        "Crew 1 validation passed: %d rows, %d classes.",
        len(df), df["pathway"].nunique(),
    )


def _validate_crew2_outputs() -> None:
    """Assert all Crew 2 artifacts exist."""
    model_path = ARTIFACTS_DIR / "model.pkl"
    report_path = ARTIFACTS_DIR / "evaluation_report.md"

    assert model_path.exists(), f"Missing artifact: {model_path}"
    assert report_path.exists(), f"Missing artifact: {report_path}"

    logger.info("Crew 2 validation passed: model.pkl and evaluation_report.md present.")


# ---------------------------------------------------------------------------
# Direct pipeline (no LLM required)
# ---------------------------------------------------------------------------

def _run_data_analyst_direct() -> None:
    """Ingest → clean → EDA, calling tool functions directly."""
    logger.info("=== Crew 1 (Direct Mode): Data Analyst ===")

    # Step 1: Ingestion
    logger.info("Step 1/3 — Ingesting FDA data…")
    from src.tools.fda_api_tool import build_raw_dataframe
    raw_df = build_raw_dataframe()
    logger.info("Ingested %d raw records.", len(raw_df))

    # Step 2: Cleaning
    logger.info("Step 2/3 — Cleaning data…")
    from src.tools.data_processing import clean_data
    clean_df = clean_data(raw_df)
    logger.info("Cleaned: %d rows remain.", len(clean_df))

    # Step 3: EDA
    logger.info("Step 3/3 — Generating EDA report…")
    from src.tools.data_processing import generate_eda
    generate_eda(clean_df)
    logger.info("EDA complete.")


def _run_data_scientist_direct() -> None:
    """Feature engineering → training → evaluation, calling functions directly."""
    logger.info("=== Crew 2 (Direct Mode): Data Scientist ===")

    import pandas as pd
    from src.tools.ml_pipeline import engineer_features, evaluate_models, train_models

    df = pd.read_csv(ARTIFACTS_DIR / "clean_data.csv")

    # Step 1: Feature engineering
    logger.info("Step 1/3 — Engineering features…")
    X, y, feature_cols, le_target = engineer_features(df)

    # Step 2: Training
    logger.info("Step 2/3 — Training models…")
    results, best_name = train_models(X, y, le_target)

    # Step 3: Evaluation
    logger.info("Step 3/3 — Evaluating models…")
    evaluate_models(results, best_name, le_target)
    logger.info("Training & evaluation complete. Best model: %s", best_name)


def _fallback_to_sample_data() -> None:
    """Generate synthetic data and write dataset contract (Crew 1 fallback)."""
    logger.warning("Falling back to synthetic sample data (generate_sample_data.py)…")
    from src.tools.generate_sample_data import generate_sample_data
    generate_sample_data(n=500, seed=42)
    logger.info("Synthetic fallback data written.")


# ---------------------------------------------------------------------------
# CrewAI Flow (LLM mode)
# ---------------------------------------------------------------------------

if FLOW_AVAILABLE:
    class FDAPathwayPredictorFlow(Flow):
        """Sequential flow: Data Analyst → validation → Data Scientist → validation."""

        @start()
        def run_data_analyst_crew(self):
            logger.info("=== Crew 1 (LLM Mode): Data Analyst ===")
            from src.crews.data_analyst.crew import DataAnalystCrew
            crew = DataAnalystCrew().crew()
            return crew.kickoff()

        @listen(run_data_analyst_crew)
        def validate_crew1(self, crew1_output):
            logger.info("Running Crew 1 validation gate…")
            try:
                _validate_crew1_outputs()
            except AssertionError as exc:
                if "fallback" in str(exc).lower() or "few records" in str(exc).lower():
                    _fallback_to_sample_data()
                    # Re-run EDA on synthetic data
                    import pandas as pd
                    from src.tools.data_processing import generate_eda
                    df = pd.read_csv(ARTIFACTS_DIR / "clean_data.csv")
                    generate_eda(df)
                else:
                    logger.error("Crew 1 validation failed: %s", exc)
                    raise
            return "crew1_validated"

        @listen(validate_crew1)
        def run_data_scientist_crew(self, validated):
            logger.info("=== Crew 2 (LLM Mode): Data Scientist ===")
            from src.crews.data_scientist.crew import DataScientistCrew
            crew = DataScientistCrew().crew()
            return crew.kickoff()

        @listen(run_data_scientist_crew)
        def validate_crew2(self, crew2_output):
            logger.info("Running Crew 2 validation gate…")
            try:
                _validate_crew2_outputs()
            except AssertionError as exc:
                logger.error("Crew 2 validation failed: %s", exc)
                sys.exit(1)
            return "pipeline_complete"

        @listen(validate_crew2)
        def pipeline_done(self, result):
            logger.info(
                "Pipeline complete! All artifacts are ready in: %s", ARTIFACTS_DIR
            )
            _log_artifact_summary()
            return result


# ---------------------------------------------------------------------------
# Orchestration entry point
# ---------------------------------------------------------------------------

def _log_artifact_summary() -> None:
    logger.info("--- Artifact Summary ---")
    for name in [
        "raw_data.csv", "clean_data.csv", "eda_report.html", "insights.md",
        "dataset_contract.json", "features.csv", "model.pkl",
        "evaluation_report.md", "model_card.md", "confusion_matrix.png",
    ]:
        path = ARTIFACTS_DIR / name
        status = f"{path.stat().st_size:,} bytes" if path.exists() else "MISSING"
        logger.info("  %-30s %s", name, status)


def run_pipeline() -> None:
    """Main orchestration: choose LLM or direct mode, run both crews with gates."""
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    if HAS_LLM and FLOW_AVAILABLE:
        logger.info("Starting FDA Pathway Predictor in LLM mode (CrewAI Flow).")
        flow = FDAPathwayPredictorFlow()
        flow.kickoff()
    else:
        if not HAS_LLM:
            logger.info(
                "No LLM API key found. Running in direct mode "
                "(set OPENAI_API_KEY in .env to enable CrewAI agents)."
            )

        # --- Crew 1 ---
        try:
            _run_data_analyst_direct()
        except Exception as exc:
            logger.error("Crew 1 failed: %s", exc, exc_info=True)
            logger.warning("Activating graceful fallback…")
            _fallback_to_sample_data()
            import pandas as pd
            from src.tools.data_processing import generate_eda
            df = pd.read_csv(ARTIFACTS_DIR / "clean_data.csv")
            generate_eda(df)

        # Validation gate 1
        try:
            _validate_crew1_outputs()
        except AssertionError as exc:
            if "fallback" in str(exc).lower() or "few records" in str(exc).lower():
                logger.warning("Validation triggered fallback: %s", exc)
                _fallback_to_sample_data()
                import pandas as pd
                from src.tools.data_processing import generate_eda
                df = pd.read_csv(ARTIFACTS_DIR / "clean_data.csv")
                generate_eda(df)
                _validate_crew1_outputs()  # must pass now
            else:
                logger.error("Crew 1 validation failed: %s", exc)
                sys.exit(1)

        # --- Crew 2 ---
        try:
            _run_data_scientist_direct()
        except Exception as exc:
            logger.error("Crew 2 failed: %s", exc, exc_info=True)
            sys.exit(1)

        # Validation gate 2
        try:
            _validate_crew2_outputs()
        except AssertionError as exc:
            logger.error("Crew 2 validation failed: %s", exc)
            sys.exit(1)

        logger.info("Pipeline complete! Artifacts ready in: %s", ARTIFACTS_DIR)
        _log_artifact_summary()


if __name__ == "__main__":
    run_pipeline()
