from __future__ import annotations

import logging
from pathlib import Path

from crewai import Agent
from crewai.tools import tool

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parents[3] / "artifacts"


# ---------------------------------------------------------------------------
# CrewAI tools — thin wrappers around ml_pipeline functions
# ---------------------------------------------------------------------------

@tool("Engineer Features")
def engineer_features_tool(dummy: str = "") -> str:
    """
    Load artifacts/clean_data.csv and artifacts/dataset_contract.json,
    encode categorical features and binary flags, and save
    artifacts/features.csv and artifacts/label_mapping.json.
    Returns a feature engineering summary.
    """
    import pandas as pd
    from src.tools.ml_pipeline import engineer_features

    clean_path = ARTIFACTS_DIR / "clean_data.csv"
    df = pd.read_csv(clean_path)
    X, y, feature_cols, le_target = engineer_features(df)
    return (
        f"Engineered {len(feature_cols)} features from {len(df)} rows. "
        f"Feature columns: {feature_cols}. "
        f"Classes: {list(le_target.classes_)}. "
        f"Saved features.csv and label_mapping.json."
    )


@tool("Train ML Models")
def train_models_tool(dummy: str = "") -> str:
    """
    Load artifacts/features.csv, train Random Forest, Gradient Boosting,
    and Logistic Regression models, and save the best model to
    artifacts/model.pkl.
    Returns training results summary.
    """
    import pandas as pd
    from src.tools.ml_pipeline import engineer_features, train_models

    clean_path = ARTIFACTS_DIR / "clean_data.csv"
    df = pd.read_csv(clean_path)
    X, y, feature_cols, le_target = engineer_features(df)
    results, best_name = train_models(X, y, le_target)

    summary_parts = [f"Trained 3 models. Best: {best_name}."]
    for name, res in results.items():
        summary_parts.append(f"  {name}: macro F1 = {res['macro_f1']:.4f}")
    summary_parts.append("Saved best model to artifacts/model.pkl.")
    return "\n".join(summary_parts)


@tool("Evaluate Models")
def evaluate_models_tool(dummy: str = "") -> str:
    """
    Load artifacts/clean_data.csv, re-run feature engineering and training,
    evaluate all models, and save artifacts/evaluation_report.md,
    artifacts/model_card.md, and artifacts/confusion_matrix.png.
    Returns evaluation summary.
    """
    import pandas as pd
    from src.tools.ml_pipeline import engineer_features, evaluate_models, train_models

    clean_path = ARTIFACTS_DIR / "clean_data.csv"
    df = pd.read_csv(clean_path)
    X, y, feature_cols, le_target = engineer_features(df)
    results, best_name = train_models(X, y, le_target)
    evaluate_models(results, best_name, le_target)

    best_f1 = results[best_name]["macro_f1"]
    return (
        f"Evaluation complete. Best model: {best_name} (macro F1={best_f1:.4f}). "
        f"Saved evaluation_report.md, model_card.md, confusion_matrix.png."
    )


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

def feature_engineer_agent() -> Agent:
    return Agent(
        role="Feature Engineering Specialist",
        goal=(
            "Transform the clean device dataset into a numeric feature matrix "
            "suitable for machine learning, using appropriate encoding strategies."
        ),
        backstory=(
            "You are an ML practitioner who specializes in converting "
            "categorical and binary device attributes into informative numeric "
            "features. You document every encoding decision for reproducibility."
        ),
        tools=[engineer_features_tool],
        allow_delegation=False,
        verbose=True,
    )


def trainer_agent() -> Agent:
    return Agent(
        role="ML Model Trainer",
        goal=(
            "Train multiple classification models on the feature matrix, "
            "select the best performer by macro F1, and persist it for inference."
        ),
        backstory=(
            "You are an experienced ML engineer who benchmarks multiple "
            "algorithms, applies stratified splits for reproducibility, and "
            "always uses random_state=42 for deterministic results."
        ),
        tools=[train_models_tool],
        allow_delegation=False,
        verbose=True,
    )


def evaluator_agent() -> Agent:
    return Agent(
        role="Model Evaluator",
        goal=(
            "Produce a comprehensive evaluation report with per-class metrics, "
            "a confusion matrix, and a model card that documents the model's "
            "intended use, limitations, and performance."
        ),
        backstory=(
            "You are a rigorous ML evaluator who never reports just accuracy. "
            "You compute macro F1, per-class precision/recall, and build "
            "confusion matrices to catch class-imbalance blind spots."
        ),
        tools=[evaluate_models_tool],
        allow_delegation=False,
        verbose=True,
    )
