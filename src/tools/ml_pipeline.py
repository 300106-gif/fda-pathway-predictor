from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parents[2] / "artifacts"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str], LabelEncoder]:
    """
    Encode categoricals and binary flags.
    Returns (X, y_encoded, feature_names, label_encoder_for_target).
    Also saves artifacts/features.csv and artifacts/label_mapping.json.
    """
    logger.info("Engineering features from %d rows…", len(df))

    categorical_cols = ["device_class", "medical_specialty_description"]
    binary_cols = ["implant_flag", "life_sustain_support_flag"]

    X = pd.DataFrame(index=df.index)
    feature_cols: list[str] = []

    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            enc_name = f"{col}_enc"
            X[enc_name] = le.fit_transform(df[col].astype(str))
            feature_cols.append(enc_name)

    for col in binary_cols:
        if col in df.columns:
            bin_name = f"{col}_bin"
            X[bin_name] = (df[col].astype(str).str.upper() == "Y").astype(int)
            feature_cols.append(bin_name)

    # Encode target
    le_target = LabelEncoder()
    y = pd.Series(
        le_target.fit_transform(df["pathway"].astype(str)),
        name="pathway_enc",
        index=df.index,
    )

    # Save features CSV
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    features_df = X[feature_cols].copy()
    features_df["pathway"] = df["pathway"].values
    features_path = ARTIFACTS_DIR / "features.csv"
    features_df.to_csv(features_path, index=False)
    logger.info("Saved features → %s", features_path)

    # Save label mapping
    label_mapping = {int(i): str(cls) for i, cls in enumerate(le_target.classes_)}
    mapping_path = ARTIFACTS_DIR / "label_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(label_mapping, f, indent=2)

    logger.info("Classes: %s", list(le_target.classes_))
    return X[feature_cols], y, feature_cols, le_target


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    le_target: LabelEncoder,
) -> tuple[dict[str, Any], str]:
    """
    Train Random Forest, Gradient Boosting, and Logistic Regression.
    Saves the best model to artifacts/model.pkl.
    Returns (results_dict, best_model_name).
    """
    # Stratified split requires each class to have >= 2 samples.
    # Real openFDA data can have very few De Novo records; fall back gracefully.
    class_counts = pd.Series(y).value_counts()
    can_stratify = int(class_counts.min()) >= 2
    if not can_stratify:
        rare = {
            le_target.inverse_transform([int(k)])[0]: int(v)
            for k, v in class_counts.items()
            if int(v) < 2
        }
        logger.warning(
            "Rare classes with < 2 samples — using non-stratified split: %s", rare
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y if can_stratify else None,
        random_state=42,
    )
    logger.info("Train size: %d | Test size: %d", len(X_train), len(X_test))

    model_definitions: dict[str, Any] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
    }

    results: dict[str, Any] = {}
    best_f1 = -1.0
    best_name = ""
    best_model: Any = None

    for name, model in model_definitions.items():
        logger.info("Training %s…", name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        # Only report on classes that actually appear in test or predictions
        present_labels = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
        present_names = [le_target.inverse_transform([lbl])[0] for lbl in present_labels]

        results[name] = {
            "model": model,
            "y_test": y_test,
            "y_pred": y_pred,
            "macro_f1": macro_f1,
            "present_labels": present_labels,
            "present_names": present_names,
            "report": classification_report(
                y_test, y_pred,
                labels=present_labels,
                target_names=present_names,
                zero_division=0,
            ),
        }
        logger.info("  %s — macro F1: %.4f", name, macro_f1)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_name = name
            best_model = model

    logger.info("Best model: %s (macro F1 = %.4f)", best_name, best_f1)

    # Persist best model
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    model_payload = {
        "model": best_model,
        "feature_cols": list(X.columns),
        "label_encoder": le_target,
        "model_name": best_name,
        "macro_f1": best_f1,
        "classes": list(le_target.classes_),
    }
    model_path = ARTIFACTS_DIR / "model.pkl"
    joblib.dump(model_payload, model_path)
    logger.info("Saved model → %s", model_path)

    return results, best_name


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_models(
    results: dict[str, Any],
    best_name: str,
    le_target: LabelEncoder,
) -> None:
    """
    Write evaluation_report.md, model_card.md, and confusion_matrix.png.
    """
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # evaluation_report.md
    lines = [
        "# Model Evaluation Report — FDA Pathway Predictor\n",
        f"**Best Model:** `{best_name}`\n",
        f"**Best Macro F1:** `{results[best_name]['macro_f1']:.4f}`\n",
        "\n---\n",
    ]
    for name, res in results.items():
        marker = " ✓ (selected)" if name == best_name else ""
        lines += [
            f"\n## {name}{marker}\n",
            f"**Macro F1:** `{res['macro_f1']:.4f}`\n\n",
            "**Full Classification Report:**\n",
            f"```\n{res['report']}\n```\n",
        ]

    report_path = ARTIFACTS_DIR / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    logger.info("Saved evaluation report → %s", report_path)

    # Confusion matrix for best model
    best = results[best_name]
    present_labels = best.get("present_labels", sorted(set(best["y_test"].tolist())))
    present_names = best.get("present_names", list(le_target.classes_))
    cm = confusion_matrix(best["y_test"], best["y_pred"], labels=present_labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=present_names,
        yticklabels=present_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = ARTIFACTS_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", cm_path)

    # model_card.md
    best_f1 = results[best_name]["macro_f1"]
    model_card = f"""# Model Card — FDA Regulatory Pathway Predictor

## Model Details
- **Name**: {best_name}
- **Version**: 1.0.0
- **Framework**: scikit-learn
- **Task**: Multi-class classification (3 classes)
- **Classes**: {', '.join(f'`{c}`' for c in le_target.classes_)}

## Intended Use
Predict the most likely FDA regulatory pathway for a medical device based on its
characteristics. **For planning purposes only — not a substitute for regulatory counsel.**

## Training Data
| Property | Value |
|----------|-------|
| Source | openFDA API (510k + PMA endpoints) or synthetic fallback |
| Split | 80% train / 20% test, stratified on `pathway` |
| Random seed | 42 |
| Class weighting | balanced (where supported) |

## Input Features
| Feature | Encoding | Description |
|---------|----------|-------------|
| `device_class_enc` | Label-encoded int | Device risk classification (Class I/II/III) |
| `medical_specialty_description_enc` | Label-encoded int | Primary medical specialty |
| `implant_flag_bin` | Binary (0/1) | Whether the device is implantable |
| `life_sustain_support_flag_bin` | Binary (0/1) | Whether the device is life-sustaining |

## Performance Summary
- **Best Model**: {best_name}
- **Macro F1**: `{best_f1:.4f}`

See `evaluation_report.md` for full per-class precision, recall, and confusion matrix.

## Limitations
- Trained on publicly available FDA data; actual pathway decisions depend on many
  additional factors (predicate device history, clinical evidence, etc.).
- De Novo pathway is under-represented; recall for that class may be lower.
- Model does not account for recent regulatory guidances or policy changes.

## Ethical Considerations
This model is strictly a **planning aid** for early-stage regulatory strategy.
Final decisions must involve qualified regulatory affairs professionals.

## Disclaimer
> This tool is for planning purposes only. Consult qualified regulatory counsel.
"""

    card_path = ARTIFACTS_DIR / "model_card.md"
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(model_card)
    logger.info("Saved model card → %s", card_path)
