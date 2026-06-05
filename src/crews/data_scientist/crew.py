from __future__ import annotations

import logging

from crewai import Crew, Process, Task

from src.crews.data_scientist.agents import (
    evaluator_agent,
    feature_engineer_agent,
    trainer_agent,
)

logger = logging.getLogger(__name__)


class DataScientistCrew:
    """Crew 2 — engineers features, trains models, and evaluates performance."""

    def crew(self) -> Crew:
        _feature_eng = feature_engineer_agent()
        _trainer = trainer_agent()
        _evaluator = evaluator_agent()

        task_features = Task(
            description=(
                "Use the 'Engineer Features' tool to encode categorical columns "
                "(device_class, medical_specialty_description) and binary flags "
                "(implant_flag, life_sustain_support_flag) from artifacts/clean_data.csv. "
                "Save artifacts/features.csv and artifacts/label_mapping.json. "
                "Report the resulting feature list and class labels."
            ),
            expected_output=(
                "A summary of the feature engineering steps: which columns were encoded, "
                "how many features were created, and what the target class labels are. "
                "Confirm that features.csv and label_mapping.json exist."
            ),
            agent=_feature_eng,
        )

        task_train = Task(
            description=(
                "Use the 'Train ML Models' tool to train three classifiers — "
                "Random Forest, Gradient Boosting, and Logistic Regression — "
                "using an 80/20 stratified split with random_state=42. "
                "Save the best model (by macro F1) to artifacts/model.pkl. "
                "Report each model's macro F1 score."
            ),
            expected_output=(
                "A table or list of all three models with their macro F1 scores, "
                "identification of the best model, and confirmation that "
                "model.pkl has been saved."
            ),
            agent=_trainer,
        )

        task_evaluate = Task(
            description=(
                "Use the 'Evaluate Models' tool to generate full evaluation artifacts: "
                "artifacts/evaluation_report.md (per-class precision, recall, F1 for all models), "
                "artifacts/confusion_matrix.png (for the best model), and "
                "artifacts/model_card.md (model name, version, intended use, limitations). "
                "Highlight any classes with low recall and suggest possible improvements."
            ),
            expected_output=(
                "A summary of evaluation results including the best model's macro F1, "
                "per-class metrics, any class with recall < 0.5 flagged, "
                "and confirmation that evaluation_report.md, model_card.md, "
                "and confusion_matrix.png have been saved."
            ),
            agent=_evaluator,
        )

        return Crew(
            agents=[_feature_eng, _trainer, _evaluator],
            tasks=[task_features, task_train, task_evaluate],
            process=Process.sequential,
            verbose=True,
        )
