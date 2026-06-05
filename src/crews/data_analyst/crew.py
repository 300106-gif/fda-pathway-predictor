from __future__ import annotations

import logging

from crewai import Crew, Process, Task

from src.crews.data_analyst.agents import (
    cleaning_agent,
    eda_agent,
    ingestion_agent,
)

logger = logging.getLogger(__name__)


class DataAnalystCrew:
    """Crew 1 — ingests raw openFDA data, cleans it, and produces EDA artifacts."""

    def crew(self) -> Crew:
        _ingestion = ingestion_agent()
        _cleaning = cleaning_agent()
        _eda = eda_agent()

        task_ingest = Task(
            description=(
                "Use the 'Fetch FDA Data' tool to retrieve medical device records "
                "from the openFDA 510(k) and PMA endpoints. "
                "The tool will save the raw records to artifacts/raw_data.csv. "
                "Report how many records were fetched and the pathway distribution."
            ),
            expected_output=(
                "A confirmation message stating the number of records fetched "
                "and the pathway distribution saved to artifacts/raw_data.csv."
            ),
            agent=_ingestion,
        )

        task_clean = Task(
            description=(
                "Use the 'Clean FDA Data' tool to deduplicate and normalize "
                "the raw records in artifacts/raw_data.csv. "
                "The tool will save artifacts/clean_data.csv and "
                "artifacts/dataset_contract.json. "
                "Report the before/after row counts and any data quality issues found."
            ),
            expected_output=(
                "A cleaning summary with row counts, normalization steps applied, "
                "and confirmation that clean_data.csv and dataset_contract.json exist."
            ),
            agent=_cleaning,
        )

        task_eda = Task(
            description=(
                "Use the 'Generate EDA Report' tool to analyze artifacts/clean_data.csv. "
                "The tool will produce artifacts/eda_report.html with embedded charts "
                "and artifacts/insights.md with key findings. "
                "Summarize the top 3 insights relevant to pathway prediction."
            ),
            expected_output=(
                "A summary of the EDA findings including pathway distribution, "
                "strongest predictive features, and class imbalance notes. "
                "Confirm that eda_report.html and insights.md have been saved."
            ),
            agent=_eda,
        )

        return Crew(
            agents=[_ingestion, _cleaning, _eda],
            tasks=[task_ingest, task_clean, task_eda],
            process=Process.sequential,
            verbose=True,
        )
