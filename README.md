# FDA Regulatory Pathway Predictor

An end-to-end ML pipeline that predicts which FDA regulatory pathway (510(k), PMA, or De Novo) a medical device should follow, based on device characteristics.

## Quick Start

```bash
# 1. Install dependencies with uv
uv init fda-pathway-predictor   # already done
uv add crewai crewai-tools pandas scikit-learn matplotlib seaborn streamlit requests

# 2. Copy .env.example and add your OpenAI key (optional — pipeline works without it)
cp .env.example .env

# 3. Run the full pipeline
uv run python -m src.flow.main_flow

# 4. Launch the Streamlit app
uv run streamlit run src/app/streamlit_app.py
```

## Architecture

```
[Flow Start]
     │
     ▼
[Crew 1: Data Analyst]  ──► validation gate ──► fail → graceful fallback
     │
     ▼
[Crew 2: Data Scientist] ──► validation gate ──► log + exit cleanly
     │
     ▼
[Flow End: artifacts ready for Streamlit]
```

### Crew 1 — Data Analyst
| Agent | Role | Tools |
|-------|------|-------|
| Ingestion Agent | Fetch from openFDA API | requests, retry logic |
| Cleaning Agent | Deduplicate, normalize, handle nulls | Pandas |
| EDA Agent | Produce charts, stats, insights | Matplotlib, Seaborn |

### Crew 2 — Data Scientist
| Agent | Role | Tools |
|-------|------|-------|
| Feature Engineer | Encode categoricals, create features | Pandas, scikit-learn |
| Trainer Agent | Train 2+ model variations | scikit-learn |
| Evaluator Agent | Metrics, confusion matrix, model card | scikit-learn, Matplotlib |

## Artifacts Produced

| File | Description |
|------|-------------|
| `artifacts/raw_data.csv` | Raw openFDA records |
| `artifacts/clean_data.csv` | Cleaned, deduplicated data |
| `artifacts/eda_report.html` | Embedded charts + narrative |
| `artifacts/insights.md` | Key findings in markdown |
| `artifacts/dataset_contract.json` | Schema: column names, dtypes, null counts |
| `artifacts/features.csv` | Engineered feature matrix |
| `artifacts/model.pkl` | Best model (joblib serialized) |
| `artifacts/evaluation_report.md` | Accuracy, F1, confusion matrix |
| `artifacts/model_card.md` | Model details, limitations, intended use |
| `artifacts/confusion_matrix.png` | Confusion matrix image |

## Streamlit App Pages

- **Page 1 — Pathway Predictor**: Enter device details, get predicted pathway + confidence
- **Page 2 — EDA Dashboard**: Explore the training data with interactive charts
- **Page 3 — Model Performance**: Review model metrics, confusion matrix, and model card

## LLM Configuration

Set `OPENAI_API_KEY` in `.env` for full CrewAI agent mode.
Without a key, the pipeline runs in **direct mode** (same outputs, no LLM reasoning).

## Reproducibility

- All `random_state=42` on scikit-learn objects
- All artifact paths relative to project root (no hardcoded absolute paths)
- `dataset_contract.json` locks the schema between Crew 1 and Crew 2

## Git Strategy

```
main                        ← protected, merged via PR only
├── feature/crew1-data-analyst
├── feature/crew2-data-scientist
├── feature/streamlit-app
├── feature/flow-orchestration
└── fix/fda-api-corrections
```

## Disclaimer

> This tool is for planning purposes only. Consult qualified regulatory counsel before making regulatory decisions.
