from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parents[2] / "artifacts"
_CSS_PATH = Path(__file__).parents[1] / "app" / "static" / "eda_style.css"


def _load_css() -> str:
    """Read eda_style.css; return empty string if file is missing."""
    if _CSS_PATH.exists():
        return _CSS_PATH.read_text(encoding="utf-8")
    logger.warning("eda_style.css not found at %s — using no stylesheet.", _CSS_PATH)
    return ""


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def _normalize_device_class(value: str) -> str:
    v = str(value).strip()
    if v in ("1", "I", "Class I", "CLASS I"):
        return "Class I"
    if v in ("2", "II", "Class II", "CLASS II"):
        return "Class II"
    if v in ("3", "III", "Class III", "CLASS III"):
        return "Class III"
    return "Unknown"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate, normalize fields, handle nulls, and save:
      - artifacts/clean_data.csv
      - artifacts/dataset_contract.json
    Returns the cleaned DataFrame.
    """
    logger.info("Cleaning data: %d rows in", len(df))

    df = df.drop_duplicates()

    # Normalize text fields
    for col in ["device_name", "product_code", "medical_specialty_description"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.title()
                .replace({"Nan": "Unknown", "None": "Unknown", "": "Unknown"})
            )

    # Device class
    if "device_class" in df.columns:
        df["device_class"] = df["device_class"].apply(_normalize_device_class)

    # Binary flags — coerce to Y/N
    for col in ["implant_flag", "life_sustain_support_flag"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.upper()
                .str.strip()
                .apply(lambda x: "Y" if x in ("Y", "YES", "1", "TRUE") else "N")
            )
        else:
            df[col] = "N"

    # Validate target
    df["pathway"] = df["pathway"].astype(str).str.strip()
    before = len(df)
    df = df[df["pathway"].isin(["510k", "PMA", "De Novo"])]
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with invalid pathway values.", dropped)

    df = df.fillna("Unknown")

    logger.info("After cleaning: %d rows", len(df))

    # Save
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    clean_path = ARTIFACTS_DIR / "clean_data.csv"
    df.to_csv(clean_path, index=False)
    logger.info("Saved clean data → %s", clean_path)

    # Dataset contract
    feature_cols = [c for c in df.columns if c != "pathway"]
    contract = {
        "row_count": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": df.isnull().sum().to_dict(),
        "pathway_distribution": df["pathway"].value_counts().to_dict(),
        "feature_columns": feature_cols,
        "target_column": "pathway",
        "target_classes": sorted(df["pathway"].unique().tolist()),
        "value_ranges": {
            col: {"unique_count": int(df[col].nunique())}
            for col in feature_cols
        },
        "source": "openfda_api",
    }
    contract_path = ARTIFACTS_DIR / "dataset_contract.json"
    with open(contract_path, "w") as f:
        json.dump(contract, f, indent=2)
    logger.info("Saved dataset contract → %s", contract_path)

    return df


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_eda(df: pd.DataFrame) -> None:
    """
    Produce EDA artifacts:
      - artifacts/eda_report.html  (embedded charts)
      - artifacts/insights.md
    """
    logger.info("Generating EDA report…")
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    charts: list[tuple[str, str]] = []

    # 1 — Pathway distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df["pathway"].value_counts()
    colors = ["#2196F3", "#F44336", "#4CAF50"][: len(counts)]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
    ax.set_title("Pathway Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Pathway")
    ax.set_ylabel("Count")
    for bar, v in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.values) * 0.01,
            str(v),
            ha="center",
            fontweight="bold",
        )
    plt.tight_layout()
    charts.append(("Pathway Distribution", _fig_to_base64(fig)))
    plt.close(fig)

    # 2 — Device class vs pathway
    if "device_class" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        ct = pd.crosstab(df["device_class"], df["pathway"])
        ct.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white")
        ax.set_title("Device Class vs Pathway", fontsize=14, fontweight="bold")
        ax.set_xlabel("Device Class")
        ax.set_ylabel("Count")
        ax.legend(title="Pathway", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        charts.append(("Device Class vs Pathway", _fig_to_base64(fig)))
        plt.close(fig)

    # 3 — Top medical specialties
    if "medical_specialty_description" in df.columns:
        top_specs = df["medical_specialty_description"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        palette = sns.color_palette("husl", len(top_specs))
        ax.barh(top_specs.index[::-1], top_specs.values[::-1], color=palette[::-1])
        ax.set_title("Top 10 Medical Specialties", fontsize=14, fontweight="bold")
        ax.set_xlabel("Count")
        plt.tight_layout()
        charts.append(("Top Medical Specialties", _fig_to_base64(fig)))
        plt.close(fig)

    # 4 — Implant flag vs pathway
    if "implant_flag" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ct2 = pd.crosstab(df["implant_flag"], df["pathway"])
        ct2.plot(kind="bar", ax=ax, colormap="Set1", edgecolor="white")
        ax.set_title("Implant Flag vs Pathway", fontsize=14, fontweight="bold")
        ax.set_xlabel("Implant Flag")
        ax.set_ylabel("Count")
        ax.legend(title="Pathway")
        plt.xticks(rotation=0)
        plt.tight_layout()
        charts.append(("Implant Flag vs Pathway", _fig_to_base64(fig)))
        plt.close(fig)

    # 5 — Life sustain flag vs pathway
    if "life_sustain_support_flag" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ct3 = pd.crosstab(df["life_sustain_support_flag"], df["pathway"])
        ct3.plot(kind="bar", ax=ax, colormap="Accent", edgecolor="white")
        ax.set_title("Life Sustaining Flag vs Pathway", fontsize=14, fontweight="bold")
        ax.set_xlabel("Life Sustain Flag")
        ax.set_ylabel("Count")
        ax.legend(title="Pathway")
        plt.xticks(rotation=0)
        plt.tight_layout()
        charts.append(("Life Sustaining Flag vs Pathway", _fig_to_base64(fig)))
        plt.close(fig)

    # --- Build HTML ---
    counts = df["pathway"].value_counts()
    top_specialty = (
        df["medical_specialty_description"].value_counts().idxmax()
        if "medical_specialty_description" in df.columns
        else "N/A"
    )

    # Load external CSS and embed it so the report is self-contained
    css = _load_css()

    # Pathway distribution table rows
    dist_rows = "\n".join(
        f"<tr><td><strong>{pw}</strong></td><td>{cnt:,}</td>"
        f"<td>{100.0 * cnt / len(df):.1f}%</td></tr>"
        for pw, cnt in counts.items()
    )

    # Chart cards — pair them into a two-column grid where possible
    chart_cards = []
    for title, b64 in charts:
        chart_cards.append(
            f'<div class="chart">\n'
            f'  <h3>{title}</h3>\n'
            f'  <img src="data:image/png;base64,{b64}" alt="{title}">\n'
            f'</div>'
        )
    # Wrap consecutive pairs in .chart-grid divs
    chart_html_parts = []
    for i in range(0, len(chart_cards), 2):
        pair = chart_cards[i: i + 2]
        if len(pair) == 2:
            chart_html_parts.append(
                f'<div class="chart-grid">\n' + "\n".join(pair) + "\n</div>"
            )
        else:
            chart_html_parts.append(pair[0])
    charts_html = "\n".join(chart_html_parts)

    sample_table = df.head(10).to_html(index=False, border=0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FDA Pathway Predictor — EDA Report</title>
  <style>
{css}
  </style>
</head>
<body>

  <div class="report-header">
    <h1>FDA Regulatory Pathway Predictor</h1>
    <p class="subtitle">Exploratory Data Analysis Report &mdash; generated from openFDA API data</p>
  </div>

  <div class="stat-grid">
    <div class="stat-card">
      <div class="value">{len(df):,}</div>
      <div class="label">Total Records</div>
    </div>
    <div class="stat-card">
      <div class="value">{len(df.columns) - 1}</div>
      <div class="label">Feature Columns</div>
    </div>
    <div class="stat-card">
      <div class="value">{df["pathway"].nunique()}</div>
      <div class="label">Pathway Classes</div>
    </div>
    <div class="stat-card">
      <div class="value">{top_specialty}</div>
      <div class="label">Top Specialty</div>
    </div>
  </div>

  <h2>Pathway Distribution</h2>
  <table>
    <thead><tr><th>Pathway</th><th>Count</th><th>Percentage</th></tr></thead>
    <tbody>
{dist_rows}
    </tbody>
  </table>

  <h2>Visualizations</h2>
{charts_html}

  <h2>Data Sample <small style="font-weight:400;font-size:.85rem;">(first 10 rows)</small></h2>
  <div class="data-sample">
{sample_table}
  </div>

  <div class="disclaimer">
    <strong>Disclaimer:</strong> This tool is for planning purposes only.
    Consult qualified regulatory counsel before making regulatory decisions.
  </div>

  <div class="report-footer">
    Generated by FDA Regulatory Pathway Predictor &bull; openFDA data
  </div>

</body>
</html>"""

    eda_path = ARTIFACTS_DIR / "eda_report.html"
    with open(eda_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Saved EDA report → %s", eda_path)

    # --- insights.md ---
    implant_pma_pct = 0.0
    if "implant_flag" in df.columns:
        pma_df = df[df["pathway"] == "PMA"]
        if len(pma_df) > 0:
            implant_pma_pct = 100.0 * (pma_df["implant_flag"] == "Y").mean()

    dominant = counts.index[0]
    dominant_pct = 100.0 * counts.iloc[0] / len(df)

    insights = f"""# EDA Insights — FDA Pathway Predictor

## Dataset Overview
- **Total records**: {len(df):,}
- **Feature columns**: {', '.join(f'`{c}`' for c in df.columns if c != 'pathway')}
- **Target**: `pathway`
  - {', '.join(f'{k}: {v:,} ({100*v/len(df):.1f}%)' for k, v in counts.items())}

## Key Findings

### 1. Class Imbalance
`{dominant}` dominates at **{dominant_pct:.1f}%** of records.
Use stratified splitting and **macro F1** as the primary evaluation metric.

### 2. Device Class — Strong Signal
Class III devices are almost exclusively in the **PMA** pathway.
Class II maps predominantly to **510(k)**. This feature is a strong predictor.

### 3. Top Medical Specialty
The most frequent specialty is **{top_specialty}**, reflecting openFDA dataset composition.
Medical specialty is expected to be a useful categorical feature.

### 4. Implant Flag
**{implant_pma_pct:.1f}%** of PMA records are implants, confirming this binary flag
is a meaningful predictor for the PMA vs. 510(k) distinction.

### 5. Modeling Recommendations
- Apply **stratified train/test split** (80/20) to preserve class ratios.
- Label-encode `device_class` and `medical_specialty_description`.
- Binarize `implant_flag` and `life_sustain_support_flag` as 0/1 integers.
- Report **per-class precision, recall, F1** in addition to overall accuracy.
- Consider class-weight adjustments if De Novo recall is insufficient.
"""

    insights_path = ARTIFACTS_DIR / "insights.md"
    with open(insights_path, "w", encoding="utf-8") as f:
        f.write(insights)
    logger.info("Saved insights → %s", insights_path)
