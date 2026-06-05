from __future__ import annotations

import json
import logging
import random
import string
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parents[2] / "artifacts"

DEVICE_NAMES = [
    "Cardiac Monitor",
    "Infusion Pump",
    "Surgical Stapler",
    "Glucose Meter",
    "Ventilator",
    "Pacemaker",
    "Defibrillator",
    "Coronary Stent",
    "Hip Implant",
    "Knee Prosthesis",
    "Urinary Catheter",
    "Blood Pressure Monitor",
    "Pulse Oximeter",
    "ECG Machine",
    "MRI Scanner",
    "Ultrasound Probe",
    "Dialysis Machine",
    "Insulin Pump",
    "Hearing Aid",
    "Cochlear Implant",
]

MEDICAL_SPECIALTIES = [
    "Cardiovascular",
    "Orthopedic",
    "Neurology",
    "General Hospital",
    "Radiology",
    "Oncology",
    "Ophthalmology",
    "General Plastic Surgery",
]

DEVICE_CLASS_MAP = {"510k": "Class II", "PMA": "Class III", "De Novo": "Class II"}


def _random_product_code(rng_random) -> str:
    """Generate a random 3-letter uppercase product code."""
    return "".join(rng_random.choice(list(string.ascii_uppercase), size=3))


def generate_sample_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate ~n synthetic device records with realistic pathway distributions:
      60% 510k, 30% PMA, 10% De Novo.

    Saves artifacts/clean_data.csv and artifacts/dataset_contract.json.
    Returns the DataFrame.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    pathways = rng.choice(
        ["510k", "PMA", "De Novo"], size=n, p=[0.60, 0.30, 0.10]
    )

    rows = []
    for pathway in pathways:
        # Device class mostly follows pathway with some noise
        if rng.random() > 0.08:
            device_class = DEVICE_CLASS_MAP[pathway]
        else:
            device_class = rng.choice(["Class I", "Class II", "Class III"])

        # Implant flag: PMA devices more likely to be implants
        if pathway == "PMA":
            implant = "Y" if rng.random() > 0.45 else "N"
        else:
            implant = "Y" if rng.random() > 0.88 else "N"

        # Life sustain flag: PMA/ventilators more likely
        if pathway == "PMA":
            life_sustain = "Y" if rng.random() > 0.65 else "N"
        else:
            life_sustain = "Y" if rng.random() > 0.93 else "N"

        rows.append({
            "device_name": random.choice(DEVICE_NAMES),
            "product_code": _random_product_code(rng),
            "device_class": device_class,
            "medical_specialty_description": random.choice(MEDICAL_SPECIALTIES),
            "implant_flag": implant,
            "life_sustain_support_flag": life_sustain,
            "pathway": pathway,
        })

    df = pd.DataFrame(rows)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    out_path = ARTIFACTS_DIR / "clean_data.csv"
    df.to_csv(out_path, index=False)
    logger.info("Generated %d synthetic records → %s", n, out_path)

    # Write dataset contract alongside
    contract = {
        "row_count": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": df.isnull().sum().to_dict(),
        "pathway_distribution": df["pathway"].value_counts().to_dict(),
        "feature_columns": [c for c in df.columns if c != "pathway"],
        "target_column": "pathway",
        "target_classes": list(df["pathway"].unique()),
        "source": "synthetic_fallback",
    }
    contract_path = ARTIFACTS_DIR / "dataset_contract.json"
    with open(contract_path, "w") as f:
        json.dump(contract, f, indent=2)
    logger.info("Saved dataset contract → %s", contract_path)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = generate_sample_data()
    print(df.head())
    print(df["pathway"].value_counts())
