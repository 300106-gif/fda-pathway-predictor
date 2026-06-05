from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

PAGE_LIMIT = 1000
ARTIFACTS_DIR = Path(__file__).parents[2] / "artifacts"

# Optional openFDA API key — 240 req/min with key vs 40 req/min without.
# Set FDA_API_KEY in .env to unlock higher rate limits.
_FDA_API_KEY: Optional[str] = os.getenv("FDA_API_KEY") or None


def _build_params(extra: dict) -> dict:
    """Return a params dict with api_key first (if configured), then extra keys."""
    params: dict = {}
    if _FDA_API_KEY:
        params["api_key"] = _FDA_API_KEY
    params.update(extra)
    return params


def _get(url: str, params: dict, retries: int = 3) -> Optional[dict]:
    """GET with exponential backoff. Returns None on 404 (no more pages)."""
    delays = [1, 2, 4]
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 404:
                logger.debug("404 received — no more pages.")
                return None
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            if attempt < retries - 1:
                logger.warning(
                    "Request failed (attempt %d/%d): %s. Retrying in %ds…",
                    attempt + 1, retries, exc, delays[attempt],
                )
                time.sleep(delays[attempt])
            else:
                logger.error("Request failed after %d attempts: %s", retries, exc)
                return None
    return None


def _paginate(url: str, search: Optional[str] = None, max_records: int = 2000) -> list[dict]:
    """Paginate through an openFDA endpoint, respecting the 25 000 skip limit."""
    if _FDA_API_KEY:
        logger.debug("Using FDA API key (240 req/min limit).")
    records: list[dict] = []
    skip = 0
    while len(records) < max_records:
        remaining = min(PAGE_LIMIT, max_records - len(records))
        extra: dict = {"limit": remaining, "skip": skip}
        if search:
            extra["search"] = search
        params = _build_params(extra)
        data = _get(url, params)
        if data is None:
            break
        batch = data.get("results", [])
        if not batch:
            break
        records.extend(batch)
        skip += PAGE_LIMIT
        if skip > 25_000:
            logger.info(
                "Reached openFDA skip limit (25 000). "
                "Use date-range chunking to fetch beyond this."
            )
            break
    logger.info("Fetched %d records from %s", len(records), url)
    return records


def fetch_510k(max_records: int = 2000) -> list[dict]:
    """Fetch 510(k) clearance records."""
    url = "https://api.fda.gov/device/510k.json"
    logger.info("Fetching 510(k) records (max %d)…", max_records)
    return _paginate(url, max_records=max_records)


def fetch_pma(max_records: int = 1000) -> list[dict]:
    """Fetch PMA (premarket approval) records."""
    url = "https://api.fda.gov/device/pma.json"
    logger.info("Fetching PMA records (max %d)…", max_records)
    return _paginate(url, max_records=max_records)


def fetch_classification(max_records: int = 500) -> list[dict]:
    """Fetch device classification records."""
    url = "https://api.fda.gov/device/classification.json"
    logger.info("Fetching classification records (max %d)…", max_records)
    return _paginate(url, max_records=max_records)


def build_raw_dataframe() -> pd.DataFrame:
    """
    Fetch 510(k) + PMA records, assign pathway labels, and save
    artifacts/raw_data.csv.  Returns the combined DataFrame.

    Pathway assignment rules:
      - 510(k) record with k_number starting "DEN" → pathway = "De Novo"
      - Other 510(k) records                        → pathway = "510k"
      - PMA records                                 → pathway = "PMA"
    """
    records_510k = fetch_510k(max_records=2000)
    records_pma = fetch_pma(max_records=1000)

    rows: list[dict] = []

    for r in records_510k:
        k_number = str(r.get("k_number", ""))
        pathway = "De Novo" if k_number.startswith("DEN") else "510k"
        rows.append({
            "device_name": r.get("device_name", ""),
            "product_code": r.get("product_code", ""),
            "device_class": r.get("device_class", ""),
            "medical_specialty_description": r.get(
                "medical_specialty_description", ""
            ),
            "decision_code": r.get("decision_code", ""),
            "implant_flag": r.get("implant_flag", "N"),
            "life_sustain_support_flag": r.get("life_sustain_support_flag", "N"),
            "k_number": k_number,
            "source": "510k",
            "pathway": pathway,
        })

    for r in records_pma:
        device_name = (
            r.get("trade_name")
            or r.get("device_name")
            or r.get("generic_name")
            or ""
        )
        rows.append({
            "device_name": device_name,
            "product_code": r.get("product_code", ""),
            "device_class": r.get("device_class", "III"),
            "medical_specialty_description": r.get(
                "advisory_committee_description",
                r.get("medical_specialty_description", ""),
            ),
            "decision_code": r.get("decision_code", ""),
            "implant_flag": r.get("implant_flag", "N"),
            "life_sustain_support_flag": r.get("life_sustain_support_flag", "N"),
            "k_number": "",
            "source": "pma",
            "pathway": "PMA",
        })

    df = pd.DataFrame(rows)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    out_path = ARTIFACTS_DIR / "raw_data.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved %d raw records → %s", len(df), out_path)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = build_raw_dataframe()
    print(df.head())
    print(df["pathway"].value_counts())
