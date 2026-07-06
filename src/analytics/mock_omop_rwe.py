"""Synthetic OMOP/RWE-style cohort summary for portfolio demos."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping


MOCK_RWE_COLUMNS = (
    "cohort_name",
    "n_mock_patients",
    "age_mean",
    "diagnosis_group",
    "medication_exposure_example",
    "endpoint_signal_available",
    "biomarker_available",
    "note",
)

MOCK_RWE_NOTE = (
    "Mock/synthetic OMOP-style demonstration data only; not real patient data "
    "and not patient-level RWE."
)


def read_mock_omop_rows(path: Path) -> list[dict[str, str]]:
    """Read synthetic OMOP-style demo rows."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def build_mock_rwe_summary(rows: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    """Return mock cohort summary rows with explicit synthetic labels."""
    output = []
    for row in rows:
        output.append(
            {
                "cohort_name": str(row.get("cohort_name", "")).strip(),
                "n_mock_patients": str(row.get("n_mock_patients", "")).strip(),
                "age_mean": str(row.get("age_mean", "")).strip(),
                "diagnosis_group": str(row.get("diagnosis_group", "")).strip(),
                "medication_exposure_example": str(
                    row.get("medication_exposure_example", "")
                ).strip(),
                "endpoint_signal_available": str(
                    row.get(
                        "endpoint_signal_available",
                        row.get("cognitive_endpoint_available", ""),
                    )
                ).strip(),
                "biomarker_available": str(row.get("biomarker_available", "")).strip(),
                "note": str(row.get("note", "")).strip() or MOCK_RWE_NOTE,
            }
        )
    return output


def write_mock_rwe_summary_csv(path: Path, rows: Iterable[Mapping[str, str]]) -> int:
    """Write mock RWE cohort summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MOCK_RWE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    return len(materialized)
