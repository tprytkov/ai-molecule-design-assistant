"""Selection gates for endpoint-specific ADMET models."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Mapping

from src.model_source_status import MANIFEST_DIR

ALLOW_EXPERIMENTAL_ADMET_MODEL_USE_ENV = "ALLOW_EXPERIMENTAL_ADMET_MODEL_USE"
DEFAULT_ADMET_EVALUATION_SUMMARY_PATH = MANIFEST_DIR / "admet_model_evaluation_summary.csv"
VALIDATION_STATUS_NOT_EVALUATED = "not_evaluated"
VALIDATION_STATUS_BENCHMARK_PASSED = "benchmark_passed"
VALIDATION_STATUS_BENCHMARK_FAILED = "benchmark_failed"
VALIDATION_STATUS_EXPERIMENTAL = "experimental_public_model"
VALIDATION_STATUS_METADATA_INCOMPLETE = "metadata_incomplete"
VALIDATION_STATUS_UNAVAILABLE = "unavailable"

VALIDATION_STATUSES = {
    VALIDATION_STATUS_NOT_EVALUATED,
    VALIDATION_STATUS_BENCHMARK_PASSED,
    VALIDATION_STATUS_BENCHMARK_FAILED,
    VALIDATION_STATUS_EXPERIMENTAL,
    VALIDATION_STATUS_METADATA_INCOMPLETE,
    VALIDATION_STATUS_UNAVAILABLE,
}


def allow_experimental_admet_model_use() -> bool:
    """Return whether experimental endpoint-model use is explicitly enabled."""
    return os.environ.get(ALLOW_EXPERIMENTAL_ADMET_MODEL_USE_ENV) == "1"


def normalize_validation_status(value: object) -> str:
    """Return a known validation status string."""
    text = str(value or "").strip()
    return text if text in VALIDATION_STATUSES else VALIDATION_STATUS_NOT_EVALUATED


def validation_status_allows_prediction(
    validation_status: object,
    *,
    allow_experimental: bool = False,
) -> bool:
    """Return whether a tuned ADMET model may be used for app predictions."""
    status = normalize_validation_status(validation_status)
    return status == VALIDATION_STATUS_BENCHMARK_PASSED or (
        allow_experimental and status == VALIDATION_STATUS_EXPERIMENTAL
    )


def read_admet_evaluation_summary(path: Path | None = None) -> list[dict[str, str]]:
    """Read a compact ADMET model evaluation summary if it exists."""
    summary_path = path or DEFAULT_ADMET_EVALUATION_SUMMARY_PATH
    if not summary_path.exists():
        return []
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def validation_status_for_model(
    *,
    model_id: str,
    endpoint_name: str,
    summary_path: Path | None = None,
    rows: list[Mapping[str, str]] | None = None,
) -> str:
    """Return the latest validation status for a model/endpoint pair."""
    candidates = rows if rows is not None else read_admet_evaluation_summary(summary_path)
    for row in reversed(candidates):
        if str(row.get("model_id", "")).strip() != model_id:
            continue
        if str(row.get("endpoint_name", "")).strip() != endpoint_name:
            continue
        return normalize_validation_status(row.get("validation_status"))
    return VALIDATION_STATUS_NOT_EVALUATED
