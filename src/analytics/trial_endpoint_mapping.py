"""Conceptual trial endpoint mapping for translational positioning demos."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping


TRIAL_ENDPOINT_COLUMNS = (
    "molecule_id",
    "indication",
    "target_context",
    "endpoint_name",
    "endpoint_type",
    "conceptual_use",
    "mapping_note",
    "disclaimer",
)

TRIAL_ENDPOINT_DISCLAIMER = (
    "Conceptual translational mapping only; not a clinical trial recommendation, "
    "not medical advice, and not clinical evidence."
)


def read_endpoint_dictionary(path: Path) -> list[dict[str, str]]:
    """Read endpoint dictionary rows."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def build_trial_endpoint_map(
    *,
    molecule_rows: Iterable[Mapping[str, str]],
    endpoint_rows: Iterable[Mapping[str, str]],
    indication: str,
    target_context: str,
) -> list[dict[str, str]]:
    """Build conceptual endpoint mapping rows for each molecule."""
    molecules = [
        str(row.get("molecule_id", "")).strip()
        for row in molecule_rows
        if str(row.get("molecule_id", "")).strip()
    ] or ["project_level"]
    output = []
    for molecule_id in molecules:
        for endpoint in endpoint_rows:
            endpoint_name = str(endpoint.get("endpoint_name", "")).strip()
            if not endpoint_name:
                continue
            output.append(
                {
                    "molecule_id": molecule_id,
                    "indication": indication,
                    "target_context": target_context,
                    "endpoint_name": endpoint_name,
                    "endpoint_type": str(endpoint.get("endpoint_type", "")).strip(),
                    "conceptual_use": str(endpoint.get("conceptual_use", "")).strip(),
                    "mapping_note": str(endpoint.get("mapping_note", "")).strip(),
                    "disclaimer": TRIAL_ENDPOINT_DISCLAIMER,
                }
            )
    return output


def write_trial_endpoint_map_csv(path: Path, rows: Iterable[Mapping[str, str]]) -> int:
    """Write trial endpoint mapping CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_ENDPOINT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    return len(materialized)
