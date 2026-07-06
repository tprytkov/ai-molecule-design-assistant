"""Normalize simple molecule-name/SMILES input tables."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, Mapping

MOLECULE_NAME_COLUMNS = (
    "molecule",
    "molecule_id",
    "molecule_name",
    "ligand",
    "compound",
    "name",
)
SMILES_COLUMNS = (
    "smiles",
    "smile",
    "canonical_smiles",
    "generated_smiles",
)
NORMALIZED_MOLECULE_COLUMNS = (
    "molecule",
    "molecule_normalized",
    "molecule_id",
    "smiles",
    "input_status",
    "error_message",
)


def normalize_molecule_name(value: object) -> str:
    """Return a stable, case-insensitive key for molecule-name matching."""
    text = str(value or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def _clean(value: object) -> str:
    return str(value or "").strip()


def _find_column(fieldnames: Iterable[str], candidates: Iterable[str]) -> str:
    by_normalized = {str(name).strip().lower(): str(name) for name in fieldnames}
    for candidate in candidates:
        if candidate in by_normalized:
            return by_normalized[candidate]
    return ""


def _first_value(row: Mapping[str, object], candidates: Iterable[str]) -> str:
    by_normalized = {str(key).strip().lower(): key for key in row.keys()}
    for candidate in candidates:
        key = by_normalized.get(candidate)
        if key is not None:
            value = _clean(row.get(key))
            if value:
                return value
    return ""


def normalize_molecule_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, str]]:
    """Normalize molecule rows while preserving invalid or incomplete entries."""
    materialized = list(rows)
    fieldnames: set[str] = set()
    for row in materialized:
        fieldnames.update(str(key) for key in row.keys())
    molecule_column = _find_column(fieldnames, MOLECULE_NAME_COLUMNS)
    smiles_column = _find_column(fieldnames, SMILES_COLUMNS)
    if not molecule_column:
        raise ValueError(
            "Molecule table is missing a molecule-name column. Accepted columns: "
            + ", ".join(MOLECULE_NAME_COLUMNS)
        )
    if not smiles_column:
        raise ValueError(
            "Molecule table is missing a SMILES column. Accepted columns: "
            + ", ".join(SMILES_COLUMNS)
        )

    normalized: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for row in materialized:
        molecule = _first_value(row, MOLECULE_NAME_COLUMNS)
        key = normalize_molecule_name(molecule)
        if key:
            counts[key] = counts.get(key, 0) + 1

    for row in materialized:
        molecule = _first_value(row, MOLECULE_NAME_COLUMNS)
        smiles = _first_value(row, SMILES_COLUMNS)
        key = normalize_molecule_name(molecule)
        status = "ready"
        errors = []
        if not molecule:
            status = "missing_molecule"
            errors.append("Molecule name is required.")
        if not smiles:
            status = "missing_smiles"
            errors.append("SMILES is required.")
        if key and counts.get(key, 0) > 1:
            status = "duplicate_molecule"
            errors.append("Duplicate molecule name; docking merge is ambiguous.")
        normalized.append(
            {
                "molecule": molecule,
                "molecule_normalized": key,
                "molecule_id": molecule,
                "smiles": smiles,
                "input_status": status,
                "error_message": " ".join(errors),
            }
        )
    return normalized


def normalize_molecule_csv(input_path: Path, output_path: Path) -> int:
    """Normalize a simple molecule CSV into the app's molecule_id/smiles schema."""
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    normalized = normalize_molecule_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NORMALIZED_MOLECULE_COLUMNS)
        writer.writeheader()
        writer.writerows(normalized)
    return len(normalized)
