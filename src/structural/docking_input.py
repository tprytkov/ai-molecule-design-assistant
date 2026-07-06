"""Normalize optional docking result CSV files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

from src.structural.simple_docking_input import (
    DOCKING_MERGE_REPORT_COLUMNS,
    SIMPLE_DOCKING_OUTPUT_COLUMNS,
    write_simple_docking_outputs,
)

DOCKING_REQUIRED_COLUMNS = (
    "molecule_id",
    "target_id",
    "docking_score",
    "docking_rank",
    "binding_site",
    "docking_program",
    "docking_note",
)
DOCKING_OUTPUT_COLUMNS = (
    "molecule",
    "molecule_normalized",
    "molecule_id",
    "standardized_smiles",
    "target_id",
    "docking_score",
    "docking_rank",
    "binding_site",
    "pose_file",
    "docking_program",
    "docking_units",
    "docking_score_direction",
    "docking_note",
    "docking_source",
    "docking_available",
    "target_docking_match",
    "docking_status",
    "evidence_note",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _clean(value: object) -> str:
    return str(value or "").strip()


def _standardized_indexes(standardized_path: Path) -> tuple[dict[str, Mapping[str, str]], dict[str, Mapping[str, str]]]:
    rows = _read_csv(standardized_path)
    by_id: dict[str, Mapping[str, str]] = {}
    by_smiles: dict[str, Mapping[str, str]] = {}
    for row in rows:
        molecule_id = _clean(row.get("molecule_id"))
        smiles = _clean(row.get("canonical_smiles") or row.get("standardized_smiles") or row.get("smiles"))
        if molecule_id:
            by_id[molecule_id] = row
        if smiles:
            by_smiles[smiles] = row
    return by_id, by_smiles


def _standardized_rows(standardized_path: Path) -> list[dict[str, str]]:
    return _read_csv(standardized_path)


def _write_merge_report(path: Path | None, rows: list[dict[str, str]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DOCKING_MERGE_REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def normalize_docking_rows(
    docking_rows: list[Mapping[str, str]],
    *,
    standardized_path: Path,
    selected_target_id: str = "",
    docking_program: str = "",
    docking_units: str = "",
    docking_score_direction: str = "",
    docking_source: str = "user_provided",
) -> list[dict[str, str]]:
    """Normalize docking rows and validate target consistency."""
    by_id, by_smiles = _standardized_indexes(standardized_path)
    normalized: list[dict[str, str]] = []
    for row in docking_rows:
        molecule_id = _clean(row.get("molecule_id"))
        smiles = _clean(row.get("standardized_smiles") or row.get("smiles"))
        standard_row = by_id.get(molecule_id) if molecule_id else None
        match_mode = "molecule_id"
        if standard_row is None and smiles:
            standard_row = by_smiles.get(smiles)
            match_mode = "standardized_smiles"
        standardized_smiles = _clean(
            (standard_row or {}).get("canonical_smiles")
            or (standard_row or {}).get("standardized_smiles")
            or smiles
        )
        if standard_row is not None:
            molecule_id = _clean((standard_row or {}).get("molecule_id")) or molecule_id

        target_id = _clean(row.get("target_id"))
        target_match = bool(target_id and selected_target_id and target_id == selected_target_id)
        if selected_target_id and target_id and target_id != selected_target_id:
            status = "target_mismatch"
            note = "Docking target_id does not match the selected target profile; row is not valid target evidence."
        elif standard_row is None:
            status = "molecule_not_matched"
            note = "Docking row could not be matched to standardized molecules by molecule_id or standardized SMILES."
        elif selected_target_id and not target_id:
            status = "target_missing"
            note = "Docking row matched a molecule but did not include target_id."
        else:
            status = "available"
            note = f"Docking row matched standardized molecule by {match_mode}."

        normalized.append(
            {
                "molecule": molecule_id,
                "molecule_normalized": molecule_id.strip().lower(),
                "molecule_id": molecule_id,
                "standardized_smiles": standardized_smiles,
                "target_id": target_id,
                "docking_score": _clean(row.get("docking_score")),
                "docking_rank": _clean(row.get("docking_rank")),
                "binding_site": _clean(row.get("binding_site")),
                "pose_file": _clean(row.get("pose_file")),
                "docking_program": _clean(row.get("docking_program")) or docking_program,
                "docking_units": _clean(row.get("docking_units")) or docking_units,
                "docking_score_direction": _clean(row.get("docking_score_direction")) or docking_score_direction,
                "docking_note": _clean(row.get("docking_note")),
                "docking_source": _clean(row.get("docking_source")) or docking_source,
                "docking_available": str(status == "available"),
                "target_docking_match": str(target_match),
                "docking_status": status,
                "evidence_note": note,
            }
        )
    return normalized


def normalize_docking_csv(
    docking_path: Path,
    output_path: Path,
    *,
    standardized_path: Path,
    selected_target_id: str = "",
    merge_report_path: Path | None = None,
    docking_program: str = "External docking",
    docking_units: str = "kcal/mol",
    docking_score_direction: str = "lower/more negative is better",
    docking_source: str = "user_provided",
) -> int:
    """Normalize an optional docking CSV into app-managed output schema."""
    with docking_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        lower_fieldnames = {field.lower() for field in fieldnames}
        if lower_fieldnames.intersection(
            {
                "affinity",
                "score",
                "vina_score",
                "binding_affinity",
                "binding_energy",
            }
        ):
            rows = [dict(row) for row in reader]
            counts = write_simple_docking_outputs(
                molecule_rows=_standardized_rows(standardized_path),
                docking_rows=rows,
                docking_output_path=output_path,
                merge_report_path=merge_report_path or output_path.with_name("docking_merge_report.csv"),
                docking_program=docking_program,
                docking_units=docking_units,
                docking_score_direction=docking_score_direction,
                selected_target_id=selected_target_id,
                docking_source=docking_source,
            )
            return counts["docking_results_normalized"]
        if "smiles" not in fieldnames and "standardized_smiles" not in fieldnames:
            raise ValueError("Docking CSV must contain smiles or standardized_smiles.")
        missing = set(DOCKING_REQUIRED_COLUMNS) - fieldnames
        if missing:
            raise ValueError(
                "Docking CSV is missing required columns: "
                + ", ".join(sorted(missing))
            )
        rows = [dict(row) for row in reader]
    normalized = normalize_docking_rows(
        rows,
        standardized_path=standardized_path,
        selected_target_id=selected_target_id,
        docking_program=docking_program,
        docking_units=docking_units,
        docking_score_direction=docking_score_direction,
        docking_source=docking_source,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DOCKING_OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(normalized)
    _write_merge_report(
        merge_report_path,
        [
            {
                "report_type": "matched_rows" if row["docking_status"] == "available" else row["docking_status"],
                "molecule": row["molecule"],
                "molecule_normalized": row["molecule_normalized"],
                "count": "1",
                "status": row["docking_status"],
                "note": row["evidence_note"],
            }
            for row in normalized
        ],
    )
    return len(normalized)
