"""Merge simple molecule and docking tables into normalized docking context."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping

from src.io.molecule_table_input import (
    MOLECULE_NAME_COLUMNS,
    normalize_molecule_name,
)

AFFINITY_COLUMNS = (
    "affinity",
    "docking_score",
    "score",
    "vina_score",
    "binding_affinity",
    "binding_energy",
)
RANK_COLUMNS = ("rank", "docking_rank")
NOTE_COLUMNS = ("note", "docking_note")
DOCKING_MERGE_REPORT_COLUMNS = (
    "report_type",
    "molecule",
    "molecule_normalized",
    "count",
    "status",
    "note",
)
SIMPLE_DOCKING_OUTPUT_COLUMNS = (
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
DEMO_DOCKING_NOTE = (
    "Illustrative computational docking-style value for workflow triage only; "
    "not real docking validation and not experimental binding affinity."
)
USER_DOCKING_NOTE = (
    "User-provided external docking result. The app did not run docking."
)


def _clean(value: object) -> str:
    return str(value or "").strip()


def _find_column(fieldnames: Iterable[str], candidates: Iterable[str]) -> str:
    by_normalized = {str(name).strip().lower(): str(name) for name in fieldnames}
    for candidate in candidates:
        if candidate in by_normalized:
            return by_normalized[candidate]
    return ""


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _standardized_molecule_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, str]]:
    output = []
    for row in rows:
        molecule = _clean(row.get("molecule") or row.get("molecule_id") or row.get("molecule_name"))
        smiles = _clean(row.get("canonical_smiles") or row.get("standardized_smiles") or row.get("smiles"))
        output.append(
            {
                "molecule": molecule,
                "molecule_id": _clean(row.get("molecule_id")) or molecule,
                "molecule_normalized": normalize_molecule_name(molecule),
                "standardized_smiles": smiles,
            }
        )
    return output


def _duplicate_keys(rows: Iterable[Mapping[str, str]], key: str) -> set[str]:
    counts: dict[str, int] = {}
    for row in rows:
        value = _clean(row.get(key))
        if value:
            counts[value] = counts.get(value, 0) + 1
    return {value for value, count in counts.items() if count > 1}


def _parse_float(value: object) -> float | None:
    try:
        return float(_clean(value))
    except ValueError:
        return None


def _computed_ranks(scores: list[tuple[int, float]], direction: str) -> dict[int, int]:
    reverse = direction.strip().lower() in {
        "higher is better",
        "higher_better",
        "higher",
        "larger is better",
    }
    ordered = sorted(scores, key=lambda item: item[1], reverse=reverse)
    return {index: rank for rank, (index, _) in enumerate(ordered, start=1)}


def build_docking_merge(
    *,
    molecule_rows: Iterable[Mapping[str, object]],
    docking_rows: Iterable[Mapping[str, object]],
    docking_program: str = "External docking",
    docking_units: str = "kcal/mol",
    docking_score_direction: str = "lower/more negative is better",
    selected_target_id: str = "",
    docking_source: str = "user_provided",
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Merge simple docking rows onto molecules by normalized molecule name."""
    molecules = _standardized_molecule_rows(molecule_rows)
    docking_materialized = list(docking_rows)
    fieldnames: set[str] = set()
    for row in docking_materialized:
        fieldnames.update(str(key) for key in row.keys())
    molecule_column = _find_column(fieldnames, MOLECULE_NAME_COLUMNS)
    affinity_column = _find_column(fieldnames, AFFINITY_COLUMNS)
    if docking_materialized and not molecule_column:
        raise ValueError(
            "Docking table is missing a molecule-name column. Accepted columns: "
            + ", ".join(MOLECULE_NAME_COLUMNS)
        )
    if docking_materialized and not affinity_column:
        raise ValueError(
            "Docking table is missing an affinity column. Accepted columns: "
            + ", ".join(AFFINITY_COLUMNS)
        )
    rank_column = _find_column(fieldnames, RANK_COLUMNS)
    note_column = _find_column(fieldnames, NOTE_COLUMNS)

    molecule_duplicates = _duplicate_keys(molecules, "molecule_normalized")
    docking_normalized = []
    scores_for_rank: list[tuple[int, float]] = []
    for index, row in enumerate(docking_materialized):
        molecule = _clean(row.get(molecule_column))
        key = normalize_molecule_name(molecule)
        score_text = _clean(row.get(affinity_column))
        score = _parse_float(score_text)
        rank = _clean(row.get(rank_column)) if rank_column else ""
        note = _clean(row.get(note_column)) if note_column else ""
        docking_normalized.append(
            {
                "index": str(index),
                "molecule": molecule,
                "molecule_normalized": key,
                "docking_score": score_text,
                "docking_rank": rank,
                "docking_note": note,
                "score_valid": str(score is not None),
            }
        )
        if score is not None and not rank:
            scores_for_rank.append((index, score))
    computed = _computed_ranks(scores_for_rank, docking_score_direction)
    for row in docking_normalized:
        if not row["docking_rank"] and int(row["index"]) in computed:
            row["docking_rank"] = str(computed[int(row["index"])])

    docking_by_key: dict[str, list[dict[str, str]]] = {}
    for row in docking_normalized:
        key = row["molecule_normalized"]
        if key:
            docking_by_key.setdefault(key, []).append(row)
    docking_duplicates = {key for key, rows in docking_by_key.items() if len(rows) > 1}

    normalized: list[dict[str, str]] = []
    report: list[dict[str, str]] = []
    matched_keys: set[str] = set()
    for molecule in molecules:
        key = molecule["molecule_normalized"]
        matches = docking_by_key.get(key, [])
        duplicate = key in molecule_duplicates or key in docking_duplicates
        if duplicate:
            report.append(
                {
                    "report_type": "duplicate_molecule_names",
                    "molecule": molecule["molecule"],
                    "molecule_normalized": key,
                    "count": str(len(matches) or 1),
                    "status": "ambiguous_duplicate",
                    "note": "Duplicate molecule names prevent unambiguous docking merge.",
                }
            )
        if not matches or duplicate:
            status = "ambiguous_duplicate" if duplicate else "no_docking_file_provided"
            note = (
                "Duplicate molecule name; docking merge was not applied."
                if duplicate
                else "No matching docking row was provided for this molecule."
            )
            normalized.append(
                _output_row(
                    molecule=molecule,
                    docking=None,
                    selected_target_id=selected_target_id,
                    docking_program=docking_program,
                    docking_units=docking_units,
                    docking_score_direction=docking_score_direction,
                    docking_source=docking_source,
                    status=status,
                    note=note,
                )
            )
            if not matches:
                report.append(
                    {
                        "report_type": "molecules_without_docking",
                        "molecule": molecule["molecule"],
                        "molecule_normalized": key,
                        "count": "1",
                        "status": status,
                        "note": note,
                    }
                )
            continue
        docking = matches[0]
        matched_keys.add(key)
        if docking["score_valid"].lower() != "true":
            status = "invalid_score"
            report.append(
                {
                    "report_type": "invalid_affinity_rows",
                    "molecule": docking["molecule"],
                    "molecule_normalized": key,
                    "count": "1",
                    "status": status,
                    "note": "Docking affinity is missing or non-numeric.",
                }
            )
        else:
            status = "available"
            report.append(
                {
                    "report_type": "matched_rows",
                    "molecule": docking["molecule"],
                    "molecule_normalized": key,
                    "count": "1",
                    "status": status,
                    "note": "Docking row matched molecule by normalized molecule name.",
                }
            )
        normalized.append(
            _output_row(
                molecule=molecule,
                docking=docking,
                selected_target_id=selected_target_id,
                docking_program=docking_program,
                docking_units=docking_units,
                docking_score_direction=docking_score_direction,
                docking_source=docking_source,
                status=status,
                note=(
                    DEMO_DOCKING_NOTE
                    if docking_source == "illustrative_demo"
                    else USER_DOCKING_NOTE
                ),
            )
        )

    molecule_keys = {row["molecule_normalized"] for row in molecules if row["molecule_normalized"]}
    for key, rows in docking_by_key.items():
        if key in molecule_keys:
            continue
        for row in rows:
            report.append(
                {
                    "report_type": "unmatched_docking_rows",
                    "molecule": row["molecule"],
                    "molecule_normalized": key,
                    "count": "1",
                    "status": "unmatched_docking",
                    "note": "Docking row did not match any uploaded molecule name.",
                }
            )
    return normalized, report


def _output_row(
    *,
    molecule: Mapping[str, str],
    docking: Mapping[str, str] | None,
    selected_target_id: str,
    docking_program: str,
    docking_units: str,
    docking_score_direction: str,
    docking_source: str,
    status: str,
    note: str,
) -> dict[str, str]:
    available = status == "available"
    return {
        "molecule": molecule.get("molecule", ""),
        "molecule_normalized": molecule.get("molecule_normalized", ""),
        "molecule_id": molecule.get("molecule_id", ""),
        "standardized_smiles": molecule.get("standardized_smiles", ""),
        "target_id": selected_target_id,
        "docking_score": _clean((docking or {}).get("docking_score")),
        "docking_rank": _clean((docking or {}).get("docking_rank")),
        "binding_site": "",
        "pose_file": "",
        "docking_program": docking_program,
        "docking_units": docking_units,
        "docking_score_direction": docking_score_direction,
        "docking_note": _clean((docking or {}).get("docking_note")) or note,
        "docking_source": docking_source,
        "docking_available": str(available),
        "target_docking_match": str(bool(available and selected_target_id)),
        "docking_status": status,
        "evidence_note": note,
    }


def write_simple_docking_outputs(
    *,
    molecule_rows: Iterable[Mapping[str, object]],
    docking_rows: Iterable[Mapping[str, object]],
    docking_output_path: Path,
    merge_report_path: Path,
    docking_program: str = "External docking",
    docking_units: str = "kcal/mol",
    docking_score_direction: str = "lower/more negative is better",
    selected_target_id: str = "",
    docking_source: str = "user_provided",
) -> dict[str, int]:
    """Write normalized docking and merge-report CSVs from simple tables."""
    normalized, report = build_docking_merge(
        molecule_rows=molecule_rows,
        docking_rows=docking_rows,
        docking_program=docking_program,
        docking_units=docking_units,
        docking_score_direction=docking_score_direction,
        selected_target_id=selected_target_id,
        docking_source=docking_source,
    )
    docking_output_path.parent.mkdir(parents=True, exist_ok=True)
    with docking_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SIMPLE_DOCKING_OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(normalized)
    merge_report_path.parent.mkdir(parents=True, exist_ok=True)
    with merge_report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DOCKING_MERGE_REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(report)
    return {"docking_results_normalized": len(normalized), "docking_merge_report": len(report)}


def write_simple_docking_outputs_from_csv(
    *,
    molecule_path: Path,
    docking_path: Path,
    docking_output_path: Path,
    merge_report_path: Path,
    docking_program: str = "External docking",
    docking_units: str = "kcal/mol",
    docking_score_direction: str = "lower/more negative is better",
    selected_target_id: str = "",
    docking_source: str = "user_provided",
) -> dict[str, int]:
    """Read simple molecule/docking CSV files and write app-normalized outputs."""
    return write_simple_docking_outputs(
        molecule_rows=_read_csv(molecule_path),
        docking_rows=_read_csv(docking_path),
        docking_output_path=docking_output_path,
        merge_report_path=merge_report_path,
        docking_program=docking_program,
        docking_units=docking_units,
        docking_score_direction=docking_score_direction,
        selected_target_id=selected_target_id,
        docking_source=docking_source,
    )
