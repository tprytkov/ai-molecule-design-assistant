"""Create structural-property and docking-aware prioritization context outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping

from src.structural.docking_input import normalize_docking_csv
from src.target.target_schema import (
    TARGET_SOURCE_TARGET_SPECIFIC_DEMO,
    TARGET_SOURCE_USER,
)
from src.target.target_profile import TargetProfile, target_profile_csv

STRUCTURAL_PROPERTIES_COLUMNS = (
    "molecule",
    "molecule_id",
    "smiles",
    "molecular_weight",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "qed",
    "druglikeness_category",
    "bbb_prediction_label",
    "cns_property_flag",
    "toxicity_risk_flag",
    "admet_readiness_category",
    "best_reference_name",
    "tanimoto_similarity",
    "similarity_category",
    "docking_score",
    "docking_rank",
    "docking_program",
    "docking_units",
    "docking_score_direction",
    "docking_status",
    "docking_available",
    "binding_site",
    "docking_priority_label",
    "structural_priority_note",
    "target_id",
    "target_name",
    "pdb_id",
    "target_context_note",
)
STRUCTURAL_PRIORITIZATION_COLUMNS = (
    "molecule",
    "smiles",
    "molecule_id",
    "descriptor_available",
    "admet_available",
    "docking_available",
    "similarity_available",
    "public_lookup_available",
    "target_available",
    "target_docking_match",
    "docking_score",
    "docking_rank",
    "docking_program",
    "docking_units",
    "docking_score_direction",
    "docking_status",
    "docking_priority_label",
    "structural_priority_note",
    "evidence_note",
)
DOCKING_CONTEXT_COLUMNS = (
    "target_id",
    "target_name",
    "docking_available",
    "docking_score",
    "docking_rank",
    "docking_program",
    "docking_units",
    "docking_score_direction",
    "docking_priority_label",
    "structural_priority_note",
)
DOCKING_DISCLAIMER = (
    "Docking scores are computational triage only and are not experimental "
    "binding affinity, activity, selectivity, safety, or efficacy."
)


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _index(rows: Iterable[Mapping[str, str]], key: str = "molecule_id") -> dict[str, Mapping[str, str]]:
    return {str(row.get(key, "") or "").strip(): row for row in rows if str(row.get(key, "") or "").strip()}


def _clean(value: object) -> str:
    return str(value or "").strip()


def docking_priority_label(row: Mapping[str, str] | None) -> str:
    """Return a conservative docking-priority label."""
    if not row:
        return "docking_unavailable"
    status = _clean(row.get("docking_status"))
    if status == "target_mismatch":
        return "target_mismatch"
    if status != "available":
        return "docking_unavailable"
    try:
        score = float(_clean(row.get("docking_score")))
    except ValueError:
        return "weak_or_missing_docking_signal"
    if score <= -9.0:
        return "strong_docking_signal"
    if score <= -7.0:
        return "moderate_docking_signal"
    return "weak_or_missing_docking_signal"


def _target_available(profile: TargetProfile) -> bool:
    return (
        profile.target_id not in {"", "target_missing"}
        and profile.target_source in {TARGET_SOURCE_TARGET_SPECIFIC_DEMO, TARGET_SOURCE_USER}
    )


def structural_summary_csv(
    *,
    target_output_path: Path,
    structural_properties_path: Path,
    structural_prioritization_path: Path,
    descriptors_path: Path,
    admet_summary_path: Path | None = None,
    similarity_path: Path | None = None,
    public_lookup_path: Path | None = None,
    docking_input_path: Path | None = None,
    docking_output_path: Path | None = None,
    docking_merge_report_path: Path | None = None,
    docking_program: str = "External docking",
    docking_units: str = "kcal/mol",
    docking_score_direction: str = "lower/more negative is better",
    docking_source: str = "user_provided",
    target_source_path: Path | None = None,
    standardized_path: Path | None = None,
) -> dict[str, int]:
    """Write target, structural-property, and prioritization-input outputs."""
    profile = target_profile_csv(target_output_path, source_path=target_source_path)
    if docking_input_path and docking_input_path.exists() and docking_output_path and standardized_path:
        normalize_docking_csv(
            docking_input_path,
            docking_output_path,
            standardized_path=standardized_path,
            selected_target_id=profile.target_id,
            merge_report_path=docking_merge_report_path,
            docking_program=docking_program,
            docking_units=docking_units,
            docking_score_direction=docking_score_direction,
            docking_source=docking_source,
        )

    descriptors = _read_csv(descriptors_path)
    admet = _index(_read_csv(admet_summary_path))
    similarity = _index(_read_csv(similarity_path))
    public = _index(_read_csv(public_lookup_path))
    docking = _index(_read_csv(docking_output_path))

    properties: list[dict[str, str]] = []
    inputs: list[dict[str, str]] = []
    has_target = _target_available(profile)
    for descriptor in descriptors:
        molecule_id = _clean(descriptor.get("molecule_id"))
        admet_row = admet.get(molecule_id, {})
        similarity_row = similarity.get(molecule_id, {})
        docking_row = docking.get(molecule_id)
        docking_label = docking_priority_label(docking_row)
        docking_available = docking_label not in {"docking_unavailable", "target_mismatch"}
        target_match = _clean((docking_row or {}).get("target_docking_match"))
        structural_note = (
            "Docking context is shown separately; existing prioritization scores are unchanged. "
            + DOCKING_DISCLAIMER
        )
        properties.append(
            {
                "molecule": _clean(descriptor.get("molecule")) or molecule_id,
                "molecule_id": molecule_id,
                "smiles": _clean(descriptor.get("canonical_smiles")),
                "molecular_weight": _clean(descriptor.get("molecular_weight")),
                "logp": _clean(descriptor.get("logp")),
                "tpsa": _clean(descriptor.get("tpsa")),
                "hbd": _clean(descriptor.get("hbd")),
                "hba": _clean(descriptor.get("hba")),
                "rotatable_bonds": _clean(descriptor.get("rotatable_bonds")),
                "qed": _clean(descriptor.get("qed")),
                "druglikeness_category": _clean(descriptor.get("druglikeness_category")),
                "bbb_prediction_label": _clean(admet_row.get("bbb_prediction_label")),
                "cns_property_flag": _clean(admet_row.get("cns_property_flag")),
                "toxicity_risk_flag": _clean(admet_row.get("toxicity_risk_flag")),
                "admet_readiness_category": _clean(admet_row.get("admet_readiness_category")),
                "best_reference_name": _clean(similarity_row.get("best_reference_name")),
                "tanimoto_similarity": _clean(similarity_row.get("tanimoto_similarity")),
                "similarity_category": _clean(similarity_row.get("similarity_category")),
                "docking_score": _clean((docking_row or {}).get("docking_score")),
                "docking_rank": _clean((docking_row or {}).get("docking_rank")),
                "docking_program": _clean((docking_row or {}).get("docking_program")),
                "docking_units": _clean((docking_row or {}).get("docking_units")),
                "docking_score_direction": _clean((docking_row or {}).get("docking_score_direction")),
                "docking_status": _clean((docking_row or {}).get("docking_status")) or "docking_unavailable",
                "docking_available": str(docking_available),
                "binding_site": _clean((docking_row or {}).get("binding_site")),
                "docking_priority_label": docking_label,
                "structural_priority_note": structural_note,
                "target_id": profile.target_id,
                "target_name": profile.target_name,
                "pdb_id": profile.pdb_id,
                "target_context_note": profile.target_relevance_note,
            }
        )
        notes = [DOCKING_DISCLAIMER]
        if docking_label == "target_mismatch":
            notes.append("Docking row target_id did not match selected target.")
        elif docking_label == "docking_unavailable":
            notes.append("Docking evidence was unavailable or unmatched.")
        inputs.append(
            {
                "molecule": _clean(descriptor.get("molecule")) or molecule_id,
                "smiles": _clean(descriptor.get("canonical_smiles")),
                "molecule_id": molecule_id,
                "descriptor_available": str(bool(_clean(descriptor.get("molecular_weight")))),
                "admet_available": str(bool(admet_row)),
                "docking_available": str(docking_available),
                "similarity_available": str(bool(similarity_row)),
                "public_lookup_available": str(molecule_id in public),
                "target_available": str(has_target),
                "target_docking_match": target_match or "False",
                "docking_score": _clean((docking_row or {}).get("docking_score")),
                "docking_rank": _clean((docking_row or {}).get("docking_rank")),
                "docking_program": _clean((docking_row or {}).get("docking_program")),
                "docking_units": _clean((docking_row or {}).get("docking_units")),
                "docking_score_direction": _clean((docking_row or {}).get("docking_score_direction")),
                "docking_status": _clean((docking_row or {}).get("docking_status")) or "docking_unavailable",
                "docking_priority_label": docking_label,
                "structural_priority_note": structural_note,
                "evidence_note": " ".join(notes),
            }
        )

    structural_properties_path.parent.mkdir(parents=True, exist_ok=True)
    with structural_properties_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STRUCTURAL_PROPERTIES_COLUMNS)
        writer.writeheader()
        writer.writerows(properties)
    with structural_prioritization_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STRUCTURAL_PRIORITIZATION_COLUMNS)
        writer.writeheader()
        writer.writerows(inputs)
    return {"structural_properties": len(properties), "structural_prioritization_inputs": len(inputs)}


def add_structural_context_to_prioritization(
    prioritization_path: Path,
    structural_properties_path: Path,
) -> int:
    """Append docking/target context columns without changing existing scores."""
    prioritization = _read_csv(prioritization_path)
    structural = _index(_read_csv(structural_properties_path))
    if not prioritization:
        return 0
    fieldnames = list(prioritization[0].keys())
    for column in DOCKING_CONTEXT_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    updated = []
    for row in prioritization:
        molecule_id = _clean(row.get("molecule_id"))
        structural_row = structural.get(molecule_id, {})
        docking_label = _clean(structural_row.get("docking_priority_label")) or "docking_unavailable"
        row = dict(row)
        row.update(
            {
                "target_id": _clean(structural_row.get("target_id")),
                "target_name": _clean(structural_row.get("target_name")),
                "docking_available": str(docking_label not in {"docking_unavailable", "target_mismatch"}),
                "docking_score": _clean(structural_row.get("docking_score")),
                "docking_rank": _clean(structural_row.get("docking_rank")),
                "docking_program": _clean(structural_row.get("docking_program")),
                "docking_units": _clean(structural_row.get("docking_units")),
                "docking_score_direction": _clean(structural_row.get("docking_score_direction")),
                "docking_priority_label": docking_label,
                "structural_priority_note": (
                    "Docking context is shown separately; existing prioritization scores are unchanged. "
                    + DOCKING_DISCLAIMER
                ),
            }
        )
        updated.append(row)
    with prioritization_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(updated)
    return len(updated)
