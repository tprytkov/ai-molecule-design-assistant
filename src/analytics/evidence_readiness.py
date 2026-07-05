"""Evidence-readiness summaries for translational research positioning."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping


DISCLAIMER = (
    "Research and portfolio analytics only; not medical advice, not clinical "
    "evidence, not safety, efficacy, or toxicity evidence, and not legal advice "
    "or a patentability, novelty, FTO, infringement, or ownership determination."
)


READINESS_COLUMNS = (
    "molecule_id",
    "smiles",
    "descriptor_readiness",
    "public_lookup_readiness",
    "biomedical_evidence_readiness",
    "patent_context_readiness",
    "admet_readiness",
    "translational_readiness_category",
    "key_gaps",
    "recommended_next_steps",
    "disclaimer",
)


def status_text(value: object) -> str:
    """Normalize a status-like value."""
    return str(value if value is not None else "").strip().lower()


def index_by_molecule(rows: Iterable[Mapping[str, str]]) -> dict[str, Mapping[str, str]]:
    """Index rows by molecule ID, keeping the first row per molecule."""
    indexed: dict[str, Mapping[str, str]] = {}
    for row in rows:
        molecule_id = str(row.get("molecule_id", "")).strip()
        if molecule_id and molecule_id not in indexed:
            indexed[molecule_id] = row
    return indexed


def readiness_from_status(value: str, *, support_terms: tuple[str, ...]) -> str:
    """Return available/support/gap readiness from compact status text."""
    text = status_text(value)
    if not text or text in {"not_run", "not_available", "unavailable", "skipped"}:
        return "evidence_gap"
    if any(term in text for term in support_terms):
        return "available"
    if "error" in text or "invalid" in text:
        return "evidence_gap"
    return "available"


def descriptor_readiness(row: Mapping[str, str] | None) -> str:
    """Return descriptor readiness."""
    if not row:
        return "evidence_gap"
    if status_text(row.get("valid_smiles", "true")) in {"false", "0", "no"}:
        return "evidence_gap"
    if row.get("molecular_weight") or row.get("logp") or row.get("tpsa"):
        return "available"
    return "evidence_gap"


def public_lookup_readiness(row: Mapping[str, str] | None) -> str:
    """Return public-lookup readiness."""
    if not row:
        return "evidence_gap"
    statuses = " ".join(status_text(value) for value in row.values())
    if "match" in statuses or "found" in statuses:
        return "public_database_overlap"
    if "no_match" in statuses or "not_found" in statuses:
        return "available"
    return "evidence_gap"


def biomedical_readiness(row: Mapping[str, str] | None) -> str:
    """Return biomedical evidence readiness."""
    if not row:
        return "evidence_gap"
    status = str(row.get("biomedical_evidence_status") or row.get("nlp_status") or "")
    return readiness_from_status(status, support_terms=("available", "match"))


def patent_readiness(row: Mapping[str, str] | None) -> str:
    """Return patent/IP-context evidence readiness."""
    if not row:
        return "evidence_gap"
    status = " ".join(
        status_text(row.get(column, ""))
        for column in (
            "patent_evidence_status",
            "surechembl_structure_status",
            "patent_document_metadata_status",
        )
    )
    if "match" in status or "available" in status or "found" in status:
        return "available"
    if "unavailable" in status or "skipped" in status or not status.strip():
        return "evidence_gap"
    return "available"


def admet_readiness(row: Mapping[str, str] | None) -> str:
    """Return ADMET readiness."""
    if not row:
        return "evidence_gap"
    category = status_text(row.get("admet_readiness_category", ""))
    if category in {"favorable", "moderate", "caution", "unavailable"}:
        return category
    return "evidence_gap"


def translational_category(parts: Mapping[str, str]) -> str:
    """Return one conservative translational-readiness category."""
    if parts["admet_readiness"] == "caution":
        return "admet_caution"
    if "evidence_gap" in parts.values():
        return "translational_followup_needed"
    if parts["public_lookup_readiness"] == "public_database_overlap":
        return "public_database_overlap"
    return "discovery_research_fit"


def key_gaps(parts: Mapping[str, str]) -> str:
    """Summarize evidence gaps."""
    gaps = [
        label
        for label, value in parts.items()
        if value in {"evidence_gap", "unavailable"}
    ]
    return "; ".join(gaps) if gaps else "No major computational evidence gaps flagged."


def next_steps(category: str) -> str:
    """Return conservative next-step text."""
    if category == "admet_caution":
        return "Review ADMET descriptor cautions; consider endpoint-specific assays or validated models."
    if category == "public_database_overlap":
        return "Review public-database overlap and reference-neighbor similarity before follow-up."
    if category == "translational_followup_needed":
        return "Fill missing descriptor, public evidence, biomedical, patent-context, or ADMET triage outputs."
    return "Consider translational hypothesis review, target-context literature review, and experimental planning."


def build_evidence_readiness_rows(
    *,
    molecule_rows: Iterable[Mapping[str, str]],
    descriptors: Iterable[Mapping[str, str]],
    public_lookup: Iterable[Mapping[str, str]],
    biomedical: Iterable[Mapping[str, str]],
    patent: Iterable[Mapping[str, str]],
    admet_summary: Iterable[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Build evidence-readiness rows."""
    descriptor_by_id = index_by_molecule(descriptors)
    public_by_id = index_by_molecule(public_lookup)
    biomedical_by_id = index_by_molecule(biomedical)
    patent_by_id = index_by_molecule(patent)
    admet_by_id = index_by_molecule(admet_summary)
    rows = []
    for molecule in molecule_rows:
        molecule_id = str(molecule.get("molecule_id", "")).strip()
        if not molecule_id:
            continue
        smiles = str(
            molecule.get("canonical_smiles") or molecule.get("smiles") or ""
        ).strip()
        parts = {
            "descriptor_readiness": descriptor_readiness(descriptor_by_id.get(molecule_id)),
            "public_lookup_readiness": public_lookup_readiness(public_by_id.get(molecule_id)),
            "biomedical_evidence_readiness": biomedical_readiness(biomedical_by_id.get(molecule_id)),
            "patent_context_readiness": patent_readiness(patent_by_id.get(molecule_id)),
            "admet_readiness": admet_readiness(admet_by_id.get(molecule_id)),
        }
        category = translational_category(parts)
        rows.append(
            {
                "molecule_id": molecule_id,
                "smiles": smiles,
                **parts,
                "translational_readiness_category": category,
                "key_gaps": key_gaps(parts),
                "recommended_next_steps": next_steps(category),
                "disclaimer": DISCLAIMER,
            }
        )
    return rows


def write_evidence_readiness_csv(path: Path, rows: Iterable[Mapping[str, str]]) -> int:
    """Write evidence-readiness CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=READINESS_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    return len(materialized)
