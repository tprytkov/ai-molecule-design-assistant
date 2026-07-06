"""Biopharma analytics and translational positioning output generation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping

from src.analytics.evidence_readiness import (
    DISCLAIMER,
    build_evidence_readiness_rows,
    write_evidence_readiness_csv,
)
from src.analytics.mock_omop_rwe import (
    build_mock_rwe_summary,
    read_mock_omop_rows,
    write_mock_rwe_summary_csv,
)
from src.analytics.trial_endpoint_mapping import (
    build_trial_endpoint_map,
    read_endpoint_dictionary,
    write_trial_endpoint_map_csv,
)


DEMO_BIOPHARMA_DIR = Path("data/demo_biopharma")
POSITIONING_COLUMNS = (
    "molecule_id",
    "smiles",
    "indication",
    "target_context",
    "positioning_category",
    "public_database_differentiation_signal",
    "reference_neighbor_similarity",
    "biomedical_context_signal",
    "patent_associated_structure_crowding",
    "admet_signal",
    "translational_positioning_summary",
    "disclaimer",
)


def read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    """Read CSV rows from an optional path."""
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Iterable[Mapping[str, str]], columns: tuple[str, ...]) -> int:
    """Write rows to a fixed-schema CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    return len(materialized)


def index_by_molecule(rows: Iterable[Mapping[str, str]]) -> dict[str, Mapping[str, str]]:
    """Index rows by molecule ID."""
    indexed = {}
    for row in rows:
        molecule_id = str(row.get("molecule_id", "")).strip()
        if molecule_id and molecule_id not in indexed:
            indexed[molecule_id] = row
    return indexed


def molecule_rows_from_sources(*sources: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    """Collect molecule rows from available upstream outputs."""
    rows_by_id: dict[str, dict[str, str]] = {}
    for rows in sources:
        for row in rows:
            molecule_id = str(row.get("molecule_id", "")).strip()
            if not molecule_id:
                continue
            current = rows_by_id.setdefault(molecule_id, {"molecule_id": molecule_id})
            for key, value in row.items():
                if str(value or "").strip() and key not in current:
                    current[key] = str(value)
    return [rows_by_id[key] for key in sorted(rows_by_id)]


def read_indication_metadata(path: Path) -> dict[str, str]:
    """Read the first indication metadata row."""
    rows = read_csv_rows(path)
    if not rows:
        return {
            "indication": "selected molecule set / target context",
            "target_context": "user-provided or demo target context",
            "demo_framing": "Public-safe translational positioning demo.",
        }
    return rows[0]


def status(value: object) -> str:
    """Normalize compact status text."""
    return str(value if value is not None else "").strip().lower()


def public_signal(row: Mapping[str, str] | None) -> str:
    """Return public-database differentiation signal."""
    if not row:
        return "evidence_gap"
    text = " ".join(status(value) for value in row.values())
    if "match" in text and "no_match" not in text:
        return "public_database_overlap"
    if "no_match" in text or "not_found" in text:
        return "public-database differentiation signal"
    return "evidence_gap"


def similarity_signal(row: Mapping[str, str] | None) -> str:
    """Return reference-neighbor similarity signal."""
    if not row:
        return "evidence_gap"
    score_text = str(row.get("tanimoto_similarity", "")).strip()
    try:
        score = float(score_text)
    except ValueError:
        return "reference-neighbor similarity unavailable"
    if score >= 0.70:
        return "close reference-neighbor similarity"
    if score >= 0.40:
        return "moderate reference-neighbor similarity"
    return "distant reference-neighbor similarity"


def biomedical_signal(row: Mapping[str, str] | None) -> str:
    """Return biomedical evidence support signal."""
    if not row:
        return "evidence_gap"
    text = status(row.get("biomedical_evidence_status", ""))
    if text == "available":
        return "biomedical_evidence_support"
    if text in {"skipped", "model_unavailable", "not_run", ""}:
        return "evidence_gap"
    return text or "evidence_gap"


def patent_crowding_signal(row: Mapping[str, str] | None) -> str:
    """Return patent-associated structure crowding signal."""
    if not row:
        return "evidence_gap"
    text = " ".join(status(value) for value in row.values())
    if "match" in text or "available" in text or "found" in text:
        return "patent_associated_structure_crowding"
    if "no_match" in text:
        return "no patent-associated structure crowding flagged"
    return "evidence_gap"


def admet_signal(row: Mapping[str, str] | None) -> str:
    """Return ADMET triage signal."""
    if not row:
        return "evidence_gap"
    category = status(row.get("admet_readiness_category", ""))
    return "admet_caution" if category == "caution" else (category or "evidence_gap")


def positioning_category(signals: Iterable[str]) -> str:
    """Return one or more conservative positioning categories."""
    values = set(signals)
    categories = []
    if "biomedical_evidence_support" in values:
        categories.append("biomedical_evidence_support")
    if "public_database_overlap" in values:
        categories.append("public_database_overlap")
    if "patent_associated_structure_crowding" in values:
        categories.append("patent_associated_structure_crowding")
    if "admet_caution" in values:
        categories.append("admet_caution")
    if "evidence_gap" in values:
        categories.append("evidence_gap")
    if not categories:
        categories.append("discovery_research_fit")
    if "evidence_gap" in categories or "admet_caution" in categories:
        categories.append("translational_followup_needed")
    return "; ".join(dict.fromkeys(categories))


def build_positioning_rows(
    *,
    molecule_rows: Iterable[Mapping[str, str]],
    public_lookup: Iterable[Mapping[str, str]],
    similarity: Iterable[Mapping[str, str]],
    biomedical: Iterable[Mapping[str, str]],
    patent: Iterable[Mapping[str, str]],
    admet_summary: Iterable[Mapping[str, str]],
    indication: str,
    target_context: str,
) -> list[dict[str, str]]:
    """Build biopharma positioning rows."""
    public_by_id = index_by_molecule(public_lookup)
    similarity_by_id = index_by_molecule(similarity)
    biomedical_by_id = index_by_molecule(biomedical)
    patent_by_id = index_by_molecule(patent)
    admet_by_id = index_by_molecule(admet_summary)
    rows = []
    for molecule in molecule_rows:
        molecule_id = str(molecule.get("molecule_id", "")).strip()
        smiles = str(molecule.get("canonical_smiles") or molecule.get("smiles") or "").strip()
        signals = {
            "public_database_differentiation_signal": public_signal(public_by_id.get(molecule_id)),
            "reference_neighbor_similarity": similarity_signal(similarity_by_id.get(molecule_id)),
            "biomedical_context_signal": biomedical_signal(biomedical_by_id.get(molecule_id)),
            "patent_associated_structure_crowding": patent_crowding_signal(patent_by_id.get(molecule_id)),
            "admet_signal": admet_signal(admet_by_id.get(molecule_id)),
        }
        category = positioning_category(signals.values())
        rows.append(
            {
                "molecule_id": molecule_id,
                "smiles": smiles,
                "indication": indication,
                "target_context": target_context,
                "positioning_category": category,
                **signals,
                "translational_positioning_summary": (
                    "Translational positioning summary for the selected molecule "
                    "set and target context based on existing public-safe "
                    "computational outputs."
                ),
                "disclaimer": DISCLAIMER,
            }
        )
    return rows


def write_biopharma_report(
    path: Path,
    *,
    indication: str,
    target_context: str,
    positioning_rows: list[Mapping[str, str]],
    readiness_rows: list[Mapping[str, str]],
) -> None:
    """Write a compact Markdown biopharma summary report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    category_counts: dict[str, int] = {}
    for row in positioning_rows:
        for category in str(row.get("positioning_category", "")).split(";"):
            clean = category.strip()
            if clean:
                category_counts[clean] = category_counts.get(clean, 0) + 1
    lines = [
        "# Biopharma Analytics Summary",
        "",
        f"Indication framing: {indication}",
        f"Target context: {target_context}",
        "",
        "This is research and portfolio analytics only. It is not medical advice, "
        "clinical evidence, safety evidence, efficacy evidence, toxicity evidence, "
        "legal advice, or a patentability, novelty, FTO, infringement, or ownership "
        "determination.",
        "",
        "## Positioning Category Counts",
    ]
    if category_counts:
        lines.extend(f"- {key}: {value}" for key, value in sorted(category_counts.items()))
    else:
        lines.append("- No molecule-level positioning rows were available.")
    lines.extend(
        [
            "",
            "## Translational Evidence Readiness",
            f"- Molecules reviewed: {len(readiness_rows)}",
            "- Mock OMOP/RWE content in this demo is synthetic and not real patient data.",
            "- Trial endpoint mapping is conceptual translational mapping only, not a clinical trial recommendation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_biopharma_outputs(
    *,
    output_dir: Path,
    demo_data_dir: Path = DEMO_BIOPHARMA_DIR,
    standardized_path: Path | None = None,
    descriptors_path: Path | None = None,
    similarity_path: Path | None = None,
    public_lookup_path: Path | None = None,
    biomedical_path: Path | None = None,
    patent_path: Path | None = None,
    prioritization_path: Path | None = None,
    admet_summary_path: Path | None = None,
    positioning_path: Path | None = None,
    readiness_path: Path | None = None,
    mock_rwe_path: Path | None = None,
    trial_endpoint_path: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, int]:
    """Generate all biopharma analytics outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    standardized = read_csv_rows(standardized_path or output_dir / "standardized.csv")
    descriptors = read_csv_rows(descriptors_path or output_dir / "descriptors.csv")
    similarity = read_csv_rows(similarity_path or output_dir / "similarity.csv")
    public_lookup = read_csv_rows(public_lookup_path or output_dir / "public_lookup.csv")
    biomedical = read_csv_rows(biomedical_path or output_dir / "biomedical_evidence.csv")
    patent = read_csv_rows(patent_path or output_dir / "patent_evidence_embeddings.csv")
    prioritization = read_csv_rows(prioritization_path or output_dir / "prioritization_results.csv")
    admet_summary = read_csv_rows(admet_summary_path or output_dir / "admet_summary.csv")
    molecules = molecule_rows_from_sources(standardized, prioritization, descriptors)
    metadata = read_indication_metadata(demo_data_dir / "indication_metadata.csv")
    indication = str(metadata.get("indication", "selected molecule set / target context")).strip()
    target_context = str(
        metadata.get("target_context", "user-provided or demo target context")
    ).strip()
    positioning_rows = build_positioning_rows(
        molecule_rows=molecules,
        public_lookup=public_lookup,
        similarity=similarity,
        biomedical=biomedical,
        patent=patent,
        admet_summary=admet_summary,
        indication=indication,
        target_context=target_context,
    )
    readiness_rows = build_evidence_readiness_rows(
        molecule_rows=molecules,
        descriptors=descriptors,
        public_lookup=public_lookup,
        biomedical=biomedical,
        patent=patent,
        admet_summary=admet_summary,
    )
    mock_rwe_rows = build_mock_rwe_summary(
        read_mock_omop_rows(demo_data_dir / "mock_translational_omop_cohort.csv")
    )
    endpoint_rows = build_trial_endpoint_map(
        molecule_rows=molecules,
        endpoint_rows=read_endpoint_dictionary(demo_data_dir / "endpoint_dictionary.csv"),
        indication=indication,
        target_context=target_context,
    )
    positioning_path = positioning_path or output_dir / "biopharma_positioning.csv"
    readiness_path = readiness_path or output_dir / "evidence_readiness.csv"
    mock_rwe_path = mock_rwe_path or output_dir / "mock_rwe_cohort_summary.csv"
    trial_endpoint_path = trial_endpoint_path or output_dir / "trial_endpoint_map.csv"
    report_path = report_path or output_dir / "biopharma_summary_report.md"
    write_biopharma_report(
        report_path,
        indication=indication,
        target_context=target_context,
        positioning_rows=positioning_rows,
        readiness_rows=readiness_rows,
    )
    return {
        "biopharma_positioning": write_csv(positioning_path, positioning_rows, POSITIONING_COLUMNS),
        "evidence_readiness": write_evidence_readiness_csv(readiness_path, readiness_rows),
        "mock_rwe_cohort_summary": write_mock_rwe_summary_csv(mock_rwe_path, mock_rwe_rows),
        "trial_endpoint_map": write_trial_endpoint_map_csv(trial_endpoint_path, endpoint_rows),
        "biopharma_summary_report": 1,
    }
