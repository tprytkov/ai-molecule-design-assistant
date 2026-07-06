"""Answer compound-intelligence questions from existing local CSV outputs."""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from rdkit import Chem
from rdkit.Chem import Draw


DEFAULT_PRIORITIZED = Path("outputs/prioritization_results.csv")
DEFAULT_SIMILARITY = Path("outputs/similarity_top_hits.csv")
DEFAULT_PUBLIC_LOOKUP = Path("outputs/public_lookup.csv")
DEFAULT_COMPOUND_CONTEXT = Path("outputs/compound_context.csv")
DEFAULT_CHEMICAL_IDENTITY = Path("outputs/chemical_identity.csv")
DEFAULT_NLP = Path("outputs/text_nlp.csv")
DEFAULT_DESCRIPTORS = Path("outputs/descriptors.csv")
DEFAULT_SURECHEMBL = Path("outputs/surechembl_evidence.csv")
DEFAULT_VISUALIZATION = Path("outputs/visualization_coordinates.csv")
LEGACY_PATHS = {
    DEFAULT_PRIORITIZED: Path("outputs/prioritized_with_nlp_demo.csv"),
    DEFAULT_SIMILARITY: Path("outputs/similarity_top_hits_demo.csv"),
    DEFAULT_PUBLIC_LOOKUP: Path("outputs/public_lookup_demo.csv"),
    DEFAULT_COMPOUND_CONTEXT: Path("outputs/compound_context_demo.csv"),
    DEFAULT_NLP: Path("outputs/text_nlp_demo.csv"),
    DEFAULT_DESCRIPTORS: Path("outputs/descriptors_demo.csv"),
    DEFAULT_SURECHEMBL: Path("outputs/surechembl_lookup_demo.csv"),
    DEFAULT_VISUALIZATION: Path("outputs/visualization_coordinates_demo.csv"),
}
LEGACY_FILENAMES = {
    "prioritization_results.csv": "prioritized_with_nlp_demo.csv",
    "similarity_top_hits.csv": "similarity_top_hits_demo.csv",
    "public_lookup.csv": "public_lookup_demo.csv",
    "compound_context.csv": "compound_context_demo.csv",
    "text_nlp.csv": "text_nlp_demo.csv",
    "descriptors.csv": "descriptors_demo.csv",
    "surechembl_evidence.csv": "surechembl_lookup_demo.csv",
    "visualization_coordinates.csv": "visualization_coordinates_demo.csv",
}

SUPPORTED_QUESTIONS = (
    "is_known",
    "closest_public_compounds",
    "why_ranked",
    "ip_potential_summary",
    "full_report",
)


@dataclass(frozen=True)
class CompoundEvidence:
    """Local evidence associated with one generated molecule."""

    prioritized: Mapping[str, str]
    descriptor: Mapping[str, str]
    similarity_hits: tuple[Mapping[str, str], ...]
    public_lookup: tuple[Mapping[str, str], ...]
    compound_context: Mapping[str, str]
    nlp_evidence: tuple[Mapping[str, str], ...]
    surechembl_evidence: tuple[Mapping[str, str], ...]
    visualization_row: Mapping[str, str]
    visualization_rows: tuple[Mapping[str, str], ...]
    chemical_identity: Mapping[str, str] = field(default_factory=dict)


def parse_boolean(value: object) -> bool:
    """Interpret common CSV boolean representations."""
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as dictionaries."""
    active_path = resolve_input_path(path)
    with active_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        return [dict(row) for row in csv.DictReader(input_file)]


def read_optional_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file when present, otherwise return no rows."""
    active_path = resolve_input_path(path)
    if not active_path.exists():
        return []
    return read_csv(active_path)


def resolve_input_path(path: Path) -> Path:
    """Resolve a current output path, with read-only fallback to legacy names."""
    if path.exists():
        return path
    legacy_path = LEGACY_PATHS.get(path)
    if legacy_path is not None and legacy_path.exists():
        return legacy_path
    legacy_name = LEGACY_FILENAMES.get(path.name)
    if legacy_name:
        candidate = path.with_name(legacy_name)
        if candidate.exists():
            return candidate
    return path


def find_unique_row(
    rows: Iterable[Mapping[str, str]], molecule_id: str, label: str
) -> Mapping[str, str]:
    """Find one molecule row and reject missing or duplicate records."""
    matches = [
        row
        for row in rows
        if row.get("molecule_id", "").strip() == molecule_id
    ]
    if not matches:
        raise ValueError(f"Molecule ID '{molecule_id}' was not found in {label}.")
    if len(matches) > 1:
        raise ValueError(
            f"Molecule ID '{molecule_id}' appears multiple times in {label}."
        )
    return matches[0]


def matching_rows(
    rows: Iterable[Mapping[str, str]], molecule_id: str
) -> tuple[Mapping[str, str], ...]:
    """Return all rows matching a molecule ID."""
    return tuple(
        row
        for row in rows
        if row.get("molecule_id", "").strip() == molecule_id
    )


def load_compound_evidence(
    molecule_id: str,
    *,
    prioritized_path: Path = DEFAULT_PRIORITIZED,
    similarity_path: Path = DEFAULT_SIMILARITY,
    public_lookup_path: Path = DEFAULT_PUBLIC_LOOKUP,
    compound_context_path: Path | None = None,
    chemical_identity_path: Path | None = None,
    nlp_path: Path = DEFAULT_NLP,
    descriptor_path: Path = DEFAULT_DESCRIPTORS,
    surechembl_path: Path = DEFAULT_SURECHEMBL,
    visualization_path: Path = DEFAULT_VISUALIZATION,
) -> CompoundEvidence:
    """Load all local evidence for one molecule."""
    prioritized_rows = read_csv(prioritized_path)
    descriptor_rows = read_csv(descriptor_path)
    similarity_rows = read_csv(similarity_path)
    public_rows = read_csv(public_lookup_path)
    active_context_path = (
        compound_context_path
        or prioritized_path.parent / "compound_context.csv"
    )
    context_rows = read_optional_csv(active_context_path)
    active_identity_path = (
        chemical_identity_path
        or prioritized_path.parent / "chemical_identity.csv"
    )
    identity_rows = read_optional_csv(active_identity_path)
    nlp_rows = read_optional_csv(nlp_path)
    surechembl_rows = read_optional_csv(surechembl_path)
    visualization_rows = read_optional_csv(visualization_path)
    visualization_matches = matching_rows(visualization_rows, molecule_id)

    return CompoundEvidence(
        prioritized=find_unique_row(
            prioritized_rows, molecule_id, prioritized_path.name
        ),
        descriptor=find_unique_row(
            descriptor_rows, molecule_id, descriptor_path.name
        ),
        similarity_hits=matching_rows(similarity_rows, molecule_id),
        public_lookup=matching_rows(public_rows, molecule_id),
        compound_context=(
            matching_rows(context_rows, molecule_id)[0]
            if matching_rows(context_rows, molecule_id)
            else {}
        ),
        chemical_identity=(
            matching_rows(identity_rows, molecule_id)[0]
            if matching_rows(identity_rows, molecule_id)
            else {}
        ),
        nlp_evidence=matching_rows(nlp_rows, molecule_id),
        surechembl_evidence=matching_rows(surechembl_rows, molecule_id),
        visualization_row=visualization_matches[0] if visualization_matches else {},
        visualization_rows=tuple(visualization_rows),
    )


def format_value(value: object, fallback: str = "not available") -> str:
    """Format a possibly missing CSV value."""
    text = str(value or "").strip()
    return text or fallback


def parse_float(value: object) -> float | None:
    """Parse a finite float from CSV/report values."""
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def draw_molecule_image(
    evidence: CompoundEvidence,
    output_path: Path,
) -> Path | None:
    """Draw a 2D RDKit molecule image for valid structures."""
    if not parse_boolean(evidence.prioritized.get("valid_smiles", "")):
        return None
    smiles = str(evidence.prioritized.get("canonical_smiles", "")).strip()
    if not smiles:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Draw.MolToFile(molecule, str(output_path), size=(420, 320))
    return output_path


def markdown_image_path(report_path: Path, image_path: Path) -> str:
    """Return a Markdown-friendly relative image path."""
    return Path(os.path.relpath(image_path, report_path.parent)).as_posix()


def is_known_answer(evidence: CompoundEvidence) -> str:
    """Explain known/public match signals."""
    if not parse_boolean(evidence.prioritized.get("valid_smiles", "")):
        return (
            "The molecule is invalid in the local pipeline, so known/public "
            "match evidence is not available."
        )

    exact = [
        row
        for row in evidence.public_lookup
        if row.get("lookup_status") == "match_found"
        and row.get("match_type") == "exact_inchikey"
    ]
    similarity = [
        row
        for row in evidence.public_lookup
        if row.get("lookup_status") == "match_found"
        and row.get("match_type") == "similarity"
    ]
    errors = [
        row
        for row in evidence.public_lookup
        if row.get("lookup_status") == "lookup_error"
    ]

    lines: list[str] = []
    if exact:
        matches = ", ".join(
            f"{format_value(row.get('source_database'))} "
            f"{format_value(row.get('public_id'))} "
            f"({format_value(row.get('public_name'))})"
            for row in exact
        )
        lines.append(f"An exact known/public match signal was found: {matches}.")
    else:
        lines.append("No exact public match found in the local output.")

    if similarity:
        matches = ", ".join(
            f"{format_value(row.get('source_database'))} "
            f"{format_value(row.get('public_id'))} at similarity "
            f"{format_value(row.get('similarity'))}"
            for row in similarity
        )
        lines.append(f"Public similarity evidence was also found: {matches}.")
    if errors:
        sources = ", ".join(
            format_value(row.get("source_database")) for row in errors
        )
        lines.append(f"Some public lookups were incomplete due to errors: {sources}.")
    return " ".join(lines)


def closest_compounds_answer(evidence: CompoundEvidence) -> str:
    """Summarize closest public-database and local-reference compounds."""
    lines = ["## Closest Public Compounds", ""]
    public_matches = [
        row
        for row in evidence.public_lookup
        if row.get("lookup_status") == "match_found"
    ]
    if public_matches:
        lines.extend(
            [
                "### Public Database Results",
                "",
                "| Database | Public ID | Name | Similarity | Match type |",
                "|---|---|---|---:|---|",
            ]
        )
        for row in public_matches:
            lines.append(
                f"| {format_value(row.get('source_database'))} | "
                f"{format_value(row.get('public_id'))} | "
                f"{format_value(row.get('public_name'))} | "
                f"{format_value(row.get('similarity'))} | "
                f"{format_value(row.get('match_type'))} |"
            )
        lines.append("")
    else:
        lines.extend(["No successful public-database matches are available.", ""])

    lines.extend(
        [
            "### Local Reference Similarity Hits",
            "",
            "| Rank | Reference | Similarity | Category | Evidence |",
            "|---:|---|---:|---|---|",
        ]
    )
    if evidence.similarity_hits:
        sorted_hits = sorted(
            evidence.similarity_hits,
            key=lambda row: int(row.get("hit_rank", "9999")),
        )
        for row in sorted_hits:
            lines.append(
                f"| {format_value(row.get('hit_rank'))} | "
                f"{format_value(row.get('reference_name'))} | "
                f"{format_value(row.get('tanimoto_similarity'))} | "
                f"{format_value(row.get('similarity_category'))} | "
                f"{format_value(row.get('evidence_note'))} |"
            )
    else:
        lines.append("| - | No local hits available | - | - | - |")
    lines.extend(["", reference_similarity_interpretation(evidence)])
    return "\n".join(lines)


def best_reference_similarity(evidence: CompoundEvidence) -> float | None:
    """Return the strongest local reference similarity available."""
    scores = [
        score
        for row in evidence.similarity_hits
        for score in [parse_float(row.get("tanimoto_similarity", ""))]
        if score is not None
    ]
    if scores:
        return max(scores)
    return parse_float(evidence.prioritized.get("tanimoto_similarity", ""))


def reference_similarity_interpretation(evidence: CompoundEvidence) -> str:
    """Interpret RDKit fingerprint support from the uploaded reference panel."""
    score = best_reference_similarity(evidence)
    if score is None:
        return (
            "Reference-panel interpretation: RDKit fingerprint similarity to the "
            "uploaded reference panel was not available."
        )
    if score < 0.30:
        return (
            "Reference-panel interpretation: This molecule is property-favorable "
            "and chemically differentiated, but has weak RDKit fingerprint "
            "similarity to the uploaded reference panel."
        )
    if score < 0.50:
        return (
            "Reference-panel interpretation: This molecule has moderate RDKit "
            "fingerprint similarity to the uploaded reference panel."
        )
    return (
        "Reference-panel interpretation: This molecule has strong RDKit "
        "fingerprint similarity to the uploaded reference panel."
    )


def why_ranked_answer(evidence: CompoundEvidence) -> str:
    """Explain the component scores driving prioritization."""
    row = evidence.prioritized
    if not parse_boolean(row.get("valid_smiles", "")):
        return (
            "The molecule was deprioritized because its structure was invalid. "
            "Validity, property, QED, Lipinski, differentiation, and NLP-enhanced "
            "scores were therefore set conservatively."
        )

    public_flag = format_value(row.get("novelty_flag"))
    return (
        f"The molecule received a research-prioritization score "
        f"of {format_value(row.get('prioritization_score'))}, adjusted to "
        f"{format_value(row.get('prioritization_score_with_nlp'))} after the "
        f"text-evidence signal. Component scores were validity "
        f"{format_value(row.get('validity_score'))}, properties "
        f"{format_value(row.get('property_score'))}, QED "
        f"{format_value(row.get('qed_score'))}, Lipinski "
        f"{format_value(row.get('lipinski_score'))}, and chemical "
        f"differentiation {format_value(row.get('differentiation_score'))}. "
        f"The final category was "
        f"{format_value(row.get('prioritization_category_with_nlp'))}. "
        f"Separately, its public-database differentiation signal is "
        f"{public_flag}; this does not erase the molecule's general property value. "
        f"{reference_similarity_interpretation(evidence)}"
    )


def nlp_summary(evidence: CompoundEvidence) -> str:
    """Summarize local NLP evidence rows."""
    row = evidence.prioritized
    if row.get("nlp_status", "").strip() == "not_run":
        return "- Text-evidence scoring was not run for this workflow."
    lines = [
        f"NLP status: {format_value(row.get('nlp_status'))}",
        f"Aggregated NLP evidence score: "
        f"{format_value(row.get('nlp_evidence_score'))}",
        f"Aggregated category: "
        f"{format_value(row.get('nlp_relevance_category'))}",
        f"Evidence row count: {format_value(row.get('nlp_evidence_count'), '0')}",
    ]
    scored = [
        item
        for item in evidence.nlp_evidence
        if str(item.get("max_relevance_score", "")).strip()
    ]
    if scored:
        best = max(
            scored, key=lambda item: float(item["max_relevance_score"])
        )
        lines.append(
            f"Highest-scoring evidence: "
            f"{format_value(best.get('title'))} "
            f"({format_value(best.get('max_relevance_score'))})."
        )
    return "\n".join(f"- {line}" for line in lines)


def describe_evidence_status(status: object) -> str:
    """Explain evidence-stage status values in report-friendly language."""
    text = str(status or "").strip() or "not_available"
    descriptions = {
        "match_found": "queried; evidence found",
        "no_match": "queried; no match found",
        "not_queried": "not checked because of max_molecules or workflow settings",
        "lookup_error": "query failed",
        "not_run": "workflow step was skipped",
        "available": "available",
        "not_available": "not available in the local output",
        "offline": "offline placeholder; no external query made",
        "invalid_molecule": "not checked because the molecule is invalid",
    }
    return f"{text} - {descriptions.get(text, 'status recorded by the workflow')}"


def evidence_completeness_section(evidence: CompoundEvidence) -> str:
    """Summarize which evidence stages were run for this molecule."""
    row = evidence.prioritized
    status_rows = (
        ("Chemical identity status", row.get("chemical_identity_lookup_status")),
        ("PubChem status", row.get("pubchem_status")),
        ("ChEMBL status", row.get("chembl_status")),
        ("SureChEMBL status", row.get("surechembl_query_status")),
        ("ChemBERTa status", row.get("chemberta_status")),
        ("NLP status", row.get("nlp_status")),
        ("Biomedical context status", row.get("context_status")),
    )
    lines = [
        "| Evidence source | Status | Meaning |",
        "|---|---|---|",
    ]
    for label, status in status_rows:
        explained = describe_evidence_status(status)
        status_text, _, meaning = explained.partition(" - ")
        lines.append(f"| {label} | {status_text} | {meaning} |")
    return "\n".join(lines)


def evidence_contribution_section(evidence: CompoundEvidence) -> str:
    """Explain which available evidence directly or contextually informed ranking."""
    row = evidence.prioritized
    lines = [
        "| Evidence | Status | Contribution to final output |",
        "|---|---|---|",
        (
            "| Standardized structure | "
            f"{'available' if parse_boolean(row.get('valid_smiles')) else 'not_available'} | "
            "Controls validity scoring and whether downstream calculations are available. |"
        ),
        (
            "| RDKit descriptors | "
            f"{'available' if str(row.get('molecular_weight', '')).strip() else 'not_run'} | "
            "Directly contributes property, QED, and Lipinski score components. |"
        ),
        (
            "| Reference similarity | "
            f"{'available' if str(row.get('tanimoto_similarity', '')).strip() else 'not_run'} | "
            "Directly contributes the chemical differentiation score component. |"
        ),
        (
            "| Chemical identity | "
            f"{format_value(row.get('chemical_identity_lookup_status'), 'not_run')} | "
            "Provides identity evidence and status; it does not receive an invented numeric value when unavailable. |"
        ),
        (
            "| Public lookup | "
            f"{format_value(row.get('public_lookup_status'), 'not_run')} | "
            "Provides exact-match and public-similarity evidence used for public-match and differentiation interpretation. |"
        ),
        (
            "| SureChEMBL structure evidence | "
            f"{format_value(row.get('surechembl_query_status'), 'not_run')} | "
            "Adds a separate structure-evidence signal and query status. |"
        ),
        (
            "| ChemBERTa embeddings | "
            f"{format_value(row.get('chemberta_status'), 'not_run')} | "
            "Adds learned chemical-space context and availability status. |"
        ),
        (
            "| Text evidence | "
            f"{format_value(row.get('nlp_status'), 'not_run')} | "
            "Adjusts the base prioritization score only when a usable NLP evidence score is available. |"
        ),
        (
            "| Biomedical context | "
            f"{format_value(row.get('context_status'), 'not_run')} | "
            "Adds grounded interpretation for the report without substituting missing evidence with zero. |"
        ),
    ]
    return "\n".join(lines)


def visualization_context(evidence: CompoundEvidence) -> str:
    """Summarize optional ChemBERTa/UMAP dashboard coordinates."""
    row = evidence.visualization_row
    if not row:
        return "- ChemBERTa/UMAP coordinates are available for dashboard visualization only when `visualization_coordinates.csv` exists."

    molecule_id = format_value(row.get("molecule_id"))
    x = parse_float(row.get("x"))
    y = parse_float(row.get("y"))
    method = format_value(row.get("coordinate_method"))
    score = format_value(row.get("prioritization_score_with_nlp"))
    lines = [
        "- ChemBERTa/UMAP coordinates are available for dashboard visualization.",
        f"- Coordinate method: {method}",
        f"- Map position for {molecule_id}: x={format_value(row.get('x'))}, y={format_value(row.get('y'))}",
        f"- Research-prioritization score on the chemical-space map: {score}",
    ]

    cluster_ids = {
        str(item.get("cluster_id", "")).strip()
        for item in evidence.visualization_rows
        if str(item.get("cluster_id", "")).strip()
        and str(item.get("cluster_id", "")).strip() != "-1"
    }
    if len(cluster_ids) > 1:
        lines.append(f"- Cluster ID: {format_value(row.get('cluster_id'))}")
    elif cluster_ids:
        lines.append(
            "- Clustering was not informative for this run because all molecules were assigned to one cluster."
        )

    neighbors: list[tuple[float, str]] = []
    if x is not None and y is not None:
        for item in evidence.visualization_rows:
            other_id = str(item.get("molecule_id", "")).strip()
            if not other_id or other_id == molecule_id:
                continue
            other_x = parse_float(item.get("x"))
            other_y = parse_float(item.get("y"))
            if other_x is None or other_y is None:
                continue
            distance = math.sqrt((x - other_x) ** 2 + (y - other_y) ** 2)
            neighbors.append((distance, other_id))
    if neighbors:
        nearest = ", ".join(
            f"{neighbor_id} (distance {distance:.3f})"
            for distance, neighbor_id in sorted(neighbors)[:3]
        )
        lines.append(f"- Nearest generated neighbors on the 2D map: {nearest}.")
    else:
        lines.append(
            "- Nearest-neighbor calculation was not available for this molecule."
        )
    return "\n".join(lines)


def surechembl_summary(evidence: CompoundEvidence) -> str:
    """Summarize SureChEMBL patent-associated structure evidence."""
    row = evidence.prioritized
    statuses = {
        item.get("lookup_status", "").strip()
        for item in evidence.surechembl_evidence
    }
    if "not_run" in statuses:
        return "- SureChEMBL structure evidence was not run for this workflow."
    lines = [
        "This section summarizes SureChEMBL structure evidence from patent-associated chemical structure search.",
        f"SureChEMBL structure evidence count: "
        f"{format_value(row.get('surechembl_evidence_count'), '0')}",
        f"Maximum patent-associated compound similarity: "
        f"{format_value(row.get('surechembl_max_similarity'))}",
        f"Signal category: "
        f"{format_value(row.get('surechembl_signal_category'))}",
    ]
    notes = format_value(row.get("surechembl_signal_notes"), "")
    if notes:
        lines.append(f"Signal notes: {notes}")
    matches = [
        item
        for item in evidence.surechembl_evidence
        if item.get("lookup_status", "").strip() == "match_found"
    ]
    if matches:
        if any(item.get("source_section", "").strip() == "SureChEMBL API" for item in matches):
            lines.append(
                "Online SureChEMBL mode sent query structures to the SureChEMBL external public API."
            )
        if all(
            format_value(item.get("patent_id")) == "not_available"
            for item in matches
        ):
            lines.append(
                "Structure-level SureChEMBL hits were found, but patent document metadata was not returned for these hits."
            )
        lines.extend(
            [
                "",
                "| Structure-level SureChEMBL hit | Patent ID | Similarity | Category | Evidence |",
                "|---|---|---:|---|---|",
            ]
        )
        sorted_matches = sorted(
            matches,
            key=lambda item: float(item.get("tanimoto_similarity", "0") or 0),
            reverse=True,
        )
        for item in sorted_matches[:5]:
            lines.append(
                f"| {format_value(item.get('compound_name'))} | "
                f"{format_value(item.get('patent_id'))} | "
                f"{format_value(item.get('tanimoto_similarity'))} | "
                f"{format_value(item.get('similarity_category'))} | "
                f"{format_value(item.get('evidence_note'))} |"
            )
    elif evidence.surechembl_evidence:
        statuses = ", ".join(
            sorted(
                {
                    format_value(item.get("lookup_status"))
                    for item in evidence.surechembl_evidence
                }
            )
        )
        lines.append(
            f"No patent-associated compound matches are available; statuses: {statuses}."
        )
    else:
        lines.append("No SureChEMBL structure evidence file or rows were available.")
    return "\n".join(f"- {line}" if not line.startswith("|") else line for line in lines)


def ip_potential_answer(evidence: CompoundEvidence) -> str:
    """Provide a cautious follow-up review interpretation."""
    row = evidence.prioritized
    if not parse_boolean(row.get("valid_smiles", "")):
        return (
            "No meaningful follow-up review signal can be formed because "
            "the molecular structure is invalid in the local workflow."
        )

    known_match = parse_boolean(row.get("known_public_match", ""))
    source = format_value(row.get("known_public_match_source"), "")
    public_id = format_value(row.get("known_public_match_id"), "")
    novelty_flag = format_value(row.get("novelty_flag"))
    notes = format_value(row.get("ip_potential_notes"))
    differentiation = format_value(row.get("differentiation_score"))
    score = format_value(row.get("prioritization_score_with_nlp"))
    if known_match:
        match_label = " ".join(part for part in (source, public_id) if part)
        known_text = (
            f"An exact known/public match signal"
            f"{f' ({match_label})' if match_label else ''} reduces the chemical "
            f"differentiation and follow-up review signal."
        )
    else:
        known_text = (
            "No exact public match found in the available "
            "local lookup output."
        )
    return (
        f"{known_text} Public-database differentiation signal: "
        f"{novelty_flag}. {notes} The general research-prioritization score remains "
        f"{score}, and the chemical differentiation component is "
        f"{differentiation}. These are separate research signals based on "
        f"limited local evidence, not a legal conclusion."
    )


def descriptor_section(evidence: CompoundEvidence) -> str:
    """Format the descriptor summary."""
    row = evidence.descriptor
    return "\n".join(
        [
            f"- Molecular weight: {format_value(row.get('molecular_weight'))}",
            f"- LogP: {format_value(row.get('logp'))}",
            f"- TPSA: {format_value(row.get('tpsa'))}",
            f"- H-bond donors: {format_value(row.get('hbd'))}",
            f"- H-bond acceptors: {format_value(row.get('hba'))}",
            f"- Rotatable bonds: {format_value(row.get('rotatable_bonds'))}",
            f"- QED: {format_value(row.get('qed'))}",
            f"- Lipinski pass: {format_value(row.get('lipinski_pass'))}",
        ]
    )


def public_biomedical_context_section(evidence: CompoundEvidence) -> str:
    """Format grounded public biomedical context without adding claims."""
    row = evidence.compound_context
    if not row:
        return (
            "- No public biomedical context was found in the available local "
            "lookup or reference evidence."
        )
    status = str(row.get("context_status", "")).strip()
    targets = str(row.get("reported_targets", "")).strip()
    if status == "structural_context_only":
        target_assignment = "No"
        interpretation = (
            "The nearest public/reference compound is shown for orientation "
            "only. Similarity is too low to transfer biological target context."
        )
    elif status == "weak_similar_reference_context":
        target_assignment = (
            "Weak reference-only context; not established for the query molecule"
            if targets
            else "No"
        )
        interpretation = (
            "The nearest reference provides weak structural context. Any target "
            "annotation shown is reference-only and should not be treated as "
            "biological evidence for the query molecule."
        )
    elif status == "similar_reference_context":
        target_assignment = (
            "Reference-derived context shown with warning"
            if targets
            else "No"
        )
        interpretation = (
            "Target annotations, when present, belong to the similar reference "
            "and are not established for the query molecule."
        )
    else:
        target_assignment = "Yes" if targets else "No"
        interpretation = ""

    lines = [
        f"- Context status: {format_value(row.get('context_status'))}",
        f"- Identity status: {format_value(row.get('identity_status'))}",
        f"- Exact public name: {format_value(row.get('exact_public_name'))}",
        f"- PubChem CID: {format_value(row.get('pubchem_cid'))}",
        f"- ChEMBL ID: {format_value(row.get('chembl_id'))}",
        f"- Closest public/reference compound: "
        f"{format_value(row.get('closest_public_compound'))}",
        f"- Closest similarity: "
        f"{format_value(row.get('closest_public_similarity'))}",
        f"- Biological target context assigned: {target_assignment}",
        f"- Reported target context: "
        f"{format_value(row.get('reported_targets'))}",
        f"- Reported assay context: "
        f"{format_value(row.get('reported_assays'))}",
        f"- Biological reference summary: "
        f"{format_value(row.get('biological_reference_summary'))}",
        f"- Biomedical relevance summary: "
        f"{format_value(row.get('biomedical_relevance_summary'))}",
        f"- Context confidence: {format_value(row.get('context_confidence'))}",
        f"- Context sources: {format_value(row.get('context_sources'))}",
    ]
    if interpretation:
        lines.append(f"- Interpretation: {interpretation}")
    return "\n".join(lines)


def chemical_identity_section(evidence: CompoundEvidence) -> str:
    """Format exact or generated chemical identity without adding names."""
    row = evidence.chemical_identity
    if not row:
        return "- Chemical identity output is not available for this molecule."
    return "\n".join(
        [
            f"- Identity status: {format_value(row.get('identity_status'))}",
            f"- Exact public name: {format_value(row.get('exact_public_name'))}",
            f"- Preferred name: {format_value(row.get('preferred_name'))}",
            f"- IUPAC name: {format_value(row.get('iupac_name'))}",
            f"- Synonyms: {format_value(row.get('synonyms'))}",
            f"- InChIKey: {format_value(row.get('inchikey'))}",
            f"- PubChem CID: {format_value(row.get('pubchem_cid'))}",
            f"- ChEMBL ID: {format_value(row.get('chembl_id'))}",
            f"- Name source: {format_value(row.get('name_source'))}",
            f"- Identity confidence: {format_value(row.get('identity_confidence'))}",
            f"- Lookup status: {format_value(row.get('lookup_status'))}",
        ]
    )


def chemberta_section(evidence: CompoundEvidence) -> str:
    """Summarize optional ChemBERTa representation availability."""
    row = evidence.prioritized
    if not parse_boolean(row.get("chemberta_embedding_available", "")):
        return ""
    return "\n".join(
        [
            "- ChemBERTa learned chemical-space representation was available for this molecule.",
            f"- ChemBERTa model: {format_value(row.get('chemberta_model'))}",
            f"- Embedding dimension: {format_value(row.get('chemberta_embedding_dim'))}",
            visualization_context(evidence),
        ]
    )


def full_report(
    evidence: CompoundEvidence,
    *,
    report_path: Path | None = None,
    image_dir: Path | None = None,
) -> str:
    """Build the complete Markdown compound-intelligence report."""
    row = evidence.prioritized
    molecule_id = format_value(row.get("molecule_id"))
    valid = parse_boolean(row.get("valid_smiles", ""))
    structure_lines: list[str] = []
    if valid and report_path is not None:
        active_image_dir = image_dir or (report_path.parent.parent / "report_images")
        image_path = active_image_dir / f"{molecule_id}.png"
        created = draw_molecule_image(evidence, image_path)
        if created is not None:
            structure_lines = [
                "## Molecule Structure",
                "",
                f"![2D structure for {molecule_id}]({markdown_image_path(report_path, created)})",
                "",
            ]
    sections = [
        f"# Compound Intelligence Report: {molecule_id}",
        "",
        "## Target profile",
        "",
        f"- Target ID: `{format_value(row.get('target_id'))}`",
        f"- Target name: `{format_value(row.get('target_name'))}`",
        "",
        "## Molecule Overview",
        "",
        f"- Molecule ID: `{molecule_id}`",
        f"- Canonical SMILES: `{format_value(row.get('canonical_smiles'))}`",
        f"- Validity: {'valid' if valid else 'invalid'}",
        "",
        *structure_lines,
        "## Chemical identity",
        "",
        chemical_identity_section(evidence),
        "",
        "## Descriptor Summary",
        "",
        descriptor_section(evidence),
        "",
        "## Docking evidence",
        "",
        f"- Docking available: {format_value(row.get('docking_available'))}",
        f"- Docking score: {format_value(row.get('docking_score'))}",
        f"- Docking rank: {format_value(row.get('docking_rank'))}",
        f"- Docking priority label: {format_value(row.get('docking_priority_label'))}",
        f"- Interpretation: {format_value(row.get('structural_priority_note'))}",
        "",
        "## Prioritization",
        "",
        f"- Score with NLP: "
        f"{format_value(row.get('prioritization_score_with_nlp'))}",
        f"- Category with NLP: "
        f"{format_value(row.get('prioritization_category_with_nlp'))}",
        "",
        "## Known/Public Match Signal",
        "",
        is_known_answer(evidence),
        "",
        f"- Known public match: "
        f"{format_value(row.get('known_public_match'))}",
        f"- Known public match source: "
        f"{format_value(row.get('known_public_match_source'))}",
        f"- Known public match ID: "
        f"{format_value(row.get('known_public_match_id'))}",
        f"- Public-database differentiation signal: {format_value(row.get('novelty_flag'))}",
        "",
        "## Evidence completeness",
        "",
        evidence_completeness_section(evidence),
        "",
        "## Evidence contribution summary",
        "",
        evidence_contribution_section(evidence),
        "",
        "## Public biomedical context",
        "",
        public_biomedical_context_section(evidence),
        "",
        closest_compounds_answer(evidence),
        "",
        "## NLP Evidence Summary",
        "",
        nlp_summary(evidence),
        "",
        "## SureChEMBL Structure Evidence",
        "",
        surechembl_summary(evidence),
        "",
    ]
    chemberta_text = chemberta_section(evidence)
    if chemberta_text:
        sections.extend(
            [
                "## ChemBERTa Representation",
                "",
                chemberta_text,
                "",
            ]
        )
    sections.extend(
        [
        "## Why It Was Ranked",
        "",
        why_ranked_answer(evidence),
        "",
        "## Follow-up Review Interpretation",
        "",
        ip_potential_answer(evidence),
        "",
        ]
    )
    return "\n".join(sections)


def answer_question(
    question: str,
    evidence: CompoundEvidence,
    *,
    report_path: Path | None = None,
    image_dir: Path | None = None,
) -> str:
    """Route a supported question to its Markdown answer."""
    if question == "is_known":
        return is_known_answer(evidence)
    if question == "closest_public_compounds":
        return closest_compounds_answer(evidence)
    if question == "why_ranked":
        return why_ranked_answer(evidence)
    if question == "ip_potential_summary":
        return ip_potential_answer(evidence)
    if question == "full_report":
        return full_report(evidence, report_path=report_path, image_dir=image_dir)
    raise ValueError(f"Unsupported question: {question}")


def write_answer(output_path: Path, answer: str) -> None:
    """Write a Markdown answer."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(answer.rstrip() + "\n", encoding="utf-8")


def compound_qa(
    molecule_id: str,
    question: str,
    output_path: Path,
    *,
    prioritized_path: Path = DEFAULT_PRIORITIZED,
    similarity_path: Path = DEFAULT_SIMILARITY,
    public_lookup_path: Path = DEFAULT_PUBLIC_LOOKUP,
    compound_context_path: Path | None = None,
    chemical_identity_path: Path | None = None,
    nlp_path: Path = DEFAULT_NLP,
    descriptor_path: Path = DEFAULT_DESCRIPTORS,
    surechembl_path: Path = DEFAULT_SURECHEMBL,
    visualization_path: Path = DEFAULT_VISUALIZATION,
    image_dir: Path | None = None,
) -> str:
    """Load evidence, answer a question, and write Markdown output."""
    evidence = load_compound_evidence(
        molecule_id,
        prioritized_path=prioritized_path,
        similarity_path=similarity_path,
        public_lookup_path=public_lookup_path,
        compound_context_path=compound_context_path,
        chemical_identity_path=chemical_identity_path,
        nlp_path=nlp_path,
        descriptor_path=descriptor_path,
        surechembl_path=surechembl_path,
        visualization_path=visualization_path,
    )
    answer = answer_question(
        question,
        evidence,
        report_path=output_path,
        image_dir=image_dir,
    )
    write_answer(output_path, answer)
    return answer


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Answer compound-intelligence questions from local outputs."
    )
    parser.add_argument("--molecule-id", required=True)
    parser.add_argument(
        "--question", required=True, choices=SUPPORTED_QUESTIONS
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prioritized", type=Path, default=DEFAULT_PRIORITIZED)
    parser.add_argument("--similarity", type=Path, default=DEFAULT_SIMILARITY)
    parser.add_argument(
        "--public-lookup", type=Path, default=DEFAULT_PUBLIC_LOOKUP
    )
    parser.add_argument("--compound-context", type=Path)
    parser.add_argument("--nlp", type=Path, default=DEFAULT_NLP)
    parser.add_argument("--descriptors", type=Path, default=DEFAULT_DESCRIPTORS)
    parser.add_argument("--surechembl", type=Path, default=DEFAULT_SURECHEMBL)
    parser.add_argument(
        "--visualization-coordinates",
        type=Path,
        default=DEFAULT_VISUALIZATION,
    )
    parser.add_argument("--image-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local compound QA command-line interface."""
    args = build_parser().parse_args(argv)
    try:
        compound_qa(
            args.molecule_id,
            args.question,
            args.output,
            prioritized_path=args.prioritized,
            similarity_path=args.similarity,
            public_lookup_path=args.public_lookup,
            compound_context_path=args.compound_context,
            nlp_path=args.nlp,
            descriptor_path=args.descriptors,
            surechembl_path=args.surechembl,
            visualization_path=args.visualization_coordinates,
            image_dir=args.image_dir,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(f"Wrote compound intelligence answer to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
