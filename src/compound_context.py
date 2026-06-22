"""Build grounded public biomedical context from existing local evidence."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


OUTPUT_COLUMNS = (
    "molecule_id",
    "smiles",
    "identity_status",
    "exact_public_name",
    "iupac_name",
    "common_names",
    "pubchem_cid",
    "chembl_id",
    "closest_public_compound",
    "closest_public_similarity",
    "reported_targets",
    "reported_assays",
    "biological_reference_summary",
    "biomedical_relevance_summary",
    "context_confidence",
    "context_sources",
    "context_status",
)


@dataclass(frozen=True)
class CompoundContext:
    """Grounded context for one molecule."""

    molecule_id: str
    smiles: str
    identity_status: str
    exact_public_name: str = ""
    iupac_name: str = ""
    common_names: str = ""
    pubchem_cid: str = ""
    chembl_id: str = ""
    closest_public_compound: str = ""
    closest_public_similarity: str = ""
    reported_targets: str = ""
    reported_assays: str = ""
    biological_reference_summary: str = ""
    biomedical_relevance_summary: str = ""
    context_confidence: str = "none"
    context_sources: str = ""
    context_status: str = "no_public_context"


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries."""
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        return [dict(row) for row in csv.DictReader(input_file)]


def clean(value: object) -> str:
    """Normalize a possibly missing CSV value."""
    return str(value or "").strip()


def parse_similarity(value: object) -> float | None:
    """Parse a similarity score in the zero-to-one range."""
    try:
        score = float(clean(value))
    except ValueError:
        return None
    if score > 1:
        score /= 100.0
    if 0 <= score <= 1:
        return score
    return None


def unique_join(values: Iterable[object]) -> str:
    """Join nonempty values once, preserving input order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = clean(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return "; ".join(result)


def first_value(rows: Iterable[Mapping[str, str]], *columns: str) -> str:
    """Return the first nonempty value from selected columns."""
    for row in rows:
        for column in columns:
            value = clean(row.get(column))
            if value:
                return value
    return ""


def is_exact_public_match(row: Mapping[str, str]) -> bool:
    """Return whether a lookup row explicitly records an exact identity match."""
    return (
        clean(row.get("lookup_status")) == "match_found"
        and clean(row.get("match_type")).lower()
        in {"exact", "exact_match", "exact_inchikey", "exact_smiles"}
    )


def best_similarity_row(
    rows: Iterable[Mapping[str, str]], score_column: str
) -> tuple[Mapping[str, str] | None, float | None]:
    """Return the row with the strongest valid similarity."""
    scored = [
        (score, row)
        for row in rows
        for score in [parse_similarity(row.get(score_column))]
        if score is not None
    ]
    if not scored:
        return None, None
    score, row = max(scored, key=lambda item: item[0])
    return row, score


def index_reference_rows(
    rows: Iterable[Mapping[str, str]],
) -> dict[str, Mapping[str, str]]:
    """Index reference metadata by common identifiers."""
    index: dict[str, Mapping[str, str]] = {}
    for row in rows:
        for column in (
            "reference_id",
            "reference_name",
            "reference_source_id",
        ):
            key = clean(row.get(column)).casefold()
            if key:
                index.setdefault(key, row)
    return index


def reference_metadata(
    hit: Mapping[str, str] | None,
    reference_index: Mapping[str, Mapping[str, str]],
) -> Mapping[str, str]:
    """Find source metadata for a local similarity hit."""
    if hit is None:
        return {}
    for column in ("reference_id", "reference_source_id", "reference_name"):
        key = clean(hit.get(column)).casefold()
        if key and key in reference_index:
            return reference_index[key]
    return {}


def public_id(
    rows: Iterable[Mapping[str, str]], database: str, prefix: str = ""
) -> str:
    """Return an ID from exact or identity-equivalent public rows."""
    candidates = []
    for row in rows:
        if clean(row.get("source_database")).casefold() != database.casefold():
            continue
        score = parse_similarity(row.get("similarity"))
        if is_exact_public_match(row) or score == 1.0:
            candidates.append(row)
    value = first_value(candidates, "public_id")
    if prefix and value.upper().startswith(prefix.upper()):
        return value[len(prefix) :]
    return value


def build_context_rows(
    molecule_rows: Iterable[Mapping[str, str]],
    public_rows: Iterable[Mapping[str, str]],
    similarity_rows: Iterable[Mapping[str, str]],
    reference_rows: Iterable[Mapping[str, str]] = (),
    identity_rows: Iterable[Mapping[str, str]] = (),
) -> list[CompoundContext]:
    """Create conservative context rows from local public/reference evidence."""
    public_by_id: dict[str, list[Mapping[str, str]]] = {}
    similarity_by_id: dict[str, list[Mapping[str, str]]] = {}
    for row in public_rows:
        public_by_id.setdefault(clean(row.get("molecule_id")), []).append(row)
    for row in similarity_rows:
        similarity_by_id.setdefault(clean(row.get("molecule_id")), []).append(row)
    reference_index = index_reference_rows(reference_rows)
    identity_by_id = {
        clean(row.get("molecule_id")): row for row in identity_rows
    }

    results: list[CompoundContext] = []
    for molecule in molecule_rows:
        molecule_id = clean(molecule.get("molecule_id"))
        smiles = first_value(
            [molecule], "canonical_smiles", "smiles", "query_smiles"
        )
        molecule_public = public_by_id.get(molecule_id, [])
        identity = identity_by_id.get(molecule_id, {})
        successful_public = [
            row
            for row in molecule_public
            if clean(row.get("lookup_status")) == "match_found"
        ]
        exact_rows = [row for row in successful_public if is_exact_public_match(row)]
        best_public, best_public_score = best_similarity_row(
            successful_public, "similarity"
        )
        best_reference, best_reference_score = best_similarity_row(
            similarity_by_id.get(molecule_id, []), "tanimoto_similarity"
        )
        metadata = reference_metadata(best_reference, reference_index)

        exact_name = first_value(
            [identity],
            "exact_public_name",
            "preferred_name",
        ) or first_value(
            exact_rows, "public_name", "preferred_name", "name"
        )
        iupac_name = first_value(
            [identity], "iupac_name"
        ) or first_value(exact_rows, "iupac_name", "IUPACName")
        common_names = unique_join(
            [
                identity.get("synonyms"),
                *(
                    row.get(column)
                    for row in exact_rows
                    for column in ("common_names", "synonyms")
                ),
            ]
        )
        pubchem_cid = clean(identity.get("pubchem_cid")) or public_id(
            successful_public, "PubChem", "CID:"
        )
        chembl_id = clean(identity.get("chembl_id")) or public_id(
            successful_public, "ChEMBL"
        )
        identity_exact = (
            clean(identity.get("identity_status")) == "exact_public_identity"
        )

        closest_row = best_public
        closest_score = best_public_score
        closest_name = first_value(
            [best_public] if best_public else [],
            "public_name",
            "preferred_name",
            "public_id",
        )
        closest_source = (
            clean(best_public.get("source_database")) if best_public else ""
        )
        if (
            best_reference is not None
            and (
                closest_score is None
                or (
                    best_reference_score is not None
                    and best_reference_score > closest_score
                )
            )
        ):
            closest_row = best_reference
            closest_score = best_reference_score
            closest_name = first_value(
                [best_reference], "reference_name", "reference_source_id"
            )
            closest_source = first_value(
                [best_reference], "reference_source"
            ) or "local public reference panel"

        context_rows = [*exact_rows]
        if metadata:
            context_rows.append(metadata)
        if closest_row is not None:
            context_rows.append(closest_row)
        targets = unique_join(
            row.get(column)
            for row in context_rows
            for column in (
                "reported_targets",
                "target_name",
                "target",
                "target_family",
            )
        )
        assays = unique_join(
            row.get(column)
            for row in context_rows
            for column in (
                "reported_assays",
                "assay_name",
                "assay",
                "assay_description",
            )
        )
        reference_note = first_value(
            [metadata, closest_row or {}],
            "biological_reference_summary",
            "evidence_note",
            "notes",
        )
        sources = unique_join(
            [
                *(
                    clean(row.get("source_database"))
                    for row in successful_public
                ),
                closest_source,
            ]
        )

        if exact_rows or identity_exact:
            summary_parts = [
                (
                    f"An exact public record is available for {exact_name}."
                    if exact_name
                    else "An exact public compound record is available."
                )
            ]
            if targets:
                summary_parts.append(
                    f"Reported target context in the available local evidence: {targets}."
                )
            if assays:
                summary_parts.append(
                    f"Reported assay context in the available local evidence: {assays}."
                )
            results.append(
                CompoundContext(
                    molecule_id=molecule_id,
                    smiles=smiles,
                    identity_status="exact_public_match",
                    exact_public_name=exact_name,
                    iupac_name=iupac_name,
                    common_names=common_names,
                    pubchem_cid=pubchem_cid,
                    chembl_id=chembl_id,
                    closest_public_compound=closest_name or exact_name,
                    closest_public_similarity=(
                        f"{closest_score:.3f}"
                        if closest_score is not None
                        else "1.000"
                    ),
                    reported_targets=targets,
                    reported_assays=assays,
                    biological_reference_summary=reference_note,
                    biomedical_relevance_summary=" ".join(summary_parts),
                    context_confidence="high",
                    context_sources=sources,
                    context_status="exact_public_context",
                )
            )
            continue

        if closest_row is not None and closest_name:
            similarity = closest_score if closest_score is not None else 0.0
            summary_parts = [
                f"No exact public identity was found. The closest grounded "
                f"reference is {closest_name}"
                + (
                    f" at similarity {closest_score:.3f}."
                    if closest_score is not None
                    else "."
                )
            ]
            reported_targets = targets
            reported_assays = assays
            biological_summary = reference_note
            if similarity >= 0.50:
                context_status = "similar_reference_context"
                context_confidence = "moderate"
                if targets:
                    reported_targets = (
                        "Reference-derived context; not established for the "
                        f"query molecule: {targets}"
                    )
                    summary_parts.append(
                        "Target context is reported for the similar reference "
                        "and is not established for the query molecule."
                    )
            elif similarity >= 0.30:
                context_status = "weak_similar_reference_context"
                context_confidence = "low"
                if targets:
                    reported_targets = (
                        "Weak reference-only context; not assigned as evidence "
                        f"for the query molecule: {targets}"
                    )
                if assays:
                    reported_assays = (
                        "Weak reference-only context; not assigned as evidence "
                        f"for the query molecule: {assays}"
                    )
                summary_parts.append(
                    "Any biological annotation is weak, reference-only context "
                    "and is not established for the query molecule."
                )
            else:
                context_status = "structural_context_only"
                context_confidence = "very_low"
                reported_targets = ""
                reported_assays = ""
                biological_summary = ""
                summary_parts = [
                    "No reliable biomedical context was assigned because the "
                    "closest reference similarity was below the reporting threshold."
                ]
            results.append(
                CompoundContext(
                    molecule_id=molecule_id,
                    smiles=smiles,
                    identity_status="similar_reference",
                    closest_public_compound=closest_name,
                    closest_public_similarity=(
                        f"{closest_score:.3f}" if closest_score is not None else ""
                    ),
                    reported_targets=reported_targets,
                    reported_assays=reported_assays,
                    biological_reference_summary=biological_summary,
                    biomedical_relevance_summary=" ".join(summary_parts),
                    context_confidence=context_confidence,
                    context_sources=sources,
                    context_status=context_status,
                )
            )
            continue

        results.append(
            CompoundContext(
                molecule_id=molecule_id,
                smiles=smiles,
                identity_status="no_public_identity",
                biomedical_relevance_summary=(
                    "No public biomedical context was found in the available "
                    "local lookup or reference evidence."
                ),
                context_confidence="none",
                context_status="no_public_context",
            )
        )
    return results


def write_context_csv(
    output_path: Path, rows: Iterable[CompoundContext]
) -> None:
    """Write compound context rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def compound_context_csv(
    molecule_path: Path,
    public_lookup_path: Path,
    similarity_path: Path,
    reference_path: Path,
    output_path: Path,
    *,
    identity_path: Path | None = None,
) -> int:
    """Build ``compound_context.csv`` from existing local CSV evidence."""
    rows = build_context_rows(
        read_csv(molecule_path),
        read_csv(public_lookup_path),
        read_csv(similarity_path),
        read_csv(reference_path),
        read_csv(identity_path) if identity_path is not None and identity_path.exists() else [],
    )
    write_context_csv(output_path, rows)
    return len(rows)
