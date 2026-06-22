"""Search public-safe reference compounds and produce evidence reports."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from rdkit import DataStructs

from src.similarity import (
    categorize_similarity,
    create_morgan_fingerprint,
    parse_boolean,
)


DESCRIPTOR_COLUMNS = ("molecule_id", "canonical_smiles", "valid_smiles")
REFERENCE_COLUMNS = ("reference_name", "smiles")
OUTPUT_COLUMNS = (
    "molecule_id",
    "query_smiles",
    "hit_rank",
    "reference_id",
    "reference_name",
    "reference_smiles",
    "reference_source",
    "reference_source_id",
    "tanimoto_similarity",
    "similarity_category",
    "evidence_note",
)

CATEGORY_EXPLANATIONS = {
    "very_close_analog": "Tanimoto similarity is at least 0.85.",
    "related_chemotype": "Tanimoto similarity is from 0.70 to below 0.85.",
    "moderate_similarity": "Tanimoto similarity is from 0.50 to below 0.70.",
    "structurally_distinct": "Tanimoto similarity is below 0.50.",
}


@dataclass(frozen=True)
class SearchReference:
    """A valid public-safe reference and its fingerprint."""

    reference_id: str
    reference_name: str
    smiles: str
    reference_source: str
    reference_source_id: str
    evidence_note: str
    fingerprint: object


@dataclass(frozen=True)
class CompoundHit:
    """One ranked structural-similarity hit."""

    molecule_id: str
    query_smiles: str
    hit_rank: int
    reference_id: str
    reference_name: str
    reference_smiles: str
    reference_source: str
    reference_source_id: str
    tanimoto_similarity: str
    similarity_category: str
    evidence_note: str


def read_csv_with_columns(
    path: Path, required_columns: Sequence[str], label: str
) -> list[dict[str, str]]:
    """Read a CSV and validate required columns."""
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = set(reader.fieldnames or [])
        missing = set(required_columns) - fieldnames
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{label} CSV is missing required columns: {names}")
        return [dict(row) for row in reader]


def prepare_search_references(
    rows: Iterable[Mapping[str, str]],
) -> list[SearchReference]:
    """Create fingerprints for valid references and skip invalid structures."""
    references: list[SearchReference] = []
    for index, row in enumerate(rows, start=1):
        smiles = row.get("smiles", "").strip()
        try:
            fingerprint = create_morgan_fingerprint(smiles)
        except (ValueError, RuntimeError):
            continue
        references.append(
            SearchReference(
                reference_id=(
                    row.get("reference_id", "").strip()
                    or f"ref_{index:04d}"
                ),
                reference_name=row.get("reference_name", "").strip(),
                smiles=smiles,
                reference_source=(
                    row.get("reference_source", "").strip()
                    or "public_reference_panel"
                ),
                reference_source_id=row.get(
                    "reference_source_id", ""
                ).strip()
                or row.get("reference_name", "").strip(),
                evidence_note=(
                    row.get("evidence_note", "").strip()
                    or row.get("notes", "").strip()
                ),
                fingerprint=fingerprint,
            )
        )
    return references


def rank_hits(
    molecule_id: str,
    query_smiles: str,
    references: Sequence[SearchReference],
    top_k: int,
) -> list[CompoundHit]:
    """Rank all valid references by descending Morgan/Tanimoto similarity."""
    if top_k < 1:
        raise ValueError("top-k must be at least 1.")
    query_fingerprint = create_morgan_fingerprint(query_smiles)
    scored = [
        (
            DataStructs.TanimotoSimilarity(
                query_fingerprint, reference.fingerprint
            ),
            index,
            reference,
        )
        for index, reference in enumerate(references)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))

    return [
        CompoundHit(
            molecule_id=molecule_id,
            query_smiles=query_smiles,
            hit_rank=rank,
            reference_id=reference.reference_id,
            reference_name=reference.reference_name,
            reference_smiles=reference.smiles,
            reference_source=reference.reference_source,
            reference_source_id=reference.reference_source_id,
            tanimoto_similarity=f"{similarity:.3f}",
            similarity_category=categorize_similarity(similarity),
            evidence_note=reference.evidence_note,
        )
        for rank, (similarity, _, reference) in enumerate(
            scored[:top_k], start=1
        )
    ]


def search_rows(
    descriptor_rows: Iterable[Mapping[str, str]],
    references: Sequence[SearchReference],
    top_k: int,
) -> list[CompoundHit]:
    """Search each valid generated molecule against all valid references."""
    hits: list[CompoundHit] = []
    for row in descriptor_rows:
        if not parse_boolean(row.get("valid_smiles", "")):
            continue
        smiles = row.get("canonical_smiles", "").strip()
        if not smiles:
            continue
        hits.extend(
            rank_hits(
                row.get("molecule_id", "").strip(),
                smiles,
                references,
                top_k,
            )
        )
    return hits


def write_hits_csv(output_path: Path, hits: Iterable[CompoundHit]) -> None:
    """Write ranked hits to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for hit in hits:
            writer.writerow(asdict(hit))


def top_hits_csv(
    descriptor_path: Path,
    reference_path: Path,
    output_path: Path,
    top_k: int,
) -> int:
    """Create the batch top-hit CSV and return its row count."""
    descriptor_rows = read_csv_with_columns(
        descriptor_path, DESCRIPTOR_COLUMNS, "Descriptor"
    )
    reference_rows = read_csv_with_columns(
        reference_path, REFERENCE_COLUMNS, "Reference"
    )
    references = prepare_search_references(reference_rows)
    if not references:
        raise ValueError("No valid reference molecules are available.")
    hits = search_rows(descriptor_rows, references, top_k)
    write_hits_csv(output_path, hits)
    return len(hits)


def find_molecule(
    rows: Iterable[Mapping[str, str]], molecule_id: str
) -> Mapping[str, str]:
    """Find one generated molecule or raise a clear error."""
    for row in rows:
        if row.get("molecule_id", "").strip() == molecule_id:
            return row
    raise ValueError(f"Molecule ID '{molecule_id}' was not found.")


def write_markdown_report(
    report_path: Path,
    molecule_id: str,
    query_smiles: str,
    hits: Sequence[CompoundHit],
) -> None:
    """Write a molecule-specific Markdown evidence report."""
    lines = [
        f"# Compound Intelligence Report: {molecule_id}",
        "",
        "## Query",
        "",
        f"- Molecule ID: `{molecule_id}`",
        f"- Query SMILES: `{query_smiles}`",
        "",
        "## Top Similar Reference Compounds",
        "",
        "| Rank | Reference | Source ID | Similarity | Category |",
        "|---:|---|---|---:|---|",
    ]
    for hit in hits:
        lines.append(
            f"| {hit.hit_rank} | {hit.reference_name} | "
            f"{hit.reference_source_id} | {hit.tanimoto_similarity} | "
            f"{hit.similarity_category} |"
        )

    lines.extend(["", "## Similarity Category Explanation", ""])
    for category, explanation in CATEGORY_EXPLANATIONS.items():
        lines.append(f"- `{category}`: {explanation}")

    lines.extend(["", "## Evidence Notes", ""])
    for hit in hits:
        note = hit.evidence_note or "No evidence note provided."
        lines.append(
            f"- **Rank {hit.hit_rank}, {hit.reference_name} "
            f"({hit.reference_source})**: {note}"
        )

    lines.extend(
        [
            "",
            "## Cautious Interpretation",
            "",
            "This is a structural-similarity and public-evidence research "
            "signal, not a legal conclusion.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def compound_report(
    molecule_id: str,
    descriptor_path: Path,
    reference_path: Path,
    report_path: Path,
    top_k: int,
) -> int:
    """Create a Markdown report for one generated molecule."""
    descriptor_rows = read_csv_with_columns(
        descriptor_path, DESCRIPTOR_COLUMNS, "Descriptor"
    )
    row = find_molecule(descriptor_rows, molecule_id)
    if not parse_boolean(row.get("valid_smiles", "")):
        raise ValueError(f"Molecule ID '{molecule_id}' is marked as invalid.")
    query_smiles = row.get("canonical_smiles", "").strip()
    if not query_smiles:
        raise ValueError(f"Molecule ID '{molecule_id}' has no canonical SMILES.")

    reference_rows = read_csv_with_columns(
        reference_path, REFERENCE_COLUMNS, "Reference"
    )
    references = prepare_search_references(reference_rows)
    if not references:
        raise ValueError("No valid reference molecules are available.")
    hits = rank_hits(molecule_id, query_smiles, references, top_k)
    write_markdown_report(report_path, molecule_id, query_smiles, hits)
    return len(hits)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Search local public-safe reference compounds."
    )
    parser.add_argument("--descriptors", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--molecule-id")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run batch search or molecule-specific report generation."""
    args = build_parser().parse_args(argv)
    if args.molecule_id:
        if args.report is None:
            raise SystemExit("Error: --report is required with --molecule-id.")
        try:
            count = compound_report(
                args.molecule_id,
                args.descriptors,
                args.references,
                args.report,
                args.top_k,
            )
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Error: {exc}") from exc
        print(f"Wrote report with {count} hits to {args.report}")
        return 0

    if args.output is None:
        raise SystemExit("Error: --output is required for batch search.")
    try:
        count = top_hits_csv(
            args.descriptors, args.references, args.output, args.top_k
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(f"Wrote {count} similarity hits to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
