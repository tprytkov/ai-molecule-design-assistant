"""Match molecule context against biomedical text evidence embeddings."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from src.text_nlp import (
    cosine_similarity,
    encode_sentences,
    read_input_csv,
)
from src.optional_domain_models import (
    DomainModelUnavailableError,
    encoder_metadata,
    load_sentence_transformer,
)


DEFAULT_BIOMEDICAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BIOMEDICAL_MODEL_UNAVAILABLE_NOTE = (
    "Biomedical embedding model unavailable in this environment; biomedical "
    "evidence matching was skipped."
)
BIOMEDICAL_MODEL_AVAILABLE_NOTE = (
    "Biomedical evidence matched with a local embedding model."
)
OUTPUT_COLUMNS = (
    "molecule_id",
    "biomedical_model_name",
    "biomedical_model_status",
    "biomedical_evidence_status",
    "biomedical_similarity_score",
    "biomedical_relevance_category",
    "biomedical_evidence_count",
    "top_biomedical_evidence_id",
    "top_biomedical_evidence_text",
    "evidence_note",
    "embedding_backend",
    "pooling_method",
    "model_source",
    "preferred_model_name",
    "fallback_model_name",
    "actual_model_used",
)


class BiomedicalModelUnavailableError(RuntimeError):
    """Raised when the configured biomedical embedding model is unavailable."""


class BiomedicalEncoder(Protocol):
    """Minimal interface required from a sentence-transformer style encoder."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> object:
        """Encode sentences as embedding vectors."""


@dataclass(frozen=True)
class BiomedicalEvidenceResult:
    """Molecule-level biomedical evidence matching output row."""

    molecule_id: str
    biomedical_model_name: str = DEFAULT_BIOMEDICAL_MODEL
    biomedical_model_status: str = "not_available"
    biomedical_evidence_status: str = "not_run"
    biomedical_similarity_score: str = "0.000"
    biomedical_relevance_category: str = "not_run"
    biomedical_evidence_count: str = "0"
    top_biomedical_evidence_id: str = ""
    top_biomedical_evidence_text: str = ""
    evidence_note: str = ""
    embedding_backend: str = ""
    pooling_method: str = ""
    model_source: str = ""
    preferred_model_name: str = ""
    fallback_model_name: str = ""
    actual_model_used: str = ""


def load_model(
    model_name: str = DEFAULT_BIOMEDICAL_MODEL,
) -> BiomedicalEncoder:
    """Load a cached sentence-transformer compatible biomedical model."""
    try:
        return load_sentence_transformer(model_name)
    except DomainModelUnavailableError as exc:
        raise BiomedicalModelUnavailableError(str(exc)) from exc


def read_optional_csv(path: Path | None) -> list[dict[str, str]]:
    """Read optional CSV rows without failing when the file is absent."""
    if path is None or not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        return [dict(row) for row in csv.DictReader(input_file)]


def row_molecule_id(row: dict[str, str]) -> str:
    """Return a normalized molecule ID."""
    return str(row.get("molecule_id", "")).strip()


def index_by_molecule(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Index rows by molecule ID, keeping the last nonempty values seen."""
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        molecule_id = row_molecule_id(row)
        if not molecule_id:
            continue
        indexed.setdefault(molecule_id, {})
        indexed[molecule_id].update(
            {
                key: value
                for key, value in row.items()
                if value is not None and str(value).strip()
            }
        )
    return indexed


def molecule_context_text(row: dict[str, str]) -> str:
    """Build biomedical matching text from available molecule context fields."""
    fields = (
        "exact_public_name",
        "preferred_name",
        "iupac_name",
        "synonyms",
        "closest_public_compound",
        "best_reference_name",
        "reported_targets",
        "target",
        "target_family",
        "biological_reference_summary",
        "biomedical_relevance_summary",
        "compound_description",
        "scaffold_family",
        "notes",
    )
    return " ".join(
        str(row.get(field, "")).strip()
        for field in fields
        if str(row.get(field, "")).strip()
    )


def merged_molecule_rows(
    context_rows: Iterable[dict[str, str]],
    *,
    identity_rows: Iterable[dict[str, str]] = (),
    descriptor_rows: Iterable[dict[str, str]] = (),
) -> list[dict[str, str]]:
    """Merge available molecule-level context, identity, and descriptor fields."""
    merged = index_by_molecule(context_rows)
    for source_rows in (identity_rows, descriptor_rows):
        for molecule_id, row in index_by_molecule(source_rows).items():
            merged.setdefault(molecule_id, {})
            merged[molecule_id].update(row)
    return [
        {**row, "molecule_id": molecule_id}
        for molecule_id, row in sorted(merged.items())
    ]


def categorize_biomedical_relevance(score: float | None) -> str:
    """Convert similarity scores into molecule-level biomedical categories."""
    if score is None:
        return "not_run"
    if score >= 0.75:
        return "high_biomedical_relevance"
    if score >= 0.55:
        return "medium_biomedical_relevance"
    if score >= 0.35:
        return "low_biomedical_relevance"
    return "not_relevant"


def applicable_evidence_rows(
    molecule_id: str,
    evidence_rows: Iterable[dict[str, str]],
) -> list[dict[str, str]]:
    """Return evidence rows for a molecule, treating blank molecule IDs as global."""
    rows = []
    for row in evidence_rows:
        text = str(row.get("text", "")).strip()
        evidence_molecule = row_molecule_id(row)
        if text and (not evidence_molecule or evidence_molecule == molecule_id):
            rows.append(row)
    return rows


def unavailable_results(
    molecule_rows: Iterable[dict[str, str]],
    model_name: str,
    *,
    model_status: str = "model_unavailable",
    metadata: dict[str, str] | None = None,
    evidence_note: str = BIOMEDICAL_MODEL_UNAVAILABLE_NOTE,
) -> list[BiomedicalEvidenceResult]:
    """Create fallback rows when embeddings cannot be loaded."""
    metadata = metadata or {}
    return [
        BiomedicalEvidenceResult(
            molecule_id=row_molecule_id(row),
            biomedical_model_name=model_name,
            biomedical_model_status=model_status,
            biomedical_evidence_status="skipped",
            biomedical_similarity_score="0.000",
            biomedical_relevance_category="not_run",
            biomedical_evidence_count="0",
            evidence_note=evidence_note,
            **metadata,
        )
        for row in molecule_rows
        if row_molecule_id(row)
    ]

def score_biomedical_evidence(
    molecule_rows: Iterable[dict[str, str]],
    evidence_rows: Iterable[dict[str, str]],
    model: BiomedicalEncoder,
    *,
    model_name: str = DEFAULT_BIOMEDICAL_MODEL,
    model_status: str = "preferred_model_used",
    evidence_note: str = BIOMEDICAL_MODEL_AVAILABLE_NOTE,
    metadata: dict[str, str] | None = None,
) -> list[BiomedicalEvidenceResult]:
    """Score biomedical evidence against molecule context and aggregate by molecule."""
    molecules = [
        (row, molecule_context_text(row))
        for row in molecule_rows
        if row_molecule_id(row)
    ]
    metadata = metadata or encoder_metadata(model, model_source=model_name)
    evidence = [row for row in evidence_rows if str(row.get("text", "")).strip()]
    if not molecules:
        return []
    if not evidence:
        return [
            BiomedicalEvidenceResult(
                molecule_id=row_molecule_id(row),
                biomedical_model_name=model_name,
                biomedical_model_status=model_status,
                biomedical_evidence_status="no_evidence",
                biomedical_relevance_category="not_run",
                evidence_note="No biomedical evidence text was available to match.",
                **metadata,
            )
            for row, _ in molecules
        ]

    molecule_texts = [text or row_molecule_id(row) for row, text in molecules]
    evidence_texts = [str(row.get("text", "")).strip() for row in evidence]
    molecule_embeddings = encode_sentences(model, molecule_texts)
    evidence_embeddings = encode_sentences(model, evidence_texts)

    results: list[BiomedicalEvidenceResult] = []
    for (molecule, _), molecule_embedding in zip(molecules, molecule_embeddings):
        molecule_id = row_molecule_id(molecule)
        evidence_matches = applicable_evidence_rows(molecule_id, evidence)
        if not evidence_matches:
            results.append(
                BiomedicalEvidenceResult(
                    molecule_id=molecule_id,
                    biomedical_model_name=model_name,
                    biomedical_model_status=model_status,
                    biomedical_evidence_status="no_match",
                    biomedical_relevance_category="not_run",
                    evidence_note="No biomedical evidence rows applied to this molecule.",
                    **metadata,
                )
            )
            continue

        best_score: float | None = None
        best_row: dict[str, str] | None = None
        evidence_embeddings_by_id = {
            id(row): embedding for row, embedding in zip(evidence, evidence_embeddings)
        }
        for evidence_row in evidence_matches:
            similarity = cosine_similarity(
                molecule_embedding,
                evidence_embeddings_by_id[id(evidence_row)],
            )
            if best_score is None or similarity > best_score:
                best_score = similarity
                best_row = evidence_row

        assert best_score is not None
        assert best_row is not None
        results.append(
            BiomedicalEvidenceResult(
                molecule_id=molecule_id,
                biomedical_model_name=model_name,
                biomedical_model_status=model_status,
                biomedical_evidence_status="available",
                biomedical_similarity_score=f"{best_score:.3f}",
                biomedical_relevance_category=(
                    categorize_biomedical_relevance(best_score)
                ),
                biomedical_evidence_count=str(len(evidence_matches)),
                top_biomedical_evidence_id=str(
                    best_row.get("evidence_id", "")
                ).strip(),
                top_biomedical_evidence_text=str(best_row.get("text", "")).strip(),
                evidence_note=evidence_note,
                **metadata,
            )
        )
    return results


def write_results(
    results: Iterable[BiomedicalEvidenceResult],
    output_path: Path,
) -> int:
    """Write biomedical evidence results and return the row count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def biomedical_evidence_csv(
    compound_context_path: Path,
    text_evidence_path: Path,
    output_path: Path,
    *,
    model: BiomedicalEncoder | None = None,
    model_name: str = DEFAULT_BIOMEDICAL_MODEL,
    identity_path: Path | None = None,
    descriptor_path: Path | None = None,
    unavailable_status: str = "model_unavailable",
    unavailable_metadata: dict[str, str] | None = None,
    unavailable_note: str = BIOMEDICAL_MODEL_UNAVAILABLE_NOTE,
    model_status: str = "preferred_model_used",
    available_note: str = BIOMEDICAL_MODEL_AVAILABLE_NOTE,
    model_metadata: dict[str, str] | None = None,
) -> int:
    """Write molecule-level biomedical evidence matching results."""
    context_rows = read_optional_csv(compound_context_path)
    identity_rows = read_optional_csv(identity_path)
    descriptor_rows = read_optional_csv(descriptor_path)
    molecule_rows = merged_molecule_rows(
        context_rows,
        identity_rows=identity_rows,
        descriptor_rows=descriptor_rows,
    )
    evidence_rows = read_input_csv(text_evidence_path)

    if model is None and unavailable_metadata is not None:
        return write_results(
            unavailable_results(
                molecule_rows,
                model_name,
                model_status=unavailable_status,
                metadata=unavailable_metadata,
                evidence_note=unavailable_note,
            ),
            output_path,
        )

    try:
        active_model = model or load_model(model_name)
    except BiomedicalModelUnavailableError:
        return write_results(
            unavailable_results(
                molecule_rows,
                model_name,
                model_status=unavailable_status,
                metadata=unavailable_metadata,
                evidence_note=unavailable_note,
            ),
            output_path,
        )

    return write_results(
        score_biomedical_evidence(
            molecule_rows,
            evidence_rows,
            active_model,
            model_name=model_name,
            model_status=model_status,
            evidence_note=available_note,
            metadata=model_metadata,
        ),
        output_path,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the biomedical evidence command-line parser."""
    parser = argparse.ArgumentParser(
        description="Match molecule context against biomedical text evidence."
    )
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--text-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--descriptors", type=Path)
    parser.add_argument(
        "--model-name",
        default=DEFAULT_BIOMEDICAL_MODEL,
        help="Cached sentence-transformer compatible biomedical model name.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run biomedical evidence matching from the command line."""
    args = build_parser().parse_args(argv)
    count = biomedical_evidence_csv(
        args.context,
        args.text_evidence,
        args.output,
        model_name=args.model_name,
        identity_path=args.identity,
        descriptor_path=args.descriptors,
    )
    print(f"Wrote {count} biomedical evidence records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
