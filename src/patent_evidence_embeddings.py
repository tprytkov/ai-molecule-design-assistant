"""Match molecule IP context against patent evidence embeddings."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from src.text_nlp import cosine_similarity, encode_sentences


DEFAULT_PATENT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PATENT_MODEL_UNAVAILABLE_NOTE = (
    "Patent embedding model unavailable in this environment; patent/IP-context "
    "evidence matching was skipped."
)
PATENT_MODEL_AVAILABLE_NOTE = (
    "Patent/IP-context evidence matched with a local embedding model. This is a "
    "research triage signal, not a legal conclusion."
)
OUTPUT_COLUMNS = (
    "molecule_id",
    "patent_model_name",
    "patent_model_status",
    "patent_evidence_status",
    "patent_similarity_score",
    "patent_relevance_category",
    "patent_evidence_count",
    "top_patent_evidence_id",
    "top_patent_evidence_text",
    "surechembl_structure_status",
    "patent_document_metadata_status",
    "evidence_note",
)


class PatentModelUnavailableError(RuntimeError):
    """Raised when the configured patent embedding model is unavailable."""


class PatentEncoder(Protocol):
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
class PatentEvidenceEmbeddingResult:
    """Molecule-level patent/IP-context evidence matching output row."""

    molecule_id: str
    patent_model_name: str = DEFAULT_PATENT_MODEL
    patent_model_status: str = "not_available"
    patent_evidence_status: str = "not_run"
    patent_similarity_score: str = "0.000"
    patent_relevance_category: str = "not_run"
    patent_evidence_count: str = "0"
    top_patent_evidence_id: str = ""
    top_patent_evidence_text: str = ""
    surechembl_structure_status: str = "not_run"
    patent_document_metadata_status: str = "not_run"
    evidence_note: str = ""


def load_model(model_name: str = DEFAULT_PATENT_MODEL) -> PatentEncoder:
    """Load a cached sentence-transformer compatible patent model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise PatentModelUnavailableError(
            "sentence-transformers is not installed in this environment."
        ) from exc

    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception as exc:
        raise PatentModelUnavailableError(
            f"Model '{model_name}' is not available in the local cache."
        ) from exc


def read_optional_csv(path: Path | None) -> list[dict[str, str]]:
    """Read optional CSV rows without failing when the file is absent."""
    if path is None or not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        return [dict(row) for row in csv.DictReader(input_file)]


def row_molecule_id(row: dict[str, str]) -> str:
    """Return a normalized molecule ID."""
    return str(row.get("molecule_id", "")).strip()


def best_status(rows: Iterable[dict[str, str]], column: str, default: str) -> str:
    """Select the most informative status from molecule-level evidence rows."""
    priority = (
        "match_found",
        "available",
        "hit",
        "structure_match_only",
        "no_match",
        "not_run",
        "not_queried",
        "lookup_error",
        "error",
        "invalid_molecule",
    )
    values = [str(row.get(column, "")).strip() for row in rows]
    values = [value for value in values if value]
    if not values:
        return default
    for status in priority:
        if status in values:
            return status
    return values[0]


def index_rows(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Group rows by molecule ID."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        molecule_id = row_molecule_id(row)
        if molecule_id:
            grouped.setdefault(molecule_id, []).append(row)
    return grouped


def molecule_ids_from_sources(
    *sources: Iterable[dict[str, str]],
) -> list[str]:
    """Collect molecule IDs from all available evidence sources."""
    ids: set[str] = set()
    for rows in sources:
        ids.update(row_molecule_id(row) for row in rows if row_molecule_id(row))
    return sorted(ids)


def context_text(
    molecule_id: str,
    *,
    surechembl_rows: Iterable[dict[str, str]] = (),
    public_rows: Iterable[dict[str, str]] = (),
    identity_rows: Iterable[dict[str, str]] = (),
    context_rows: Iterable[dict[str, str]] = (),
) -> str:
    """Build molecule IP-context text from public identity and evidence fields."""
    fields = []
    for row in list(identity_rows) + list(context_rows) + list(public_rows):
        if row_molecule_id(row) != molecule_id:
            continue
        for column in (
            "exact_public_name",
            "preferred_name",
            "iupac_name",
            "synonyms",
            "public_name",
            "source_database",
            "closest_public_compound",
            "biological_reference_summary",
            "biomedical_relevance_summary",
            "compound_description",
            "notes",
        ):
            value = str(row.get(column, "")).strip()
            if value:
                fields.append(value)
    for row in surechembl_rows:
        if row_molecule_id(row) != molecule_id:
            continue
        for column in (
            "surechembl_id",
            "compound_name",
            "patent_id",
            "patent_number",
            "patent_title",
            "patent_date",
            "source_section",
            "evidence_note",
        ):
            value = str(row.get(column, "")).strip()
            if value:
                fields.append(value)
    return " ".join(fields) or molecule_id


def evidence_text_from_row(row: dict[str, str]) -> str:
    """Build patent evidence text from several supported CSV shapes."""
    parts = []
    for column in (
        "text",
        "patent_title",
        "abstract_excerpt",
        "claims_excerpt",
        "summary_excerpt",
        "patent_id",
        "patent_number",
        "compound_name",
        "assignee",
        "inventors",
        "source_section",
        "evidence_note",
        "notes",
    ):
        value = str(row.get(column, "")).strip()
        if value:
            parts.append(value)
    return " ".join(parts)


def evidence_id_from_row(row: dict[str, str]) -> str:
    """Return a stable evidence identifier from known patent evidence fields."""
    for column in ("evidence_id", "patent_id", "patent_number", "surechembl_id"):
        value = str(row.get(column, "")).strip()
        if value:
            return value
    return ""


def patent_evidence_rows(
    surechembl_rows: Iterable[dict[str, str]],
    patent_text_rows: Iterable[dict[str, str]] = (),
) -> list[dict[str, str]]:
    """Normalize patent text rows from SureChEMBL and optional patent CSVs."""
    rows: list[dict[str, str]] = []
    for row in surechembl_rows:
        text = evidence_text_from_row(row)
        if text:
            rows.append(
                {
                    "molecule_id": row_molecule_id(row),
                    "evidence_id": evidence_id_from_row(row),
                    "text": text,
                }
            )
    for row in patent_text_rows:
        text = evidence_text_from_row(row)
        if text:
            rows.append(
                {
                    "molecule_id": row_molecule_id(row),
                    "evidence_id": evidence_id_from_row(row),
                    "text": text,
                }
            )
    return rows


def metadata_status_for_molecule(rows: Iterable[dict[str, str]]) -> str:
    """Return patent document metadata status for one molecule."""
    rows = list(rows)
    explicit = best_status(rows, "patent_metadata_status", "")
    if explicit:
        return explicit
    if any(
        str(row.get(column, "")).strip()
        for row in rows
        for column in ("patent_id", "patent_number", "patent_title", "patent_date")
    ):
        return "available"
    return "not_available"


def categorize_patent_relevance(score: float | None) -> str:
    """Convert similarity scores into patent/IP-context relevance categories."""
    if score is None:
        return "not_run"
    if score >= 0.75:
        return "high_patent_relevance"
    if score >= 0.55:
        return "medium_patent_relevance"
    if score >= 0.35:
        return "low_patent_relevance"
    return "not_relevant"


def applicable_evidence(
    molecule_id: str,
    evidence_rows: Iterable[dict[str, str]],
) -> list[dict[str, str]]:
    """Return evidence rows for a molecule, treating blank molecule IDs as global."""
    rows = []
    for row in evidence_rows:
        evidence_molecule = row_molecule_id(row)
        if not evidence_molecule or evidence_molecule == molecule_id:
            rows.append(row)
    return rows


def fallback_results(
    molecule_ids: Iterable[str],
    *,
    model_name: str,
    surechembl_by_molecule: dict[str, list[dict[str, str]]],
) -> list[PatentEvidenceEmbeddingResult]:
    """Create model-unavailable fallback rows."""
    results = []
    for molecule_id in molecule_ids:
        surechembl_rows = surechembl_by_molecule.get(molecule_id, [])
        results.append(
            PatentEvidenceEmbeddingResult(
                molecule_id=molecule_id,
                patent_model_name=model_name,
                patent_model_status="model_unavailable",
                patent_evidence_status="skipped",
                patent_similarity_score="0.000",
                patent_relevance_category="not_run",
                surechembl_structure_status=best_status(
                    surechembl_rows, "lookup_status", "not_run"
                ),
                patent_document_metadata_status=metadata_status_for_molecule(
                    surechembl_rows
                ),
                evidence_note=PATENT_MODEL_UNAVAILABLE_NOTE,
            )
        )
    return results


def score_patent_evidence(
    molecule_ids: Iterable[str],
    evidence_rows: Iterable[dict[str, str]],
    model: PatentEncoder,
    *,
    model_name: str = DEFAULT_PATENT_MODEL,
    surechembl_rows: Iterable[dict[str, str]] = (),
    public_rows: Iterable[dict[str, str]] = (),
    identity_rows: Iterable[dict[str, str]] = (),
    context_rows: Iterable[dict[str, str]] = (),
) -> list[PatentEvidenceEmbeddingResult]:
    """Score patent evidence against molecule IP context and aggregate by molecule."""
    ids = list(molecule_ids)
    evidence = [row for row in evidence_rows if str(row.get("text", "")).strip()]
    surechembl_by_molecule = index_rows(surechembl_rows)
    if not ids:
        return []
    if not evidence:
        return [
            PatentEvidenceEmbeddingResult(
                molecule_id=molecule_id,
                patent_model_name=model_name,
                patent_model_status="available",
                patent_evidence_status="no_evidence",
                surechembl_structure_status=best_status(
                    surechembl_by_molecule.get(molecule_id, []),
                    "lookup_status",
                    "not_run",
                ),
                patent_document_metadata_status=metadata_status_for_molecule(
                    surechembl_by_molecule.get(molecule_id, [])
                ),
                evidence_note="No patent/IP-context text was available to match.",
            )
            for molecule_id in ids
        ]

    molecule_texts = [
        context_text(
            molecule_id,
            surechembl_rows=surechembl_rows,
            public_rows=public_rows,
            identity_rows=identity_rows,
            context_rows=context_rows,
        )
        for molecule_id in ids
    ]
    evidence_texts = [str(row.get("text", "")).strip() for row in evidence]
    molecule_embeddings = encode_sentences(model, molecule_texts)
    evidence_embeddings = encode_sentences(model, evidence_texts)
    evidence_embeddings_by_id = {
        id(row): embedding for row, embedding in zip(evidence, evidence_embeddings)
    }

    results: list[PatentEvidenceEmbeddingResult] = []
    for molecule_id, molecule_embedding in zip(ids, molecule_embeddings):
        matches = applicable_evidence(molecule_id, evidence)
        surechembl_for_molecule = surechembl_by_molecule.get(molecule_id, [])
        if not matches:
            results.append(
                PatentEvidenceEmbeddingResult(
                    molecule_id=molecule_id,
                    patent_model_name=model_name,
                    patent_model_status="available",
                    patent_evidence_status="no_match",
                    surechembl_structure_status=best_status(
                        surechembl_for_molecule, "lookup_status", "not_run"
                    ),
                    patent_document_metadata_status=metadata_status_for_molecule(
                        surechembl_for_molecule
                    ),
                    evidence_note="No patent evidence rows applied to this molecule.",
                )
            )
            continue

        best_score: float | None = None
        best_row: dict[str, str] | None = None
        for row in matches:
            score = cosine_similarity(
                molecule_embedding,
                evidence_embeddings_by_id[id(row)],
            )
            if best_score is None or score > best_score:
                best_score = score
                best_row = row

        assert best_score is not None
        assert best_row is not None
        results.append(
            PatentEvidenceEmbeddingResult(
                molecule_id=molecule_id,
                patent_model_name=model_name,
                patent_model_status="available",
                patent_evidence_status="available",
                patent_similarity_score=f"{best_score:.3f}",
                patent_relevance_category=categorize_patent_relevance(best_score),
                patent_evidence_count=str(len(matches)),
                top_patent_evidence_id=str(best_row.get("evidence_id", "")).strip(),
                top_patent_evidence_text=str(best_row.get("text", "")).strip(),
                surechembl_structure_status=best_status(
                    surechembl_for_molecule, "lookup_status", "not_run"
                ),
                patent_document_metadata_status=metadata_status_for_molecule(
                    surechembl_for_molecule
                ),
                evidence_note=PATENT_MODEL_AVAILABLE_NOTE,
            )
        )
    return results


def write_results(
    results: Iterable[PatentEvidenceEmbeddingResult],
    output_path: Path,
) -> int:
    """Write patent evidence embedding results and return the row count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def patent_evidence_embeddings_csv(
    surechembl_path: Path,
    output_path: Path,
    *,
    public_lookup_path: Path | None = None,
    identity_path: Path | None = None,
    context_path: Path | None = None,
    patent_text_path: Path | None = None,
    model: PatentEncoder | None = None,
    model_name: str = DEFAULT_PATENT_MODEL,
) -> int:
    """Write molecule-level patent/IP-context evidence embedding results."""
    surechembl_rows = read_optional_csv(surechembl_path)
    public_rows = read_optional_csv(public_lookup_path)
    identity_rows = read_optional_csv(identity_path)
    context_rows = read_optional_csv(context_path)
    patent_text_rows = read_optional_csv(patent_text_path)
    molecule_ids = molecule_ids_from_sources(
        surechembl_rows,
        public_rows,
        identity_rows,
        context_rows,
        patent_text_rows,
    )
    surechembl_by_molecule = index_rows(surechembl_rows)
    evidence_rows = patent_evidence_rows(surechembl_rows, patent_text_rows)

    try:
        active_model = model or load_model(model_name)
    except PatentModelUnavailableError:
        return write_results(
            fallback_results(
                molecule_ids,
                model_name=model_name,
                surechembl_by_molecule=surechembl_by_molecule,
            ),
            output_path,
        )

    return write_results(
        score_patent_evidence(
            molecule_ids,
            evidence_rows,
            active_model,
            model_name=model_name,
            surechembl_rows=surechembl_rows,
            public_rows=public_rows,
            identity_rows=identity_rows,
            context_rows=context_rows,
        ),
        output_path,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the patent evidence embedding command-line parser."""
    parser = argparse.ArgumentParser(
        description="Match molecule IP context against patent evidence text."
    )
    parser.add_argument("--surechembl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-lookup", type=Path)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--patent-text", type=Path)
    parser.add_argument(
        "--model-name",
        default=DEFAULT_PATENT_MODEL,
        help="Cached sentence-transformer compatible patent model name.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run patent evidence embedding from the command line."""
    args = build_parser().parse_args(argv)
    count = patent_evidence_embeddings_csv(
        args.surechembl,
        args.output,
        public_lookup_path=args.public_lookup,
        identity_path=args.identity,
        context_path=args.context,
        patent_text_path=args.patent_text,
        model_name=args.model_name,
    )
    print(f"Wrote {count} patent/IP-context evidence records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
