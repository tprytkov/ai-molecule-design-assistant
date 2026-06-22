"""Score public-safe text evidence against local relevance queries."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHEMICAL_CONTEXT_QUERY = (
    "small molecule chemical structure public evidence reference context "
    "research prioritization"
)
BIOACTIVITY_QUERY = (
    "small molecule bioactivity evidence receptor modulation assay results "
    "drug discovery relevance"
)
NOVELTY_QUERY = (
    "chemical novelty structural differentiation analog comparison generated "
    "molecule prioritization uniqueness"
)
RELEVANCE_QUERIES = (
    CHEMICAL_CONTEXT_QUERY,
    BIOACTIVITY_QUERY,
    NOVELTY_QUERY,
)

INPUT_COLUMNS = (
    "evidence_id",
    "molecule_id",
    "source_type",
    "title",
    "text",
)
OUTPUT_COLUMNS = (
    "molecule_id",
    "evidence_id",
    "molecule_text",
    "evidence_text",
    "similarity_score",
    "nlp_status",
    "source_type",
    "title",
    "model_name",
    "chemical_context_relevance_score",
    "bioactivity_relevance_score",
    "novelty_relevance_score",
    "max_relevance_score",
    "nlp_relevance_category",
    "nlp_notes",
)


class SentenceEncoder(Protocol):
    """Minimal interface required from a sentence-transformer model."""

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
class TextRelevanceResult:
    """NLP relevance scores for one evidence record."""

    molecule_id: str
    evidence_id: str
    molecule_text: str = ""
    evidence_text: str = ""
    similarity_score: str = ""
    nlp_status: str = "not_available"
    source_type: str = ""
    title: str = ""
    model_name: str = MODEL_NAME
    chemical_context_relevance_score: str = ""
    bioactivity_relevance_score: str = ""
    novelty_relevance_score: str = ""
    max_relevance_score: str = ""
    nlp_relevance_category: str = "not_relevant"
    nlp_notes: str = ""


def categorize_relevance(score: float | None) -> str:
    """Assign the requested NLP relevance category."""
    if score is None:
        return "not_relevant"
    if score >= 0.75:
        return "high_nlp_relevance"
    if score >= 0.55:
        return "medium_nlp_relevance"
    if score >= 0.35:
        return "low_nlp_relevance"
    return "not_relevant"


def cosine_similarity(
    first: Sequence[float], second: Sequence[float]
) -> float:
    """Calculate cosine similarity without an additional dependency."""
    if len(first) != len(second) or not first:
        raise ValueError("Embedding vectors must be nonempty and equal length.")

    dot_product = sum(a * b for a, b in zip(first, second))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        raise ValueError("Embedding vectors must have nonzero magnitude.")
    return dot_product / (first_norm * second_norm)


def load_model(model_name: str = MODEL_NAME) -> SentenceEncoder:
    """Load a sentence-transformer model from the local environment/cache."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Update the "
            "molecule-intelligence environment from environment.yml."
        ) from exc

    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Model '{model_name}' is not available in the local cache. "
            "The NLP CLI requires a previously cached model for offline use."
        ) from exc


def encode_sentences(
    model: SentenceEncoder, sentences: Sequence[str]
) -> list[list[float]]:
    """Encode text and convert model output to plain float vectors."""
    embeddings = model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [
        [float(value) for value in embedding]
        for embedding in embeddings  # type: ignore[union-attr]
    ]


def missing_text_result(
    row: dict[str, str], model_name: str = MODEL_NAME
) -> TextRelevanceResult:
    """Create a retained output row for missing evidence text."""
    return TextRelevanceResult(
        molecule_id=row.get("molecule_id", "").strip(),
        evidence_id=row.get("evidence_id", "").strip(),
        molecule_text="",
        evidence_text=row.get("text", "").strip(),
        nlp_status="not_available",
        source_type=row.get("source_type", "").strip(),
        title=row.get("title", "").strip(),
        model_name=model_name,
        nlp_notes="Evidence text is missing; relevance scores unavailable.",
    )


def context_text(context: dict[str, str]) -> str:
    """Build grounded NLP text from available compound-context fields."""
    parts = [
        context.get("exact_public_name", "").strip(),
        context.get("preferred_name", "").strip(),
        context.get("iupac_name", "").strip(),
        context.get("synonyms", "").strip(),
        context.get("closest_public_compound", "").strip(),
        context.get("reported_targets", "").strip(),
        context.get("biological_reference_summary", "").strip(),
        context.get("biomedical_relevance_summary", "").strip(),
        context.get("compound_description", "").strip(),
        context.get("scaffold_family", "").strip(),
        context.get("notes", "").strip(),
    ]
    return " ".join(part for part in parts if part)


def index_context_rows(
    rows: Iterable[dict[str, str]],
) -> dict[str, dict[str, str]]:
    """Index context rows by molecule ID."""
    return {
        row.get("molecule_id", "").strip(): row
        for row in rows
        if row.get("molecule_id", "").strip()
    }


def scoring_text(
    row: dict[str, str],
    context_by_molecule: dict[str, dict[str, str]] | None = None,
) -> str:
    """Combine evidence with grounded context when it is available."""
    evidence_text = row.get("text", "").strip()
    molecule_id = row.get("molecule_id", "").strip()
    context = (context_by_molecule or {}).get(molecule_id)
    grounded_context = context_text(context) if context else ""
    return " ".join(part for part in (evidence_text, grounded_context) if part)


def score_rows(
    rows: Iterable[dict[str, str]],
    model: SentenceEncoder,
    model_name: str = MODEL_NAME,
    context_by_molecule: dict[str, dict[str, str]] | None = None,
) -> list[TextRelevanceResult]:
    """Score evidence rows against the three fixed relevance queries."""
    row_list = list(rows)
    query_embeddings = encode_sentences(model, RELEVANCE_QUERIES)
    results: list[TextRelevanceResult | None] = [None] * len(row_list)

    valid_indices: list[int] = []
    texts: list[str] = []
    for index, row in enumerate(row_list):
        text = scoring_text(row, context_by_molecule)
        if not text:
            results[index] = missing_text_result(row, model_name)
            continue
        valid_indices.append(index)
        texts.append(text)

    text_embeddings = encode_sentences(model, texts) if texts else []
    for index, text_embedding in zip(valid_indices, text_embeddings):
        row = row_list[index]
        scores = [
            cosine_similarity(text_embedding, query_embedding)
            for query_embedding in query_embeddings
        ]
        max_score = max(scores)
        results[index] = TextRelevanceResult(
            molecule_id=row.get("molecule_id", "").strip(),
            evidence_id=row.get("evidence_id", "").strip(),
            molecule_text=(
                context_text(
                    (context_by_molecule or {}).get(
                        row.get("molecule_id", "").strip(), {}
                    )
                )
            ),
            evidence_text=row.get("text", "").strip(),
            similarity_score=f"{max_score:.3f}",
            nlp_status="available",
            source_type=row.get("source_type", "").strip(),
            title=row.get("title", "").strip(),
            model_name=model_name,
            chemical_context_relevance_score=f"{scores[0]:.3f}",
            bioactivity_relevance_score=f"{scores[1]:.3f}",
            novelty_relevance_score=f"{scores[2]:.3f}",
            max_relevance_score=f"{max_score:.3f}",
            nlp_relevance_category=categorize_relevance(max_score),
            nlp_notes=(
                "Local semantic-similarity signal from public-safe text evidence."
            ),
        )

    return [result for result in results if result is not None]


def read_input_csv(input_path: Path) -> list[dict[str, str]]:
    """Read evidence rows and validate the required columns."""
    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = set(reader.fieldnames or [])
        if set(INPUT_COLUMNS) <= fieldnames:
            return [dict(row) for row in reader]
        neutral_columns = {"evidence_id", "text", "source", "target_family", "notes"}
        if not neutral_columns <= fieldnames:
            missing = ", ".join(sorted(neutral_columns - fieldnames))
            raise ValueError(f"Input CSV is missing required columns: {missing}")
        return [
            {
                "evidence_id": row.get("evidence_id", ""),
                "molecule_id": "",
                "source_type": row.get("source", ""),
                "title": row.get("target_family", ""),
                "text": " ".join(
                    part
                    for part in (
                        row.get("text", "").strip(),
                        row.get("notes", "").strip(),
                    )
                    if part
                ),
            }
            for row in reader
        ]


def merge_molecule_context(
    context_rows: Iterable[dict[str, str]],
    molecule_rows: Iterable[dict[str, str]] = (),
    descriptor_rows: Iterable[dict[str, str]] = (),
) -> list[dict[str, str]]:
    """Merge context and generated/local fields by molecule ID."""
    merged = index_context_rows(context_rows)
    for row in molecule_rows:
        molecule_id = row.get("molecule_id", "").strip()
        if molecule_id:
            merged.setdefault(molecule_id, {}).update(
                {
                    key: value
                    for key, value in row.items()
                    if value is not None and str(value).strip()
                }
            )
    validity = {
        row.get("molecule_id", "").strip(): row.get("valid_smiles", "")
        for row in descriptor_rows
        if row.get("molecule_id", "").strip()
    }
    return [
        {**row, "valid_smiles": validity.get(molecule_id, row.get("valid_smiles", ""))}
        for molecule_id, row in merged.items()
    ]


def match_molecules_to_evidence(
    molecule_rows: Iterable[dict[str, str]],
    evidence_rows: Iterable[dict[str, str]],
    model: SentenceEncoder,
    model_name: str = MODEL_NAME,
) -> list[TextRelevanceResult]:
    """Score valid molecule context against applicable local evidence rows."""
    molecules = [
        row
        for row in molecule_rows
        if str(row.get("valid_smiles", "")).strip().lower()
        in {"true", "1", "yes", "y"}
        and context_text(row)
    ]
    evidence = [
        row for row in evidence_rows if row.get("text", "").strip()
    ]
    if not molecules or not evidence:
        return []

    molecule_texts = [context_text(row) for row in molecules]
    evidence_texts = [row["text"].strip() for row in evidence]
    molecule_embeddings = encode_sentences(model, molecule_texts)
    evidence_embeddings = encode_sentences(model, evidence_texts)
    query_embeddings = encode_sentences(model, RELEVANCE_QUERIES)

    results: list[TextRelevanceResult] = []
    for molecule, molecule_text_value, molecule_embedding in zip(
        molecules, molecule_texts, molecule_embeddings
    ):
        molecule_id = molecule.get("molecule_id", "").strip()
        for evidence_row, evidence_text_value, evidence_embedding in zip(
            evidence, evidence_texts, evidence_embeddings
        ):
            evidence_molecule = evidence_row.get("molecule_id", "").strip()
            if evidence_molecule and evidence_molecule != molecule_id:
                continue
            similarity = cosine_similarity(molecule_embedding, evidence_embedding)
            query_scores = [
                cosine_similarity(evidence_embedding, query_embedding)
                for query_embedding in query_embeddings
            ]
            max_score = max(query_scores)
            results.append(
                TextRelevanceResult(
                    molecule_id=molecule_id,
                    evidence_id=evidence_row.get("evidence_id", "").strip(),
                    molecule_text=molecule_text_value,
                    evidence_text=evidence_text_value,
                    similarity_score=f"{similarity:.3f}",
                    nlp_status="available",
                    source_type=evidence_row.get("source_type", "").strip(),
                    title=evidence_row.get("title", "").strip(),
                    model_name=model_name,
                    chemical_context_relevance_score=f"{query_scores[0]:.3f}",
                    bioactivity_relevance_score=f"{query_scores[1]:.3f}",
                    novelty_relevance_score=f"{query_scores[2]:.3f}",
                    max_relevance_score=f"{max_score:.3f}",
                    nlp_relevance_category=categorize_relevance(max_score),
                    nlp_notes=(
                        "Local semantic similarity between grounded molecule "
                        "context and public-safe evidence text."
                    ),
                )
            )
    return results


def write_output_csv(
    output_path: Path, records: Iterable[TextRelevanceResult]
) -> None:
    """Write text relevance results to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def text_nlp_csv(
    input_path: Path,
    output_path: Path,
    model: SentenceEncoder | None = None,
    model_name: str = MODEL_NAME,
    context_path: Path | None = None,
    molecule_path: Path | None = None,
    descriptor_path: Path | None = None,
    identity_path: Path | None = None,
) -> int:
    """Score an evidence CSV and return the output row count."""
    active_model = model or load_model(model_name)
    evidence_rows = read_input_csv(input_path)
    if context_path is not None and context_path.exists():
        context_rows = read_csv_rows(context_path)
        molecule_rows = (
            read_csv_rows(molecule_path)
            if molecule_path is not None and molecule_path.exists()
            else []
        )
        descriptor_rows = (
            read_csv_rows(descriptor_path)
            if descriptor_path is not None and descriptor_path.exists()
            else []
        )
        identity_rows = (
            read_csv_rows(identity_path)
            if identity_path is not None and identity_path.exists()
            else []
        )
        identity_by_id = index_context_rows(identity_rows)
        context_rows = [
            {**row, **identity_by_id.get(row.get("molecule_id", "").strip(), {})}
            for row in context_rows
        ]
        results = match_molecules_to_evidence(
            merge_molecule_context(
                context_rows, molecule_rows, descriptor_rows
            ),
            evidence_rows,
            active_model,
            model_name,
        )
    else:
        results = score_rows(evidence_rows, active_model, model_name)
    write_output_csv(output_path, results)
    return len(results)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read an auxiliary CSV without imposing an evidence schema."""
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        return [dict(row) for row in csv.DictReader(input_file)]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Score local text evidence with a sentence transformer."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input CSV containing public-safe evidence text.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination CSV for NLP relevance scores.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    model: SentenceEncoder | None = None,
) -> int:
    """Run the text relevance command-line interface."""
    args = build_parser().parse_args(argv)
    try:
        row_count = text_nlp_csv(args.input, args.output, model=model)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print(f"Wrote {row_count} text NLP records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
