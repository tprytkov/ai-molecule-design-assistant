"""Optional ChemBERTa molecular embeddings and 2D visualization coordinates."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence


DEFAULT_CHEMBERTA_MODEL = "DeepChem/ChemBERTa-77M-MLM"
MAX_UMAP_MOLECULES = 30
STANDARDIZED_COLUMNS = ("molecule_id", "canonical_smiles", "valid_smiles")
EMBEDDING_COLUMNS = (
    "molecule_id",
    "canonical_smiles",
    "valid_smiles",
    "chemberta_model",
    "embedding_available",
    "embedding_dim",
    "embedding_json",
    "error_message",
)
VISUALIZATION_COLUMNS = (
    "molecule_id",
    "x",
    "y",
    "coordinate_method",
    "prioritization_score_with_nlp",
    "novelty_flag",
    "ip_potential_category",
    "known_public_match",
    "best_reference_name",
    "tanimoto_similarity",
    "cluster_id",
)
PRIORITIZED_CHEMBERTA_COLUMNS = (
    "chemberta_status",
    "chemberta_model",
    "chemberta_embedding_available",
    "chemberta_embedding_dim",
)


class ChembertaEmbedder(Protocol):
    """Minimal interface for real or mocked ChemBERTa embedders."""

    model_name: str

    def embed(self, smiles: str) -> list[float]:
        """Return one learned embedding vector for a canonical SMILES."""


@dataclass(frozen=True)
class EmbeddingRecord:
    """One ChemBERTa embedding output row."""

    molecule_id: str
    canonical_smiles: str
    valid_smiles: bool
    chemberta_model: str
    embedding_available: bool
    embedding_dim: str = ""
    embedding_json: str = ""
    error_message: str = ""


def parse_boolean(value: object) -> bool:
    """Interpret common CSV boolean representations."""
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


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


class HuggingFaceChembertaEmbedder:
    """ChemBERTa embedder using Hugging Face transformers."""

    def __init__(self, model_name: str = DEFAULT_CHEMBERTA_MODEL) -> None:
        self.model_name = model_name
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "ChemBERTa requires transformers and torch in the active "
                "molecule-intelligence environment."
            ) from exc

        self._torch = torch
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name)
            self._model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Could not load ChemBERTa model '{model_name}'. Ensure the "
                "model is available locally or that Hugging Face access is allowed."
            ) from exc

    def embed(self, smiles: str) -> list[float]:
        """Embed one SMILES with attention-mask mean pooling."""
        encoded = self._tokenizer(
            smiles,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with self._torch.no_grad():
            output = self._model(**encoded)
        hidden = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = (summed / counts).squeeze(0)
        return [float(value) for value in pooled.detach().cpu().tolist()]


def embedding_records(
    standardized_rows: Iterable[Mapping[str, str]],
    embedder: ChembertaEmbedder,
) -> list[EmbeddingRecord]:
    """Generate ChemBERTa embeddings while retaining invalid molecule rows."""
    records: list[EmbeddingRecord] = []
    for row in standardized_rows:
        molecule_id = row.get("molecule_id", "").strip()
        canonical_smiles = row.get("canonical_smiles", "").strip()
        valid_smiles = parse_boolean(row.get("valid_smiles", ""))
        if not valid_smiles or not canonical_smiles:
            records.append(
                EmbeddingRecord(
                    molecule_id=molecule_id,
                    canonical_smiles=canonical_smiles,
                    valid_smiles=valid_smiles,
                    chemberta_model=embedder.model_name,
                    embedding_available=False,
                    error_message=(
                        row.get("error_message", "").strip()
                        or "ChemBERTa embedding skipped for invalid molecule."
                    ),
                )
            )
            continue
        try:
            embedding = embedder.embed(canonical_smiles)
        except RuntimeError as exc:
            records.append(
                EmbeddingRecord(
                    molecule_id=molecule_id,
                    canonical_smiles=canonical_smiles,
                    valid_smiles=True,
                    chemberta_model=embedder.model_name,
                    embedding_available=False,
                    error_message=str(exc),
                )
            )
            continue
        records.append(
            EmbeddingRecord(
                molecule_id=molecule_id,
                canonical_smiles=canonical_smiles,
                valid_smiles=True,
                chemberta_model=embedder.model_name,
                embedding_available=True,
                embedding_dim=str(len(embedding)),
                embedding_json=json.dumps(embedding, separators=(",", ":")),
            )
        )
    return records


def write_embeddings_csv(
    output_path: Path, records: Iterable[EmbeddingRecord]
) -> None:
    """Write ChemBERTa embedding rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=EMBEDDING_COLUMNS)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["valid_smiles"] = str(record.valid_smiles)
            row["embedding_available"] = str(record.embedding_available)
            writer.writerow(row)


def chemberta_embeddings_csv(
    standardized_path: Path,
    output_path: Path,
    *,
    model_name: str = DEFAULT_CHEMBERTA_MODEL,
    embedder: ChembertaEmbedder | None = None,
) -> int:
    """Generate ChemBERTa embeddings and return row count."""
    active_embedder = embedder or HuggingFaceChembertaEmbedder(model_name)
    rows = read_csv_with_columns(
        standardized_path, STANDARDIZED_COLUMNS, "Standardized"
    )
    records = embedding_records(rows, active_embedder)
    write_embeddings_csv(output_path, records)
    return len(records)


def load_available_embeddings(
    embedding_rows: Iterable[Mapping[str, str]],
) -> dict[str, list[float]]:
    """Parse available embedding vectors by molecule ID."""
    embeddings: dict[str, list[float]] = {}
    for row in embedding_rows:
        if not parse_boolean(row.get("embedding_available", "")):
            continue
        molecule_id = row.get("molecule_id", "").strip()
        try:
            values = json.loads(row.get("embedding_json", ""))
        except json.JSONDecodeError:
            continue
        if isinstance(values, list) and values:
            embeddings[molecule_id] = [float(value) for value in values]
    return embeddings


def pca_coordinates(vectors: list[list[float]]) -> list[tuple[float, float]]:
    """Compute deterministic 2D PCA coordinates with a small numpy dependency."""
    if not vectors:
        return []
    if len(vectors) == 1:
        return [(0.0, 0.0)]
    try:
        import numpy as np
    except ImportError:
        return [
            (
                float(vector[0]) if vector else 0.0,
                float(vector[1]) if len(vector) > 1 else 0.0,
            )
            for vector in vectors
        ]
    matrix = np.array(vectors, dtype=float)
    centered = matrix - matrix.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2].T
    coords = centered @ components
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])
    return [(float(x), float(y)) for x, y in coords[:, :2]]


def reduce_embeddings(vectors: list[list[float]]) -> tuple[str, list[tuple[float, float]]]:
    """Compute UMAP coordinates when available, otherwise PCA."""
    if 3 <= len(vectors) <= MAX_UMAP_MOLECULES:
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(
                n_components=2,
                random_state=42,
                n_neighbors=min(15, len(vectors) - 1),
            )
            coords = reducer.fit_transform(vectors)
            return "umap", [(float(x), float(y)) for x, y in coords]
        except Exception:
            pass
    return "pca", pca_coordinates(vectors)


def cluster_id(x: float | None, y: float | None) -> str:
    """Assign a simple visualization cluster ID from 2D coordinate quadrants."""
    if x is None or y is None:
        return "-1"
    if x >= 0 and y >= 0:
        return "0"
    if x < 0 <= y:
        return "1"
    if x < 0 and y < 0:
        return "2"
    return "3"


def visualization_coordinates_csv(
    embeddings_path: Path,
    prioritized_path: Path | None,
    output_path: Path,
) -> int:
    """Create 2D coordinates, optionally enriched with prioritization data."""
    embedding_rows = read_csv_with_columns(
        embeddings_path, EMBEDDING_COLUMNS, "ChemBERTa embeddings"
    )
    if prioritized_path is not None and prioritized_path.exists():
        prioritized_rows = read_csv_with_columns(
            prioritized_path,
            (
                "molecule_id",
                "prioritization_score_with_nlp",
                "novelty_flag",
                "ip_potential_category",
                "known_public_match",
                "best_reference_name",
                "tanimoto_similarity",
            ),
            "Prioritized",
        )
    else:
        prioritized_rows = [
            {"molecule_id": row.get("molecule_id", "")}
            for row in embedding_rows
        ]
    embeddings = load_available_embeddings(embedding_rows)
    ordered_ids = [
        row["molecule_id"].strip()
        for row in prioritized_rows
        if row.get("molecule_id", "").strip() in embeddings
    ]
    method = "not_available"
    coords_by_id: dict[str, tuple[float, float]] = {}
    if ordered_ids:
        method, coords = reduce_embeddings([embeddings[molecule_id] for molecule_id in ordered_ids])
        coords_by_id = dict(zip(ordered_ids, coords))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=VISUALIZATION_COLUMNS)
        writer.writeheader()
        for row in prioritized_rows:
            molecule_id = row.get("molecule_id", "").strip()
            coords = coords_by_id.get(molecule_id)
            x = coords[0] if coords else None
            y = coords[1] if coords else None
            writer.writerow(
                {
                    "molecule_id": molecule_id,
                    "x": f"{x:.6f}" if x is not None else "",
                    "y": f"{y:.6f}" if y is not None else "",
                    "coordinate_method": method if coords else "not_available",
                    "prioritization_score_with_nlp": row.get(
                        "prioritization_score_with_nlp", ""
                    ),
                    "novelty_flag": row.get("novelty_flag", ""),
                    "ip_potential_category": row.get("ip_potential_category", ""),
                    "known_public_match": row.get("known_public_match", ""),
                    "best_reference_name": row.get("best_reference_name", ""),
                    "tanimoto_similarity": row.get("tanimoto_similarity", ""),
                    "cluster_id": cluster_id(x, y),
                }
            )
    return len(prioritized_rows)


def merge_chemberta_into_prioritized(
    prioritized_path: Path, embeddings_path: Path
) -> int:
    """Append compact ChemBERTa availability fields to prioritized output."""
    prioritized_rows = read_csv_with_columns(
        prioritized_path, ("molecule_id",), "Prioritized"
    )
    embedding_rows = read_csv_with_columns(
        embeddings_path, EMBEDDING_COLUMNS, "ChemBERTa embeddings"
    )
    embedding_index = {
        row.get("molecule_id", "").strip(): row for row in embedding_rows
    }
    fieldnames = list(prioritized_rows[0].keys()) if prioritized_rows else ["molecule_id"]
    for column in PRIORITIZED_CHEMBERTA_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    with prioritized_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in prioritized_rows:
            embedding = embedding_index.get(row.get("molecule_id", "").strip(), {})
            row["chemberta_status"] = (
                "available"
                if parse_boolean(embedding.get("embedding_available", ""))
                else "not_available"
            )
            row["chemberta_model"] = embedding.get("chemberta_model", "")
            row["chemberta_embedding_available"] = embedding.get(
                "embedding_available", "False"
            )
            row["chemberta_embedding_dim"] = embedding.get("embedding_dim", "")
            writer.writerow(row)
    return len(prioritized_rows)


def build_parser() -> argparse.ArgumentParser:
    """Build the ChemBERTa embedding CLI parser."""
    parser = argparse.ArgumentParser(
        description="Generate optional ChemBERTa molecular embeddings."
    )
    parser.add_argument("--standardized", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_CHEMBERTA_MODEL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ChemBERTa embedding CLI."""
    args = build_parser().parse_args(argv)
    try:
        row_count = chemberta_embeddings_csv(
            args.standardized,
            args.output,
            model_name=args.model,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(f"Wrote {row_count} ChemBERTa embedding records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
