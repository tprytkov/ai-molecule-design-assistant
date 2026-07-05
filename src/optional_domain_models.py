"""Optional local domain-model loading and benchmarking utilities."""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

from src.text_nlp import cosine_similarity
from src.model_source_status import (
    HUGGINGFACE_CACHE_DIR,
    configure_huggingface_cache_env,
    ensure_app_data_dirs,
)

configure_huggingface_cache_env()

ALLOW_LOCAL_MODEL_DOWNLOADS_ENV = "ALLOW_LOCAL_MODEL_DOWNLOADS"
FALLBACK_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CLOUD_SAFE_FALLBACK_LABEL = "Cloud-safe fallback"
CUSTOM_MODEL_LABEL = "Custom Hugging Face model ID"
BIOBERT_LABEL = "BioBERT"
PUBMEDBERT_LABEL = "PubMedBERT"
PAECTER_LABEL = "PaECTER"
PATENT_SBERTA_LABEL = "PatentSBERTa"
BIOBERT_MODEL_ID = "dmis-lab/biobert-base-cased-v1.1"
PUBMEDBERT_MODEL_ID = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
PATENT_SBERTA_MODEL_ID = "AI-Growth-Lab/PatentSBERTa"
PAECTER_MODEL_ID: str | None = None
TRANSFORMERS_BACKEND = "transformers"
SENTENCE_TRANSFORMERS_BACKEND = "sentence-transformers"
MEAN_POOLING = "mean_pooling"
SENTENCE_TRANSFORMERS_POOLING = "model_default"
BENCHMARK_COLUMNS = (
    "model_name",
    "model_id",
    "model_type",
    "model_status",
    "embedding_backend",
    "pooling_method",
    "runtime_seconds",
    "embedding_dimension",
    "mean_similarity_score",
    "top_evidence_example",
    "error_message",
)
BIOMEDICAL_MODEL_OPTIONS = (
    CLOUD_SAFE_FALLBACK_LABEL,
    BIOBERT_LABEL,
    PUBMEDBERT_LABEL,
    CUSTOM_MODEL_LABEL,
)
PATENT_MODEL_OPTIONS = (
    CLOUD_SAFE_FALLBACK_LABEL,
    PAECTER_LABEL,
    PATENT_SBERTA_LABEL,
    CUSTOM_MODEL_LABEL,
)


class DomainModelUnavailableError(RuntimeError):
    """Raised when an optional local domain model cannot be loaded safely."""

    def __init__(self, message: str, *, status: str = "model_unavailable") -> None:
        super().__init__(message)
        self.status = status


class Encoder(Protocol):
    """Minimal encoder interface used by scoring modules."""

    embedding_backend: str
    pooling_method: str
    model_source: str

    def encode(
        self,
        sentences: Sequence[str],
        *,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> object:
        """Encode text."""


@dataclass(frozen=True)
class OptionalModelSelection:
    """Resolved local-model selector state."""

    label: str
    model_id: str
    model_type: str
    embedding_backend: str
    pooling_method: str
    requires_download_gate: bool
    error_message: str = ""


class SentenceTransformerEncoder:
    """Small metadata wrapper around a SentenceTransformer model."""

    embedding_backend = SENTENCE_TRANSFORMERS_BACKEND
    pooling_method = SENTENCE_TRANSFORMERS_POOLING

    def __init__(self, model: object, model_source: str) -> None:
        self.model = model
        self.model_source = model_source

    def encode(self, sentences: Sequence[str], **kwargs: object) -> object:
        return self.model.encode(sentences, **kwargs)


class TransformersMeanPoolingEncoder:
    """Transformers AutoModel encoder using attention-mask mean pooling."""

    embedding_backend = TRANSFORMERS_BACKEND
    pooling_method = MEAN_POOLING

    def __init__(
        self,
        model_id: str,
        *,
        local_files_only: bool = True,
        cache_dir: Path | None = None,
    ) -> None:
        self.model_source = model_id
        cache_dir = cache_dir or HUGGINGFACE_CACHE_DIR
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise DomainModelUnavailableError(
                "transformers and torch are required for transformer mean pooling."
            ) from exc
        try:
            self._torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                local_files_only=local_files_only,
                cache_dir=str(cache_dir),
            )
            self.model = AutoModel.from_pretrained(
                model_id,
                local_files_only=local_files_only,
                cache_dir=str(cache_dir),
            )
            self.model.eval()
        except Exception as exc:
            raise DomainModelUnavailableError(
                f"Model '{model_id}' is not available in the local cache."
            ) from exc

    def encode(
        self,
        sentences: Sequence[str],
        *,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        del convert_to_numpy, show_progress_bar
        texts = list(sentences)
        if not texts:
            return []
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with self._torch.no_grad():
            output = self.model(**encoded)
            token_embeddings = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            summed = (token_embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            embeddings = summed / counts
            if normalize_embeddings:
                embeddings = self._torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().tolist()


def allow_local_model_downloads() -> bool:
    """Return whether explicit local model loading/downloading is enabled."""
    return os.environ.get(ALLOW_LOCAL_MODEL_DOWNLOADS_ENV) == "1"


def normalize_option(value: str, options: Sequence[str]) -> str:
    """Return a known selector option, defaulting to cloud-safe fallback."""
    text = str(value or "").strip()
    return text if text in options else CLOUD_SAFE_FALLBACK_LABEL


def resolve_model_selection(
    *,
    model_type: str,
    option: str,
    custom_model_id: str = "",
    fallback_model_id: str,
) -> OptionalModelSelection:
    """Resolve UI selector state into model loading metadata."""
    if model_type == "biomedical":
        option = normalize_option(option, BIOMEDICAL_MODEL_OPTIONS)
        if option == BIOBERT_LABEL:
            return OptionalModelSelection(option, BIOBERT_MODEL_ID, model_type, TRANSFORMERS_BACKEND, MEAN_POOLING, True)
        if option == PUBMEDBERT_LABEL:
            return OptionalModelSelection(option, PUBMEDBERT_MODEL_ID, model_type, TRANSFORMERS_BACKEND, MEAN_POOLING, True)
    elif model_type == "patent":
        option = normalize_option(option, PATENT_MODEL_OPTIONS)
        if option == PATENT_SBERTA_LABEL:
            return OptionalModelSelection(option, PATENT_SBERTA_MODEL_ID, model_type, SENTENCE_TRANSFORMERS_BACKEND, SENTENCE_TRANSFORMERS_POOLING, True)
        if option == PAECTER_LABEL:
            return OptionalModelSelection(
                option,
                "",
                model_type,
                TRANSFORMERS_BACKEND,
                MEAN_POOLING,
                True,
                "PaECTER model ID was not verified; enter a custom Hugging Face model ID to test it locally.",
            )
    if option == CUSTOM_MODEL_LABEL:
        model_id = str(custom_model_id or "").strip()
        if not model_id:
            return OptionalModelSelection(
                option,
                "",
                model_type,
                TRANSFORMERS_BACKEND,
                MEAN_POOLING,
                True,
                "Custom Hugging Face model ID is required.",
            )
        return OptionalModelSelection(option, model_id, model_type, TRANSFORMERS_BACKEND, MEAN_POOLING, True)
    return OptionalModelSelection(
        CLOUD_SAFE_FALLBACK_LABEL,
        fallback_model_id,
        model_type,
        SENTENCE_TRANSFORMERS_BACKEND,
        SENTENCE_TRANSFORMERS_POOLING,
        False,
    )


def load_sentence_transformer(model_id: str) -> Encoder:
    """Load a cached sentence-transformer model and attach metadata."""
    ensure_app_data_dirs()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise DomainModelUnavailableError(
            "sentence-transformers is not installed in this environment."
        ) from exc
    try:
        return SentenceTransformerEncoder(
            SentenceTransformer(
                model_id,
                cache_folder=str(HUGGINGFACE_CACHE_DIR),
                local_files_only=True,
            ),
            model_id,
        )
    except Exception as exc:
        raise DomainModelUnavailableError(
            f"Model '{model_id}' is not available in the local cache."
        ) from exc


def load_optional_model(selection: OptionalModelSelection) -> Encoder:
    """Load a selected model only when the local download/cache gate allows it."""
    if selection.error_message:
        raise DomainModelUnavailableError(selection.error_message)
    if selection.requires_download_gate and not allow_local_model_downloads():
        raise DomainModelUnavailableError(
            "Set ALLOW_LOCAL_MODEL_DOWNLOADS=1 to enable local domain-model testing.",
            status="downloads_disabled",
        )
    if selection.embedding_backend == SENTENCE_TRANSFORMERS_BACKEND:
        return load_sentence_transformer(selection.model_id)
    return TransformersMeanPoolingEncoder(
        selection.model_id,
        local_files_only=True,
        cache_dir=HUGGINGFACE_CACHE_DIR,
    )


def encoder_metadata(model: object, *, model_source: str = "") -> dict[str, str]:
    """Return optional embedding metadata from an encoder object."""
    return {
        "embedding_backend": str(getattr(model, "embedding_backend", "") or ""),
        "pooling_method": str(getattr(model, "pooling_method", "") or ""),
        "model_source": str(getattr(model, "model_source", model_source) or model_source),
    }


def benchmark_rows(
    selections: Iterable[OptionalModelSelection],
    *,
    query_texts: Sequence[str],
    evidence_texts: Sequence[str],
    loader: Callable[[OptionalModelSelection], Encoder] = load_optional_model,
) -> list[dict[str, str]]:
    """Benchmark selected models on a tiny local demo evidence set."""
    rows: list[dict[str, str]] = []
    for selection in selections:
        start = time.perf_counter()
        row = {
            "model_name": selection.label,
            "model_id": selection.model_id,
            "model_type": selection.model_type,
            "model_status": "not_run",
            "embedding_backend": selection.embedding_backend,
            "pooling_method": selection.pooling_method,
            "runtime_seconds": "0.000",
            "embedding_dimension": "0",
            "mean_similarity_score": "0.000",
            "top_evidence_example": "",
            "error_message": selection.error_message,
        }
        try:
            model = loader(selection)
            queries = model.encode(query_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            evidence = model.encode(evidence_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            query_vectors = [[float(value) for value in vector] for vector in queries]
            evidence_vectors = [[float(value) for value in vector] for vector in evidence]
            scores = []
            top_score: float | None = None
            top_text = ""
            for query in query_vectors:
                for evidence_text, evidence_vector in zip(evidence_texts, evidence_vectors):
                    score = cosine_similarity(query, evidence_vector)
                    scores.append(score)
                    if top_score is None or score > top_score:
                        top_score = score
                        top_text = evidence_text
            metadata = encoder_metadata(model, model_source=selection.model_id)
            row.update(metadata)
            row["model_status"] = "available"
            row["embedding_dimension"] = str(len(query_vectors[0]) if query_vectors else 0)
            row["mean_similarity_score"] = f"{(sum(scores) / len(scores)) if scores else 0.0:.3f}"
            row["top_evidence_example"] = top_text
            row["error_message"] = ""
        except DomainModelUnavailableError as exc:
            row["model_status"] = exc.status
            row["error_message"] = str(exc)
        except Exception as exc:
            row["model_status"] = "model_unavailable"
            row["error_message"] = f"{type(exc).__name__}: {exc}"
        row["runtime_seconds"] = f"{time.perf_counter() - start:.3f}"
        rows.append(row)
    return rows


def write_benchmark_csv(rows: Iterable[dict[str, str]], output_path: Path) -> int:
    """Write compact optional-model benchmark rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=BENCHMARK_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    return len(materialized)


def benchmark_optional_models(
    *,
    biomedical_selection: OptionalModelSelection,
    patent_selection: OptionalModelSelection,
    output_dir: Path,
    loader: Callable[[OptionalModelSelection], Encoder] = load_optional_model,
) -> dict[str, object]:
    """Run tiny local benchmarks and write biomedical/patent CSVs."""
    biomedical_rows = benchmark_rows(
        [biomedical_selection],
        query_texts=("candidate kinase inhibitor with polar heteroatoms",),
        evidence_texts=(
            "Public biomedical evidence mentions kinase modulation in cell assays.",
            "General formulation evidence without biological context.",
        ),
        loader=loader,
    )
    patent_rows = benchmark_rows(
        [patent_selection],
        query_texts=("candidate compound with patent-associated kinase chemistry",),
        evidence_texts=(
            "Patent document describes substituted heterocycles for kinase-related applications.",
            "Unrelated manufacturing equipment patent text.",
        ),
        loader=loader,
    )
    biomedical_path = output_dir / "biomedical_model_benchmark.csv"
    patent_path = output_dir / "patent_model_benchmark.csv"
    write_benchmark_csv(biomedical_rows, biomedical_path)
    write_benchmark_csv(patent_rows, patent_path)
    return {
        "biomedical_rows": biomedical_rows,
        "patent_rows": patent_rows,
        "biomedical_path": biomedical_path,
        "patent_path": patent_path,
    }
